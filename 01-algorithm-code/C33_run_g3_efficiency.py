#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC 提分实验 G3：效率指标实测（★ 补齐 GPU 那 3 项）
===============================================================================

对应需求文档《04_需用户本人操作事项反馈.md》§5「★ 效率指标实测」。

---------------------------------------------------------------------------
它解决什么问题
---------------------------------------------------------------------------

★ 项二的 7 个效率指标里，本机（CPU-only torch）只能实测 4 个：

    已完成（本机实测）          未完成（必须 GPU）
    ─────────────────           ──────────────────
    参数量        56,812,813      GPU 单图推理延迟
    对照 ResNeXt  25,028,904      GPU 吞吐（img/s）
    FLOPs         14.985 G        GPU 峰值显存
    CPU 单图      140.0 ms

方案 §四(二) 的 L4「未实测效率指标」就是这 3 项。本脚本在 GPU 机器上
补齐它们。

---------------------------------------------------------------------------
测量口径（必须与已发布的 FLOPs 一致）
---------------------------------------------------------------------------

已发布的 FLOPs = 14,985,420,662（14.985 G），约定为：

    - 输入形状 1 × 3 × 224 × 224，**batch = 1**
    - 1 MAC = 2 FLOPs
    - 只统计 Conv2d 与 Linear（BN / 激活 / 池化不计）
    - MACs = 7,492,310,331

**本脚本沿用完全相同的约定**，否则新旧数字不可比。

延迟/吞吐测量约定：
    - `model.eval()` + `torch.no_grad()`
    - 前 10 次作为 warmup，不计入
    - 至少 50 次计时取中位数与四分位（**不用均值**，避免被偶发抖动拉偏）
    - `torch.cuda.synchronize()` 夹住每次前向
    - 显存用 `torch.cuda.max_memory_allocated()`

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------

    # 本机（无 GPU）自检，会明确报告"GPU 项无法测量"：
    python g3_efficiency.py --dry-run

    # GPU 机器：
    python g3_efficiency.py --device cuda:0 \
        --batch-sizes 1,16 \
        --out-dir /home/featurize/work/dmsa_efficiency

产物直接对齐方案里的「效率对比表」，可粘贴。
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    EXPECTED_CLASSES,
    PKG_VERSION,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    ProgressReporter,
    base_parser,
    find_baseline_dir,
    load_module,
    make_tee_sink,
    resolve_out_dir,
    sha256_file,
    write_csv,
    write_json,
)

# 已发布数字（来自 DMSA_v24/_arch_out/arch_probe.json）——用于交叉校验
# ⚠️ 勘误（2026-09-23）：arch_probe.json 落盘的 flops 字段曾写 14,985,420,662，
#    但 2 × MACs = 14,984,620,662，相差 800,000，属落盘笔误。
#    这里用**勘误后**的值，并由下方 _assert_flops_convention() 当场自证。
PUBLISHED_PARAMS = 56_812_813
PUBLISHED_MACS = 7_492_310_331
PUBLISHED_FLOPS = 14_984_620_662          # == 2 × PUBLISHED_MACS
PUBLISHED_CPU_MS = 140.034


def _assert_flops_convention() -> None:
    """口径自检：FLOPs 必须等于 2 × MACs。不符即报错，防止笔误再次落盘。"""
    if PUBLISHED_FLOPS != 2 * PUBLISHED_MACS:
        raise RuntimeError(
            f"FLOPs 口径不自洽：PUBLISHED_FLOPS={PUBLISHED_FLOPS:,} "
            f"!= 2 × PUBLISHED_MACS={2 * PUBLISHED_MACS:,}"
        )


_assert_flops_convention()

FLOP_CONVENTION = (
    "输入 1x3x224x224，batch=1；1 MAC = 2 FLOPs；只统计 Conv2d 与 Linear；"
    "BN / 激活 / 池化不计入。"
)


# =============================================================================
# 1. FLOPs 计数器（与 arch_probe 同口径）
# =============================================================================

class FlopCounter:
    """用 forward hook 统计 Conv2d / Linear 的 MACs。

    注意 hook 签名是 `hook(module, inputs, output)`，output 对普通层是 Tensor。
    """

    def __init__(self, model):
        self.macs = 0
        self._handles: List = []

        def conv_hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            kh, kw = mod.kernel_size
            self.macs += (
                (mod.in_channels // mod.groups)
                * mod.out_channels * o.shape[2] * o.shape[3] * kh * kw
            )

        def lin_hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            self.macs += mod.in_features * mod.out_features * o.shape[0]

        import torch.nn as nn
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(lin_hook))

    @property
    def flops(self) -> int:
        return 2 * self.macs

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def measure_flops(model, device) -> dict:
    import torch

    counter = FlopCounter(model)
    model.eval()
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224, device=device))
    finally:
        counter.remove()
    return {"macs": int(counter.macs), "flops": int(counter.flops)}


# =============================================================================
# 2. 延迟 / 吞吐 / 显存（仅 CUDA）
# =============================================================================

def measure_latency(model, device, batch_size: int, warmup: int, iters: int,
                    rep: Optional[ProgressReporter] = None) -> dict:
    """测 GPU 延迟/吞吐/显存。

    ⚠️ 进度打印必须放在**计时窗口之外**：计时窗口是
    `synchronize → t0 → model(x) → synchronize → t1`，
    所以 `rep.emit()` 只能放在 append 之后、下一次 synchronize 之前。
    否则打印开销会被算进延迟里，测出来的数就不对了。
    """
    import torch

    if device.type != "cuda":
        return {
            "measurable": False,
            "reason": "NOT_MEASURED_ON_THIS_MACHINE: torch.cuda.is_available()=False",
        }

    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        for i in range(warmup):
            model(x)
            if rep is not None:
                rep.set(step=i + 1, extra="warmup")
                rep.emit()
        torch.cuda.synchronize()

        times: List[float] = []
        torch.cuda.reset_peak_memory_stats(device)
        for i in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
            if rep is not None:                      # ← 计时窗口之外，安全
                rep.set(step=warmup + i + 1, extra=f"{len(times)}/{iters}")
                rep.emit()
        if rep is not None:
            rep.set(step=warmup + iters, extra="done")
            rep.emit(force=True)

    times.sort()
    n = len(times)
    med = times[n // 2]
    q1 = times[n // 4]
    q3 = times[(3 * n) // 4]
    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory

    return {
        "measurable": True,
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "latency_ms_median": med,
        "latency_ms_q1": q1,
        "latency_ms_q3": q3,
        "latency_ms_min": times[0],
        "latency_ms_p95": times[int(0.95 * (n - 1))],
        "throughput_img_per_s": (batch_size * 1000.0) / med,
        "peak_memory_mb": peak / (1024 * 1024),
        "peak_memory_pct_of_total": 100.0 * peak / total,
        "device_total_memory_mb": total / (1024 * 1024),
    }


def measure_cpu_latency(model, batch_size: int, warmup: int, iters: int,
                        rep: Optional[ProgressReporter] = None) -> dict:
    import torch

    model.eval()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for i in range(warmup):
            model(x)
            if rep is not None:
                rep.set(step=i + 1, extra="warmup")
                rep.emit()
        times: List[float] = []
        for i in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
            if rep is not None:                      # ← 同样在计时窗口之外
                rep.set(step=warmup + i + 1, extra=f"{len(times)}/{iters}")
                rep.emit()
        if rep is not None:
            rep.set(step=warmup + iters, extra="done")
            rep.emit(force=True)
    times.sort()
    med = times[len(times) // 2]
    return {
        "batch_size": batch_size,
        "latency_ms_median": med,
        "per_image_ms": med / batch_size,
        "iters": iters,
        "note": "CPU 数值只能做相对比较，不能外推为 GPU 性能。",
    }


# =============================================================================
# 3. ResNeXt50 对照（实测，不手算）
# =============================================================================

def resnext50_reference() -> Optional[dict]:
    try:
        import torch
        import torchvision
        from torchvision.models import resnext50_32x4d
    except Exception as e:
        print(f"  ⚠ 无法导入 torchvision（{e}），跳过 ResNeXt 对照")
        return None

    try:
        m = resnext50_32x4d(weights=None)
    except TypeError:
        m = resnext50_32x4d(pretrained=False)

    n = sum(p.numel() for p in m.parameters())
    out = {
        "model": "resnext50_32x4d (torchvision, weights=None)",
        "torchvision_version": torchvision.__version__,
        "params": int(n),
        "measured_not_hand_computed": True,
    }
    try:
        m.eval()
        counter = FlopCounter(m)
        with torch.no_grad():
            m(torch.randn(1, 3, 224, 224))
        counter.remove()
        out["macs"] = int(counter.macs)
        out["flops"] = int(counter.flops)
    except Exception as e:
        out["flops_error"] = f"{type(e).__name__}: {e}"
    return out


# =============================================================================
# 4. main
# =============================================================================

def main() -> int:
    ap = base_parser("DMSA-Net 效率指标实测（★ 补齐 GPU 三项）")
    ap.add_argument("--batch-sizes", default="1,16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--skip-cpu", action="store_true",
                    help="跳过 CPU 计时（CPU 很慢，batch16 可能耗时数分钟）")
    ap.add_argument("--progress-log", default=None,
                    help="进度日志文件（默认 <out-dir>/g3_progress.log）")
    ap.add_argument("--progress-interval", type=float, default=2.0,
                    help="进度行最小输出间隔（秒，默认 2）")
    a = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = resolve_out_dir(a.out_dir, "g3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 进度日志：与 g2 同口径（屏幕 + 日志同一份内容）----
    progress_log = (Path(a.progress_log) if a.progress_log
                    else out_dir / "g3_progress.log")
    if progress_log.exists():
        try:
            progress_log.unlink()     # 每次重跑都是新的一份，避免和上一次混在一起
        except Exception:
            pass
    sink = make_tee_sink(progress_log)

    bdir = find_baseline_dir(a.baseline_dir)
    print(f"进度日志     : {progress_log}")
    print("=" * 78)
    print("G3 · 效率指标实测")
    print("=" * 78)
    print(f"baseline dir : {bdir}")
    print(f"out dir      : {out_dir}")

    import torch

    dev_info = {
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_requested": a.device,
    }

    print(f"包版本      : {PKG_VERSION}")
    device = None
    gpu_measurable = bool(torch.cuda.is_available()) and a.device.startswith("cuda")
    if gpu_measurable:
        idx = int(a.device.split(":")[1]) if ":" in a.device else 0
        props = torch.cuda.get_device_properties(idx)
        device = torch.device(f"cuda:{idx}")
        dev_info.update({
            "device_name": props.name,
            "device_total_memory_bytes": int(props.total_memory),
            "device_capability": f"{props.major}.{props.minor}",
        })
        print(f"device       : {device}  {props.name}")
    else:
        device = torch.device("cpu")
        dev_info["gpu_not_measured_reason"] = (
            "NOT_MEASURED_ON_THIS_MACHINE: cuda_available="
            f"{bool(torch.cuda.is_available())}, device_requested={a.device}"
        )
        print(f"device       : cpu（GPU 三项将标记为 NOT_MEASURED）")

    if a.dry_run and not gpu_measurable:
        print("\n本机无 CUDA：只做接线自检（前向 + FLOPs + 参数量）。")

    # ---- 构造基线模型 ----
    smoke = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke")
    man = bdir / TRAIN_MANIFEST_NAME
    man_sha = sha256_file(man)
    if man_sha != TRAIN_MANIFEST_SHA256:
        raise RuntimeError(
            f"train manifest SHA 不符：{man_sha}\nExpected: {TRAIN_MANIFEST_SHA256}"
        )

    model = smoke.create_model(num_classes=4)
    fp = smoke.architecture_fingerprint(model)
    if fp != ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            f"架构指纹不符：{fp}\nExpected: {ARCHITECTURE_FINGERPRINT}"
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model        : params={n_params:,}  fp={fp[:12]}… ✅ 与基线同构")

    result: Dict[str, object] = {
        "experiment": "G3 efficiency benchmark",
        "environment": dev_info,
        "flop_convention": FLOP_CONVENTION,
        "model": {
            "name": "DMSA-Net (EnhancedResNeXt50)",
            "params_total": int(n_params),
            "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "architecture_fingerprint": fp,
            "fingerprint_matches_baseline": True,
            "spatial_attention_params": int(sum(
                p.numel() for n, p in model.named_parameters() if "spatial_attention" in n)),
            "manifest_sha256": man_sha,
            "smoke_test_sha256": sha256_file(bdir / "smoke_test.py"),
        },
    }

    # ---- FLOPs ----
    print("\n[1] FLOPs（沿用已发布口径）…")
    fl = measure_flops(model, torch.device("cpu"))
    result["flops"] = {
        **fl,
        "gflops": fl["flops"] / 1e9,
        "published_flops": PUBLISHED_FLOPS,
        "published_macs": PUBLISHED_MACS,
        "matches_published": (fl["macs"] == PUBLISHED_MACS),
        "convention": FLOP_CONVENTION,
    }
    print(f"    MACs = {fl['macs']:,}   FLOPs = {fl['flops']:,} "
          f"({fl['flops']/1e9:.3f} G)")
    print(f"    与已发布值一致：{fl['macs'] == PUBLISHED_MACS} "
          f"（published MACs={PUBLISHED_MACS:,}）")

    # ---- ResNeXt50 对照 ----
    print("\n[2] ResNeXt50-32x4d 对照（torchvision 实测）…")
    ref = resnext50_reference()
    if ref:
        result["resnext50_reference"] = ref
        result["param_ratio_dmsa_over_resnext"] = n_params / ref["params"]
        print(f"    params = {ref['params']:,}  →  DMSA / ResNeXt = "
              f"{n_params / ref['params']:.3f}x")
        if "flops" in ref:
            print(f"    FLOPs  = {ref['flops']:,} ({ref['flops']/1e9:.3f} G)")

    # ---- 延迟 / 吞吐 / 显存 ----
    batch_sizes = [int(b.strip()) for b in a.batch_sizes.split(",") if b.strip()]
    lat: Dict[str, object] = {}

    print("\n[3] 延迟 / 吞吐 / 显存 …")
    for bs in batch_sizes:
        if gpu_measurable:
            rep = ProgressReporter(label=f"GPU bs={bs}", unit="iter",
                                   total_units=a.warmup + a.iters,
                                   sink=sink, min_interval=a.progress_interval)
            r = measure_latency(model.to(device), device, bs, a.warmup, a.iters,
                                rep=rep)
            lat[f"gpu_bs{bs}"] = r
            print(f"    GPU bs={bs:<3} latency={r['latency_ms_median']:.2f} ms "
                  f"[{r['latency_ms_q1']:.2f}, {r['latency_ms_q3']:.2f}]  "
                  f"throughput={r['throughput_img_per_s']:.1f} img/s  "
                  f"peak={r['peak_memory_mb']:.1f} MB "
                  f"({r['peak_memory_pct_of_total']:.1f}%)")
        else:
            lat[f"gpu_bs{bs}"] = {
                "measurable": False,
                "batch_size": bs,
                "reason": "NOT_MEASURED_ON_THIS_MACHINE: CUDA unavailable",
            }
            print(f"    GPU bs={bs:<3} NOT_MEASURED_ON_THIS_MACHINE")

    if not a.skip_cpu:
        print("\n[4] CPU 延迟（仅相对参考）…")
        for bs in batch_sizes[:1]:
            _w = max(2, a.warmup // 5)
            _it = max(5, a.iters // 5)
            rep = ProgressReporter(label=f"CPU bs={bs}", unit="iter",
                                   total_units=_w + _it,
                                   sink=sink, min_interval=a.progress_interval)
            r = measure_cpu_latency(model.to("cpu"), bs, _w, _it, rep=rep)
            lat[f"cpu_bs{bs}"] = r
            print(f"    CPU bs={bs} per_image={r['per_image_ms']:.1f} ms "
                  f"（已发布参考 {PUBLISHED_CPU_MS} ms/img）")
    result["latency"] = lat

    # ---- 汇总状态 ----
    gpu_rows = [lat.get(f"gpu_bs{bs}", {}) for bs in batch_sizes]
    result["summary"] = {
        "measured_7_items": {
            "params_total": "MEASURED",
            "resnext50_params": "MEASURED" if ref else "NOT_MEASURED",
            "flops": "MEASURED",
            "macs": "MEASURED",
            "cpu_latency": "MEASURED" if not a.skip_cpu else "SKIPPED",
            "gpu_latency": "MEASURED" if gpu_measurable else "NOT_MEASURED_ON_THIS_MACHINE",
            "gpu_throughput": "MEASURED" if gpu_measurable else "NOT_MEASURED_ON_THIS_MACHINE",
            "gpu_peak_memory": "MEASURED" if gpu_measurable else "NOT_MEASURED_ON_THIS_MACHINE",
        },
        "n_measured": 3 + (1 if ref else 0) + (1 if not a.skip_cpu else 0)
                      + (3 if gpu_measurable else 0),
        "note": (
            "GPU 三项只在有 CUDA 的机器上可测。本脚本不会用 CPU 数值外推 GPU 性能——"
            "这是项目铁律。若在 GPU 机器上运行，请把 --device cuda:0 且确认 "
            "torch.cuda.is_available() 为 True。"
        ),
    }

    write_json(out_dir / "g3_efficiency.json", result, script_path)

    rows = []
    for bs in batch_sizes:
        r = lat.get(f"gpu_bs{bs}", {})
        rows.append({
            "batch_size": bs,
            "gpu_latency_ms_median": r.get("latency_ms_median", "NOT_MEASURED"),
            "gpu_latency_ms_q1": r.get("latency_ms_q1", ""),
            "gpu_latency_ms_q3": r.get("latency_ms_q3", ""),
            "gpu_throughput_img_s": r.get("throughput_img_per_s", "NOT_MEASURED"),
            "gpu_peak_memory_mb": r.get("peak_memory_mb", "NOT_MEASURED"),
            "gpu_peak_memory_pct": r.get("peak_memory_pct_of_total", ""),
        })
    write_csv(
        out_dir / "g3_latency_table.csv", rows,
        ["batch_size", "gpu_latency_ms_median", "gpu_latency_ms_q1",
         "gpu_latency_ms_q3", "gpu_throughput_img_s",
         "gpu_peak_memory_mb", "gpu_peak_memory_pct"],
    )

    # ---- 方案可粘贴的 Markdown 表 ----
    md: List[str] = []
    md.append("# DMSA-Net 效率指标实测\n")
    md.append(f"- 设备：`{dev_info.get('device_name', 'CPU only')}`"
              f"（torch {dev_info['torch']}）")
    md.append(f"- 架构指纹：`{fp}`（与基线一致 ✅）")
    md.append(f"- FLOPs 口径：{FLOP_CONVENTION}\n")
    md.append("| 指标 | 数值 | 状态 |")
    md.append("|---|---:|---|")
    md.append(f"| 参数量 | {n_params:,} | ✅ 实测 |")
    if ref:
        md.append(f"| 对照 ResNeXt50-32x4d 参数量 | {ref['params']:,} | ✅ 实测 |")
        md.append(f"| 参数量比值（DMSA / ResNeXt） | {n_params/ref['params']:.2f}× | ✅ |")
    md.append(f"| FLOPs (batch=1, 224²) | {fl['flops']/1e9:.3f} G | ✅ 实测 |")
    md.append(f"| MACs | {fl['macs']/1e9:.3f} G | ✅ 实测 |")
    if not a.skip_cpu and lat.get(f"cpu_bs{batch_sizes[0]}"):
        md.append(f"| CPU 单图延迟 | {lat[f'cpu_bs{batch_sizes[0]}']['per_image_ms']:.1f} ms | ✅ 实测（仅相对） |")
    for bs in batch_sizes:
        r = lat.get(f"gpu_bs{bs}", {})
        if r.get("measurable"):
            md.append(f"| **GPU 延迟 (bs={bs})** | **{r['latency_ms_median']:.2f} ms** | ✅ 实测 |")
            md.append(f"| **GPU 吞吐 (bs={bs})** | **{r['throughput_img_per_s']:.1f} img/s** | ✅ 实测 |")
            md.append(f"| **GPU 峰值显存 (bs={bs})** | **{r['peak_memory_mb']:.1f} MB** | ✅ 实测 |")
        else:
            md.append(f"| GPU 延迟 / 吞吐 / 显存 (bs={bs}) | — | ❌ NOT_MEASURED |")
    md.append("")
    md.append(f"> {result['summary']['note']}")
    p = out_dir / "g3_efficiency_table.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print(f"  [write_md   ] {p}")

    print("\n" + "=" * 78)
    print("G3 完成：7 项中已实测", result["summary"]["n_measured"], "项")
    if not gpu_measurable:
        print("GPU 三项未测（本机无 CUDA）。请在 GPU 机器上重跑本脚本。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
