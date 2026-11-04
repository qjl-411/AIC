#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC —— G2（多种子）+ G1（模块消融）一键启动（含运行前预检查）
===============================================================================

它解决什么问题
---------------------------------------------------------------------------
G1 与 G2 都是**真训练**，都要长时间独占 GPU（量级：十几小时 ~ 几十小时）。
如果不先做检查就一起丢上去，典型翻车方式有：

  · 同卡并行 → 显存占用**相加** → OOM，几十小时白跑
  · 输出目录撞车 → 产物互相覆盖，事后分不清是谁写的
  · G1 的 `full` 参照没找到 → 表格跑完却没有 Δ vs full（看起来"跑完了"其实是残的）
  · 磁盘不够 → 跑到一半写不下

本脚本在**训练真正开始前**做一次轻量预检查，给出明确结论：

    SERIAL_REQUIRED         只能串行（显存放不下 / 单卡 / 显存不可测）
    PARALLEL_OK             可以并行（两张卡，各占一张）
    PARALLEL_WITH_CAVEATS   同卡放得下，但会互相拖慢 → 默认仍推荐串行

并按结论启动。硬冲突（路径撞车、数据集被指向输出区等）直接**中止**并说清楚原因。

---------------------------------------------------------------------------
两者在工作量上的差别（决定串行顺序）
---------------------------------------------------------------------------
    G2 多种子   每 seed：126 epoch × 640 step（10,240 张全量 final 训练）
    G1 消融     每个 (变体,折)：200 epoch × 512 step（8,192 张折内训练 + 2,048 val）

⇒ 默认顺序 **先 G2 再 G1**：G2 工作量较小，先拿到结果；即便后面中断，
   也已经有一部分**完整可用**的产物，而不是两套都跑一半。
   想反过来用 `--order g1,g2`。

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------

    # 只做预检查，不跑（第一次务必先这样看一眼）：
    python g12_run.py --plan

    # 一键跑（默认串行：G2 跑完再 G1，各独占整卡）：
    python g12_run.py --seeds 43,44 --variants no_sam,no_mdc,no_ilf --folds 1,5

    # 两张卡，强制并行（预检查会先确认是否可行）：
    python g12_run.py --parallel --g2-device cuda:0 --g1-device cuda:1

    # 机器上有已完成的五折结果（<dir>/Fold1..5/best_metrics.json）时指给它 →
    # `full` 不必实跑（省 2 折 × 200 epoch），表格照样有 Δ vs full：
    python g12_run.py --reference-cv /home/featurize/work/ref_cv

    # 本机 CPU 接线自检：
    python g12_run.py --dry-run

    # ⚠️ 上一版（2026-09-24.3）曾把"同卡并行放不下"误判成硬冲突直接中止；
    #    现在的判据：只有"这张卡连一个任务都放不下"才是 FAIL。

---------------------------------------------------------------------------
GPU 是怎么被占用的（结论先说）
---------------------------------------------------------------------------
  · 串行（默认）：G2 独占整卡跑完 → 再 G1 独占整卡跑完。
    同一时刻只有一个进程在卡上 → 不会 OOM，也不会互相拖慢。
  · 并行：两个进程同时在卡上。**若同一张卡，显存是两者相加**，
    且算力被分时 → 两个都变慢；只有**各自一张卡**时才真正并行。
  · 预检查里的显存数字不是"参数量 × 4 字节"估的，而是
    **真的建一次模型、跑一个 batch 测出来的峰值**（几秒钟）。
===============================================================================
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from C22_lib_gpu_runtime import (  # noqa: E402
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_LR,
    EXPECTED_CLASSES,
    EXPECTED_TRAIN_IMAGES,
    FINAL_EPOCHS,
    FIXED_DATA_ROOT,
    FIXED_OUT_ROOT,
    PKG_VERSION,
    default_out_root,
    find_baseline_dir,
    fmt_hms,
    load_module,
    resolve_data_root,
    resolve_out_dir,
    stream_run,
    write_json,
)

HERE = Path(__file__).resolve().parent

# 两个子进程各自会写的文件名（用于"路径是否撞车"的静态检查）
G2_NAMES = {"g2_multiseed_summary.json", "g2_multiseed_table.csv",
            "g2_dryrun.json", "_variants",
            "log_seed43.txt", "log_seed44.txt",
            "progress_seed43.log", "progress_seed44.log"}
G1_NAMES = {"g1_ablation_summary.json", "g1_ablation_table.md",
            "g1_ablation_table.csv", "g1_ablation_per_fold.csv",
            "g1_dryrun.json", "g1_run_config.json", "g1_progress.log",
            "metrics.csv", "best_metrics.json"}

# 显存安全系数：实测峰值只反映"这一刻的激活+参数"，实际训练还有
# DataLoader / CUDA context / 碎片 等开销，留出余量再下结论。
MEM_SAFETY = 1.15
MEM_USABLE = 0.85

# 磁盘告警阈值（G2 每个 seed 会存模型权重，量级是 GB）
DISK_WARN_MB = 8 * 1024
DISK_FAIL_MB = 2 * 1024


# =============================================================================
# 1. 预检查
# =============================================================================

class Check:
    __slots__ = ("item", "status", "detail")

    def __init__(self, item: str, status: str, detail: str):
        self.item = item
        self.status = status      # OK / WARN / FAIL
        self.detail = detail


def _gpu_snapshot(device: str) -> dict:
    """读取指定 device 的显存现状。不可用时返回 measurable=False。"""
    import torch
    out = {"device": device, "measurable": False}
    if not torch.cuda.is_available():
        out["reason"] = "torch.cuda.is_available() = False"
        return out
    idx = int(device.split(":")[1]) if ":" in device else 0
    if idx >= torch.cuda.device_count():
        out["reason"] = f"只有 {torch.cuda.device_count()} 张卡，没有 {device}"
        return out
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    out.update({
        "measurable": True,
        "name": props.name,
        "total_mb": total / (1024 * 1024),
        "free_mb": free / (1024 * 1024),
        "used_mb": (total - free) / (1024 * 1024),
    })
    return out


def _probe_peak_mb(bdir: Path, device: str, batch_size: int) -> Optional[int]:
    """真的建一次模型、跑一个 batch（前向+反传+Adam），测训练态峰值显存。

    不用"参数量 × 4 字节"去估——激活值才是大头，估不准。
    这里实测一次只花几秒，换来的是可信的串行/并行结论。

    ⚠️ G1 的消融变体容量都不大于 `full`（no_mdc 少 1,177 万参数），
       所以按 `full` 探测得到的是**上界**，用它下结论是安全的。
    """
    import torch

    idx = int(device.split(":")[1]) if ":" in device else 0
    dev = torch.device(f"cuda:{idx}")
    smoke = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke")
    model = smoke.create_model(num_classes=4).to(dev)
    x = torch.randn(batch_size, 3, 224, 224, device=dev)
    peak = 0
    try:
        torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)
        lossf = torch.nn.CrossEntropyLoss()
        y = torch.randint(0, 4, (batch_size,), device=dev)
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        del opt, y, loss
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated(dev))
    finally:
        del model, x
        torch.cuda.empty_cache()
    return peak


def _g1_full_reference(bdir: Optional[str], explicit: Optional[str] = None) -> dict:
    """判断 G1 的 `full` 基准行能不能拿到。

    这是 G1 最容易"跑完了但结果是残的"的坑：没有 full 参照、且 --variants 里
    也没有 full 时，汇总表会**没有 Δ vs full**，而脚本照样返回 0。

    `explicit` = `--reference-cv` 给的目录：给了就**只用它**，找不到即报错
    （不悄悄回退，否则你以为用了 A、实际用了 B）。
    """
    try:
        import C31_run_g1_ablation as g1mod
    except Exception as e:                       # 同目录找不到就不判断，明说
        return {"decided": False, "reason": f"无法载入 g1_ablation：{type(e).__name__}: {e}"}
    try:
        cands = [bdir, Path(bdir).parent if bdir else HERE, Path.home() / "work"]
        ref = g1mod.load_reference_cv(explicit, extra_dirs=[c for c in cands if c])
    except Exception as e:
        return {"decided": True, "found": False,
                "error": f"{type(e).__name__}: {e}"}
    return {"decided": True, "found": ref is not None}


def precheck(
    *,
    g2_out: Path, g1_out: Path, data_root: str,
    g2_device: str, g1_device: str, seeds: List[int],
    variants: List[str], folds: List[int], epochs: int,
    baseline_dir: Optional[str], reference_cv: Optional[str] = None,
    probe: bool = True,
) -> Tuple[List[Check], dict]:
    """返回 (检查项列表, 结论字典)。

    `probe=True` 时会真的在 GPU 上建一次模型测峰值显存（几秒）。
    """
    checks: List[Check] = []
    verdict: Dict[str, object] = {}

    # ---- ⓪ 工作量提示（先算：后面任何提前 return 都要带上它）----
    n_runs = len(variants) * len(folds)
    g2_units = len(seeds) * FINAL_EPOCHS * (EXPECTED_TRAIN_IMAGES // BASE_BATCH_SIZE)
    g1_units = n_runs * epochs * ((EXPECTED_TRAIN_IMAGES - 2048) // BASE_BATCH_SIZE)
    verdict.update({
        "g2_total_steps": g2_units,
        "g1_total_steps": g1_units,
        "g1_runs": n_runs,
        "workload_note": (
            f"G2 = {len(seeds)} seed × {FINAL_EPOCHS} epoch × "
            f"{EXPECTED_TRAIN_IMAGES // BASE_BATCH_SIZE} step = {g2_units:,} step；"
            f"G1 = {n_runs} run × {epochs} epoch × "
            f"{(EXPECTED_TRAIN_IMAGES - 2048) // BASE_BATCH_SIZE} step = {g1_units:,} step"
            "（绝对时长取决于机器，本预检查不实测）"
        ),
    })

    # ---- ① 输出目录不重叠 ----
    g2r, g1r = g2_out.resolve(), g1_out.resolve()
    if g2r == g1r:
        checks.append(Check("输出目录", "FAIL", f"G2 与 G1 输出目录相同：{g2r}"))
    elif g2r in g1r.parents or g1r in g2r.parents:
        checks.append(Check("输出目录", "FAIL",
                            f"一个输出目录是另一个的上级：{g2r} vs {g1r}"))
    else:
        checks.append(Check("输出目录", "OK",
                            f"G2={g2r} ｜ G1={g1r}（互不包含）"))

    # ---- ② 文件名不撞车 ----
    overlap = sorted(G2_NAMES & G1_NAMES)
    if overlap:
        checks.append(Check("产物文件名", "FAIL", f"同名产物：{overlap}"))
    else:
        checks.append(Check("产物文件名", "OK",
                            f"G2 写 {len(G2_NAMES)} 类、G1 写 {len(G1_NAMES)} 类，无交集"))

    # ---- ③ 输出区与数据集互不嵌套（防止"覆盖"误删数据）----
    dr = Path(data_root)
    try:
        dr_res = dr.resolve()
    except Exception:
        dr_res = dr
    bad = False
    for tag, p in (("G2", g2r), ("G1", g1r)):
        if p == dr_res or dr_res in p.parents or p in dr_res.parents:
            checks.append(Check("数据集隔离", "FAIL",
                                f"{tag} 输出目录 {p} 与数据集 {dr_res} 存在嵌套"))
            bad = True
    if not bad:
        checks.append(Check("数据集隔离", "OK",
                            f"输出区与数据集 {dr_res} 互不嵌套"))

    # ---- ④ 数据可读（train/ 与四个类目录）----
    train_root = dr / "train"
    if not train_root.is_dir():
        checks.append(Check("数据可读", "FAIL", f"找不到 {train_root}"))
    else:
        missing = [c for c in EXPECTED_CLASSES if not (train_root / c).is_dir()]
        if missing:
            checks.append(Check("数据可读", "FAIL",
                                f"train/ 下缺少类目录：{missing}（注意目录名带空格）"))
        else:
            checks.append(Check("数据可读", "OK",
                                f"{train_root} 四类目录齐全（只读，不写入）"))

    # ---- ⑤ G1 的 full 基准行（G1 特有的"跑完但结果是残的"风险）----
    #     下面这行 `full 参照目录` 是 G1 内部搜索逻辑的回显，不是本脚本的输出。
    import io as _io
    import contextlib as _ctx
    _buf = _io.StringIO()
    with _ctx.redirect_stdout(_buf):
        refinfo = _g1_full_reference(baseline_dir, reference_cv)
    _echo = _buf.getvalue().strip()
    if _echo:
        print(f"  [G1 内部回显] {_echo}")
    if not refinfo.get("decided"):
        checks.append(Check("G1 full 基准", "WARN",
                            f"无法判定（{refinfo.get('reason')}）→ 跑之前请自己确认 "
                            f"--reference-cv 指向的目录含 Fold1..5/best_metrics.json"))
    elif refinfo.get("found"):
        src = "（--reference-cv 显式指定）" if reference_cv else "（自动搜索到）"
        checks.append(Check("G1 full 基准", "OK",
                            f"找到已完成五折参照{src} → 表格会有 full 行与 Δ vs full，"
                            "且 `full` 不必实跑（省 2 折 × 200 epoch）"))
    elif reference_cv:
        checks.append(Check("G1 full 基准", "FAIL",
                            f"--reference-cv 指向的目录不可用：{refinfo.get('error')}"
                            " → 显式指定时**不回退**，请修正路径或去掉该参数"))
    elif "full" in variants:
        checks.append(Check("G1 full 基准", "OK",
                            "没找到参照，但 --variants 含 full → full 会实跑，仍有 Δ"))
    else:
        checks.append(Check("G1 full 基准", "WARN",
                            "既没有五折参照、--variants 里也没有 full → "
                            "**表格将没有 full 行、也没有 Δ vs full**。"
                            "若这是你想要的（只想看各变体绝对精度），可忽略"))

    # ---- ⑥ 随机种子 ----
    checks.append(Check("随机种子", "OK",
                        f"G2 用 seeds={seeds}（每个种子一份独立副本 + 独立子进程）；"
                        f"G1 用 BASE_SEED=42（与基线同 seed，消融才有可比性），"
                        f"折内 DataLoader 另用 42+fold 派生。各进程独立 → 无 RNG 耦合"))

    # ---- ⑦ 数据读取冲突 ----
    checks.append(Check("数据读取", "WARN",
                        "两者**只读**同一个 train/，不会破坏数据；"
                        "但若并行，DataLoader worker（各 6 个）会争抢 IO 与 /dev/shm，"
                        "只会变慢、不影响正确性"))

    # ---- ⑧ 磁盘空间 ----
    try:
        usage = shutil.disk_usage(str(g2r))
        free_mb = usage.free / (1024 * 1024)
        detail = (f"输出区可用 {free_mb/1024:.1f} GB"
                  f"（G2 每个 seed 会存模型权重，量级 GB；G1 只存 csv/json）")
        if free_mb < DISK_FAIL_MB:
            checks.append(Check("磁盘空间", "FAIL", detail + " → 空间不足，跑到一半会写不下"))
        elif free_mb < DISK_WARN_MB:
            checks.append(Check("磁盘空间", "WARN", detail + " → 偏紧，建议先清出更多空间"))
        else:
            checks.append(Check("磁盘空间", "OK", detail))
        verdict["disk_free_mb"] = free_mb
    except Exception as e:
        checks.append(Check("磁盘空间", "WARN", f"无法读取（{type(e).__name__}: {e}）"))

    # ---- ⑨ GPU：显存实测 ----
    snap2 = _gpu_snapshot(g2_device)
    snap1 = _gpu_snapshot(g1_device) if g1_device != g2_device else snap2

    if not snap2["measurable"] or not snap1["measurable"]:
        reason = snap2.get("reason") or snap1.get("reason")
        checks.append(Check("GPU 显存", "WARN",
                            f"无法实测（{reason}）→ 按串行处理"))
        verdict.update({"verdict": "SERIAL_REQUIRED", "recommended": "serial",
                        "same_device": True, "reason": reason})
        return checks, verdict

    same_dev = (g2_device == g1_device)
    need2 = need1 = None
    if probe:
        bdir = find_baseline_dir(baseline_dir)
        # 两者都是训练态（前向 + 反传 + Adam 状态），按 full 容量探测（上界）
        need2 = _probe_peak_mb(bdir, g2_device, BASE_BATCH_SIZE)
        need1 = _probe_peak_mb(bdir, g1_device, BASE_BATCH_SIZE)

    if same_dev:
        free_mb = snap2["free_mb"]
        if need2 is None or need1 is None:
            verdict.update({"verdict": "SERIAL_REQUIRED", "recommended": "serial",
                            "same_device": True, "reason": "显存探测未完成"})
            checks.append(Check("GPU 显存", "WARN", "探测未完成 → 保守按串行"))
        else:
            # ⚠️ 判据分两层，**不能再把"同卡放不下"当硬冲突**：
            #   同卡放不下 ⇒ 正确结论是"只能串行"（这正是推荐方式），
            #   串行时同一时刻只有一个进程在卡上，单任务的 5 GB 完全放得下。
            #   只有"这张卡连一个任务都放不下"才是真硬冲突。
            free_mb = float(snap2["free_mb"])
            total_mb = float(snap2.get("total_mb") or 0.0)
            # ⚠️ 单位必须统一到 MB：`_probe_peak_mb` 返回**字节**，
            #    而 mem_get_info 换算出来的是 **MB**。字节 vs MB 直接比大小，
            #    会让 `single_need > cap` **永远成立** → 任何机器都会被判成
            #    "放不下"而中止。本机无 GPU 时这条路径从不执行，所以测不出来。
            need2_mb = need2 / (1024 * 1024)
            need1_mb = need1 / (1024 * 1024)
            worst = max(need2_mb, need1_mb)
            single_need = worst * MEM_SAFETY            # 串行：一次只跑一个
            both_need = (need2_mb + need1_mb) * MEM_SAFETY   # 并行：两者同时在卡上
            single_cap_card = total_mb * MEM_USABLE     # 物理上限（独占整卡）
            free_cap = free_mb * MEM_USABLE             # 当前可用上限
            detail = (f"同卡 {g2_device}（{snap2['name']}，总 {total_mb/1024:.1f} GB）："
                      f"G2 训练峰值≈{need2_mb:.0f} MB，"
                      f"G1 训练峰值≈{need1_mb:.0f} MB（按 full 容量探测，是上界）；"
                      f"串行只需 max×{MEM_SAFETY}≈{single_need:.0f} MB，"
                      f"并行需合计×{MEM_SAFETY}≈{both_need:.0f} MB；"
                      f"当前空闲 {free_mb:.0f} MB"
                      f"（可用上限按 {MEM_USABLE} 计 = {free_cap:.0f} MB）")
            low_free_note = ""
            if free_mb < 64:
                low_free_note = (
                    " ⚠️ 空闲读数≈0：通常是卡上还挂着别的进程/Jupyter 内核，"
                    "或容器上报不准（本次探测仍能分配 5 GB，说明卡是好的）。"
                    "**串行不受影响**；建议 `nvidia-smi` 看一眼有没有残留进程占着。")
            if single_need > single_cap_card:
                checks.append(Check("GPU 显存", "FAIL",
                                    detail + f" → 这张卡（{total_mb/1024:.1f} GB）"
                                             "连一个任务都放不下，跑不了"
                                    + low_free_note))
                verdict.update({"verdict": "SERIAL_REQUIRED",
                                "recommended": "serial", "same_device": True,
                                "reason": "单任务放不下"})
            elif single_need > free_cap:
                checks.append(Check("GPU 显存", "WARN",
                                    detail + " → **只能串行**；且当前空闲连一个任务"
                                             "的份额都不够，串行也可能 OOM"
                                    + low_free_note))
                verdict.update({"verdict": "SERIAL_REQUIRED",
                                "recommended": "serial", "same_device": True})
            elif both_need > free_cap:
                checks.append(Check("GPU 显存", "WARN",
                                    detail + " → **同卡并行会 OOM，只能串行**"
                                             "（串行一次只占 ~5 GB，可行）"
                                    + low_free_note))
                verdict.update({"verdict": "SERIAL_REQUIRED",
                                "recommended": "serial", "same_device": True})
            else:
                checks.append(Check("GPU 显存", "WARN",
                                    detail + " → 同卡并行**放得下但会互相拖慢**；"
                                             "两者都要跑十几小时以上，独占更划算"
                                    + low_free_note))
                verdict.update({"verdict": "PARALLEL_WITH_CAVEATS",
                                "recommended": "serial", "same_device": True})
            verdict.update({"g2_peak_mb": need2 / (1024 * 1024),
                            "g1_peak_mb": need1 / (1024 * 1024),
                            "free_mb": free_mb, "total_mb": total_mb,
                            "device": g2_device, "device_name": snap2["name"]})
    else:
        checks.append(Check("GPU 显存", "OK",
                            f"各占一张卡：G2→{g2_device}（{snap2.get('name')}，"
                            f"可用 {snap2.get('free_mb', 0)/1e6:.0f} MB）｜"
                            f"G1→{g1_device}（{snap1.get('name')}，"
                            f"可用 {snap1.get('free_mb', 0)/1e6:.0f} MB）"
                            + (f"；G2 训练峰值≈{need2/1e6:.0f} MB" if need2 else "")
                            + (f"；G1 训练峰值≈{need1/1e6:.0f} MB" if need1 else "")))
        verdict.update({"verdict": "PARALLEL_OK", "recommended": "parallel",
                        "same_device": False})

    return checks, verdict


def print_checks(checks: List[Check], verdict: dict) -> None:
    print("-" * 78)
    print("预检查")
    print("-" * 78)
    icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}
    for c in checks:
        print(f"  {icon.get(c.status, '  ')} [{c.status:4}] {c.item:<12} {c.detail}")
    print("-" * 78)
    v = verdict.get("verdict", "UNKNOWN")
    rec = verdict.get("recommended", "serial")
    zh = {"SERIAL_REQUIRED": "只能串行",
          "PARALLEL_OK": "可以并行",
          "PARALLEL_WITH_CAVEATS": "可并行但有代价"}
    print(f"  结论：{v}（{zh.get(v, v)}）   建议执行方式：{rec}")
    if verdict.get("workload_note"):
        print(f"  工作量：{verdict['workload_note']}")


# =============================================================================
# 2. 启动
# =============================================================================

def _g2_cmd(g2_out: Path, data_root: str, device: str, seeds: List[int],
            dry: bool) -> List[str]:
    # `-u`：子进程 stdout 走的是管道，Python 默认按 4–8KB 块缓冲 ——
    # 加了它，脚本自己 print 的行才会**逐行**流出来（否则 run.log 长时间是空的，
    # 只能靠尾部攒够一块才落盘）。tqdm 会自己 flush，所以它不受影响。
    cmd = [sys.executable, "-u", str(HERE / "g2_multi_seed.py"),
           "--out-dir", str(g2_out),
           "--data-root", data_root,
           "--device", device,
           "--seeds", ",".join(str(s) for s in seeds)]
    if dry:
        cmd.append("--dry-run")
    return cmd


def _g1_cmd(g1_out: Path, data_root: str, device: str, variants: List[str],
            folds: List[int], epochs: int, dry: bool,
            reference_cv: Optional[str] = None) -> List[str]:
    cmd = [sys.executable, "-u", str(HERE / "g1_ablation.py"),
           "--out-dir", str(g1_out),
           "--data-root", data_root,
           "--device", device,
           "--variants", ",".join(variants),
           "--folds", ",".join(str(f) for f in folds),
           "--epochs", str(epochs)]
    if reference_cv:
        cmd += ["--reference-cv", str(reference_cv)]
    if dry:
        cmd.append("--dry-run")
    return cmd


def run_serial(jobs: List[Tuple[str, List[str], Path]]) -> Dict[str, int]:
    """串行：一个跑完再跑下一个，同一时刻只有一者占 GPU。"""
    rcs: Dict[str, int] = {}
    t0 = time.time()
    for i, (name, cmd, log) in enumerate(jobs, 1):
        print("\n" + "=" * 78)
        print(f"[{i}/{len(jobs)}] 启动 {name}（串行，独占 GPU）")
        print(f"  $ {' '.join(cmd)}")
        print(f"  实时日志：{log}")
        print("=" * 78)
        rc, _ = stream_run(cmd, log, on_frame=lambda f: print(f))
        rcs[name] = rc
        print(f"\n  {name} 退出码 = {rc}   累计用时 {fmt_hms(time.time() - t0)}")
    return rcs


def run_parallel(jobs: List[Tuple[str, List[str], Path]]) -> Dict[str, int]:
    """并行：两个进程同时在 GPU 上（只有各占一张卡时才真正并行）。"""
    print("\n" + "=" * 78)
    print("并行启动（两个进程同时在 GPU 上；进度请各自 tail -f 对应日志）")
    print("=" * 78)
    procs = []
    for name, cmd, log in jobs:
        log.parent.mkdir(parents=True, exist_ok=True)
        f = log.open("w", encoding="utf-8", errors="replace")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        procs.append((name, p, f, log))
        print(f"  ▶ {name}: {' '.join(cmd)}")
        print(f"    日志：{log}")
    rcs: Dict[str, int] = {}
    t0 = time.time()
    for name, p, f, log in procs:
        rcs[name] = p.wait()
        f.close()
        print(f"  ■ {name} 退出码 = {rcs[name]}   用时 {fmt_hms(time.time() - t0)}")
    return rcs


# =============================================================================
# 3. main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="DMSA-Net：G2（多种子）+ G1（模块消融）一键启动，含运行前预检查")
    ap.add_argument("--device", default="cuda:0", help="G2/G1 共用的默认设备")
    ap.add_argument("--g2-device", default=None, help="G2 单独指定设备（默认同 --device）")
    ap.add_argument("--g1-device", default=None, help="G1 单独指定设备（默认同 --device）")
    ap.add_argument("--seeds", default="43,44", help="G2 的种子（默认 43,44）")
    ap.add_argument("--variants", default="no_sam,no_mdc,no_ilf",
                    help="G1 的消融变体（默认 no_sam,no_mdc,no_ilf）")
    ap.add_argument("--folds", default="1,5", help="G1 的折号（默认 1,5）")
    ap.add_argument("--epochs", type=int, default=BASE_EPOCHS,
                    help=f"G1 的 epoch 数（默认 {BASE_EPOCHS}；G2 固定 {FINAL_EPOCHS}）")
    ap.add_argument("--data-root", default=None,
                    help=f"数据集路径（默认 {FIXED_DATA_ROOT}）")
    ap.add_argument("--reference-cv", default=None,
                    help="G1 的 full 参照目录（内含 Fold1..5/best_metrics.json）；"
                         "给了就直接用它，`full` 不必实跑（省 2 折 × 200 epoch）")
    ap.add_argument("--out-root", default=None,
                    help=f"输出根（默认 {FIXED_OUT_ROOT}；下建 g2/ 与 g1/）")
    ap.add_argument("--baseline-dir", default=None, help="基线目录")
    ap.add_argument("--parallel", action="store_true",
                    help="强制并行（预检查判定只能串行时会拒绝）")
    ap.add_argument("--serial", action="store_true", help="强制串行（忽略推荐策略）")
    ap.add_argument("--order", default="g2,g1", help="串行顺序，默认 g2,g1")
    ap.add_argument("--plan", action="store_true", help="只做预检查并打印，不启动")
    ap.add_argument("--dry-run", action="store_true", help="把 --dry-run 透传给子脚本")
    ap.add_argument("--no-probe", action="store_true",
                    help="跳过 GPU 显存实测（只用 mem_get_info 粗判）")
    ap.add_argument("--force", action="store_true",
                    help="硬冲突也继续（不推荐；仅在明确知道后果时用）")
    a = ap.parse_args()

    print("=" * 78)
    print("G2（多种子）+ G1（模块消融）一键启动（含预检查）")
    print("=" * 78)
    print(f"包版本      : {PKG_VERSION}")

    g2_device = a.g2_device or a.device
    g1_device = a.g1_device or a.device
    data_root = resolve_data_root(a.data_root)
    seeds = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    folds = [int(f.strip()) for f in a.folds.split(",") if f.strip()]

    root = Path(a.out_root) if a.out_root else default_out_root()
    g2_out = resolve_out_dir(str(root / "g2"), "g2")
    g1_out = resolve_out_dir(str(root / "g1"), "g1")
    print(f"输出根      : {root}")
    print(f"data root   : {data_root}")
    print(f"设备        : G2={g2_device}  G1={g1_device}")
    print(f"任务        : G2 seeds={seeds} ｜ G1 variants={variants} folds={folds} "
          f"epochs={a.epochs}")

    checks, verdict = precheck(
        g2_out=g2_out, g1_out=g1_out, data_root=data_root,
        g2_device=g2_device, g1_device=g1_device, seeds=seeds,
        variants=variants, folds=folds, epochs=a.epochs,
        baseline_dir=a.baseline_dir, reference_cv=a.reference_cv,
        probe=not (a.no_probe or a.dry_run),
    )
    print_checks(checks, verdict)

    failed = [c for c in checks if c.status == "FAIL"]
    if failed and not a.force:
        print("\n❌ 预检查发现硬冲突，已中止（未启动任何训练）：")
        for c in failed:
            print(f"   · {c.item}：{c.detail}")
        print("\n修正后再跑；若确知后果可用 --force 强行继续。")
        return 2
    if failed:
        print("\n⚠️ 存在硬冲突但 --force 已给出，按要求继续。")

    # ---- 决定执行方式 ----
    rec = verdict.get("recommended", "serial")
    if a.serial:
        mode = "serial"
    elif a.parallel:
        if verdict.get("verdict") == "SERIAL_REQUIRED" and not a.force:
            print("\n❌ 预检查判定**只能串行**（显存放不下 / 单卡），拒绝并行。")
            print("   若坚持：改用两张不同的卡（--g2-device / --g1-device），"
                  "或加 --force 自行承担 OOM 风险。")
            return 2
        mode = "parallel"
    else:
        mode = rec

    print(f"\n执行方式    : {mode}"
          + ("（G2 先跑完腾出整卡，再让 G1 独占；反之用 --order g1,g2）"
             if mode == "serial" else
             "（两个进程同时占 GPU，进度请分别看各自日志）"))

    jobs = [
        ("G2", _g2_cmd(g2_out, data_root, g2_device, seeds, a.dry_run),
         g2_out / "g2_run.log"),
        ("G1", _g1_cmd(g1_out, data_root, g1_device, variants, folds,
                       a.epochs, a.dry_run, a.reference_cv),
         g1_out / "g1_run.log"),
    ]
    order = [s.strip().lower() for s in a.order.split(",") if s.strip()]
    if mode == "serial" and order[:2] == ["g1", "g2"]:
        jobs = jobs[::-1]

    if a.plan:
        print("\n[--plan] 只做预检查，未启动任何训练。上面就是结论。")
        write_json(root / "g12_precheck.json",
                   {"checks": [{"item": c.item, "status": c.status,
                                "detail": c.detail} for c in checks],
                    "verdict": verdict, "mode_would_use": mode,
                    "g2_out": str(g2_out), "g1_out": str(g1_out),
                    "g2_device": g2_device, "g1_device": g1_device,
                    "seeds": seeds, "variants": variants, "folds": folds,
                    "epochs": a.epochs},
                   Path(__file__).resolve())
        return 0

    t0 = time.time()
    rcs = run_parallel(jobs) if mode == "parallel" else run_serial(jobs)

    print("\n" + "=" * 78)
    print(f"全部结束（{mode}），总用时 {fmt_hms(time.time() - t0)}")
    bad = {k: v for k, v in rcs.items() if v != 0}
    for k, v in rcs.items():
        print(f"  {k}: exit_code={v}  {'✅' if v == 0 else '❌'}")
    print(f"  G2 产物：{g2_out}")
    print(f"  G1 产物：{g1_out}")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
