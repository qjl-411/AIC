#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC 提分实验 G0：一键编排
===============================================================================

在 GPU 机器上按顺序跑完三项实验，最后汇总成一份可直接写进方案的报告。

---------------------------------------------------------------------------
推荐执行顺序（按"性价比"从高到低）
---------------------------------------------------------------------------

    1. G3 效率实测     ~0.2 h    补齐 ★ 项最后 3 个指标（最便宜）
    2. G2 多种子       ~8–16 h   跨种子稳健性（收益/成本比最高）
    3. G1 消融矩阵     ~16–32 h  模块贡献（最容易被评委追问）

    G1 与 G2 之间没有依赖，理论上可以并行（如果你有两张卡）。

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------

    # 本机（无 GPU）：只自检三个脚本的接线
    python g0_run_all.py --dry-run

    # GPU 机器：全跑
    python g0_run_all.py \
        --data-root "/home/featurize/work/Combined_Dataset" \
        --out-root  "/home/featurize/work/dmsa_aic_boost" \
        --device cuda:0 \
        --seeds 43,44 \
        --variants no_sam,no_mdc,no_ilf \
        --folds 1,5

    # 只跑最便宜的一步（先验证环境）
    python g0_run_all.py --only g3 ...

---------------------------------------------------------------------------
铁的约束（脚本会强制检查，不通过就拒绝跑）
---------------------------------------------------------------------------

  ○ train manifest SHA-256 必须等于 51f5b251…edfd
  ○ `full` 变体指纹必须等于 8e542321…69d7
  ○ **sealed test 全程不被触碰**
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    FIXED_DATA_ROOT,
    FIXED_OUT_ROOT,
    PKG_VERSION,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    base_parser,
    find_baseline_dir,
    resolve_data_root,
    resolve_out_dir,
    sha256_file,
    write_json,
)

HERE = Path(__file__).resolve().parent

STEPS: Dict[str, dict] = {
    "g3": {
        "script": "g3_efficiency.py",
        "title": "G3 效率指标实测（★ 补齐 GPU 三项）",
        "cost_h": "~0.2 h",
        "args": ["--batch-sizes", "1,16"],
    },
    "g2": {
        "script": "g2_multi_seed.py",
        "title": "G2 多随机种子重复（★★）",
        "cost_h": "8–18 h",
        "args": ["--mode", "final", "--keep-code"],
    },
    "g1": {
        "script": "g1_ablation.py",
        "title": "G1 模块消融矩阵（★★）",
        "cost_h": "16–32 h",
        "args": [],
    },
}

DEFAULT_ORDER = ["g3", "g2", "g1"]


def run_step(
    key: str, *, py: str, data_root: Optional[str], out_root: Path,
    device: str, extra: Dict[str, List[str]], dry: bool,
) -> dict:
    spec = STEPS[key]
    script = HERE / spec["script"]
    if not script.is_file():
        raise FileNotFoundError(f"找不到 {script}")

    out_dir = out_root / key
    cmd = [
        py, str(script),
        "--device", device,
        "--out-dir", str(out_dir),
    ]
    if data_root and key in ("g1", "g2"):
        cmd += ["--data-root", data_root]
    cmd += spec["args"]
    cmd += extra.get(key, [])
    if dry:
        cmd.append("--dry-run")

    print("\n" + "#" * 78)
    print(f"# {spec['title']}")
    print(f"# 预计成本 {spec['cost_h']}   输出 {out_dir}")
    print("#" * 78)
    print("$ " + " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, text=True, errors="replace")
    elapsed = time.time() - t0

    return {
        "step": key,
        "title": spec["title"],
        "exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "out_dir": str(out_dir),
        "command": " ".join(cmd),
    }


def build_report(results: List[dict], out_root: Path, script_path: Path) -> None:
    lines: List[str] = []
    lines.append("# 2026AIC 提分实验 · 运行报告\n")
    lines.append("> 本报告由 `g0_run_all.py` 自动生成。")
    lines.append("> **sealed test 全程未被触碰**；所有指标均来自五折 val。\n")

    lines.append("## 1. 运行状态\n")
    lines.append("| 步骤 | 描述 | 耗时 | 退出码 |")
    lines.append("|---|---|---:|---:|")
    for r in results:
        mark = "✅" if r["exit_code"] == 0 else f"❌ {r['exit_code']}"
        lines.append(
            f"| `{r['step']}` | {r['title']} | {r['elapsed_hours']:.2f} h | {mark} |"
        )
    lines.append("")

    def _load(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ---- G3 ----
    g3 = _load(out_root / "g3" / "g3_efficiency.json")
    if g3:
        m = g3.get("model", {})
        fl = g3.get("flops", {})
        ref = g3.get("resnext50_reference") or {}
        lines.append("## 2. 效率指标（G3）\n")
        lines.append("| 指标 | 数值 | 状态 |")
        lines.append("|---|---:|---|")
        lines.append(f"| 参数量 | {m.get('params_total', 0):,} | ✅ 实测 |")
        if ref:
            lines.append(f"| ResNeXt50-32x4d 参数量 | {ref.get('params', 0):,} | ✅ 实测 |")
            if ref.get("params"):
                lines.append(
                    f"| 参数量比值 | {m.get('params_total', 0)/ref['params']:.2f}× | ✅ |"
                )
        lines.append(f"| FLOPs | {fl.get('gflops', 0):.3f} G | ✅ 实测 |")
        lines.append(f"| MACs | {fl.get('macs', 0)/1e9:.3f} G | ✅ 实测 |")
        lat = g3.get("latency", {})
        for k, v in lat.items():
            if not isinstance(v, dict):
                continue
            if v.get("measurable"):
                bs = v["batch_size"]
                lines.append(
                    f"| GPU 延迟 (bs={bs}) | {v['latency_ms_median']:.2f} ms | ✅ 实测 |"
                )
                lines.append(
                    f"| GPU 吞吐 (bs={bs}) | {v['throughput_img_per_s']:.1f} img/s | ✅ 实测 |"
                )
                lines.append(
                    f"| GPU 峰值显存 (bs={bs}) | {v['peak_memory_mb']:.1f} MB "
                    f"({v['peak_memory_pct_of_total']:.1f}%) | ✅ 实测 |"
                )
        lines.append(f"\n> {g3.get('summary', {}).get('note', '')}\n")

    # ---- G2 ----
    g2 = _load(out_root / "g2" / "g2_multiseed_summary.json")
    if g2:
        lines.append("## 3. 多随机种子（G2）\n")
        lines.append("| seed | 记录 epoch 数 | 首 epoch train_loss | 末 epoch train_loss |")
        lines.append("|---:|---:|---:|---:|")
        for r in g2.get("per_seed", []):
            lines.append(
                f"| {r['seed']} | {r['n_epochs_logged']} | "
                f"{r.get('first_train_loss')} | {r.get('final_train_loss')} |"
            )
        l = g2.get("final_train_loss", {})
        if "stdev" in l:
            lines.append(
                f"\n最终 train loss：均值 **{l['mean']:.6f}** ± **{l['stdev']:.6f}** "
                f"（min {l['min']:.6f} / max {l['max']:.6f}）\n"
            )
        lines.append(f"> {g2.get('limitation_note', '')}\n")

    # ---- G1 ----
    g1 = _load(out_root / "g1" / "g1_ablation_summary.json")
    if g1:
        lines.append("## 4. 模块消融（G1）\n")
        lines.append("| 变体 | 参数量 | 折数 | val 准确率(均) | Δ vs full | 折间跨度 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for t in g1.get("table", []):
            d = t.get("delta_vs_full")
            sp = t.get("fold_spread")
            lines.append(
                f"| `{t['variant']}` | {t['params']:,} | {t['n_folds']} | "
                f"{(t['accuracy_mean']*100) if t['accuracy_mean'] is not None else float('nan'):.3f}% | "
                f"{('%+.3f pp' % (d*100)) if d is not None else '—'} | "
                f"{(('%.3f pp' % (sp*100)) if sp is not None else '—')} |"
            )
        lines.append(f"\n> **容量不匹配**：{g1.get('capacity_warning', '')}\n")
        lines.append(f"> **统计效力**：{g1.get('limitation_note', '')}\n")

    lines.append("## 5. 免责与口径\n")
    lines.append("- 所有消融/多种子数字均为**折内验证集（val）**指标。")
    lines.append("- **sealed test 只被使用过一次**（seed=42 full 模型，")
    lines.append("  accuracy 0.989835809226），本批实验后仍然只用那一次。")
    lines.append("- 若要在方案里写「多种子 test 指标」，**必须另找从未用过的测试数据**。")

    p = out_root / "G0_运行报告.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[write_md] {p}")


def main() -> int:
    ap = base_parser("DMSA-Net 2026AIC 提分实验 · 一键编排")
    ap.add_argument("--data-root", default=None,
                    help=f"Combined Dataset 路径（默认已固定为 {FIXED_DATA_ROOT}）")
    ap.add_argument("--out-root", default=None,
                    help=f"所有实验的输出根目录（默认已固定为 {FIXED_OUT_ROOT}）")
    # base_parser 会带一个 --out-dir；在 g0 里它与 --out-root 同义，避免"传了没反应"
    ap.add_argument("--seeds", default="43,44")
    ap.add_argument("--variants", default="no_sam,no_mdc,no_ilf")
    ap.add_argument("--folds", default="1,5")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--only", default=None,
                    help=f"只跑某一步，逗号分隔。可选：{','.join(STEPS)}")
    ap.add_argument("--skip", default=None, help="跳过某一步，逗号分隔")

    # base_parser 给的 --out-dir 帮助文案提到了 "<tag>"，但 g0 没有 tag 层级；
    # 这里就地改掉，避免用户按文案去猜一个并不存在的子目录。
    for _act in ap._actions:                      # noqa: SLF001
        if _act.dest == "out_dir":
            _act.help = (f"输出目录（在 g0 中与 --out-root 同义；"
                         f"默认已固定为 {FIXED_OUT_ROOT}）")

    a = ap.parse_args()

    script_path = Path(__file__).resolve()
    # ⚠️ g0 没有自己的 <tag>，所以它的输出根 = --out-root；同时把 base_parser 带来的
    #    --out-dir 当同义词接受，否则用户传 --out-dir 会**静默失效**并落到默认位置。
    out_root_arg = a.out_root if a.out_root else a.out_dir
    if a.out_root and a.out_dir and str(a.out_root) != str(a.out_dir):
        raise SystemExit(
            "--out-root 与 --out-dir 同时给出且不一致，请只留一个：\n"
            f"  --out-root = {a.out_root}\n  --out-dir  = {a.out_dir}"
        )
    out_root = resolve_out_dir(out_root_arg, "")
    out_root.mkdir(parents=True, exist_ok=True)

    bdir = find_baseline_dir(a.baseline_dir)

    print("=" * 78)
    print("2026AIC 提分实验 · 编排")
    print("=" * 78)
    print(f"baseline dir : {bdir}")
    print(f"out root     : {out_root}")

    # ---- 全局守卫（在跑任何东西之前）----
    man = bdir / TRAIN_MANIFEST_NAME
    man_sha = sha256_file(man)
    if man_sha != TRAIN_MANIFEST_SHA256:
        raise RuntimeError(
            f"train manifest SHA 不符！\nExpected: {TRAIN_MANIFEST_SHA256}\nActual: {man_sha}"
        )
    print(f"manifest SHA : {man_sha[:16]}… ✅")
    print(f"arch fp      : {ARCHITECTURE_FINGERPRINT[:16]}… (由各子脚本复核)")

    order = DEFAULT_ORDER
    if a.only:
        order = [k.strip() for k in a.only.split(",") if k.strip()]
    if a.skip:
        sk = {k.strip() for k in a.skip.split(",") if k.strip()}
        order = [k for k in order if k not in sk]
    for k in order:
        if k not in STEPS:
            raise ValueError(f"未知步骤 {k}；可选 {list(STEPS)}")

    # ---- 每步的额外参数 ----
    extra: Dict[str, List[str]] = {
        "g1": ["--variants", a.variants, "--folds", a.folds,
               "--epochs", str(a.epochs)],
        "g2": ["--seeds", a.seeds],
    }

    # 数据根目录已定死为基线自己的 DATA_ROOT，--data-root 现在是可选覆盖项。
    a.data_root = resolve_data_root(a.data_root)
    print(f"data root    : {a.data_root}")
    print(f"包版本      : {PKG_VERSION}")

    print(f"执行顺序     : {' → '.join(order)}")
    print(f"dry-run      : {a.dry_run}")

    write_json(
        out_root / "g0_plan.json",
        {
            "order": order, "data_root": a.data_root, "device": a.device,
            "seeds": a.seeds, "variants": a.variants, "folds": a.folds,
            "epochs": a.epochs, "dry_run": bool(a.dry_run),
            "baseline_dir": str(bdir), "manifest_sha256": man_sha,
            "architecture_fingerprint": ARCHITECTURE_FINGERPRINT,
            "sealed_test_policy": "not touched by any step",
        },
        script_path,
    )

    results: List[dict] = []
    for k in order:
        r = run_step(
            k, py=sys.executable, data_root=a.data_root, out_root=out_root,
            device=a.device, extra=extra, dry=a.dry_run,
        )
        results.append(r)
        if r["exit_code"] != 0:
            print(f"\n❌ 步骤 {k} 失败（退出码 {r['exit_code']}），停止后续步骤。")
            break

    write_json(out_root / "g0_results.json", {"results": results}, script_path)
    build_report(results, out_root, script_path)

    print("\n" + "=" * 78)
    print("编排完成")
    for r in results:
        mark = "✅" if r["exit_code"] == 0 else "❌"
        print(f"  {mark} {r['step']}  {r['elapsed_hours']:.2f} h")
    print(f"报告：{out_root / 'G0_运行报告.md'}")
    print("=" * 78)
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
