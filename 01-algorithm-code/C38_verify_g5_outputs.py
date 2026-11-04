#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_g5.py · 第三段（g5）产物验收器 —— **只读、数个数、给退出码**。

为什么要有它：项目铁律「每句『已完成』都要能被可执行验证器当场证伪」。
g5 跑完后，本机只靠肉眼看 `g5_report.md` 不算验收——必须**数个数**。

用法
----
    python verify_g5.py                     # 自动找：cloud_results/20260928_g5 → _out_local/g5 → <cwd>/g5
    python verify_g5.py <g5_dir>            # 显式指定 g5 产物目录

退出码
------
    0 = 全部判据 PASS（消融 4 变体齐 + 多种子 3 种子 + 6 run 齐全 + 未碰 sealed test）
    1 = 至少一条判据 FAIL（**不要**据此往方案里写数字）
    2 = 找不到产物目录（或缺少核心文件）

本脚本**不写入任何文件**，不联网，不碰数据集。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BASELINE_FP = "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975"
EXPECTED_N_RUNS = 6
ABL_ALL = ["no_sam", "no_mdc", "no_ilf", "no_dual_path"]
HERE = Path(__file__).resolve().parent


def find_g5_dir(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_dir() else None
    cands = [
        HERE.parent / "cloud_results" / "20260928_g5",
        HERE.parent / "cloud_results" / "20260928_g5" / "g5",
        HERE / "_out_local" / "g5",
        Path.cwd() / "g5",
    ]
    for c in cands:
        if (c / "g5_summary.json").is_file():
            return c
    # 兜底：在整个 cloud_results 下找 g5_summary.json
    root = HERE.parent / "cloud_results"
    if root.is_dir():
        hits = sorted(root.rglob("g5_summary.json"))
        if hits:
            return hits[0].parent
    return None


def main() -> int:
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    g5 = find_g5_dir(explicit)
    if g5 is None:
        print("❌ 找不到 g5 产物目录（含 g5_summary.json）。")
        print("   请把云端 `/home/featurize/work/AIC_train_again/g5/` 整目录下载到：")
        print("   E:\\Code\\NeuroEvo\\DMSA_netV2\\cloud_results\\20260928_g5\\")
        print("   再跑： python verify_g5.py")
        return 2
    print("=" * 78)
    print(f"g5 产物目录 : {g5}")
    print("=" * 78)

    checks: list[tuple[str, bool, str]] = []

    # ---- 1. 产物文件 ----
    # 核心 5 件：正式跑必产出（缺 = FAIL）
    core = ["g5_progress.log", "g5_per_run.csv", "g5_summary.json",
            "g5_ablation_merged.json", "g5_report.md"]
    missing_p = [p for p in core if not (g5 / p).is_file()]
    checks.append(("核心 5 个产物文件齐全", not missing_p,
                   "缺: " + ", ".join(missing_p) if missing_p else "全在"))
    # g5_precheck.json：2026-09-28 起正式跑也会写；更早的跑（只 --plan 时写）可能没有 ⇒ 仅提示
    precheck_note = ""
    if not (g5 / "g5_precheck.json").is_file():
        precheck_note = ("ℹ️  未见到 g5_precheck.json（2026-09-28 之前跑的正式任务不会写它；"
                         "若需要，跑一次 `--plan` 即生成）—— 不作为 FAIL。")

    # ---- 2. run 目录数（数个数）----
    n_abl = len([d for d in (g5 / "abl" / "runs").iterdir() if d.is_dir()]) \
        if (g5 / "abl" / "runs").is_dir() else 0
    n_seed = len([d for d in (g5 / "seed" / "runs").iterdir() if d.is_dir()]) \
        if (g5 / "seed" / "runs").is_dir() else 0
    checks.append(("abl/runs == 2", n_abl == 2, f"实测 {n_abl}"))
    checks.append(("seed/runs == 4", n_seed == 4, f"实测 {n_seed}"))

    # ---- 3. per_run.csv：行数 + 参数匹配 + full 指纹 ----
    per_run = g5 / "g5_per_run.csv"
    rows_csv: list[dict] = []
    if per_run.is_file():
        with open(per_run, encoding="utf-8-sig", newline="") as f:
            rows_csv = list(csv.DictReader(f))
    checks.append((f"per_run.csv 行数 == {EXPECTED_N_RUNS}",
                   len(rows_csv) == EXPECTED_N_RUNS, f"实测 {len(rows_csv)}"))
    bad_pm = [r["label"] for r in rows_csv if str(r.get("params_matches_doc")).lower() not in ("true", "1")]
    checks.append(("每 run params_matches_doc == True", not bad_pm,
                   "不符: " + ", ".join(bad_pm) if bad_pm else "全部 True"))
    full_rows = [r for r in rows_csv if r.get("variant") == "full" and str(r.get("skipped")).lower() not in ("true", "1")]
    bad_fp = [r["label"] for r in full_rows
              if str(r.get("fingerprint", "")).strip() and r["fingerprint"].strip() != BASELINE_FP]
    checks.append(("full 指纹 == 基线（硬门）", not bad_fp,
                   "不符: " + ", ".join(bad_fp) if bad_fp else f"{len(full_rows)} 个 full run 全一致"))

    # ---- 4. summary ----
    s: dict = {}
    sp = g5 / "g5_summary.json"
    if sp.is_file():
        try:
            s = json.loads(sp.read_text(encoding="utf-8"))
        except Exception as e:
            s = {}
            print(f"⚠️ g5_summary.json 解析失败: {e}")
    checks.append(("summary.n_runs == 6", s.get("n_runs") == EXPECTED_N_RUNS,
                   f"实测 {s.get('n_runs')}"))
    stp = str(s.get("sealed_test_policy", ""))
    checks.append(("未碰 sealed test", "not touched" in stp.lower(), stp or "(缺)"))

    # ---- 5. 消融 4 变体齐 ----
    abl = s.get("ablation") or {}
    miss = abl.get("missing_variants")
    have = [r.get("variant") for r in (abl.get("rows") or []) if r.get("n_folds")]
    checks.append(("消融表无缺失变体", (not miss) and set(ABL_ALL).issubset(set(have)),
                   f"missing={miss} 有结果={have}"))
    checks.append(("消融 full 参照在位", bool(abl.get("full_reference_found")),
                   str(abl.get("full_reference_found"))))

    # ---- 6. 多种子 3 种子 + 跨种子极差 ----
    ms = s.get("multi_seed") or {}
    n_seeds = ms.get("n_seeds")
    checks.append(("多种子 n_seeds == 3（42参照+43/44）", n_seeds == 3, f"实测 {n_seeds}"))
    checks.append(("跨种子极差可算（n>=2）", ms.get("across_seed_spread_pp") is not None,
                   "None（种子<2 或参照缺失）" if ms.get("across_seed_spread_pp") is None
                   else f"{ms['across_seed_spread_pp']:.3f} pp"))

    # ---- 打印判据 ----
    print("\n[判据]")
    n_fail = 0
    for name, ok, detail in checks:
        flag = "✅ PASS" if ok else "❌ FAIL"
        if not ok:
            n_fail += 1
        print(f"  {flag}  {name:<34s} {detail}")
    if precheck_note:
        print(f"  {precheck_note}")

    # ---- 打印两张表 ----
    if abl.get("rows"):
        print("\n[消融合并表]")
        print(f'  {"variant":<16} {"params":>12} {"nf":>3} {"acc_mean":>9} {"ΔvsFull":>10} {"spread":>9}  origin')
        for r in abl["rows"]:
            am = "—" if r.get("accuracy_mean") is None else f'{r["accuracy_mean"]*100:.3f}%'
            dp = "—" if r.get("delta_vs_full_pp") is None else f'{r["delta_vs_full_pp"]:+.3f}pp'
            spr = "—" if r.get("fold_spread") is None else f'{r["fold_spread"]*100:.3f}pp'
            pm = "—" if r.get("params") is None else f'{r["params"]:,}'
            print(f'  {str(r.get("variant")):<16} {pm:>12} {r.get("n_folds", 0):>3} '
                  f'{am:>9} {dp:>10} {spr:>9}  {r.get("origin")}')
    if ms.get("rows"):
        print("\n[多种子表]")
        for r in ms["rows"]:
            tag = "（参照）" if r.get("reference") else ""
            pf = " ".join(f'f{k}={v*100:.2f}%' for k, v in (r.get("per_fold") or {}).items() if v is not None)
            mm = "—" if r.get("accuracy_mean") is None else f'{r["accuracy_mean"]*100:.3f}%'
            spr = "—" if r.get("spread_fold") is None else f'{r["spread_fold"]*100:.3f}pp'
            print(f'  seed{r.get("seed")}{tag:<6} {pf}  mean={mm}  折间跨度={spr}')
        if ms.get("across_seed_spread_pp") is not None:
            print(f'  → 跨种子极差 = {ms["across_seed_spread_pp"]:.3f} pp（{ms.get("n_seeds")} 个种子的折内均值）')

    print("\n" + "=" * 78)
    if n_fail == 0:
        print("✅ 全部判据 PASS —— g5 产物可用于「消融补 no_dual_path + 多种子 val/acc」两处提分。")
    else:
        print(f"❌ {n_fail} 条判据 FAIL —— **不要**据此往方案里写数字，先排查上面的明细。")
    print("提醒：折数只有 2（n=2 无 CI）；no_mdc/no_dual_path 容量不匹配 ⇒ Δ 不可归因单模块。")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
