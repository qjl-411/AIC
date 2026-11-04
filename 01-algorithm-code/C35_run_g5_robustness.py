#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""g5_run.py · 第三段实验（最省补完）· 2026-09-27

===============================================================================
一句话
===============================================================================
把 **★ 项已经免费拿完** 之后剩下的两项，按**最省方案**补到「可用」：
  阶段 1 `ABL`  ：模块消融补 **1 个变体**（`no_dual_path`）× 折 {1,5}   → 2 run
  阶段 2 `SEED` ：多随机种子 **seed 43/44** × 折 {1,5}（分组 · full）   → 4 run
合计 **6 run ≈ 15.75 h ≈ ¥38**（单 run 2.625 h，单价按反推值 ¥2.4/h）。

===============================================================================
为什么是这个组合（不是别的）
===============================================================================
1. **消融只缺 1 个变体**，不是"跑得不够"：
   `g1_ablation.py` 的 `DEFAULT_VARIANTS = ["no_sam","no_mdc","no_ilf"]` **本来就不含**
   `no_dual_path`（而 `VARIANT_SWITCHES` 支持它）。所以补齐只需显式指定，无需改代码。
2. **多种子不能用 `g2_multi_seed.py` 补**：它的 `--mode cv` 是**脚本主动封禁**的
   （源码原文：「cv 模式成本为 5 折 × N 种子……本脚本不代跑 cv 模式以免误用」），
   而 `--mode final` 是 126 epoch **全量训练、没有 val/acc** ⇒ 产不出"跨种子准确率"。
   因此多种子必须走**五折**路线（本脚本的 SEED 阶段）。
3. **只跑折 {1,5}** 是为了与**已有的 seed42 折 {1,5} 参照**（full = 99.170%）同折对齐，
   否则折不同、不可比。

===============================================================================
本脚本刻意**不做**的三件事（都是踩过的坑）
===============================================================================
* **不写 `g1/` 目录**：`g1_ablation.py` 的 `summarize()` 只用"本次 results"重建表并覆盖
  `g1_ablation_summary.json`——再跑一次会把原有 3 变体的表冲掉。
  本脚本把结果写到 **`g5/abl/runs/`**，另**只读** g1 已有结果来**合并**成完整 4 变体表。
* **不改基线**：不 import / 不引用 `final_train.py`，全部训练走 `smoke_test.py`（与基线同源）。
* **不碰 sealed test**：只评折内 val。

===============================================================================
用法
===============================================================================
    python g5_run.py --plan              # 只预检 + 报总时长/费用（不训练，几秒）
    python g5_run.py --dry-run --device cpu   # CPU 接线自检（变体构造 + 指纹硬门）
    python g5_run.py                     # 前台正式跑（6 run）
    # 后台跑（脱离终端，见 RUNBOOK §6.9）：
    nohup python -u g5_run.py --quiet > g5_console.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_SEED,
    EXPECTED_TRAIN_PER_FOLD,
    FIXED_DATA_ROOT,
    PKG_VERSION,
    ProgressReporter,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    assert_train_manifest,
    base_parser,
    cpu_smoke,
    find_baseline_dir,
    fmt_hms,
    human_mb,
    load_module,
    make_tee_sink,
    probe_device,
    read_manifest,
    resolve_data_root,
    resolve_out_dir,
    safe_rmtree,
    sha256_file,
    write_csv,
    write_json,
)

import C34_run_g4_full_cv as G4  # noqa: E402  （复用其 train_one_run / variant_probe，见文件头说明）

OUT_TAG = "g5"
EXPECTED_PKG_VERSION = "2026-09-24.4"

# 反推单价（**不是平台报价**）：由「105 h ≈ ¥250」反推 ¥2.38/h，取 ¥2.4。
COST_PER_HOUR_CNY = 2.4

# ---------------------------------------------------------------------------
# 阶段定义。**顺序 = 实际执行顺序**：先便宜且信息增益明确的消融，再多种子。
# ---------------------------------------------------------------------------
STAGE_SPECS: Dict[str, dict] = {
    "ABL": {
        "title": "第三段·1 消融补完（no_dual_path × 折 {1,5}）",
        "subdir": "abl",
        "arm": "grouped",
        "variant": "no_dual_path",
        "seeds": (BASE_SEED,),
        "folds": (1, 5),
        "why": "g1 当年只跑了 no_sam/no_mdc/no_ilf（脚本默认值不含 no_dual_path）",
    },
    "SEED": {
        "title": "第三段·2 多种子（full · seed 43/44 × 折 {1,5}）",
        "subdir": "seed",
        "arm": "grouped",
        "variant": "full",
        "seeds": (43, 44),
        "folds": (1, 5),
        "why": "g2 只有 126ep final 的 train loss，无 val/acc ⇒ 必须用五折拿跨种子准确率",
    },
}
STAGE_ORDER = ["ABL", "SEED"]
DEFAULT_STAGES = ",".join(STAGE_ORDER)

# 消融合并表要覆盖的 4 个变体（3 个来自 g1 已有产物 + 1 个由本脚本产出）
ABL_ALL_VARIANTS = ["no_sam", "no_mdc", "no_ilf", "no_dual_path"]
# g1 已有产物所在（只读；**绝不写入**）
DEFAULT_G1_CANDIDATES = (
    "g1",
)
# full 参照（折 1,5）的搜索位置：优先 g4 的 A1（同一套协议/同一张卡），再退到复现版五折
DEFAULT_REF_CANDIDATES = (
    "复现版实验数据",
)


# =============================================================================
# 1. 任务矩阵
# =============================================================================
def build_jobs(stages: Sequence[str], folds_override: Optional[Sequence[int]]) -> List[dict]:
    """把阶段展开成逐 run 的清单。`folds_override` 是**全局**的（与 g4 同口径）。"""
    jobs: List[dict] = []
    for key in STAGE_ORDER:
        if key not in stages:
            continue
        spec = STAGE_SPECS[key]
        folds = tuple(folds_override) if folds_override else tuple(spec["folds"])
        for seed in spec["seeds"]:
            for fold in folds:
                jobs.append({
                    "stage": key,
                    "stage_title": spec["title"],
                    "subdir": spec["subdir"],
                    "arm": spec["arm"],
                    "variant": spec["variant"],
                    "seed": int(seed),
                    "fold": int(fold),
                    "label": f"{spec['arm']}__{spec['variant']}__seed{int(seed)}__fold{int(fold)}",
                })
    seen = set()
    for j in jobs:
        k = (j["arm"], j["variant"], j["seed"], j["fold"])
        if k in seen:
            raise ValueError(f"任务矩阵有重复项：{k}")
        seen.add(k)
    return jobs


# =============================================================================
# 2. 读结果：schema 容错
# =============================================================================
def _read_metrics(path: Path) -> Optional[dict]:
    """读一个 best_metrics.json，**两种 schema 都认**。

    * g4/g5 产物：`{"best": {"accuracy": .., "epoch": ..}, "params": .., ...}`
    * 复现版五折参照：**扁平** `{"accuracy": .., "epoch": ..}`
    认不出来返回 None（不猜、不编）。
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    b = d.get("best")
    if isinstance(b, dict) and b.get("accuracy") is not None:
        return {"accuracy": float(b["accuracy"]), "epoch": b.get("epoch"),
                "params": d.get("params"), "raw": d}
    if d.get("accuracy") is not None:
        return {"accuracy": float(d["accuracy"]), "epoch": d.get("epoch"),
                "params": d.get("params"), "raw": d}
    return None


def locate_full_reference(folds: Sequence[int], *, out_root: Path,
                          g1_dir: Optional[Path] = None,
                          explicit: Optional[str] = None,
                          extra_dirs: Sequence[Path] = ()) -> Dict[int, dict]:
    """找 `full` 的折级 val 准确率，作为消融 Δ 与多种子 seed42 的参照。

    优先级（**先同一套协议、同一张卡**，再退到历史复现版）：
      1) `--reference-cv` 显式指定
      2) `g4/runs/grouped__full__seed42__fold<f>/best_metrics.json`   ← 最优
         （与 B/A3/A4 同一份脚本、同一张卡、逐折对齐 ⇒ Δ 才谈得上可比）
      3) `<extra_dirs>/Fold<f>/best_metrics.json`（复现版五折；**扁平** schema）
      4) `<g1_dir>/g1_ablation_summary.json` 的 `full` 参照行
         （它就是 3) 被 g1 持久化下来的那份，provenance 见该文件的 `full_reference_note`）
    """
    foldset = sorted({int(x) for x in folds})
    out: Dict[int, dict] = {}
    bases: List[Path] = []
    if explicit:
        bases.append(Path(explicit).expanduser())
    bases += [Path(p) for p in extra_dirs if p]

    # ---- 2) g4 的 A1 ----
    for f in foldset:
        p = (out_root / "g4" / "runs" /
             f"grouped__full__seed{BASE_SEED}__fold{f}" / "best_metrics.json")
        if p.is_file():
            m = _read_metrics(p)
            if m:
                out[f] = dict(m, source=f"g4 A1 {p.parent.name}")
    if len(out) >= len(foldset):
        return out

    # ---- 3) 显式 / 额外目录下的 Fold<f> ----
    for f in foldset:
        if f in out:
            continue
        for b in bases:
            hit = next((c for c in (b / f"Fold{f}" / "best_metrics.json",
                                    b / f"fold_{f}" / "best_metrics.json") if c.is_file()),
                       None)
            if hit is not None:
                m = _read_metrics(hit)
                if m:
                    out[f] = dict(m, source=f"{b.name}/Fold{f}")
                    break

    # ---- 4) g1 汇总表里的 full 参照行 ----
    if len(out) < len(foldset) and g1_dir:
        smy = Path(g1_dir) / "g1_ablation_summary.json"
        if smy.is_file():
            try:
                d = json.loads(smy.read_text(encoding="utf-8"))
            except Exception:
                d = {}
            row = next((t for t in (d.get("table") or [])
                        if t.get("variant") == "full"), None)
            for k, v in ((row or {}).get("per_fold") or {}).items():
                try:
                    f = int(k)
                except Exception:
                    continue
                if f in foldset and f not in out and v is not None:
                    out[f] = {"accuracy": float(v), "epoch": None,
                              "params": (row or {}).get("params"),
                              "source": "g1_ablation_summary 的 full 参照行"}
    return out


def collect_g1_existing(g1_dir: Optional[Path], folds: Sequence[int]) -> Dict[str, dict]:
    """**只读** g1 已有产物，取到 3 个变体的折级准确率。

    优先读 `g1_ablation_summary.json`（**关键**：`no_sam/fold_1/` 的逐 run 产物已丢失，
    但汇总表里保留了 per_fold 值），再退到逐 run 的 best_metrics.json。
    """
    out: Dict[str, dict] = {}
    if not g1_dir or not g1_dir.is_dir():
        return out

    smy = g1_dir / "g1_ablation_summary.json"
    if smy.is_file():
        try:
            d = json.loads(smy.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        for row in d.get("table") or []:
            v = row.get("variant")
            pf = row.get("per_fold") or {}
            picked = {int(k): float(vv) for k, vv in pf.items()
                      if vv is not None and int(k) in set(map(int, folds))}
            if v:
                out[v] = {"params": row.get("params"), "per_fold": picked,
                          "source": "g1_ablation_summary.json"}

    # 逐 run 兜底（补汇总表里没有的）
    for v in ABL_ALL_VARIANTS:
        rec = out.setdefault(v, {"params": None, "per_fold": {}, "source": "per-run"})
        for f in folds:
            f = int(f)
            if f in rec["per_fold"]:
                continue
            for cand in (g1_dir / v / f"fold_{f}" / "best_metrics.json",
                         g1_dir / v / f"fold_{f}" / "best_metrics.json"):
                m = _read_metrics(cand)
                if m:
                    rec["per_fold"][f] = m["accuracy"]
                    if rec["params"] is None:
                        rec["params"] = m["params"]
                    break
    return {k: v for k, v in out.items() if v["per_fold"]}


def collect_g5_ablation(abl_runs_root: Path, folds: Sequence[int]) -> Dict[str, dict]:
    """读本脚本自己的消融产物（`g5/abl/runs/<label>/best_metrics.json`）。"""
    out: Dict[str, dict] = {}
    if not abl_runs_root.is_dir():
        return out
    for d in sorted(abl_runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = _read_metrics(d / "best_metrics.json")
        if not m:
            continue
        try:
            _, variant, _, fold = d.name.split("__")
            fold = int(fold.replace("fold", ""))
        except Exception:
            continue
        rec = out.setdefault(variant, {"params": m.get("params"), "per_fold": {},
                                       "source": "g5 abl"})
        rec["per_fold"][fold] = m["accuracy"]
        if rec["params"] is None:
            rec["params"] = m.get("params")
    return out


# =============================================================================
# 3. 汇总
# =============================================================================
def _stat(vals: Sequence[float]) -> dict:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None, "spread": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals),
            "min": min(vals), "max": max(vals),
            "spread": (max(vals) - min(vals)) if len(vals) >= 2 else None}


def merge_ablation(*, folds: Sequence[int], full_ref: Dict[int, dict],
                   g1: Dict[str, dict], g5: Dict[str, dict]) -> dict:
    """合成 **4 变体 × 折{folds}** 的消融表（3 个来自 g1 + 1 个来自 g5）。"""
    folds = [int(f) for f in folds]
    merged: Dict[str, dict] = {}
    for v in ABL_ALL_VARIANTS:
        if v in g5:
            merged[v] = dict(g5[v], origin="g5（本次）")
        elif v in g1:
            merged[v] = dict(g1[v], origin="g1（已有）")
        else:
            merged[v] = {"params": None, "per_fold": {}, "origin": "缺失"}

    full_pf = {f: (full_ref.get(f) or {}).get("accuracy") for f in folds}
    full_st = _stat([v for v in full_pf.values() if v is not None])

    rows: List[dict] = []
    if full_st["n"]:
        rows.append({
            "variant": "full", "params": 56812813,
            "per_fold": full_pf, "n_folds": full_st["n"],
            "accuracy_mean": full_st["mean"], "fold_spread": full_st["spread"],
            "origin": (full_ref.get(folds[0]) or {}).get("source", "参照"),
            "delta_vs_full_pp": 0.0,
        })
    for v in ABL_ALL_VARIANTS:
        rec = merged[v]
        pf = {f: rec["per_fold"].get(f) for f in folds}
        st = _stat([x for x in pf.values() if x is not None])
        rows.append({
            "variant": v, "params": rec.get("params"),
            "per_fold": pf, "n_folds": st["n"],
            "accuracy_mean": st["mean"], "fold_spread": st["spread"],
            "origin": rec.get("origin"),
            "delta_vs_full_pp": (None if (st["mean"] is None or full_st["mean"] is None)
                                 else (st["mean"] - full_st["mean"]) * 100.0),
        })

    missing = [r["variant"] for r in rows if r["n_folds"] == 0]
    return {
        "folds": folds,
        "rows": rows,
        "full_reference_found": bool(full_st["n"]),
        "missing_variants": missing,
        "note": (
            "Δ 只在 `full` 参照可用时给出；参照优先取 g4 的 A1（同一套脚本/同一张卡/逐折对齐），"
            "缺失时退到复现版五折（不同脚本、不同卡 ⇒ Δ 只能作参考性对照）。"
        ),
    }


def build_seed_table(*, folds: Sequence[int], full_ref: Dict[int, dict],
                     results: Sequence[dict]) -> dict:
    """多种子表：seed42（参照，来自 full_ref）+ 本次跑出的 seed 43/44。"""
    folds = [int(f) for f in folds]
    per_seed: Dict[int, Dict[int, Optional[float]]] = {}
    for f in folds:
        per_seed.setdefault(BASE_SEED, {})[f] = (full_ref.get(f) or {}).get("accuracy")
    for r in results:
        if r.get("stage") != "SEED":
            continue
        per_seed.setdefault(int(r["seed"]), {})[int(r["fold"])] = r.get("accuracy")

    rows: List[dict] = []
    for seed in sorted(per_seed):
        pf = per_seed[seed]
        st = _stat([v for v in pf.values() if v is not None])
        rows.append({
            "seed": seed, "per_fold": pf, "n": st["n"],
            "accuracy_mean": st["mean"], "spread_fold": st["spread"],
            "reference": seed == BASE_SEED,
        })
    means = [r["accuracy_mean"] for r in rows if r["accuracy_mean"] is not None]
    return {
        "folds": folds,
        "rows": rows,
        "across_seed_mean": (sum(means) / len(means)) if means else None,
        "across_seed_spread_pp": ((max(means) - min(means)) * 100.0) if len(means) >= 2 else None,
        "n_seeds": len(means),
        "note": (
            "`spread_fold` 是**折间**离散度；`across_seed_*` 才是**跨种子**口径。"
            "n 个种子算极差/标准差统计效力很弱，n=2 时只能说「方向上一致/不一致」。"
        ),
    }


# =============================================================================
# 4. 报告
# =============================================================================
def write_report(*, out_dir: Path, jobs: Sequence[dict], results: Sequence[dict],
                 elapsed: float, abl: dict, seed: dict, plan_only: bool = False) -> None:
    L: List[str] = []
    L.append("# G5 · 第三段实验报告（最省补完）\n")
    L.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　包版本：{PKG_VERSION}\n")
    L.append(f"> 本段 = **{len(jobs)} run**（消融补 1 变体×2 折 + 多种子 2 seed×2 折）；"
             f"实测用时 **{fmt_hms(elapsed)}**。\n")
    L.append("> ⚠️ 评估集一律为**折内 val**；sealed test **未被触碰**。\n")

    L.append("\n## 1. 逐 run 清单\n")
    L.append("| # | 阶段 | arm | 变体 | seed | 折 | 结果 | best epoch | 用时 |")
    L.append("|---:|---|---|---|---:|---:|---|---:|---|")
    for i, j in enumerate(jobs, 1):
        r = next((x for x in results if x.get("label") == j["label"]), None)
        acc = "—" if not r or r.get("accuracy") is None else f"{r['accuracy']*100:.3f}%"
        be = "—" if not r or r.get("best_epoch") is None else r["best_epoch"]
        el = "—" if not r or not r.get("elapsed_seconds") else fmt_hms(r["elapsed_seconds"])
        L.append(f"| {i} | {j['stage']} | {j['arm']} | `{j['variant']}` | {j['seed']} | "
                 f"{j['fold']} | {acc} | {be} | {el} |")

    L.append("\n## 2. 消融合并表（4 变体 × 折 " + ",".join(str(f) for f in abl["folds"]) + "）\n")
    if not abl["full_reference_found"]:
        L.append("> ⚠️ **未找到 `full` 参照**（g4 的 A1 与复现版五折都不在），"
                 "因此本表**没有 Δ 列**。补 Δ 的办法：把 `复现版实验数据/Fold{1..5}` 放上来，"
                 "或用 `--reference-cv` 指定；也可见 `g5_ablation_merged.json` 在本地补算。\n")
    L.append("| 变体 | 参数量 | ΔParams | 折数 | val 准确率(均) | Δ vs full | 折间跨度 | 各折 | 来源 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in abl["rows"]:
        pm = "—" if r["params"] is None else f"{r['params']:,}"
        dpa = ("—" if (r["params"] is None or r["params"] == 56812813)
               else f"{(r['params'] - 56812813):+,}")
        if r["variant"] == "full":
            dpa = "—"
        acc = "—" if r["accuracy_mean"] is None else f"{r['accuracy_mean']*100:.3f}%"
        dl = "—" if r["delta_vs_full_pp"] is None else f"{r['delta_vs_full_pp']:+.3f} pp"
        sp = "—" if r["fold_spread"] is None else f"{r['fold_spread']*100:.3f} pp"
        pf = " ".join(f"f{k}={v*100:.2f}%" for k, v in r["per_fold"].items() if v is not None)
        L.append(f"| `{r['variant']}` | {pm} | {dpa} | {r['n_folds']} | {acc} | {dl} | {sp} | "
                 f"{pf or '—'} | {r['origin']} |")
    if abl["missing_variants"]:
        L.append(f"\n> ⚠️ 缺以下变体（未跑也未找到）：{', '.join(abl['missing_variants'])}\n")
    L.append("\n> **归因边界（必须随表一起引用）**：`no_mdc`（−11,775,012）与 "
             "`no_dual_path`（−10,059,776）**同时改变了模型容量**，"
             "因此其精度差**不能归因给单个模块**，只能表述为「移除 X 后的整体效果（含容量变化）」。"
             "`no_sam`（ΔParams −459）是唯一容量几乎不变、可干净归因的消融。"
             "`no_ilf` 实测**未收敛**（各折 best_epoch=2、acc≈0.25、val_loss≈ln4），"
             "只能写「移除 ILF 后训练不收敛」，**不得**主张「ILF 的门控/融合机制不可或缺」。\n")

    L.append("\n## 3. 多种子表（跨种子口径）\n")
    L.append("| seed | " + " | ".join(f"折 {f}" for f in seed["folds"]) +
             " | 均值 | 折间跨度 |")
    L.append("|---:|" + "---:|" * (len(seed["folds"]) + 2))
    for r in seed["rows"]:
        cells = " | ".join(
            "—" if r["per_fold"].get(f) is None else f"{r['per_fold'][f]*100:.3f}%"
            for f in seed["folds"])
        mean = "—" if r["accuracy_mean"] is None else f"{r['accuracy_mean']*100:.3f}%"
        sp = "—" if r["spread_fold"] is None else f"{r['spread_fold']*100:.3f} pp"
        tag = "（参照）" if r["reference"] else ""
        L.append(f"| {r['seed']}{tag} | {cells} | {mean} | {sp} |")
    if seed["across_seed_spread_pp"] is not None:
        L.append(f"\n**跨种子极差 = {seed['across_seed_spread_pp']:.3f} pp**"
                 f"（基于 {seed['n_seeds']} 个种子的折内均值）\n")
    else:
        L.append("\n> 跨种子极差：**算不出**（可用种子数 < 2，或参照缺失）。\n")

    L.append("\n## 4. 诚实限制（随数字一起写，不要藏）\n")
    L.append("1. **折数只有 2**（折 1、5），n=2 **没有置信区间** ⇒ 不得写「显著」；"
             "只能在 |Δ| > 折间跨度时谈「方向」。")
    L.append("2. **统计单位**：项目铁律要求「源图 + cluster bootstrap CI」，"
             "本表两条都不满足（只有折均值）。")
    L.append("3. **参照对齐**：若 Δ 的 full 行来自**复现版五折**（而非 g4 的 A1），"
             "则两侧的**训练脚本/显卡/数据根目录**可能不同（G1 曾记录 3 处未对齐）⇒ Δ 只作参考性对照。")
    L.append("4. **多种子只有 2 个新种子**（真源要求 2–3 个），"
             "极差/标准差的统计效力弱。")
    L.append("5. 本段**不评 sealed test**，也不改变 L1 / L2 / L5（客观限制）三条局限。")

    L.append("\n## 5. 产物\n")
    for p in ("g5_precheck.json", "g5_progress.log", "g5_per_run.csv",
              "g5_summary.json", "g5_ablation_merged.json", "g5_report.md"):
        L.append(f"- `{out_dir / p}`")
    (out_dir / "g5_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# =============================================================================
# 5. main
# =============================================================================
def main() -> int:
    ap = base_parser("G5 · 第三段（最省补完）：消融补 no_dual_path×2折 + 多种子 seed43/44×2折")
    ap.add_argument("--stages", default=DEFAULT_STAGES,
                    help=f"逗号分隔（默认 {DEFAULT_STAGES}）；可选 {STAGE_ORDER}")
    ap.add_argument("--folds", default=None,
                    help="**全局**折号覆盖（默认各阶段自带 1,5）")
    ap.add_argument("--data-root", default=None,
                    help=f"Combined Dataset 路径（默认 {FIXED_DATA_ROOT}）")
    ap.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BASE_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--arch-probe", default=None, help="arch_probe.py 路径")
    ap.add_argument("--reference-cv", default=None,
                    help="full 折级参照目录（内含 Fold1..5/best_metrics.json）")
    ap.add_argument("--g1-dir", default=None,
                    help="g1 产物目录（默认 <out-root>/g1）；**只读**，用于合并 4 变体表")
    ap.add_argument("--max-batches", type=int, default=None, help="调试用：每 epoch 最多几个 batch")
    ap.add_argument("--quiet", action="store_true",
                    help="后台跑用：不转发基线 tqdm（否则日志会被刷成几百 MB 单行）")
    ap.add_argument("--plan", action="store_true", help="只预检 + 报时长/费用，不训练")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：已有 best_metrics.json 的 run 自动跳过")
    ap.add_argument("--progress-interval", type=float, default=5.0)
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="遇到已存在的 run 产物就报错（默认：重跑即覆盖，路径不变）")
    ap.add_argument("--overwrite-dry-run", action="store_true",
                    help="只预演覆盖会删掉什么，不真删、不训练")
    a = ap.parse_args()

    stages = [s.strip().upper() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGE_SPECS]
    if unknown:
        print(f"❌ 未知阶段 {unknown}；可选 {STAGE_ORDER}", flush=True)
        return 2
    folds_override = ([int(f.strip()) for f in a.folds.split(",") if f.strip()]
                      if a.folds else None)
    if folds_override:
        for f in folds_override:
            if not 1 <= f <= 5:
                print(f"❌ 非法折号 {f}（只允许 1–5）", flush=True)
                return 2

    out_dir = resolve_out_dir(a.out_dir, OUT_TAG)
    out_root = out_dir.parent          # `g4/` `g1/` `复现版实验数据/` 都在这一层
    runs_roots = {k: out_dir / STAGE_SPECS[k]["subdir"] / "runs" for k in STAGE_ORDER}
    jobs = build_jobs(stages, folds_override)

    bdir = find_baseline_dir(a.baseline_dir)
    print("=" * 78)
    print("G5 · 第三段实验（最省补完）")
    print("=" * 78)
    print(f"gpu_lib 版本 : {PKG_VERSION}（期望 {EXPECTED_PKG_VERSION}）", flush=True)
    print(f"baseline dir : {bdir}")
    print(f"out dir      : {out_dir}")
    print(f"g5_run.py    : {Path(__file__).resolve()}  sha256={sha256_file(Path(__file__))[:16]}…")

    # ---- 协议守卫（硬门：manifest SHA）----
    man = bdir / TRAIN_MANIFEST_NAME
    if not man.is_file():
        print(f"❌ 找不到 {man}", flush=True)
        return 2
    man_sha = sha256_file(man)
    if man_sha != TRAIN_MANIFEST_SHA256:
        print(f"❌ train manifest SHA 不符！\n  expected {TRAIN_MANIFEST_SHA256}\n"
              f"  actual   {man_sha}", flush=True)
        return 2
    rows = read_manifest(man)
    assert_train_manifest(rows)
    class_to_idx = {n: i for i, n in enumerate(sorted({r["class_name"] for r in rows}))}
    print(f"协议校验     : PASS  rows={len(rows):,}  classes={list(class_to_idx)}")

    data_root = resolve_data_root(a.data_root)
    train_root = Path(data_root).expanduser() / "train"
    print(f"data root    : {data_root}     ← 云上排错第一件事就是核对这一行")
    if not a.dry_run and not a.plan and not train_root.is_dir():
        print(f"❌ 找不到 train 目录：{train_root}", flush=True)
        return 2

    g1_dir = Path(a.g1_dir).expanduser() if a.g1_dir else (out_root / "g1")
    ref_dirs = [out_root / c for c in DEFAULT_REF_CANDIDATES]
    full_ref = locate_full_reference(
        sorted({j["fold"] for j in jobs}), out_root=out_root, g1_dir=g1_dir,
        explicit=a.reference_cv, extra_dirs=ref_dirs)

    # ---- 任务与成本 ----
    hpr = G4.MEASURED_SECONDS_PER_RUN / 3600.0
    total_h = len(jobs) * hpr
    print(f"任务         : {len(jobs)} run"
          f"（阶段 {','.join(stages)}；折 {sorted({j['fold'] for j in jobs})}）")
    for j in jobs:
        print(f"   - [{j['stage']}] {j['label']}  ← {STAGE_SPECS[j['stage']]['why']}")
    print(f"预计用时     : {total_h:.2f} h（单 run {hpr:.3f} h，按 G1 六个 run 实测均值）")
    print(f"预计费用     : ≈ ¥{total_h * COST_PER_HOUR_CNY:.0f}"
          f"（单价 ¥{COST_PER_HOUR_CNY}/h 为**反推值**，不是平台报价）")
    print(f"full 参照    : {'找到 ' + str(len(full_ref)) + ' 折（' + str((full_ref or {}).get(sorted(full_ref)[0], {}).get('source', '?')) + '）' if full_ref else '❌ 未找到（消融表将没有 Δ）'}")
    print(f"g1 已有产物  : {'在 ' + str(g1_dir) if g1_dir.is_dir() else '❌ 不在（' + str(g1_dir) + '）——合并表将只有 1 个变体'}")

    # ---- dry-run：CPU 接线自检 ----
    if a.dry_run:
        print("\n-- dry-run：变体构造 + 指纹硬门（CPU）------------------", flush=True)
        res = cpu_smoke(bdir, max_batches=a.max_batches or 1)
        print(f"  基线冒烟     : {'PASS' if res.get('ok') else 'FAIL'}  "
              f"params={res.get('params')}  fp={str(res.get('fingerprint'))[:12]}…  "
              f"loss={res.get('loss')}")
        probe = G4.variant_probe(bdir, sorted({"full"} | {j["variant"] for j in jobs}))
        gate = False
        for p in probe:
            if p.get("params") is None:
                print(f"    {str(p.get('variant')):14s} ❌ {p.get('error', '无法构造')}")
            else:
                print(f"    {p['variant']:14s} params={p['params']:>12,} "
                      f"Δ{(p.get('params_delta') or 0):>+11,} "
                      f"fwd={'OK' if p.get('forward_ok') else 'FAIL'} "
                      f"{'[=baseline]' if p.get('equals_baseline') else ''}")
            if p.get("variant") == "full":
                gate = bool(p.get("ok"))
        ok_all = bool(res.get("ok")) and gate
        print(f"\n{'✅' if ok_all else '❌'} dry-run {'通过：`full` 与基线同构，待跑变体均可构造并前向。' if ok_all else '未通过：见上。'}")
        write_json(out_dir / "g5_dryrun.json",
                   {"cpu_smoke": res, "variants": probe, "gate_passed": gate,
                    "jobs": jobs, "folds": sorted({j["fold"] for j in jobs})},
                   Path(__file__))
        return 0 if ok_all else 2

    # ---- 预检记录：**--plan 与正式跑都会落盘** ----
    # ⚠️ 曾经只在 `--plan` 分支里写，但 `g5_report.md` §5 把它列入产物清单
    #    ⇒ 正式跑（不带 --plan）时报告会"列出一个不存在的文件"（报告说谎）。
    precheck = {
        "mode": "plan" if a.plan else "train",
        "pkg_version": PKG_VERSION, "baseline_dir": str(bdir),
        "data_root": data_root, "out_dir": str(out_dir),
        "manifest_sha256": man_sha, "jobs": jobs,
        "estimated_hours": total_h,
        "cost_cny_per_hour": COST_PER_HOUR_CNY,
        "cost_cny_estimated": total_h * COST_PER_HOUR_CNY,
        "full_reference_folds": sorted(full_ref),
        "g1_dir": str(g1_dir), "g1_found": g1_dir.is_dir(),
        "sealed_test_policy": "not touched; val only",
    }
    write_json(out_dir / "g5_precheck.json", precheck, Path(__file__))

    if a.plan:
        print(f"\n[--plan] 预检通过，未训练。预检明细 → {out_dir / 'g5_precheck.json'}")
        return 0

    # ---- 覆盖策略（只清本次 planned 的 run 目录）----
    # ⚠️ 顺序很关键：`--resume` 要复用的产物**绝不能先被清掉**，否则 resume 永远失效
    #    （这是本脚本自检时抓到的真 bug：清理放在 resume 判断之前 ⇒ 6/6 全部重跑）。
    cleared: List[dict] = []
    kept: List[str] = []
    for j in jobs:
        d = runs_roots[j["stage"]] / j["label"]
        if not d.exists():
            continue
        if a.resume and (d / "best_metrics.json").is_file():
            kept.append(j["label"])          # 要复用 → 不删
            continue
        if not a.overwrite:
            print(f"❌ 产物已存在：{d}（当前是 --no-overwrite）", flush=True)
            return 2
        cleared.append(safe_rmtree(d, must_be_under=out_dir, dry=a.overwrite_dry_run))
    if kept:
        print(f"覆盖清理     : --resume 保护 {len(kept)} 个已完成 run，未删除"
              f"（{'、'.join(kept[:3])}{'…' if len(kept) > 3 else ''}）")
    if cleared:
        tf = sum(c.get("files", 0) for c in cleared)
        tb = sum(c.get("bytes", 0) for c in cleared)
        verb = "将删除" if a.overwrite_dry_run else "已清理"
        print(f"覆盖清理     : {verb} {len(cleared)} 个旧目录、{tf} 个文件、{human_mb(tb)}")
    if a.overwrite_dry_run:
        print("\n[--overwrite-dry-run] 预演结束：上面就是将被删除的内容。")
        return 0

    # ---- 设备 & 模块 ----
    device, dev_info = probe_device(a.device)
    print(f"device       : {device}  {dev_info.get('device_name')}")
    smoke_mod = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke")
    arch_probe = G4.load_arch_probe([a.arch_probe])
    print(f"smoke_test   : {bdir / 'smoke_test.py'}")
    print(f"arch_probe   : {arch_probe.__file__}")

    sink = make_tee_sink(out_dir / "g5_progress.log")
    print(f"进度日志     : {out_dir / 'g5_progress.log'}")
    if not a.quiet:
        print("提示         : 前台跑可看到基线 tqdm；后台请加 `--quiet`（见 RUNBOOK §6.9）")

    # ---- 进度：跨 run 全局口径 ----
    steps_per_epoch = (EXPECTED_TRAIN_PER_FOLD + a.batch_size - 1) // a.batch_size
    if a.max_batches:
        steps_per_epoch = min(steps_per_epoch, a.max_batches)
    units_per_run = a.epochs * steps_per_epoch
    done_pre = 0
    if a.resume:
        for j in jobs:
            if (runs_roots[j["stage"]] / j["label"] / "best_metrics.json").is_file():
                done_pre += units_per_run
    rep = ProgressReporter(label="G5", total_epochs=a.epochs, steps_per_epoch=steps_per_epoch,
                           total_units=len(jobs) * units_per_run, sink=sink,
                           min_interval=a.progress_interval, base_units=done_pre)

    results: List[dict] = []
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        run_dir = runs_roots[j["stage"]] / j["label"]
        if a.resume and (run_dir / "best_metrics.json").is_file():
            m = _read_metrics(run_dir / "best_metrics.json")
            results.append({"stage": j["stage"], "label": j["label"], "arm": j["arm"],
                            "variant": j["variant"], "seed": j["seed"], "fold": j["fold"],
                            "accuracy": (m or {}).get("accuracy"),
                            "best_epoch": (m or {}).get("epoch"),
                            "elapsed_seconds": None, "skipped": True})
            print(f"\n[跳过 {i}/{len(jobs)}] {j['label']}（已有产物）", flush=True)
            continue

        print("\n" + "=" * 78, flush=True)
        print(f"[{i}/{len(jobs)}] {j['stage_title']}", flush=True)
        print(f"  run = {j['label']}   （{STAGE_SPECS[j['stage']]['why']}）", flush=True)
        print("=" * 78, flush=True)

        rec = G4.train_one_run(
            smoke_mod=smoke_mod, arch_probe=arch_probe,
            arm=j["arm"], variant=j["variant"], seed=j["seed"], fold=j["fold"],
            rows=rows, class_to_idx=class_to_idx, train_root=train_root,
            device=device, epochs=a.epochs, batch_size=a.batch_size,
            num_workers=a.num_workers, out_run_dir=run_dir,
            max_batches=a.max_batches, rep=rep, sink=sink,
            run_index=i, run_total=len(jobs),
            tqdm_passthrough=not a.quiet,
        )
        rec.update({"stage": j["stage"], "label": j["label"]})
        results.append(rec)

    elapsed = time.time() - t0
    rep.total_epochs = 0
    rep.set_base(0)
    rep.set(epoch=0, step=0, loss=None, extra="G5 全部 run 完成")
    rep.emit(force=True)

    # ---- 汇总（消融合并 + 多种子）----
    all_folds = sorted({j["fold"] for j in jobs})
    g1 = collect_g1_existing(g1_dir, all_folds)
    g5_abl = collect_g5_ablation(runs_roots["ABL"], all_folds)
    abl = merge_ablation(folds=all_folds, full_ref=full_ref, g1=g1, g5=g5_abl)
    seed = build_seed_table(folds=all_folds, full_ref=full_ref, results=results)

    summary = {
        "experiment": "G5 · 第三段（最省补完）",
        "pkg_version": PKG_VERSION,
        "stages": stages,
        "folds": all_folds,
        "epochs": a.epochs, "batch_size": a.batch_size,
        "manifest_sha256": man_sha,
        "architecture_fingerprint": ARCHITECTURE_FINGERPRINT,
        "sealed_test_policy": "not touched; val only",
        "n_runs": len(jobs),
        "elapsed_seconds": elapsed,
        "estimated_hours": total_h,
        "cost_cny_per_hour": COST_PER_HOUR_CNY,
        "ablation": abl,
        "multi_seed": seed,
        "results": results,
        "capacity_warning": (
            "`no_mdc`（ΔParams -11,775,012）与 `no_dual_path`（-10,059,776）同时改变模型容量，"
            "其精度差**不可归因单模块**；`no_sam`（-459）是唯一容量匹配的消融。"
            "`no_ilf` 实测未收敛（best_epoch=2，acc≈0.25），只能写「移除后训练不收敛」。"
        ),
        "limitation_note": (
            "折数 2、n=2 无 CI；统计单位非源图、无 cluster bootstrap；"
            "多种子仅 2 个新种子；Δ 的参照若来自复现版五折则含脚本/显卡/数据根目录差异。"
        ),
    }
    write_json(out_dir / "g5_summary.json", summary, Path(__file__))
    write_json(out_dir / "g5_ablation_merged.json", abl, Path(__file__))
    write_csv(out_dir / "g5_per_run.csv", results,
              ["stage", "label", "arm", "variant", "seed", "fold", "params",
               "fingerprint", "params_matches_doc", "best_epoch", "accuracy",
               "precision", "recall", "f1", "kappa", "val_loss",
               "elapsed_seconds", "skipped"])
    write_report(out_dir=out_dir, jobs=jobs, results=results, elapsed=elapsed,
                 abl=abl, seed=seed)

    print("\n" + "=" * 78)
    print(f"G5 完成：{len(jobs)} run，实跑用时 {fmt_hms(elapsed)}")
    print("=" * 78)
    for p in ("g5_summary.json", "g5_ablation_merged.json", "g5_per_run.csv",
              "g5_report.md", "g5_progress.log"):
        print(f"  → {out_dir / p}")
    if abl["missing_variants"]:
        print(f"  ⚠️ 消融表缺变体：{', '.join(abl['missing_variants'])}")
    if not abl["full_reference_found"]:
        print("  ⚠️ 未找到 full 参照 ⇒ 消融表没有 Δ 列（放上 复现版实验数据/ 或用 "
              "--reference-cv 即可补）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
