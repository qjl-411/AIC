#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 0.0d 检测器判别力实测（v2.7 新增）
======================================

【为什么必须有这一条】
2026-09-23 在真实数据上跑 A1 时，产出把**每类 2,560 张全部连成一个连通分量**
（`uniq_merged = 1`、`confidence` 全为 `unresolved`、`cross_matches.csv` 1.8 GB）。
两个候选解释必须分开：

  **H1（工程）** 阈值太低 —— `dedup_config.json` 的 `primary_th = 4` 是**未标定的占位符**
                 （A1-cal 的合成压力测试给的是 17）。
  **H2（方法）** **pHash 在 128×128 灰度脑 MRI 上几乎没有判别力** —— 更严重，
                 因为它意味着"再标定一次阈值"也救不回来。

本脚本用**真实数据**区分二者。它测三条距离分布（45°步 × 翻转，|G|=16，min-over-pairs）：

  A. **null 池**：`train/<最大类>` 内的**随机无关对**（默认 No Impairment，manifest 自述无重复）
  B. **exact 正例**：manifest `group_reason == exact_duplicate` 的组内对
  C. **near 正例**：manifest `group_reason == near_duplicate_component` 的组内对

并输出每个候选阈值下的 TPR / FPR。

⚠️ **本脚本能说与不能说的**
- **能说**：某个阈值在**这个 null 池**上的 FPR；合成建议阈值的**不可迁移性**。
- **不能说**：真实的派生比例、独立源数。
- ⚠️ **null 池本身不能保证干净**：若 `d = 0` 的对里有真实近重复混入，则测出的 FPR 是**上界**；
  反之若 pHash 有碰撞，则 FPR 也含碰撞。**本数据没有 verified-negative 池** ——
  这本身就是 G0-C（`blocked_missing_lineage`）的一个独立理由。

用法：
    python a0c_detector_power.py --data "<Combined Dataset>" --cache _a1v2_cache --out _a0_out
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import pathlib
import random
from typing import Dict, List, Tuple

import numpy as np

from B15_lib_deduplication import TransformSet, HashCache, hamming_min_over_T, sha256_file
from B16_lib_provenance_audit import write_artifact

PAIR = Tuple[Tuple[str, str, str], Tuple[str, str, str]]


def _hist(d: List[int]) -> dict:
    c = collections.Counter(int(x) for x in d)
    q = np.percentile(d, [1, 5, 25, 50, 75, 95, 99]).tolist() if d else [0] * 7
    return dict(n=len(d), hist={str(k): v for k, v in sorted(c.items())},
                quantiles={k: round(v, 2) for k, v in
                           zip(["P1", "P5", "P25", "P50", "P75", "P95", "P99"], q)},
                min=int(min(d)) if d else None, max=int(max(d)) if d else None)


def main():
    ap = argparse.ArgumentParser(description="A1 检测器判别力实测（真实数据）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default="../DMSA_net/pcv_v2_folds_seed42.csv")
    ap.add_argument("--cache", default="_a1v2_cache")
    ap.add_argument("--out", default="_a0_out")
    ap.add_argument("--config", default="dedup_config.json")
    ap.add_argument("--null-class", default="No Impairment",
                    help="用作 null 池的类（应选 manifest 自述无重复的类）")
    ap.add_argument("--n-null-files", type=int, default=160)
    ap.add_argument("--max-pos-pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    data = pathlib.Path(a.data).resolve()
    cfg = json.loads(pathlib.Path(a.config).read_text(encoding="utf-8")) \
        if pathlib.Path(a.config).is_file() else {}
    ts = TransformSet(cfg.get("angle_step", 45), cfg.get("use_flip", True))
    cache = HashCache(pathlib.Path(a.cache), ts)
    memo: Dict[str, np.ndarray] = {}

    def H1(split: str, cls: str, fname: str) -> np.ndarray:
        k = f"{split}|{cls}|{fname}"
        if k in memo:
            return memo[k]
        p = data / split / cls / fname
        s = sha256_file(p)
        v = cache.get(k, s)
        if v is None:
            from PIL import Image
            v = ts.hashes(Image.open(p))
        v = np.asarray(v, dtype=np.uint64).reshape(1, -1)
        memo[k] = v
        return v

    def dists(pairs: List[PAIR]) -> List[int]:
        out = []
        for (s1, c1, f1), (s2, c2, f2) in pairs:
            D = hamming_min_over_T(H1(s1, c1, f1), H1(s2, c2, f2),
                                   mode="two-sided", block=64)
            out.append(int(D[0, 0]))
        return out

    print("=" * 74)
    print("阶段 0.0d 检测器判别力实测（真实数据）")
    print("=" * 74)
    rng = random.Random(a.seed)
    rep: dict = dict(data_root=str(data), transform_set=dict(
        angle_step=cfg.get("angle_step", 45), use_flip=cfg.get("use_flip", True),
        n_transforms=len(ts)), null_class=a.null_class,
        cache_entries=len(list(pathlib.Path(a.cache).glob("*.npy"))))

    # ---- A. null 池 -------------------------------------------------------
    nf = sorted(p.name for p in (data / "train" / a.null_class).glob("*.jpg"))
    pick = rng.sample(nf, min(a.n_null_files, len(nf)))
    Hn = np.vstack([H1("train", a.null_class, f) for f in pick])
    Dn = hamming_min_over_T(Hn, Hn, mode="two-sided", block=256)
    iu = np.triu_indices(len(pick), k=1)
    d_null = Dn[iu].tolist()
    rep["A_null"] = _hist(d_null)
    print(f"\n--- A. null 池：train/{a.null_class} 随机无关对 (n={len(d_null)}) ---")
    print("  " + "  ".join(f"{k}:{v}" for k, v in rep["A_null"]["hist"].items()))
    print(f"  分位 {rep['A_null']['quantiles']}  范围 [{rep['A_null']['min']}, {rep['A_null']['max']}]")

    # ---- B/C. manifest 自述分组 -------------------------------------------
    with io.open(a.manifest, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_gid: Dict[str, List[dict]] = collections.defaultdict(list)
    reason_of: Dict[str, str] = {}
    for r in rows:
        by_gid[r["group_id"]].append(r)
        reason_of[r["group_id"]] = r["group_reason"]

    for key, reason in (("B_exact", "exact_duplicate"),
                        ("C_near", "near_duplicate_component")):
        pairs: List[PAIR] = []
        for gid, mem in by_gid.items():
            if reason_of[gid] != reason:
                continue
            cls = mem[0]["class_name"]
            fns = sorted(m["filename"] for m in mem)
            for i in range(len(fns)):
                for j in range(i + 1, len(fns)):
                    pairs.append((("train", cls, fns[i]), ("train", cls, fns[j])))
        if len(pairs) > a.max_pos_pairs:
            pairs = rng.sample(pairs, a.max_pos_pairs)
        d = dists(pairs)
        rep[key] = _hist(d)
        rep[key]["group_reason"] = reason
        rep[key]["sampled"] = len(pairs) < sum(
            len(m) * (len(m) - 1) // 2 for g, m in by_gid.items()
            if reason_of[g] == reason)
        print(f"\n--- {key}：manifest 自述 {reason} (n={len(d)}) ---")
        print("  " + "  ".join(f"{k}:{v}" for k, v in rep[key]["hist"].items()))
        print(f"  分位 {rep[key]['quantiles']}  范围 [{rep[key]['min']}, {rep[key]['max']}]")

    # ---- PR 表 ------------------------------------------------------------
    dn = np.asarray(d_null)
    dpos = np.asarray([int(k) for k, v in rep["C_near"]["hist"].items() for _ in range(v)])
    print("\n--- 阈值表（正例 = manifest 自述 near 组；负例 = null 池）---")
    print("  th   TPR(near)      FPR(null)      null 绝对对数")
    rows_pr = []
    for th in list(range(0, 13)) + [16, 17, 20]:
        fpr = float((dn <= th).mean())
        tpr = float((dpos <= th).mean()) if dpos.size else float("nan")
        rows_pr.append(dict(th=th, tpr=tpr, fpr=fpr,
                            n_null_le=int((dn <= th).sum())))
        print(f"  {th:2d}   {tpr:8.4f}      {fpr:9.5f}      {int((dn<=th).sum()):6d}")
    rep["pr_table"] = rows_pr

    cal_th = None
    cal_p = pathlib.Path(cfg.get("output_dirs", {}).get("a1cal_out", "_a1cal_out")) \
        / "suggested_thresholds.json"
    if cal_p.is_file():
        cal_th = json.loads(cal_p.read_text(encoding="utf-8")).get("primary_th")
    rep["a1cal_suggested_primary_th"] = cal_th
    rep["config_primary_th"] = cfg.get("primary_th")
    if cal_th is not None:
        row = next((r for r in rows_pr if r["th"] == cal_th), None)
        if row:
            print(f"\n  ⚠️ A1-cal（**合成压力测试**）建议 primary_th = {cal_th}"
                  f" → 在**真实数据**上 FPR = {row['fpr']:.5f}"
                  f"（null 池 {row['n_null_le']}/{len(dn)} 对会被判为同源）")
            rep["a1cal_th_on_real_data"] = row

    # ---- 判读 -------------------------------------------------------------
    verdict = []
    d0 = rep["A_null"]["hist"].get("0", 0)
    verdict.append(f"null 池中 d=0 的对：{d0}/{len(dn)}（{d0/max(len(dn),1)*100:.2f}%）"
                   " —— 这些是真实近重复还是低分辨率 pHash 碰撞**无法区分**")
    if cal_th is not None:
        row = next((r for r in rows_pr if r["th"] == cal_th), None)
        if row and row["fpr"] > 0.5:
            verdict.append(f"A1-cal 建议阈值 {cal_th} 的 FPR = {row['fpr']:.4f} > 0.5 "
                           "⇒ **合成压力测试的阈值不可迁移到真实数据**")
    low = [r for r in rows_pr if r["fpr"] <= 0.05 and r["tpr"] >= 0.9]
    verdict.append("存在 FPR≤5% 且 TPR≥90% 的阈值：" +
                   (f"是（th={[r['th'] for r in low]}）" if low else "**否**"))
    verdict.append("本数据**没有 verified-negative 池** ⇒ 任何阈值都无法被**验证**；"
                   "G0-C（blocked_missing_lineage）因此有独立于 lineage 的第二个理由")
    rep["verdict"] = verdict
    print("\n判读：")
    for v in verdict:
        print(f"  · {v}")
    print("\n  ⚠️ 本脚本测的是**检测器的判别力**，不是派生比例。"
          "不得据此估算独立源数或泄漏量。")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_artifact(out / "detector_power.json", rep)
    md = ["# 阶段 0.0d 检测器判别力实测（真实数据）\n",
          f"- 数据：`{data}`",
          f"- 变换集：{rep['transform_set']}",
          f"- null 池：`train/{a.null_class}` 随机对（n={rep['A_null']['n']}）",
          f"- 缓存条目：{rep['cache_entries']}\n",
          "## 距离分布\n",
          "| 集合 | n | 分布 | 分位 P1/P5/P25/P50/P75/P95/P99 |",
          "|---|---|---|---|"]
    for k, name in (("A_null", f"null（`train/{a.null_class}` 随机对）"),
                    ("B_exact", "exact 正例（manifest 自述）"),
                    ("C_near", "near 正例（manifest 自述）")):
        h = rep[k]
        md.append(f"| {name} | {h['n']} | " +
                  " ".join(f"{a_}×{b_}" for a_, b_ in h["hist"].items()) + " | " +
                  "/".join(str(x) for x in h["quantiles"].values()) + " |")
    md += ["", "## 阈值表\n", "| th | TPR(near) | FPR(null) | null 绝对对数 |", "|---|---|---|---|"]
    for r in rows_pr:
        md.append(f"| {r['th']} | {r['tpr']:.4f} | {r['fpr']:.5f} | {r['n_null_le']} |")
    md += ["", "## 判读\n"] + [f"- {v}" for v in verdict]
    md += ["", "> ⚠️ 本脚本测的是**检测器的判别力**，不是派生比例。",
           "> 不得据此估算独立源数或泄漏量。",
           "> null 池本身不能保证干净：若含真实近重复则 FPR 为上界，若含 pHash 碰撞则 FPR 也含碰撞。"]
    (out / "detector_power.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出：{out/'detector_power.json'}、{out/'detector_power.md'}")


if __name__ == "__main__":
    main()
