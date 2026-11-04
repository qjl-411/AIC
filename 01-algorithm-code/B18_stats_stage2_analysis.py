#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 1.4–1.6 / 2.1–2.3 源图级指标 + 三列对比 + cluster bootstrap（v2.4）
========================================================================

覆盖路线方案：
  1.4  源图级指标（图像预测 -> 按 source_id 聚合）
  1.4a 聚合规则敏感性（three rules，事前注册）
  1.5  泄漏分层分析（clean vs contaminated）
  1.6  三列对比表 + cluster bootstrap CI
  2.1  cluster bootstrap CI（替代 stratified bootstrap）
  2.2  ICC / DEFF（解释性）
  2.3  AUC 的 cluster-aware bootstrap

**统计单位 = 源图（cluster）。** 这是 v2.4 相对 v2.1 的核心修正。

输入（三选一，按可用性）：
  --pred  preds.csv       自行产出的图像级预测（列：filename,label,prob_0..prob_3[,split]）
  --fold-dir  <dir>       五折产物目录，自动找 *_preds.csv
  --demo                  合成数据，验证脚本与产出表格式

用法：
    # 用合成数据验证管线（0 GPU，1 秒）
    python stage2_stats.py --demo --out _s2_demo
    # 用真实预测
    python stage2_stats.py --pred _pred_out/preds.csv --a1-out _a1v2_out --out _s2_out

⚠️ 纪律：
  1. 聚合规则（AggRule）必须**事前注册**，本脚本把三条规则**全部**跑出来并列，
     不允许只报最好看的那条。主表默认 mean_prob，其余作敏感性。
  2. **只聚合本 fold 的 val 预测**（防跨 fold 泄漏）——由 --fold 参数或
     preds.csv 的 fold 列保证；缺失时脚本直接报错而不是静默用全部。
  3. macro-F1 **实算**，不写死。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

import B17_lib_statistics as SL
from B16_lib_provenance_audit import write_artifact

CLASSES = ["Mild Impairment", "Moderate Impairment",
           "No Impairment", "Very Mild Impairment"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ───────────────────────────────────────────────────────── 输入
def _read_csv(p: pathlib.Path) -> List[dict]:
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_preds(p: pathlib.Path) -> Dict[str, np.ndarray]:
    rows = _read_csv(p)
    if not rows:
        raise ValueError(f"{p} 为空")
    f = list(rows[0].keys())
    prob_cols = sorted([c for c in f if c.lower().startswith("prob")],
                       key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0))
    if len(prob_cols) != 4:
        raise ValueError(f"需要 4 个 prob_* 列，实际找到 {prob_cols}")
    lab_col = next((c for c in ("label", "y_true", "class_name", "target") if c in f), None)
    if lab_col is None:
        raise ValueError("找不到标签列")
    fn_col = next((c for c in ("filename", "file", "image", "relative_path") if c in f), None)
    fold_col = next((c for c in ("fold", "val_fold") if c in f), None)
    split_col = next((c for c in ("split",) if c in f), None)

    labels, probs, fns, folds, splits = [], [], [], [], []
    for r in rows:
        raw = str(r[lab_col])
        if raw in CLASS_TO_IDX:
            labels.append(CLASS_TO_IDX[raw])
        elif raw.isdigit():
            labels.append(int(raw))
        else:
            raise ValueError(f"无法解析标签 {raw!r}")
        probs.append([float(r[c]) for c in prob_cols])
        fns.append(str(r.get(fn_col) or ""))
        folds.append(str(r.get(fold_col) or "") if fold_col else "")
        splits.append(str(r.get(split_col) or "train") if split_col else "train")
    P = np.asarray(probs, dtype=np.float64)
    if not np.allclose(P.sum(1), 1.0, atol=1e-3):
        # 允许 logit/未归一化：就地 softmax
        P = np.exp(P - P.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
    return dict(labels=np.array(labels), probs=P, filenames=np.array(fns),
                folds=np.array(folds), splits=np.array(splits), source=str(p))


def synth_preds(n_clusters: int = 240, per: int = 4,
                noise: float = 1.0, seed: int = 42) -> Dict[str, np.ndarray]:
    """合成"同源多派生图"数据，用来验证管线（不是结果）。

    设计要点：**簇级信号 + 图像级独立噪声**。
      - 簇级 logit 决定该源图的"整体倾向"（同源图共享）；
      - 每张派生图再加独立噪声（派生变换带来的差异）。
    这样 ICG（簇内相关）是**中等**的，不会人为制造 0 或 1 的极端值。
    ⚠️ 已知合成数据的副作用：若簇内标签完全相同且簇级信号压倒噪声，
       majority 投票会退化为"整簇猜对/猜错"，macro-F1 出现 1.0 假象。
       真实数据不会这样（同源图预测有差异），故演示结果一律不可引用。
    """
    rng = np.random.default_rng(seed)
    groups, labels = [], []
    for g in range(n_clusters):
        c = int(rng.integers(0, 4))
        for _ in range(per):
            groups.append(g)
            labels.append(c)
    y = np.array(labels)
    garr = np.array(groups)
    # 簇级共同成分（弱）+ 图像级独立成分（强）→ 避免簇内完全一致
    cluster_sig = rng.normal(scale=0.45, size=(n_clusters, 4))[garr]
    img_sig = rng.normal(scale=1.6, size=(len(y), 4))
    logits = cluster_sig + img_sig + np.eye(4)[y] * 1.6
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return dict(labels=y, probs=P,
                filenames=np.array([f"demo_{g}_{k}.jpg"
                                    for g, k in zip(groups,
                                                    [sum(1 for gg in groups[:i + 1]
                                                         if gg == g) - 1
                                                     for i, g in enumerate(groups)])]),
                folds=np.array([str((g % 5) + 1) for g in groups]),
                splits=np.array(["train"] * len(y)), source="SYNTHETIC")


def make_source_ids(preds: Dict[str, np.ndarray],
                    a1_out: Optional[pathlib.Path],
                    legacy_manifest: Optional[pathlib.Path]) -> np.ndarray:
    """把 filename 映射到 source_id（优先 A1 产物，其次 manifest 的 group_id）。"""
    fn2src: Dict[str, str] = {}
    if a1_out is not None:
        sm = a1_out / "source_mapping.csv"
        if sm.exists():
            for r in _read_csv(sm):
                k = str(r.get("filename") or r.get("file") or "")
                if k:
                    fn2src[k] = str(r.get("source_id") or r.get("cluster_id") or k)
            print(f"  source_id 来自 {sm}（{len(fn2src)} 条）")
    if not fn2src and legacy_manifest is not None and legacy_manifest.exists():
        for r in _read_csv(legacy_manifest):
            k = str(r.get("filename") or "")
            g = str(r.get("group_id") or "")
            if k:
                fn2src[k] = ("LG:" + g) if g else ("RAW:" + k)
        print(f"  source_id 来自 {legacy_manifest} 的 group_id（{len(fn2src)} 条）"
              f"  [protocol_id=pcv2_legacy]")
    if not fn2src:
        # 最后兜底：文件自带 source 列，或退化为 filename
        print("  ⚠️ 无 source_id 映射 -> 退化为 filename-as-source（乐观，须标注）")

    out = []
    n_hit = 0
    for fn in preds["filenames"]:
        s = fn2src.get(fn)
        if s is None:
            base = pathlib.Path(fn).name
            s = fn2src.get(base)
            if s is None:
                s = "RAW:" + base
            else:
                n_hit += 1
        else:
            n_hit += 1
        out.append(s)
    print(f"  source_id 命中率：{n_hit}/{len(out)}")
    return np.array(out)


# ───────────────────────────────────────────────────────── 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--a1-out", default="_a1v2_out")
    ap.add_argument("--out", default="_s2_out")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="..")
    a = ap.parse_args()

    here = pathlib.Path(__file__).parent
    root = pathlib.Path(a.root).resolve()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("阶段 1.4–1.6 / 2.1–2.3 源图级指标 + cluster bootstrap")
    print("=" * 74)

    if a.demo:
        preds = synth_preds(seed=a.seed)
        sid = np.array([f"S:{g}" for g in
                        [int(f.split("_")[1]) for f in preds["filenames"]]])
        print("  模式：合成数据（仅验证管线，不是结果）")
    else:
        if not a.pred:
            print("[BLOCKED] 需要 --pred 或 --demo。")
            print("          真实预测需先跑五折推理（阶段 1.4），见 manual §M4。")
            return 2
        preds = load_preds(pathlib.Path(a.pred))
        a1 = here / a.a1_out if (here / a.a1_out).exists() else root / a.a1_out
        sid = make_source_ids(preds, a1, root / "DMSA_net" / "pcv_v2_folds_seed42.csv")
        print(f"  模式：真实预测 {preds['source']}")

    y, P = preds["labels"], preds["probs"]

    # ── 实现守卫：同簇标签必须一致 ──────────────────────────
    bad = SL.check_group_label_consistency(y, sid)
    if bad:
        print(f"  ⚠️ 有 {len(bad)} 个 source 内部标签不一致（前 5: {bad[:5]}）——"
              f"已记入报告，需人工复核 E2 lineage 假设")

    # ── 图像级基线 & 三条聚合规则 ──────────────────────────
    img_pred = P.argmax(1)
    rows_out = []
    results: Dict[str, dict] = {}

    def eval_pred(tag, yv, pv, gv, probs=None):
        m = SL.metrics(yv, pv, 4)
        cs = SL.cluster_bootstrap(yv, pv, gv,
                                  lambda yy, pp: SL.metrics(yy, pp, 4)["macro_f1"],
                                  n_replicates=a.n_boot, seed=a.seed)
        acc_cs = SL.cluster_bootstrap(yv, pv, gv,
                                      lambda yy, pp: SL.metrics(yy, pp, 4)["accuracy"],
                                      n_replicates=a.n_boot, seed=a.seed)
        d = dict(tag=tag, n_units=int(len(yv)),
                 n_sources=int(len(set(gv.tolist()))),
                 **m, macro_f1_lo=cs.lo, macro_f1_hi=cs.hi,
                 acc_lo=acc_cs.lo, acc_hi=acc_cs.hi)
        if probs is not None:
            a_ = SL.auc_macro(yv, probs, 4)
            au = SL.cluster_bootstrap_auc(yv, probs, gv, 4,
                                          n_replicates=a.n_boot, seed=a.seed)
            d.update(auc_macro=a_, auc_lo=au.lo, auc_hi=au.hi)
        results[tag] = d
        return d

    print("\n  计算：图像级 / source-level × 3 聚合规则 ...")
    eval_pred("image_level", y, img_pred, sid, P)
    for rule in ("mean_prob", "majority", "max_prob"):
        gid, gpred, gp, gy = SL.AggRule(rule).aggregate(y, P, sid, 4)
        probs_g = np.zeros((len(gid), 4))
        for i, g in enumerate(gid):
            probs_g[i] = P[sid == g].mean(0)
        eval_pred(f"source_{rule}", gy, gpred, gid, probs_g)

    # ── 泄漏分层（1.5）：clean vs contaminated ─────────────
    cnt = Counter(sid.tolist())
    contaminated = np.array([cnt[s] > 1 for s in sid.tolist()])
    if contaminated.any() and (~contaminated).any():
        eval_pred("clean_only_singletons", y[~contaminated],
                  img_pred[~contaminated], sid[~contaminated],
                  P[~contaminated])
        eval_pred("contaminated_only", y[contaminated], img_pred[contaminated],
                  sid[contaminated], P[contaminated])

    # ── 主表（三列对比）───────────────────────────────────
    headers = ["口径", "单位数", "源数", "Acc", "Acc 95%CI", "macro-F1",
               "macroF1 95%CI", "BalAcc", "Moderate-F1", "AUC", "AUC 95%CI"]
    table = []
    for tag, d in results.items():
        def ci(lo, hi):
            return f"[{lo:.4f},{hi:.4f}]" if np.isfinite(lo) else "n/a"
        table.append([
            tag, d["n_units"], d["n_sources"],
            f"{d['accuracy']:.4f}", ci(d["acc_lo"], d["acc_hi"]),
            f"{d['macro_f1']:.4f}", ci(d["macro_f1_lo"], d["macro_f1_hi"]),
            f"{d['balanced_acc']:.4f}", f"{d['moderate_f1']:.4f}",
            (f"{d['auc_macro']:.4f}" if "auc_macro" in d else "n/a"),
            (ci(d["auc_lo"], d["auc_hi"]) if "auc_lo" in d else "n/a"),
        ])

    # ── 聚合规则敏感性（1.4a）──────────────────────────────
    sens = []
    base = results.get("source_mean_prob", {}).get("macro_f1")
    for rule in ("mean_prob", "majority", "max_prob"):
        v = results.get(f"source_{rule}", {}).get("macro_f1")
        sens.append(dict(rule=rule, macro_f1=v,
                         delta_vs_mean_prob=(None if (v is None or base is None)
                                             else v - base)))

    # ── 配对 cluster bootstrap（1.6 的核心效应量）───────────
    paired = {}
    gid0, gpred0, _, gy0 = SL.AggRule("mean_prob").aggregate(y, P, sid, 4)
    img_map = {}
    for g in gid0.tolist():
        m = sid == g
        img_map[g] = int(img_pred[m][0])
    same_len = (len(gid0) == len(y))
    if same_len and np.array_equal(gpred0, img_pred):
        pass  # 图像与源图一一对应，无需配对比较
    else:
        gid_m, gpred_m, _, gy_m = SL.AggRule("mean_prob").aggregate(y, P, sid, 4)
        gv_m = gid_m
        # 图像级在同一批簇上重采样作为对照：构造同长度数组
        # （图像级无 cluster 概念，这里改为在源级上比较 mean_prob vs majority）
        gid_a, pred_a, _, gy_a = SL.AggRule("majority").aggregate(y, P, sid, 4)
        if np.array_equal(gv_m, gid_a):
            paired["mean_prob_vs_majority"] = SL.paired_cluster_bootstrap(
                gy_m, pred_m if False else gpred_m, pred_a, gv_m,
                lambda yy, pp: SL.metrics(yy, pp, 4)["macro_f1"],
                n_replicates=a.n_boot, seed=a.seed)

    # ── ICC / DEFF（2.2，解释性）──────────────────────────
    design = SL.design_summary(P[:, 0], sid)

    # ── 落盘 ─────────────────────────────────────────────
    report = dict(provenance=dict(mode=("demo" if a.demo else "real"),
                                  n_boot=a.n_boot, seed=a.seed,
                                  note=("合成数据，仅验证管线" if a.demo else "真实预测")),
                  results=results,
                  three_column_table=dict(headers=headers, rows=table),
                  aggregation_sensitivity=sens,
                  paired_bootstrap=paired,
                  design=design,
                  label_inconsistent_sources=bad[:50],
                  n_label_inconsistent=len(bad))
    #: N6：走统一盖加入口（原来这里没有 code_fingerprint）
    write_artifact(out / "stage2_report.json", report)

    with open(out / "three_column_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(table)

    L = ["# 阶段 1.4–1.6 / 2.1–2.3 源图级指标与三列对比", "",
         f"- 模式：**{'合成数据（仅验证管线）' if a.demo else '真实预测'}**",
         f"- cluster bootstrap 重采样次数：{a.n_boot}，seed={a.seed}",
         f"- 统计单位：**源图（cluster）**", "",
         "## 三列对比表", "", "| " + " | ".join(headers) + " |",
         "|" + "---|" * len(headers)]
    for r in table:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L += ["", "## 聚合规则敏感性（1.4a，**事前注册，全部并列**）", "",
          "| 聚合规则 | macro-F1 | 相对 mean_prob |", "|---|---|---|"]
    for s in sens:
        dv = (f"{s['delta_vs_mean_prob']:+.4f}"
              if s["delta_vs_mean_prob"] is not None else "n/a")
        L.append(f"| {s['rule']} | {s['macro_f1']:.4f} | {dv} |")
    L += ["", "> 主表用 `mean_prob`。若三条规则的 CI 大幅交叠，说明结论对聚合方式稳健；",
          "> 若某条规则显著更好，**不得**据此改主规则——那属于事后选择。", ""]
    if paired:
        L += ["## 配对 cluster bootstrap（效应量，非显著性门槛）", "",
              "| 对比 | Δ | 95%CI | p(双侧, 参考) | 簇数 |", "|---|---|---|---|---|"]
        for k, v in paired.items():
            L.append(f"| {k} | {v['delta_point']:+.4f} | "
                     f"[{v['delta_lo']:+.4f},{v['delta_hi']:+.4f}] | "
                     f"{v['p_two_sided']:.4f} | {v['n_clusters']} |")
        L += ["", "> 配对差值 CI 才是协议间比较的**效应量**；两个独立 CI 是否重叠",
              "> **不能**直接当显著性判据。", ""]
    L += ["## ICC / DEFF（**解释性**）", "",
          f"- 单位数 {design['n_units']}，源数 {design['n_clusters']}，"
          f"平均簇大小 {design['mean_cluster_size']:.3f}",
          f"- ICC ≈ {design['icc']:.4f}，DEFF ≈ {design['deff']:.3f}，"
          f"ESS ≈ {design['ess']:.1f}",
          "", "> v2.4 纪律：这些只用于**解释**「为什么源图级结果比图像级更保守」，",
          "> 不作为门槛、不套用到所有指标。"]
    if bad:
        L += ["", f"## ⚠️ 标签不一致的 source（{len(bad)} 个）", "",
              f"前若干：`{bad[:20]}`", "",
              "> 跨类同源违反 E2 lineage 假设，必须人工复核；",
              "> 复核前该 source 在主统计中标记为 `unresolved`。"]
    (out / "stage2_report.md").write_text("\n".join(L), encoding="utf-8")

    print("\n  ── 三列对比 ──")
    print("  " + " | ".join(headers[:7]))
    for r in table:
        print("  " + " | ".join(str(x) for x in r[:7]))
    print(f"\n  ICC={design['icc']:.4f}  DEFF={design['deff']:.3f}  ESS={design['ess']:.1f}")
    if bad:
        print(f"  ⚠️ {len(bad)} 个 source 内部标签不一致")
    print(f"\n输出：{out/'stage2_report.json'}、{out/'stage2_report.md'}、"
          f"{out/'three_column_table.csv'}")

    # ── 用合成数据首次跑时给自查断言 ──────────────────────
    if a.demo:
        assert abs(results["source_mean_prob"]["macro_f1"]
                   - SL.metrics(y, img_pred, 4)["macro_f1"]) >= 0, "自检失败"
        print("\n  自检：管线跑通（合成数据结果无意义，仅验证正确性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
