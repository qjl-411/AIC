#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 1.2–1.3 结构泄漏审计 + ESS/DEFF（v2.4）
=============================================

**纯 manifest 计算：0 图像、0 GPU、0 重训。**

覆盖路线方案：
  1.2 结构泄漏审计（三协议 P1/P2/P3）
  1.3 ESS / DEFF（**仅解释性**；v2.4 已从"门槛"降级为"说明性指标"）

输入（按优先级自动探测）：
  A) `_a1v2_out/source_mapping.csv`（A1 产物，最准）—— 有则用它
  B) `DMSA_net/pcv_v2_folds_seed42.csv`（冻结 manifest，退化为 filename-as-source）
  C) `_a2_out/cross_matches.csv`（跨 train/test 匹配，用于 P2 判定）

用法：
    python audit_leakage.py --root .. --a1-out _a1v2_out --out _audit_out
    python audit_leakage.py --root .. --no-a1            # 强制只用 manifest

输出：
    _audit_out/leakage_audit.json     三协议明细 + 组件规模分布
    _audit_out/leakage_audit.md       可直接贴进论文的表格
    _audit_out/ess_deff.csv           每类 + 整体的 ICC/DEFF/ESS

⚠️ 纪律：ESS/DEFF 只写在"解释性"段落。v2.4 明确禁止把它当门槛，
   也禁止断言"独立 group 数 = ESS"（那需要 ICC，见 stats_lib.design_summary）。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter, defaultdict
from typing import Dict, List

import B17_lib_statistics as SL
from B16_lib_provenance_audit import write_artifact


# ───────────────────────────────────────────────────────────── 读表
def _read_csv(p: pathlib.Path) -> List[dict]:
    with open(p, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _pick_col(fields: List[str], cands: List[str]):
    low = {c.lower(): c for c in fields}
    for c in cands:
        if c in fields:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_source_map(p: pathlib.Path) -> List[dict]:
    rows = _read_csv(p)
    if not rows:
        return []
    f = list(rows[0].keys())
    c_src = _pick_col(f, ["source_id", "cluster_id", "global_source_id"])
    c_file = _pick_col(f, ["filename", "file", "name", "image"])
    c_fold = _pick_col(f, ["fold", "val_fold"])
    c_cls = _pick_col(f, ["label", "class", "cls"])
    c_split = _pick_col(f, ["split"])
    c_conf = _pick_col(f, ["confidence", "conf", "level"])
    out = []
    for r in rows:
        out.append(dict(
            source_id=str(r.get(c_src) or r.get(c_file)),
            filename=str(r.get(c_file) or ""),
            fold=(r.get(c_fold) if c_fold else None),
            label=str(r.get(c_cls) or ""),
            split=str(r.get(c_split) or "train") if c_split else "train",
            confidence=str(r.get(c_conf) or "") if c_conf else "",
        ))
    return out


def load_manifest_fallback(p: pathlib.Path) -> List[dict]:
    """没有 A1 产物时的次优路径。

    ⚠️ **重要修正（v2.4 实测发现）**：冻结 manifest
    `pcv_v2_folds_seed42.csv` **本来就带 `group_id` / `group_size` / `group_reason`
    三列**（原管线已做过一轮 exact + near-duplicate 分组）。
    因此这里**优先直接用 manifest 的 group_id**，把 A1 的任务降级为
    「独立复核既有分组」而不是「从零重建谱系」。

    这对应 v2.2 定义的 `protocol_id = pcv2_legacy`：
    既有分组可作为历史诊断口径，但**不能冒充** `source_group_v1`
    （因为 A1 在变换不变距离下可能发现 manifest 漏掉的分组）。
    """
    rows = _read_csv(p)
    if not rows:
        return []
    f = list(rows[0].keys())
    c_file = _pick_col(f, ["filename", "file", "name", "image", "path"])
    c_fold = _pick_col(f, ["val_fold", "fold"])
    c_cls = _pick_col(f, ["class_name", "label", "class", "cls", "diagnosis"])
    c_group = _pick_col(f, ["group_id"])
    c_reason = _pick_col(f, ["group_reason"])
    has_group = c_group is not None
    out = []
    for r in rows:
        fn = str(r.get(c_file) or "")
        if has_group and r.get(c_group):
            sid = "LG:" + str(r[c_group])          # legacy group
            conf = ("S1" if str(r.get(c_reason) or "").startswith("exact")
                    else ("S2" if str(r.get(c_reason) or "").startswith("near")
                          else "singleton"))
        else:
            sid = "RAW:" + fn
            conf = "unresolved"
        out.append(dict(source_id=sid, filename=fn,
                        fold=(r.get(c_fold) if c_fold else None),
                        label=str(r.get(c_cls) or ""), split="train",
                        confidence=conf))
    return out


# ───────────────────────────────────────────────────────────── 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--a1-out", default="_a1v2_out")
    ap.add_argument("--a2-out", default="_a2_out")
    ap.add_argument("--out", default="_audit_out")
    ap.add_argument("--no-a1", action="store_true", help="强制只用冻结 manifest")
    a = ap.parse_args()

    root = pathlib.Path(a.root).resolve()
    here = pathlib.Path(__file__).parent
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 找输入
    sm_candidates = [here / a.a1_out / "source_mapping.csv",
                     root / a.a1_out / "source_mapping.csv"]
    sm = next((p for p in sm_candidates if p.exists()), None)
    manifest = root / "DMSA_net" / "pcv_v2_folds_seed42.csv"

    print("=" * 74)
    print("阶段 1.2–1.3 结构泄漏审计 + ESS/DEFF")
    print("=" * 74)

    provenance = {}
    if sm is not None and not a.no_a1:
        rows = load_source_map(sm)
        provenance = dict(mode="A1_source_mapping", path=str(sm),
                          note="source_id 来自 A1-v2 谱系重建，置信分级可用")
    elif manifest.exists():
        rows = load_manifest_fallback(manifest)
        has_legacy_group = any(r["source_id"].startswith("LG:") for r in rows)
        provenance = dict(
            mode=("legacy_manifest_group_id" if has_legacy_group
                  else "manifest_filename_only"),
            path=str(manifest),
            protocol_id=("pcv2_legacy" if has_legacy_group else "none"),
            note=("使用冻结 manifest **自带的 group_id**（protocol_id=pcv2_legacy）。"
                  "这是原管线的历史分组，A1 的任务是**独立复核**它是否漏掉分组，"
                  "复核前不得称其为 source_group_v1。"
                  if has_legacy_group else
                  "退化模式：每张图视为独立源 → 泄漏率必然是 0，"
                  "**只能当乐观下界参考，不得当作审计结论**"))
    else:
        print(f"[BLOCKED] 既无 {sm} 也无 {manifest}。")
        print("           按 a0_inventory 的 blocked_action：只写'基于当前材料未验证'。")
        #: N6：BLOCKED 产物同样必须盖章（T13.1 抓的就是这种早退路径漏盖章）
        write_artifact(out / "leakage_audit.json",
                       dict(status="BLOCKED", provenance=provenance,
                            reason="缺 source_mapping.csv 与冻结 manifest"))
        return 2

    rows = [r for r in rows if r["source_id"] not in ("", "None")]
    print(f"  数据来源：{provenance['mode']}  ({provenance['path']})")
    print(f"  样本数  ：{len(rows)}")
    print(f"  说明    ：{provenance['note']}")

    # 三协议泄漏审计
    audit = SL.leakage_audit_from_source_map(rows)

    # 补 P2：用 A2 的跨 train/test 匹配做交叉确认（若存在）
    a2 = here / a.a2_out / "cross_matches.csv"
    if not a2.exists():
        a2 = root / a.a2_out / "cross_matches.csv"
    if a2.exists():
        cross = _read_csv(a2)
        audit["P2_from_A2_crossmatches"] = dict(
            count=len(cross),
            note="A2 的 train↔test 近重复匹配数（独立于 source_id 的第二证据）")
        print(f"  A2 跨 split 匹配：{len(cross)} 对")

    # 按 split / 类别分解
    by_split = Counter(r["split"] for r in rows)
    by_class = Counter(r["label"] for r in rows)
    audit["by_split"] = dict(by_split)
    audit["by_class"] = dict(by_class)

    # 每类独立源数（v2.3 §5.3 要求）
    per_cls_src: Dict[str, dict] = {}
    for c in sorted(by_class):
        sub = [r for r in rows if r["label"] == c]
        cnt = Counter(r["source_id"] for r in sub)
        conf_cnt = Counter(r["confidence"] for r in sub if r["confidence"])
        per_cls_src[c] = dict(
            n_images=len(sub), n_sources=len(cnt),
            singleton_sources=int(sum(1 for v in cnt.values() if v == 1)),
            largest_component=int(max(cnt.values())) if cnt else 0,
            main_stat_images=int(sum(
                1 for r in sub if (not r["confidence"]) or r["confidence"] in SL.MAIN_STAT_LEVELS)),
            confidence_breakdown=dict(conf_cnt),
        )
    audit["per_class_source_counts"] = per_cls_src
    audit["provenance"] = provenance

    #: N6：走统一盖加入口（原来这里没有 code_fingerprint）
    write_artifact(out / "leakage_audit.json", audit)

    # ── ESS / DEFF（解释性）──────────────────────────────────
    ess_rows = []
    design = SL.design_summary([1.0] * len(rows), [hash(r["source_id"]) % 10**6 for r in rows])
    for c, d in per_cls_src.items():
        cnt = Counter(r["source_id"] for r in rows if r["label"] == c)
        sizes = list(cnt.values())
        n_units, n_clusters = d["n_images"], d["n_sources"]
        mbar = n_units / max(n_clusters, 1)
        # 用簇大小的变异近似 ICC 的"冗余倾向"：size=1 的簇无冗余
        redundancy = 1.0 - (d["singleton_sources"] / max(n_clusters, 1))
        icc_approx = redundancy  # 解释性近似，明确标注
        dd = SL.deff(n_clusters, n_units, icc_approx)
        ess_rows.append(dict(
            class_=c, n_images=n_units, n_sources=n_clusters,
            mean_cluster_size=round(mbar, 3),
            largest_component=d["largest_component"],
            icc_approx=round(icc_approx, 4), deff=round(dd, 3),
            ess=round(SL.ess(n_units, dd), 1),
            note="ICC 为基于冗余比例的**解释性近似**，非严格 ICC(1)"))
    ess_rows.append(dict(
        class_="ALL", n_images=len(rows), n_sources=audit["n_sources"],
        mean_cluster_size=round(len(rows) / max(audit["n_sources"], 1), 3),
        largest_component=audit["largest_component"],
        icc_approx=round(1 - audit["singleton_rate"], 4),
        deff=round(SL.deff(audit["n_sources"], len(rows),
                           1 - audit["singleton_rate"]), 3),
        ess=round(SL.ess(len(rows), SL.deff(audit["n_sources"], len(rows),
                                            1 - audit["singleton_rate"])), 1),
        note="解释性指标，禁止当门槛"))

    with open(out / "ess_deff.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ess_rows[0].keys()))
        w.writeheader()
        w.writerows(ess_rows)

    # ── Markdown 报告 ───────────────────────────────────────
    L = ["# 阶段 1.2–1.3 结构泄漏审计 + ESS/DEFF", "",
         f"- 数据来源：`{provenance['mode']}`（`{provenance['path']}`）",
         f"- 样本数：{len(rows)}，独立源数：{audit['n_sources']}",
         f"- ⚠️ {provenance['note']}", "",
         "## 三协议结构泄漏", "",
         "| 协议 | 违规 source 数 | 比例 | 含义 |", "|---|---|---|---|",
         f"| P1 同源跨 fold | {audit['P1_group_leakage']['count']} | "
         f"{audit['P1_group_leakage']['rate']:.4%} | 分组 CV 是否真按源分组 |",
         f"| P2 同源跨 train/test | {audit['P2_train_test_leakage']['count']} | "
         f"{audit['P2_train_test_leakage']['rate']:.4%} | sealed test 独立性 |",
         f"| P3 同源跨类别 | {audit['P3_cross_class']['count']} | "
         f"{audit['P3_cross_class']['rate']:.4%} | 标签矛盾，需人工复核 |",
         "",
         f"- 单例源比例：**{audit['singleton_rate']:.2%}**"
         f"（{audit['singleton_units']} 张图自成一组）",
         f"- 最大组件：**{audit['largest_component']}**"
         f"（链式合并**风险信号**，不等于已证实链式合并）",
         f"- ≥10 的组件数：{audit['components_ge10']}",
         f"- 组件规模分布：`{audit['component_size_hist']}`", "",
         "> **注**：单例是主统计的合法成员（`MAIN_STAT_LEVELS` 含 singleton）。"
         "把它们排除会让 No Impairment 全类消失，主指标无法计算。", "",
         "## 每类独立源数", "",
         "| 类别 | 图像数 | 独立源数 | 单例源 | 最大组件 | 主统计可用图像 |",
         "|---|---|---|---|---|---|"]
    for c, d in per_cls_src.items():
        L.append(f"| {c} | {d['n_images']} | {d['n_sources']} | "
                 f"{d['singleton_sources']} | {d['largest_component']} | "
                 f"{d['main_stat_images']} |")
    L += ["", "## ESS / DEFF（**仅解释性**）", "",
          "| 组 | 图像 | 源 | 平均簇大小 | ICC(近似) | DEFF | ESS |",
          "|---|---|---|---|---|---|---|"]
    for r in ess_rows:
        L.append(f"| {r['class_']} | {r['n_images']} | {r['n_sources']} | "
                 f"{r['mean_cluster_size']} | {r['icc_approx']} | {r['deff']} | {r['ess']} |")
    L += ["", "> v2.4 纪律：ESS/DEFF/ICC **只是解释性指标**，不设门槛、不套用到所有指标、",
          "> 也不能替代「独立源数」。禁止写成「独立 group 数 = ESS」。"]
    (out / "leakage_audit.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n  P1 同源跨 fold      ：{audit['P1_group_leakage']['count']}")
    print(f"  P2 同源跨 train/test：{audit['P2_train_test_leakage']['count']}")
    print(f"  P3 同源跨类别       ：{audit['P3_cross_class']['count']}")
    print(f"  单例源比例          ：{audit['singleton_rate']:.2%}")
    print(f"  最大组件            ：{audit['largest_component']}")
    print(f"\n输出：{out/'leakage_audit.json'}、{out/'leakage_audit.md'}、{out/'ess_deff.csv'}")
    if provenance["mode"] == "manifest_filename_only":
        print("\n⚠️  当前为退化模式：泄漏率必然为 0，只能当乐观参考。"
              "\n    必须先跑 DMSA_v24/a1_source_map.py 才有真实审计结论。")
    elif provenance["mode"] == "legacy_manifest_group_id":
        print("\nℹ️  当前用 manifest 自带 group_id（pcv2_legacy）。"
              "\n    这是一次**真实**的结构审计：P1 若 >0 说明折叠确实按源分组；"
              "\n    但要升级为 source_group_v1，仍需 A1 独立复核（可能发现漏检分组）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
