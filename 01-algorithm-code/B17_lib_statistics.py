#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMSA-Net v2.4 —— 源图级统计公共库
==================================

覆盖路线方案：
  阶段 1.2 结构泄漏审计（三协议）
  阶段 1.3 ESS / DEFF（**仅解释性**，v2.4 已降级）
  阶段 1.6 三列对比表 + cluster bootstrap CI
  阶段 2.1 cluster bootstrap CI
  阶段 2.2 ICC / DEFF
  阶段 2.3 AUC 的 cluster-aware bootstrap

核心纪律（来自 v2.4 的修订）：
  1. **统计单位一律是源图（cluster），不是图像。**
     stratified bootstrap（对图像重采样）会低估方差，本项目禁止。
  2. **singleton 必须进主统计**（`MAIN_STAT_LEVELS`）。
     否则 No Impairment 全部 2,560 张被排除，主指标无法计算。
  3. **聚合规则必须先行注册**（`AggRule`），不得事后挑一个最好看的。
  4. 宏平均 F1 **必须实算**，不得写死（v2.2 §1.5 曾写错固定值）。
  5. ICC / DEFF / ESS **只作解释性指标**，不得套用到所有指标、不得当门槛。

依赖：numpy 即可（bootstrap / ICC / AUC 全部手写，无 scipy / sklearn 依赖）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "2.4.0"

# ═══════════════════════════════════════════════════════════════
# 1. 置信等级与主统计口径
# ═══════════════════════════════════════════════════════════════
S_PROVEN = "S1"
S_INVARIANT = "S2"
S_BOUNDARY = "S3"
S_UNRESOLVED = "unresolved"
S_SINGLETON = "singleton"

#: 进入主统计的等级集合。**singleton 必须在内**（见 lib_dedup.py 同名常量）。
MAIN_STAT_LEVELS: Tuple[str, ...] = (S_PROVEN, S_INVARIANT, S_SINGLETON)

#: 敏感性分析用的宽口径（把 S3 也纳入，看结论是否翻转）。
SENS_LEVELS: Tuple[str, ...] = (S_PROVEN, S_INVARIANT, S_BOUNDARY, S_SINGLETON)


# ═══════════════════════════════════════════════════════════════
# 2. 聚合规则（必须事前注册）
# ═══════════════════════════════════════════════════════════════
@dataclass
class AggRule:
    """图像预测 -> 源图预测的聚合规则。

    v2.3 的教训：聚合规则若允许事后挑选，等于给结果开后门。
    本类把规则名与实现绑死，落盘时写进 run_config，事后可见。

    rule="majority"  : 组内多数投票（对**预测标签**投票，不是真标签！）；
                       平票时用 mean_prob 打破（确定性）。
    rule="mean_prob" : 组内平均预测概率取 argmax。
    rule="max_prob"  : 组内最大预测概率（保守，偏向上报"更难"的类）。

    ⚠️ v2.4 修正（实测发现）：v2.3 的 majority 实现用 `np.bincount(labels[m])`
       统计的是**真标签**，等于把 ground truth 泄漏进聚合规则，
       会让 majority 退化为"整簇猜真标签"→ macro-F1 恒为 1.0 的假象。
       正确做法是对 `probs.argmax(1)`（预测）投票。
    """
    rule: str = "mean_prob"

    def aggregate(self,
                  labels: np.ndarray,
                  probs: np.ndarray,
                  groups: Sequence[int],
                  n_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (gid_list, pred, mean_prob)。

        labels  : (N,) 真标签（仅用于一致性检查与返回 gc；**不参与投票**）
        probs   : (N, C) 预测概率
        groups  : (N,) 每个样本所属 cluster 的标签
        """
        labels = np.asarray(labels).reshape(-1)
        probs = np.asarray(probs, dtype=np.float64)
        groups = np.asarray(groups).reshape(-1)
        preds = probs.argmax(axis=1)          # 图像级预测标签
        #: 只聚合本 fold 的 val 预测（实现守卫）：由调用方保证传入的就是本 fold 子集
        uniq = sorted(set(groups.tolist()), key=lambda x: str(x))
        gid, gpred, glow, gc = [], [], [], []
        for g in uniq:
            m = groups == g
            lab = labels[m]
            gc.append(int(lab[0]) if lab.size else -1)
            P = probs[m]
            mp = P.mean(axis=0)
            glow.append(float(mp.max()))
            if self.rule == "mean_prob":
                p = int(np.argmax(mp))
            elif self.rule == "max_prob":
                p = int(np.argmax(P.max(axis=0)))
            elif self.rule == "majority":
                cnt = np.bincount(preds[m].astype(int), minlength=n_classes)
                top = int(cnt.max())
                cands = np.flatnonzero(cnt == top)
                p = int(cands[np.argmax(mp[cands])]) if len(cands) > 1 else int(cands[0])
            else:
                raise ValueError(f"未知聚合规则 {self.rule}")
            gid.append(g)
            gpred.append(p)
        return (np.array(gid, dtype=object), np.array(gpred),
                np.array(glow), np.array(gc))


def check_group_label_consistency(labels: np.ndarray, groups: np.ndarray) -> List[int]:
    """返回标签不一致的 cluster 列表。数据泄漏审计的硬检查项之一。"""
    labels = np.asarray(labels).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    bad = []
    for g in sorted(set(groups.tolist())):
        lab = set(labels[groups == g].tolist())
        if len(lab) > 1:
            bad.append(int(g))
    return bad


# ═══════════════════════════════════════════════════════════════
# 3. 分类指标（手写，macro-F1 必须实算）
# ═══════════════════════════════════════════════════════════════
def confusion_matrix(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=int).reshape(-1)
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a, b in zip(y, p):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            C[a, b] += 1
    return C


def per_class_prf(C: np.ndarray) -> Dict[str, np.ndarray]:
    """每类 precision / recall / f1 / support。**全部实算**（不写死）。"""
    tp = np.diag(C).astype(np.float64)
    fp = C.sum(axis=0).astype(np.float64) - tp
    fn = C.sum(axis=1).astype(np.float64) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return dict(precision=prec, recall=rec, f1=f1, support=C.sum(axis=1).astype(np.int64))


def metrics(y: np.ndarray, p: np.ndarray, n_classes: int) -> Dict[str, float]:
    C = confusion_matrix(y, p, n_classes)
    d = per_class_prf(C)
    return dict(
        accuracy=float(np.trace(C) / max(C.sum(), 1)),
        macro_f1=float(np.mean(d["f1"])),
        weighted_f1=float(np.average(d["f1"], weights=np.maximum(d["support"], 1))),
        balanced_acc=float(np.mean(d["recall"])),
        #: Moderate 类样本极少（sealed test 仅 12 张），单独报，避免被宏平均掩盖
        moderate_f1=float(d["f1"][3]) if n_classes > 3 else float("nan"),
        support=int(C.sum()),
    )


# ═══════════════════════════════════════════════════════════════
# 4. AUC（手写 rank 统计，支持 macro / 每类 one-vs-rest）
# ═══════════════════════════════════════════════════════════════
def _auc_binary(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U 形式的 AUC，含并列值的中位秩处理。"""
    y_bin = np.asarray(y_bin, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # 并列值取平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[y_bin == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ovr(y: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    """每类 one-vs-rest AUC。"""
    y = np.asarray(y, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    return np.array([_auc_binary((y == c).astype(int), probs[:, c])
                     for c in range(n_classes)], dtype=np.float64)


def auc_macro(y: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    a = auc_ovr(y, probs, n_classes)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


# ═══════════════════════════════════════════════════════════════
# 5. Cluster bootstrap（统计单位 = 源图）
# ═══════════════════════════════════════════════════════════════
@dataclass
class BootstrapResult:
    point: float
    lo: float
    hi: float
    n_clusters: int
    n_replicates: int
    ci: float = 0.95
    note: str = ""

    def as_dict(self) -> dict:
        return dict(point=self.point, lo=self.lo, hi=self.hi,
                    n_clusters=self.n_clusters, n_replicates=self.n_replicates,
                    ci_level=self.ci, method="cluster_bootstrap", note=self.note)


def cluster_bootstrap(y: np.ndarray,
                      pred: np.ndarray,
                      groups: np.ndarray,
                      metric_fn: Callable[[np.ndarray, np.ndarray], float],
                      n_replicates: int = 2000,
                      ci: float = 0.95,
                      seed: int = 42) -> BootstrapResult:
    """**cluster bootstrap**：对**源图**有放回重采样，而非对图像。

    这是 v2.4 相对 v2.1 的关键修正：stratified bootstrap 会把同源的多张派生图
    当成独立证据，系统性地把 CI 压窄。

    metric_fn(y_sub, pred_sub) -> float
    """
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred)
    groups = np.asarray(groups).reshape(-1)
    # pred 可以是 (N,) 标签、(N,C) 概率或 (N,T,K) 等，只比对第 0 维
    if not (y.shape[0] == pred.shape[0] == groups.shape[0]):
        raise ValueError(f"y / pred / groups 长度不一致: "
                         f"{y.shape[0]} / {pred.shape[0]} / {groups.shape[0]}")
    # 任意可哈希 group 标签均可（source_id 常是字符串）
    uniq = np.array(sorted(set(groups.tolist()), key=lambda x: str(x)))
    idx_of = {str(g): np.flatnonzero(groups == g) for g in uniq.tolist()}

    point = float(metric_fn(y, pred))
    rng = np.random.default_rng(seed)
    vals = np.full(n_replicates, np.nan, dtype=np.float64)
    for b in range(n_replicates):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[str(g)] for g in pick.tolist()])
        try:
            vals[b] = float(metric_fn(y[sel], pred[sel]))
        except Exception:
            vals[b] = np.nan
    ok = vals[np.isfinite(vals)]
    if ok.size < max(50, n_replicates // 20):
        return BootstrapResult(point, float("nan"), float("nan"),
                               len(uniq), n_replicates, ci,
                               note=f"有效重采样仅 {ok.size}/{n_replicates}，CI 不可用")
    a = (1 - ci) / 2
    return BootstrapResult(point, float(np.quantile(ok, a)),
                           float(np.quantile(ok, 1 - a)),
                           len(uniq), n_replicates, ci,
                           note=f"有效重采样 {ok.size}/{n_replicates}")


def cluster_bootstrap_auc(y: np.ndarray,
                          probs: np.ndarray,
                          groups: np.ndarray,
                          n_classes: int,
                          n_replicates: int = 2000,
                          ci: float = 0.95,
                          seed: int = 42) -> BootstrapResult:
    """AUC 的 cluster-aware bootstrap（阶段 2.3）。"""
    return cluster_bootstrap(
        y, probs, groups,
        metric_fn=lambda yy, pp: auc_macro(yy, pp, n_classes),
        n_replicates=n_replicates, ci=ci, seed=seed)


def paired_cluster_bootstrap(y: np.ndarray,
                             pred_a: np.ndarray,
                             pred_b: np.ndarray,
                             groups: np.ndarray,
                             metric_fn: Callable[[np.ndarray, np.ndarray], float],
                             n_replicates: int = 2000,
                             ci: float = 0.95,
                             seed: int = 42) -> Dict[str, float]:
    """**配对** cluster bootstrap：同一批重采样簇上同时评估 A、B。

    配对的意义：比较两个协议（如 legacy vs source_group）时，非配对 CI 的
    重叠与否不能直接当显著性判据；配对差值的 CI 才是有意义的效应量。
    """
    y = np.asarray(y).reshape(-1)
    pred_a = np.asarray(pred_a).reshape(-1)
    pred_b = np.asarray(pred_b).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    uniq = np.array(sorted(set(groups.tolist()), key=lambda x: str(x)))
    idx_of = {str(g): np.flatnonzero(groups == g) for g in uniq.tolist()}

    d_point = float(metric_fn(y, pred_a)) - float(metric_fn(y, pred_b))
    rng = np.random.default_rng(seed)
    diffs = np.full(n_replicates, np.nan, dtype=np.float64)
    for b in range(n_replicates):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[str(g)] for g in pick.tolist()])
        try:
            diffs[b] = (float(metric_fn(y[sel], pred_a[sel]))
                        - float(metric_fn(y[sel], pred_b[sel])))
        except Exception:
            diffs[b] = np.nan
    ok = diffs[np.isfinite(diffs)]
    if ok.size < max(50, n_replicates // 20):
        return dict(delta_point=d_point, delta_lo=float("nan"),
                    delta_hi=float("nan"), p_two_sided=float("nan"),
                    n_clusters=len(uniq), n_replicates=n_replicates,
                    method="paired_cluster_bootstrap")
    a = (1 - ci) / 2
    lo, hi = float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a))
    # 双侧 p：差值分布中 0 的位置（bootstrap 百分位法，仅供参考）
    p = 2.0 * min(float((ok <= 0).mean()), float((ok >= 0).mean()))
    return dict(delta_point=d_point, delta_lo=lo, delta_hi=hi,
                p_two_sided=float(min(p, 1.0)),
                n_clusters=len(uniq), n_replicates=n_replicates,
                method="paired_cluster_bootstrap")


# ═══════════════════════════════════════════════════════════════
# 6. ICC / DEFF / ESS（**仅解释性**）
# ═══════════════════════════════════════════════════════════════
def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """单因素随机效应 ICC(1)：簇间方差 / 总方差。

    用途：量化"同源图像之间的相似程度"。
    ⚠️ v2.4 明确：ICC 只是**解释性**指标，不得作为结果门槛，
    也不得套用到所有指标上（尤其不得用于分类正确率这类 0/1 结果）。
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    uniq = sorted(set(groups.tolist()))
    if len(uniq) < 2:
        return float("nan")
    grand = values.mean()
    ss_b, ss_w, df_b, df_w = 0.0, 0.0, 0, 0
    for g in uniq:
        v = values[groups == g]
        if v.size == 0:
            continue
        ss_b += v.size * (v.mean() - grand) ** 2
        ss_w += float(((v - v.mean()) ** 2).sum())
        df_b += 1
        df_w += v.size - 1
    if df_b == 0 or df_w <= 0:
        return float("nan")
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    n_bar = len(values) / len(uniq)
    denom = ms_b + (n_bar - 1) * ms_w
    return float((ms_b - ms_w) / denom) if denom > 0 else float("nan")


def deff(n_clusters: int, n_units: int, icc: float) -> float:
    """设计效应 DEFF ≈ 1 + (m̄ − 1)·ICC，m̄ = 平均簇大小。"""
    if n_clusters <= 0 or not np.isfinite(icc):
        return float("nan")
    m_bar = n_units / n_clusters
    return float(1.0 + (m_bar - 1.0) * icc)


def ess(n_units: int, deff_value: float) -> float:
    """有效样本量 ESS ≈ N / DEFF。**解释性指标**，不是样本量本身。"""
    if not np.isfinite(deff_value) or deff_value <= 0:
        return float("nan")
    return float(n_units / deff_value)


def design_summary(values: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """一次性给出 ICC / DEFF / ESS / 平均簇大小（阶段 1.3、2.2）。"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    n_units = len(values)
    n_clusters = len(set(groups.tolist()))
    ic = icc_oneway(values, groups)
    dd = deff(n_clusters, n_units, ic)
    return dict(n_units=n_units, n_clusters=n_clusters,
                mean_cluster_size=(n_units / n_clusters if n_clusters else float("nan")),
                icc=ic, deff=dd, ess=ess(n_units, dd))


# ═══════════════════════════════════════════════════════════════
# 7. 结构泄漏审计（阶段 1.2，纯 manifest 计算）
# ═══════════════════════════════════════════════════════════════
def leakage_audit(source_ids: Sequence[str],
                  folds: Sequence[int],
                  splits: Sequence[str],
                  classes: Sequence[str]) -> Dict[str, object]:
    """三协议结构泄漏审计。

    检查三种**结构性**（不需要图像）的泄漏：
      P1 同一 source_id 跨多个 fold        -> 分组协议是否真的按源分组
      P2 同一 source_id 跨 train/test      -> sealed test 是否被训练侧派生样本污染
      P3 同一 source_id 出现在多个类别      -> 标签矛盾（需人工复核）

    返回每个协议违规的明细 + 汇总计数。**纯 manifest 计算，0 图像、0 GPU。**
    """
    n = len(source_ids)
    if not (len(folds) == len(splits) == len(classes) == n):
        raise ValueError("输入长度不一致")
    from collections import defaultdict
    by_src_fold: Dict[str, set] = defaultdict(set)
    by_src_split: Dict[str, set] = defaultdict(set)
    by_src_cls: Dict[str, set] = defaultdict(set)
    for s, f, sp, c in zip(source_ids, folds, splits, classes):
        by_src_fold[s].add(int(f))
        by_src_split[s].add(sp)
        by_src_cls[s].add(c)

    viol_p1 = [dict(source_id=s, folds=sorted(v)) for s, v in by_src_fold.items() if len(v) > 1]
    viol_p2 = [dict(source_id=s, splits=sorted(v)) for s, v in by_src_split.items() if len(v) > 1]
    viol_p3 = [dict(source_id=s, classes=sorted(v)) for s, v in by_src_cls.items() if len(v) > 1]

    n_src = len(by_src_fold)
    from collections import Counter
    cnt = Counter(source_ids)
    largest = int(max(cnt.values())) if cnt else 0
    return dict(
        n_units=n, n_sources=n_src,
        P1_group_leakage=dict(count=len(viol_p1), rate=len(viol_p1) / max(n_src, 1),
                              detail=viol_p1[:50]),
        P2_train_test_leakage=dict(count=len(viol_p2), rate=len(viol_p2) / max(n_src, 1),
                                   detail=viol_p2[:50]),
        P3_cross_class=dict(count=len(viol_p3), rate=len(viol_p3) / max(n_src, 1),
                            detail=viol_p3[:50]),
        #: 单例比例 —— 直接反映"已检测到的同源组"覆盖度
        singleton_rate=float(sum(1 for s, c in cnt.items() if c == 1) / max(len(cnt), 1)),
        singleton_units=int(sum(c for c in cnt.values() if c == 1)),
        #: 最大组件（链式合并风险信号，不是"链式合并的证据"）
        largest_component=largest,
        components_ge10=int(sum(1 for c in cnt.values() if c >= 10)),
        component_size_hist=_size_hist(cnt.values()),
    )


def _size_hist(sizes) -> Dict[str, int]:
    """组件规模分布（v2.4 要求 A1 输出之一）。"""
    from collections import Counter
    h: Dict[str, int] = {}
    for s in sizes:
        if s > 64:
            h["64+"] = h.get("64+", 0) + 1
        else:
            h[str(int(s))] = h.get(str(int(s)), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (kv[0] == "64+", int(kv[0].rstrip("+")))))


def leakage_audit_from_source_map(rows: Sequence[dict],
                                  fold_key: str = "fold",
                                  src_key: str = "source_id",
                                  split_key: str = "split",
                                  cls_key: str = "label") -> Dict[str, object]:
    """从 source_mapping.csv 的行列表做结构泄漏审计。

    rows 里 split 缺省时按 train 处理（A1 的 train 那一遍没有 split 列）。
    """
    sid, fld, spl, cls = [], [], [], []
    for r in rows:
        sid.append(str(r.get(src_key)))
        f = r.get(fold_key)
        fld.append(int(f) if f not in (None, "", "nan") else -1)
        spl.append(str(r.get(split_key) or "train"))
        cls.append(str(r.get(cls_key)))
    out = leakage_audit(sid, fld, spl, cls)
    # 单例率：只出现一次的 source_id 占比
    from collections import Counter
    cnt = Counter(sid)
    out["singleton_rate"] = float(
        sum(1 for s, c in cnt.items() if c == 1) / max(len(cnt), 1))
    out["largest_component"] = int(max(cnt.values())) if cnt else 0
    return out


if __name__ == "__main__":
    # 自检：用合成数据跑通所有函数（真实数据请走 audit_leakage.py / stage2_stats.py）
    rng = np.random.default_rng(0)
    n_clu, per = 60, 4
    groups = np.repeat(np.arange(n_clu), per)
    y = rng.integers(0, 4, size=n_clu * per)
    # 让同簇标签一致（真实数据应当如此）
    for g in range(n_clu):
        y[groups == g] = int(rng.integers(0, 4))
    logits = rng.normal(size=(n_clu * per, 4)) + np.eye(4)[y] * 1.2
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)

    print("metrics:", metrics(y, pred, 4))
    print("auc macro:", auc_macro(y, probs, 4))
    r = cluster_bootstrap(y, pred, groups, lambda a, b: metrics(a, b, 4)["macro_f1"],
                          n_replicates=300)
    print("bootstrap:", r.as_dict())
    print("design:", design_summary(probs[:, 0], groups))
    print("label-consistency violations:",
          check_group_label_consistency(y, groups))
    print("self-check OK")
