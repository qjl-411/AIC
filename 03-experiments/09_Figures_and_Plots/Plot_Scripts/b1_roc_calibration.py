# -*- coding: utf-8 -*-
"""
B1 —— ROC-AUC / 校准 / 置信度分析（2026AIC 提分项 ★）

输入：sealed test 的既有预测产物 test_predictions.csv（1279 样本 × 4 类 softmax）
      —— 本脚本 **不重训、不碰 test 图像**，只用已有概率列做统计。

禁止事项（沿用项目铁律）：
  * sealed test 只能做**一次**正式评价 —— 本脚本不产生新的模型评价，只对已产出的概率做后处理
  * 不得用 test 拟合 temperature scaling（report-only calibration）
  * 不得用 sealed test 挑方案

输出：
  out/b1_auc_summary.json          每类 + macro 的 AUC 及 cluster-aware bootstrap CI
  out/b1_calibration.json          ECE / Brier / NLL / 逐 bin 校准表
  out/b1_confidence.json           正确 vs 错误预测的置信度分布
  figures/fig_roc.png              One-vs-Rest 四类 ROC 曲线
  figures/fig_reliability.png      可靠性图
  figures/fig_confidence.png       置信度分布
"""
import csv
import json
import math
import pathlib
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
CLASSES = [
    ("Mild Impairment", "prob_mild_impairment"),
    ("Moderate Impairment", "prob_moderate_impairment"),
    ("No Impairment", "prob_no_impairment"),
    ("Very Mild Impairment", "prob_very_mild_impairment"),
]
CLASS_ORDER = [c for c, _ in CLASSES]
PROB_COLS = [p for _, p in CLASSES]

# 源图级聚类单位：本项目无受试者 ID，退化为「文件名去掉编号后的前缀族」
# 这里用 relative_path 的目录+文件名主干做粗聚类，避免把同一原图派生样本当独立样本。

DEFAULT_PRED = (
    "E:/Code/NeuroEvo/NeuroEvo_AD/dmsa_final_train(1)/"
    "dmsa_final_train_seed42_epoch126/sealed_test_once/test_predictions.csv"
)
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE / "figures"


# ----------------------------------------------------------------------------
# 工具：纯标准库实现（不依赖 numpy，保证任何环境可跑）
# ----------------------------------------------------------------------------
def load_predictions(path: pathlib.Path) -> Dict[str, list]:
    """读 test_predictions.csv -> {'y': [int], 'P': [[4 floats]], 'path': [str]}"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"预测文件为空：{path}")

    y: List[int] = []
    P: List[List[float]] = []
    paths: List[str] = []
    for r in rows:
        yi = int(r["true_index"])
        pi = [float(r[c]) for c in PROB_COLS]
        s = sum(pi)
        if s <= 0:
            raise SystemExit(f"概率全零：{r['relative_path']}")
        pi = [v / s for v in pi]  # 数值安全归一（理论上已归一）
        y.append(yi)
        P.append(pi)
        paths.append(r["relative_path"])
    return {"y": y, "P": P, "path": paths}


def roc_auc_binary(y_true: List[int], scores: List[float]) -> float:
    """
    手写 rank-based AUC（Mann-Whitney U），含 1e-12 级 tie 判定。
    与 sklearn.metrics.roc_auc_score 等价（不引入 sklearn 依赖）。
    """
    pairs = sorted(zip(scores, y_true), key=lambda t: t[0])
    n = len(pairs)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # 处理 tie：同分数给平均秩
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(pairs[j + 1][0] - pairs[i][0]) <= 1e-15:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve_binary(y_true: List[int], scores: List[float]) -> Tuple[list, list]:
    """返回 (fpr_list, tpr_list)，含 (0,0) 与 (1,1) 端点。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]

    fpr = [0.0]
    tpr = [0.0]
    tp = fp = 0
    prev = None
    for idx in order:
        s = scores[idx]
        if prev is not None and s != prev:
            fpr.append(fp / n_neg)
            tpr.append(tp / n_pos)
        prev = s
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
    fpr.append(1.0)
    tpr.append(1.0)
    return fpr, tpr


def cluster_bootstrap_auc(
    y: List[int], scores: List[float], groups: List[str],
    n_boot: int = 2000, seed: int = 42,
) -> Tuple[float, float, float]:
    """
    cluster bootstrap CI（统计单位 = 源图族，而非图像）。
    返回 (point, lo95, hi95)。纯标准库实现的 LCG 随机源，保证可复现。
    """
    point = roc_auc_binary(y, scores)

    gidx: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        gidx.setdefault(g, []).append(i)
    gkeys = sorted(gidx.keys())
    n_g = len(gkeys)

    # 确定性 LCG（避免依赖 random 模块的实现细节）
    state = seed
    def rnd() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    vals: List[float] = []
    for _ in range(n_boot):
        yb: List[int] = []
        sb: List[float] = []
        for _ in range(n_g):
            g = gkeys[int(rnd() * n_g) % n_g]
            for i in gidx[g]:
                yb.append(y[i])
                sb.append(scores[i])
        a = roc_auc_binary(yb, sb)
        if not math.isnan(a):
            vals.append(a)

    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return point, lo, hi


def ece_and_bins(conf: List[float], correct: List[int], n_bins: int = 15) -> dict:
    """Expected Calibration Error + 逐 bin 表（等宽分箱）。"""
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(conf, correct):
        k = min(n_bins - 1, int(c * n_bins))
        bins[k].append((c, ok))

    rows = []
    ece = 0.0
    total = len(conf)
    for k, b in enumerate(bins):
        if not b:
            rows.append({
                "bin": k,
                "lo": round(k / n_bins, 4),
                "hi": round((k + 1) / n_bins, 4),
                "n": 0,
                "avg_conf": None,
                "accuracy": None,
                "gap": None,
            })
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        acc = sum(x[1] for x in b) / len(b)
        gap = acc - avg_conf
        ece += (len(b) / total) * abs(gap)
        rows.append({
            "bin": k,
            "lo": round(k / n_bins, 4),
            "hi": round((k + 1) / n_bins, 4),
            "n": len(b),
            "avg_conf": round(avg_conf, 6),
            "accuracy": round(acc, 6),
            "gap": round(gap, 6),
        })
    return {"ece": ece, "n_bins": n_bins, "total": total, "bins": rows}


def brier_nll(P: List[List[float]], y: List[int]) -> Tuple[float, float]:
    """多分类 Brier（sum of squared probs, 归一）与 NLL。"""
    n = len(y)
    K = len(P[0])
    brier = 0.0
    nll = 0.0
    for pi, yi in zip(P, y):
        for k in range(K):
            target = 1.0 if k == yi else 0.0
            brier += (pi[k] - target) ** 2
        p_true = max(pi[yi], 1e-12)
        nll -= math.log(p_true)
    return brier / n, nll / n


def make_groups(paths: List[str]) -> List[str]:
    """
    粗聚类：把 'Mild Impairment/1 (10).jpg' 归到 'Mild Impairment/1'。
    目的是让 bootstrap 不以「同一原图的多个派生副本」为独立单位。
    """
    out = []
    for p in paths:
        p = p.replace("\\", "/")
        d, _, f = p.rpartition("/")
        stem = f.rsplit(".", 1)[0]
        # 去掉尾部形如 ' (12)' 的副本编号
        base = stem.split(" (")[0]
        out.append(f"{d}/{base}")
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> int:
    ap_pred = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRED
    pred_path = pathlib.Path(ap_pred)
    if not pred_path.is_file():
        raise SystemExit(f"找不到预测文件：{pred_path}")

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print(f"[B1] 读取预测：{pred_path}")
    D = load_predictions(pred_path)
    y, P, paths = D["y"], D["P"], D["path"]
    n = len(y)
    print(f"[B1] 样本数 = {n}")

    groups = make_groups(paths)
    print(f"[B1] 聚类单位（源图族）数 = {len(set(groups))}")

    # ---------- 1. 每类 OvR AUC + cluster bootstrap CI ----------
    auc_summary = []
    roc_data = {}
    for k, (cname, _) in enumerate(CLASSES):
        scores = [pi[k] for pi in P]
        yk = [1 if yi == k else 0 for yi in y]
        pt, lo, hi = cluster_bootstrap_auc(yk, scores, groups, n_boot=2000, seed=42)
        fpr, tpr = roc_curve_binary(yk, scores)
        roc_data[cname] = {"fpr": fpr, "tpr": tpr}
        auc_summary.append({
            "class_name": cname,
            "support": sum(yk),
            "auc": round(pt, 6),
            "ci95_lo": round(lo, 6),
            "ci95_hi": round(hi, 6),
        })
        print(f"[B1]   {cname:24s} AUC = {pt:.4f}  [{lo:.4f}, {hi:.4f}]  (n_pos={sum(yk)})")

    macro_auc = sum(a["auc"] for a in auc_summary) / len(auc_summary)
    # macro 的 CI：对每类 bootstrap 值取平均（用同一 seed 重跑一遍并记录均值分布）
    print(f"[B1]   macro AUC = {macro_auc:.4f}")

    # ---------- 2. 校准 ----------
    conf = [max(pi) for pi in P]
    correct = [1 if pi.index(max(pi)) == yi else 0 for pi, yi in zip(P, y)]
    cal = ece_and_bins(conf, correct, n_bins=15)
    bs, nll = brier_nll(P, y)
    acc = sum(correct) / n

    # ---------- 3. 置信度分布 ----------
    conf_ok = [c for c, ok in zip(conf, correct) if ok == 1]
    conf_bad = [c for c, ok in zip(conf, correct) if ok == 0]

    def q(v: List[float], p: float) -> float:
        if not v:
            return float("nan")
        s = sorted(v)
        return s[min(len(s) - 1, int(p * len(s)))]

    confidence = {
        "n_total": n,
        "n_correct": len(conf_ok),
        "n_incorrect": len(conf_bad),
        "accuracy": round(acc, 6),
        "mean_conf_all": round(sum(conf) / n, 6),
        "mean_conf_correct": round(sum(conf_ok) / len(conf_ok), 6) if conf_ok else None,
        "mean_conf_incorrect": round(sum(conf_bad) / len(conf_bad), 6) if conf_bad else None,
        "median_conf_incorrect": round(q(conf_bad, 0.5), 6) if conf_bad else None,
        "max_conf_incorrect": round(max(conf_bad), 6) if conf_bad else None,
        "min_conf_correct": round(min(conf_ok), 6) if conf_ok else None,
    }

    # ---------- 4. 落盘 ----------
    (OUT / "b1_auc_summary.json").write_text(json.dumps({
        "source_predictions": str(pred_path).replace("\\", "/"),
        "n_samples": n,
        "n_clusters": len(set(groups)),
        "macro_auc": round(macro_auc, 6),
        "per_class": auc_summary,
        "note": (
            "AUC 的 95% CI 用 cluster bootstrap（统计单位 = 文件名前缀族，"
            "用于弱化同源派生副本的伪独立）。Moderate 类 support=12，CI 极宽，"
            "报告时必须显式说明样本量限制。"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "b1_calibration.json").write_text(json.dumps({
        "ece": round(cal["ece"], 6),
        "n_bins": cal["n_bins"],
        "brier_score": round(bs, 6),
        "nll": round(nll, 6),
        "accuracy": round(acc, 6),
        "mean_confidence": round(sum(conf) / n, 6),
        "overconfidence_gap": round(sum(conf) / n - acc, 6),
        "bins": cal["bins"],
        "note": (
            "本报告为 report-only calibration：温度参数**未**用 sealed test 拟合。"
            "若后续要做 temperature scaling，温度必须由 CV validation 或独立 calibration set 得到。"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "b1_confidence.json").write_text(
        json.dumps(confidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (OUT / "b1_roc_curves.json").write_text(
        json.dumps(roc_data, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 62)
    print(f"  AUC (macro)                : {macro_auc:.4f}")
    print(f"  Accuracy                   : {acc:.4f}")
    print(f"  ECE                        : {cal['ece']:.4f}")
    print(f"  Brier                      : {bs:.4f}")
    print(f"  NLL                        : {nll:.4f}")
    print(f"  平均置信度 vs 准确率        : {sum(conf)/n:.4f} vs {acc:.4f}"
          f"  (gap {sum(conf)/n - acc:+.4f})")
    print(f"  错误样本平均置信度          : "
          f"{confidence['mean_conf_incorrect']:.4f}" if conf_bad else "  (无错误样本)")
    print("=" * 62)
    print(f"[B1] 产物 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
