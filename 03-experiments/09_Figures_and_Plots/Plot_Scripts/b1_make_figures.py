# -*- coding: utf-8 -*-
"""
b1_make_figures.py -- 把 b1_roc_calibration.py 的 JSON 产物渲染成可直接进比赛材料的图。

输入（只读）:
    out/b1_roc_curves.json
    out/b1_auc_summary.json
    out/b1_calibration.json
    out/b1_confidence.json

输出:
    figures/fig1_roc_curves.png      四类 ROC 曲线（含 macro AUC / 95% CI）
    figures/fig2_reliability.png     可靠性图（15 bins）+ 置信度直方图
    figures/fig3_confidence_gap.png  正确/错误样本置信度分布对比

设计约束:
  - **本脚本是纯渲染器**，不做任何计算/选择/拟合。所有数字来自 B1 的 JSON。
  - 禁止在此脚本里重算 AUC 或重新分箱——避免"两处实现不一致"这一经典事故。
  - 图片为 300 dpi PNG，白底，适合直接贴进 Word / PPT。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

DPI = 300
# 四类配色：Mild / Moderate / No / VeryMild —— 色盲友好且彼此可区分
COLORS = {
    "Mild Impairment":        "#1f77b4",
    "Moderate Impairment":    "#d62728",
    "No Impairment":          "#2ca02c",
    "Very Mild Impairment":   "#ff7f0e",
}


def _setup_cjk_font():
    """尽量找一个能渲染中文的字体；找不到就退化为英文标签（图内已用英文类名）。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK = _setup_cjk_font()
L = (lambda zh, en: zh) if CJK else (lambda zh, en: en)


def load(name):
    with open(os.path.join(OUT, name), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- fig 1: ROC
def fig_roc():
    curves = load("b1_roc_curves.json")
    summary = load("b1_auc_summary.json")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for c in summary["per_class"]:
        name = c["class_name"]
        cd = curves.get(name)
        if not cd:
            continue
        fpr, tpr = cd["fpr"], cd["tpr"]
        ax.plot(
            fpr, tpr, lw=1.8, color=COLORS.get(name, "#333333"),
            label="%s  AUC=%.4f (%.4f-%.4f)  n=%d"
                  % (name, c["auc"], c["ci95_lo"], c["ci95_hi"], c["support"]),
        )

    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#999999",
            label=L("随机基线", "Chance"))

    macro = summary["macro_auc"]
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.03)
    ax.set_xlabel(L("假正率 (FPR)", "False Positive Rate"), fontsize=10)
    ax.set_ylabel(L("真正率 (TPR)", "True Positive Rate"), fontsize=10)
    ax.set_title(L("Sealed Test 四类 ROC 曲线（one-vs-rest, n=1279）\nmacro AUC = %.4f"
                   % macro,
                   "Sealed Test ROC (one-vs-rest, n=1279)\nmacro AUC = %.4f" % macro),
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.95)
    ax.grid(alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    p = os.path.join(FIG, "fig1_roc_curves.png")
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("[FIG] %s" % p)


# ------------------------------------------------------- fig 2: reliability
def fig_reliability():
    cal = load("b1_calibration.json")
    bins = cal["bins"]

    used = [b for b in bins if b["n"] > 0]
    conf = [b["avg_conf"] for b in used]
    acc = [b["accuracy"] for b in used]
    n = [b["n"] for b in used]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5))

    # --- 左：可靠性图 ---
    ax1.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#999999",
             label=L("完美校准", "Perfect calibration"))
    ax1.plot(conf, acc, "o-", lw=1.8, ms=5.5, color="#1f77b4",
             label=L("实测", "Model"))
    for x, y, nn in zip(conf, acc, n):
        ax1.annotate(str(nn), (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6.5, color="#666666")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlabel(L("平均置信度", "Mean confidence"), fontsize=10)
    ax1.set_ylabel(L("实际准确率", "Empirical accuracy"), fontsize=10)
    ax1.set_title(L("可靠性图  ECE = %.4f" % cal["ece"],
                    "Reliability diagram  ECE = %.4f" % cal["ece"]), fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25, ls=":")
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # --- 右：置信度直方图 ---
    ax2.bar(conf, n, width=0.045, color="#1f77b4", alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax2.set_yscale("log")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_xlabel(L("平均置信度", "Mean confidence"), fontsize=10)
    ax2.set_ylabel(L("样本数（对数刻度）", "Count (log scale)"), fontsize=10)
    ax2.set_title(L("置信度分布：1257/1279 落在最后一箱",
                    "Confidence distribution:1257/1279 in last bin"), fontsize=11)
    ax2.grid(alpha=0.25, ls=":", axis="y")
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle(L(
        "Sealed Test 置信度校准（report-only，温度参数未在测试集上拟合）",
        "Sealed Test calibration (report-only; no temperature fitted on test)"),
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(FIG, "fig2_reliability.png")
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("[FIG] %s" % p)


# ------------------------------------------------- fig 3: confidence gap
def fig_confidence_gap():
    """
    用 test_predictions.csv 里每条样本的最大 softmax 作为置信度，
    画正确 / 错误样本的置信度分布对比。

    注意：这里读原始 CSV（不是 JSON），因为 B1 的 confidence.json 只存了汇总统计。
    本函数不做统计推断，只做可视化；聚合数字仍以 JSON 为准。
    """
    import csv

    src = load("b1_auc_summary.json")["source_predictions"]
    probs = ["prob_mild_impairment", "prob_moderate_impairment",
             "prob_no_impairment", "prob_very_mild_impairment"]

    conf_c, conf_w = [], []
    with open(src, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = max(float(row[k]) for k in probs)
            if int(row["correct"]) == 1:
                conf_c.append(m)
            else:
                conf_w.append(m)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bw = 0.008
    ax.hist(conf_c, bins=[i / 250.0 for i in range(251)], color="#2ca02c",
            alpha=0.80, label=L("预测正确 (n=%d)" % len(conf_c),
                                "Correct (n=%d)" % len(conf_c)))
    ax.hist(conf_w, bins=[i / 250.0 for i in range(251)], color="#d62728",
            alpha=0.85, label=L("预测错误 (n=%d)" % len(conf_w),
                                "Incorrect (n=%d)" % len(conf_w)))
    ax.set_yscale("log")
    ax.set_xlim(0.45, 1.005)
    ax.set_xlabel(L("最大 softmax 置信度", "Max softmax confidence"), fontsize=10)
    ax.set_ylabel(L("样本数（对数刻度）", "Count (log scale)"), fontsize=10)
    ax.set_title(L("正确 / 错误样本的置信度分布（13 个错误中 7 个置信度 > 0.99）",
                   "Confidence distribution for correct vs incorrect samples"),
                 fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", axis="y")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    p = os.path.join(FIG, "fig3_confidence_gap.png")
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("[FIG] %s" % p)


if __name__ == "__main__":
    print("[B1-FIG] CJK font = %s" % (CJK or "(none, fallback to English)"))
    fig_roc()
    fig_reliability()
    fig_confidence_gap()
    print("[B1-FIG] done -> %s" % FIG)
