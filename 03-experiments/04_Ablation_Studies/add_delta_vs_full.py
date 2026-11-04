#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1 消融表补算 `Δ vs full`（同折对齐）—— 云上没有 `--reference-cv`，故在本地补。

用法:
    python add_delta_vs_full.py \
        --per-fold "AIC_train_again/g1/g1_ablation_per_fold.csv" \
        --reference-cv "E:/Code/NeuroEvo/NeuroEvo_AD/复现版实验数据" \
        --out "g1_ablation_table_with_delta.md"

三条铁律（本脚本存在的主要理由）:
  1. **先硬校验、再算数**：参照的 `fold_manifest_sha256` 与 `architecture_fingerprint`
     必须**逐位等于基线常量**，否则 exit 2 报错 —— 不允许"看起来像就拿来减"。
  2. **只对齐本次跑过的折**：本次 G1 只跑 fold 1,5 ⇒ full 参照也只取 fold 1,5 的均值；
     **绝不用参照的五折均值**（口径不同，见 memory/2026-09-25.md §8.1）。
  3. **未对齐项如实入表**：训练脚本 / GPU 型号 / data_root 三处参照与 G1 不一致，
     必须在交付物里写出来，不能只给一个漂亮的 Δ。
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# ---- 基线常量（与项目 MEMORY.md §2 一致，写死以便当场证伪）-------------------
BASELINE_MANIFEST_SHA = "51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd"
BASELINE_FINGERPRINT = "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975"
BASELINE_PARAMS = 56_812_813

# 参照自身声明的、且**不能**从参照文件里核实的两项（只能标注）
UNVERIFIED = [
    "参照的 entry_script = `train_5fold.py`，与 G1 使用的 `smoke_test.py:train_one_epoch()` **不是同一份训练代码**（等价性未验证）",
    "参照 device = **RTX 3060 (12 GB)**，G1 跑在 **RTX 4090**，且两侧 `deterministic=false` ⇒ Δ 含硬件与非确定性噪声",
    "参照 data_root = `/home/featurize/data/dmsa_pcv2/Combined Dataset`，G1 = `/home/featurize/work/Combined_Dataset`；manifest SHA 相同，但**两份图片内容是否逐字节同一批未验证**",
]


def die(msg: str, code: int = 2):
    print(f"❌ {msg}")
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", required=True)
    ap.add_argument("--reference-cv", required=True)
    ap.add_argument("--out", default="g1_ablation_table_with_delta.md")
    a = ap.parse_args()

    per_fold = Path(a.per_fold)
    ref_root = Path(a.reference_cv)
    if not per_fold.is_file():
        die(f"找不到 per_fold：{per_fold}")
    if not ref_root.is_dir():
        die(f"找不到参照目录：{ref_root}")

    # ---- 1. 读 G1 结果 ----
    with per_fold.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die(f"per_fold 是空的：{per_fold}")
    folds = sorted({int(r["fold"]) for r in rows})
    print(f"[G1] 变体×折 共 {len(rows)} 行，折集合 = {folds}")

    # ---- 2. 硬校验参照 ----
    print("\n[校验] 参照目录合规性")
    ref = {}
    checks = []
    for fd in folds:
        d = ref_root / f"Fold{fd}"
        bm, mi, rc = d / "best_metrics.json", d / "model_info.json", d / "run_config.json"
        for p in (bm, mi, rc):
            if not p.is_file():
                die(f"参照缺文件：{p}")
        best = json.loads(bm.read_text(encoding="utf-8-sig"))
        info = json.loads(mi.read_text(encoding="utf-8-sig"))
        conf = json.loads(rc.read_text(encoding="utf-8-sig"))
        ref[fd] = {"acc": float(best["accuracy"]), "epoch": int(best["epoch"]),
                   "info": info, "conf": conf}

        sha = str(conf.get("fold_manifest_sha256", ""))
        fp = str(info.get("architecture_fingerprint", ""))
        np_ = int(info.get("total_params", -1))
        checks += [
            (f"Fold{fd} manifest SHA256", sha == BASELINE_MANIFEST_SHA,
             sha[:16] + "…" if sha else "(缺失)"),
            (f"Fold{fd} fingerprint", fp == BASELINE_FINGERPRINT, fp[:16] + "…" if fp else "(缺失)"),
            (f"Fold{fd} total_params", np_ == BASELINE_PARAMS, f"{np_:,}"),
            (f"Fold{fd} protocol", conf.get("protocol") == "P-CV v2 FINAL", str(conf.get("protocol"))),
            (f"Fold{fd} seed", int(conf.get("seed", -1)) == 42, str(conf.get("seed"))),
            (f"Fold{fd} epochs", int(conf.get("epochs", -1)) == 200, str(conf.get("epochs"))),
            (f"Fold{fd} batch_size", int(conf.get("batch_size", -1)) == 16, str(conf.get("batch_size"))),
        ]

    bad = [c for c in checks if not c[1]]
    for name, ok, val in checks:
        print(f"  {'✅' if ok else '❌'} {name:<26} = {val}")
    if bad:
        die(f"参照未通过硬校验（{len(bad)} 项不符）→ 拒绝出 Δ（详见上方 ❌）")

    full_mean = statistics.fmean(ref[f]["acc"] for f in folds)
    print(f"\n[参照] full 在折 {folds} 的均值 = {full_mean:.8f}"
          f"  （⚠️ 仅 {len(folds)} 折，非五折均值）")

    # ---- 3. 逐变体算 Δ ----
    variants, order = {}, []
    for r in rows:
        v = r["variant"]
        if v not in variants:
            variants[v] = {"params": int(r["params"]), "acc": [], "ep": [], "fold": []}
            order.append(v)
        variants[v]["acc"].append(float(r["accuracy"]))
        variants[v]["ep"].append(int(r["best_epoch"]))
        variants[v]["fold"].append(int(r["fold"]))

    full_info = ref[folds[0]]["info"]
    mdc_params = int(full_info.get("multi_scale_dilated_params", 0))
    sam_params = int(full_info.get("spatial_attention_params", 0))

    lines = []
    lines.append("# G1 模块消融矩阵（本地补算 Δ vs full）\n")
    lines.append("> 评估集：**折内验证集（val）**；sealed test **未被触碰**。")
    lines.append("> 选模：折内按 val accuracy 选 best epoch（与基线一致）；200 epochs；seed=42。")
    lines.append(f"> Δ 口径：**同折对齐**，参照只取本次跑过的折 {folds} 的均值"
                 f"（**不是**五折均值）。原始表见 `AIC_train_again/g1/g1_ablation_table.md`。\n")

    lines.append("| 变体 | 参数量 | ΔParams | 折数 | val 准确率(均) | **Δ vs full** | 折间跨度 | 各折 best epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(f"| **`full`**（参照·本机） | {BASELINE_PARAMS:,} | — | {len(folds)} | "
                 f"{full_mean*100:.3f}% | — | {max(ref[f]['acc'] for f in folds)*100 - min(ref[f]['acc'] for f in folds)*100:.3f} pp | "
                 f"{[ref[f]['epoch'] for f in folds]} |")
    for v in order:
        d = variants[v]
        m = statistics.fmean(d["acc"])
        delta = (m - full_mean) * 100.0
        spread = (max(d["acc"]) - min(d["acc"])) * 100.0
        dp = d["params"] - BASELINE_PARAMS
        lines.append(f"| `{v}` | {d['params']:,} | {dp:+,} | {len(d['acc'])} | "
                     f"{m*100:.3f}% | **{delta:+.3f} pp** | {spread:.3f} pp | {d['ep']} |")
    lines.append("")

    # ---- 4. 诚实说明（硬性）----
    lines.append("## 必读：这张表的效力边界\n")
    lines.append(f"1. **统计效力 = 2 折（{folds}），没有 CI**。n=2 的均值差**不做显著性检验** ⇒")
    lines.append("   **不得**写成「显著提升/下降」，只能说「方向上变化 X pp」。项目铁律要求")
    lines.append("   统计单位=源图、CI=cluster bootstrap，本表两者都不满足。")
    lines.append(f"   参考量级：{(full_mean*100 - variants[order[0]]['acc'][0]*100):.3f} pp ≈ "
                 f"{abs(round((full_mean - variants[order[0]]['acc'][0]) * 2048))} 张图（每折 2,048 张）。\n")

    lines.append("2. **容量可比性**（决定能不能归因给「模块语义」）：")
    if "no_sam" in variants:
        d_sam = variants["no_sam"]["params"] - BASELINE_PARAMS
        lines.append(f"   - `no_sam`：ΔParams = **{d_sam:+,}**（`spatial_attention` 自身 {sam_params:,}）"
                     f"⇒ **容量几乎不变，唯一可干净归因的消融**。")
    lines.append(f"   - `no_mdc`：ΔParams = **{variants.get('no_mdc', {}).get('params', 0) - BASELINE_PARAMS:+,}**，"
                 f"其中 MDC 自身仅 {mdc_params:,} ⇒ 余下 "
                 f"{abs(variants.get('no_mdc', {}).get('params', 0) - BASELINE_PARAMS) - mdc_params:,} "
                 f"来自相邻层通道变化 ⇒ **不可归因单模块**。")
    if "no_ilf" in variants:
        lines.append(f"   - `no_ilf`：ΔParams = **{variants['no_ilf']['params'] - BASELINE_PARAMS:+,}**；"
                     f"且替身同时移除了浅层支路 + `align_bn` + `Swish` + attention 门控 ⇒ **不可归因单组件**。")
    lines.append("")

    lines.append("3. **参照与实验有 3 处未对齐（未验证，不掩盖）**：")
    for u in UNVERIFIED:
        lines.append(f"   - {u}")
    lines.append("")

    if "no_ilf" in variants:
        accs = variants["no_ilf"]["acc"]
        lines.append(f"4. **`no_ilf` 是「训练未收敛」，不是「精度下降」**：各折 best_epoch = "
                     f"{variants['no_ilf']['ep']}（第 2 个 epoch 之后就没再好过），"
                     f"acc = {['%.4f' % x for x in accs]} ≈ 随机。")
        lines.append("   val_loss 应 ≈ ln4 = 1.3862944。**结论只能写「移除 ILF 后训练不收敛」**，")
        lines.append("   要主张「ILF 的注意力融合机制不可或缺」必须先补公平消融（保留 BN/Swish/门控，只换成等权融合）。\n")

    lines.append("5. **待复核的巧合**：`no_sam` 两折 acc 完全相同（都 "
                 f"{variants[order[0]]['acc'][0]:.8f} = 都对 {round(variants[order[0]]['acc'][0]*2048)}/2048 张）。"
                 "两折 `best_epoch`/`elapsed_seconds` 不同 ⇒ 非复制错误；但 `fold_1` 的 "
                 "`metrics.csv` 已被误删，**无法逐 epoch 复核**，故标注存疑。\n")

    out = Path(a.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 已写出：{out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
