#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A1-cal (v2.4) —— 阈值与变换集标定
=================================

【v2.4 修正 P0-4】
  v2.3 提议用 "Mild 编号 1–2412 全为互异原始图" 构造 hard negatives。
  **这个构造与 v2.3 自身的 E2 假说数学矛盾**：E2 说 Mild 原始训练源只有 717 张、
  其余 1,843 张为派生；那么 2,412 个编号里不可能全是互异原始图。
  从 "已发现的 125 个 exact 对都涉及尾部编号" 只能推出**该区间没有已检出的
  字节级重复**，推不出里面没有旋转/缩放/重编码副本。
  ⇒ 该 hard-negative 池**整体作废**。

  替代设计：**三态池**
    positive  仅来自可追溯证据（发布者 lineage / 字节级相同 / 人工确认）
    negative  仅来自可追溯证据（不同原始源且可证明，例如发布者 subject/patient ID 不同）
    pending   距离落在边界、无法判定 —— **不进 P/R 计算**，单独报告占比
  无 lineage 时，synthetic stress positives 只能用于"已知变换的压力测试"，
  不得用来声称 "真实数据 precision ≥ 95% / recall ≥ 90%"。

输出：calibration.csv、transform_set_report.csv、
      suggested_thresholds.json（含 ssim_th，供 dedup_config.json 使用）
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from B15_lib_deduplication import (Table, TransformSet, hamming_min_over_T, phash_u64,
                       code_fingerprint, config_hash, write_artifact)

THREE_STATES = ("positive", "negative", "pending")


def three_state_split(positives: Sequence[Tuple[int, int]],
                      negatives: Sequence[Tuple[int, int]],
                      boundary: Sequence[Tuple[int, int]]) -> Dict[str, list]:
    """把标定 pair 分成三态。pending 不参与 P/R 计算。"""
    return {"positive": list(positives), "negative": list(negatives),
            "pending": list(boundary)}


def _synthetic_pairs(n: int, seed: int = 0):
    """在无法取得真实谱系时，只能产出**压力测试正例**（已知变换）。
    这些不是真实数据分布，只能回答"检测器能否识别我自己施加的变换"。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224]
    pos, neg = [], []
    for i in range(n):
        base = np.clip(
            np.exp(-(((xx - 112) ** 2 + (yy - 100) ** 2) / (2 * 45 ** 2))) * 200
            + rng.normal(0, 6, (224, 224)), 0, 255).astype(np.uint8)
        a = Image.fromarray(base, "L")
        for deg, flip in [(15, False), (30, False), (45, True), (90, False)]:
            b = a.rotate(deg, expand=False, resample=Image.BICUBIC)
            if flip:
                b = b.transpose(Image.FLIP_LEFT_RIGHT)
            pos.append((a, b, f"rot{deg}{'_flip' if flip else ''}"))
        other = np.clip(
            np.exp(-(((xx - 90) ** 2 + (yy - 140) ** 2) / (2 * 30 ** 2))) * 200
            + rng.normal(0, 6, (224, 224)), 0, 255).astype(np.uint8)
        neg.append((a, Image.fromarray(other, "L"), "different_source"))
    return pos, neg


def evaluate(pairs: List[Tuple[Image.Image, Image.Image, str, int]],
             ts: TransformSet, ths: Sequence[int]) -> Table:
    """pairs: (imgA, imgB, label, state) 其中 state in {0:pos,1:neg,2:pending}

    返回每阈值一行：tp/fp/fn/tn、precision、recall。
    **pending 一律不进 P/R 计算**（只统计其数量）。
    """
    rows = []
    if not pairs:
        return Table(rows)
    HA = np.vstack([ts.hashes(a)[None, :] for a, _, _, _ in pairs])
    HB = np.vstack([ts.hashes(b)[None, :] for _, b, _, _ in pairs])
    D = hamming_min_over_T(HA, HB, mode="two-sided")
    d = np.diag(D).astype(int)
    y = np.array([s for _, _, _, s in pairs])
    mask = y != 2  # 排除 pending
    for th in ths:
        pred = d <= th
        tp = int((pred & mask & (y == 0)).sum())
        fp = int((pred & mask & (y == 1)).sum())
        fn = int((~pred & mask & (y == 0)).sum())
        tn = int((~pred & mask & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        rows.append(dict(threshold=th, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=(round(prec, 4) if prec is not None else None),
                         recall=(round(rec, 4) if rec is not None else None),
                         n_pending=int((y == 2).sum())))
    return Table(rows)


def split_cal_holdout(pairs: List[tuple], holdout_frac: float = 0.4,
                      seed: int = 20260923) -> Tuple[List[tuple], List[tuple]]:
    """把 pair 池按 state 分层切成**标定集**与**留出验证集**。

    【v2.4 修正（GPT T5 要求）】GPT 明确要求"标定集与独立验证集分开"。
    v2.4 此前只在同一个池上选阈值并报指标 —— 那是**自证**（in-sample），
    会系统性高估 precision/recall。现在：
      · 阈值**只在标定集上选**；
      · 报告的是**留出集**上的指标（honest estimate）。
    """
    rng = np.random.default_rng(seed)
    cal, hold = [], []
    for st in (0, 1, 2):
        idx = [i for i, p in enumerate(pairs) if p[3] == st]
        rng.shuffle(idx)
        n_hold = int(round(len(idx) * holdout_frac))
        hold += [pairs[i] for i in idx[:n_hold]]
        cal += [pairs[i] for i in idx[n_hold:]]
    return cal, hold


def main():
    ap = argparse.ArgumentParser(description="A1-cal (v2.4) 阈值标定")
    ap.add_argument("--n", type=int, default=25, help="每个变换的正例样本数")
    ap.add_argument("--angle-step", type=int, default=45)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--out", default="_a1cal_out")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--holdout-frac", type=float, default=0.4,
                    help="留出验证集比例（GPT T5：标定集与验证集必须分开）")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ts = TransformSet(a.angle_step, not a.no_flip)

    print("⚠️ 无真实谱系时的默认模式只能做**已知变换压力测试**，"
          "不得据此声称真实数据 precision/recall。")
    pos, neg = _synthetic_pairs(a.n)
    pairs = [(x, y, lab, 0) for x, y, lab in pos] + [(x, y, lab, 1) for x, y, lab in neg]

    cal, hold = split_cal_holdout(pairs, holdout_frac=a.holdout_frac)
    print(f"  标定集 {len(cal)} 对 / 留出验证集 {len(hold)} 对（分层切分）")

    ths = list(range(0, 21))
    df_cal = evaluate(cal, ts, ths)
    df_hold = evaluate(hold, ts, ths)
    df_cal.to_csv(out / "calibration.csv", index=False, encoding="utf-8-sig")
    df_hold.to_csv(out / "calibration_holdout.csv", index=False, encoding="utf-8-sig")

    # 变换集报告：各角度在本变换集下的距离
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:224, 0:224]
    base = np.clip(np.exp(-(((xx - 112) ** 2 + (yy - 100) ** 2) / (2 * 45 ** 2))) * 200
                   + rng.normal(0, 6, (224, 224)), 0, 255).astype(np.uint8)
    im = Image.fromarray(base, "L")
    rep = []
    for deg in [0, 5, 10, 15, 30, 45, 90]:
        rot = im.rotate(deg, expand=False, resample=Image.BICUBIC)
        HA = ts.hashes(im)[None, :]
        HB = ts.hashes(rot)[None, :]
        d = int(hamming_min_over_T(HA, HB, mode="two-sided")[0, 0])
        rep.append(dict(angle=deg, distance=d, n_transforms=len(ts)))
    Table(rep).to_csv(out / "transform_set_report.csv", index=False,
                             encoding="utf-8-sig")

    # 阈值只在**标定集**上选
    prec_cal = df_cal["precision"].fillna(0)
    rec_cal = df_cal["recall"].fillna(0)
    ok = df_cal[(prec_cal >= a.target_precision) & (rec_cal > 0)]
    suggested = int(ok["threshold"].max()) if len(ok) else None

    # 报的是**留出集**上的指标（honest）
    hold_row = None
    if suggested is not None:
        for r in df_hold.rows:
            if int(r["threshold"]) == suggested:
                hold_row = r
                break

    res = dict(primary_th=suggested,
               ssim_th=None,   # 必须由真实确认层标定；无真实谱系时保持 None，A2 会拒绝运行
               #: v2.7（N11）：**必须显式声明本次标定是否基于真实数据**。
               #: 合成压力测试给出的阈值**不可迁移**：2026-09-23 实测，
               #: A1-cal 建议的 17 在真实数据上 FPR = 99.94%（见 `_a0_out/detector_power.md`）。
               #: A1 读到本字段为 False 时会把来源标为 `synthetic_calibration`
               #: 并在报告里降级为 exploratory —— 不再允许"合成阈值冒充正式标定"。
               real_data_calibrated=False,
               calibration_pool="synthetic_stress_test",
               distance_mode="two-sided",
               angle_step=a.angle_step, use_flip=not a.no_flip,
               selection_set="calibration", report_set="holdout",
               n_cal=len(cal), n_holdout=len(hold),
               holdout_precision=(hold_row or {}).get("precision"),
               holdout_recall=(hold_row or {}).get("recall"),
               note=("primary_th 在标定集上选、在留出集上报告（GPT T5 要求）；"
                     "注意两者均来自**合成压力测试**，不代表真实 MRI 指标。"
                     "**真实数据上的判别力必须另跑 `a0c_detector_power.py` 验证**"
                     "（实测：本建议阈值在真实数据上 FPR 极高，不可直接使用）。"
                     "ssim_th 未标定 -> A2 将拒绝运行（不得静默回退 0.98）"))

    #: N6：统一盖加入口。**这正是 v2.4 唯一写了指纹、却因为没有回读校验
    #: 而悄悄过期的那一个产物** —— 现在改由 stamp() 统一生成 code_fingerprint。
    stamped = write_artifact(out / "suggested_thresholds.json", res)

    print("\n── 标定集（用于选阈值）──")
    print(df_cal.to_string(index=False))
    print("\n── 留出验证集（用于报告，honest）──")
    print(df_hold.to_string(index=False))
    print(f"\n建议 primary_th = {suggested}（标定集目标 precision >= {a.target_precision}）")
    if hold_row:
        print(f"  留出集实测：precision={hold_row['precision']}  recall={hold_row['recall']}")
    print(f"  code_fingerprint = {stamped['code_fingerprint'][:16]}…")
    print(f"输出：{out/'calibration.csv'}, {out/'calibration_holdout.csv'}, "
          f"{out/'transform_set_report.csv'}, {out/'suggested_thresholds.json'}")


if __name__ == "__main__":
    main()
