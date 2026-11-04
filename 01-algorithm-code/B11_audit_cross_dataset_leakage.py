#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A2 (v2.4) —— train <-> test 近重复审计
======================================

【v2.4 修正 P0-3】
  · **删除 MAX_CONFIRM_PER_TEST=200 截断**。v2.3 说"复核所有 distance<=threshold
    的候选"，实现里却又截断到 200 —— 原文话术与代码不符。本版默认**全量复核**；
    仅当显式传入 --max-confirm 时才分批，并把审计标记为 incomplete 并写出
    unreviewed_candidates.csv（不得以"零命中"作为完整审计结论）。
  · **ssim_th 必须来自 dedup_config.json / run_config.json**，读取不到就报错退出
    （v2.3 的 `CFG.get("ssim_th", 0.98)` 是静默回退，等于把未标定的阈值写死）。
  · 距离复用 lib_dedup 的**正确**汉明实现（v2.3 的逐字节 minimum 会把 56 算成 0）。
  · 几何对齐 = 变换集内的旋转+翻转搜索；**平移/裁剪/缩放不是已实现的配准**，
    报告中按实际能力命名（不得自称 registration）。

输出：
    near_dup_curve.csv        阈值 -> 候选数 -> 确认数 三列曲线
    confirmed_matches.csv     确认命中的 train<->test 对
    unreviewed_candidates.csv 被截断/未复核的候选（正常全量时为空）
    audit_status.json         完整 / 不完整标记
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

from B15_lib_deduplication import (Table, TransformSet, hamming_min_over_T, load_config,
                       hamming_two_stage, write_artifact)


def best_aligned_ssim(pa: pathlib.Path, pb: pathlib.Path, ts: TransformSet):
    """在变换集上搜索最佳几何对齐后的最大 SSIM。

    注意：**只做旋转 + 翻转**。平移 / crop / 尺度变化未实现，
    报告里不得把这些称为"已配准"。
    """
    from skimage.metrics import structural_similarity as ssim
    A0 = Image.open(pa).convert("L")
    B0 = Image.open(pb).convert("L")
    best = (-1.0, None, None, "")
    for deg, flip in ts.members:
        B = ts.apply(B0, deg, flip)
        if A0.size == B.size:
            X = np.asarray(A0, np.float32)
            Y = np.asarray(B, np.float32)
            mode = "same_size"
        else:
            w = min(A0.size[0], B.size[0])
            h = min(A0.size[1], B.size[1])
            X = np.asarray(A0.resize((w, h)), np.float32)
            Y = np.asarray(B.resize((w, h)), np.float32)
            mode = f"resized({A0.size}->{B.size})"
        s = float(ssim(X, Y, data_range=255.0))
        if s > best[0]:
            best = (s, deg, flip, mode)
    return best


def main():
    ap = argparse.ArgumentParser(description="A2 (v2.4) train<->test 近重复审计")
    ap.add_argument("--config", default="_a1v2_out/run_config.json",
                    help="优先用 A1 产出的 run_config.json（保证同一变换集）")
    ap.add_argument("--train", default="Combined Dataset/train")
    ap.add_argument("--test", default="Combined Dataset/test")
    ap.add_argument("--out", default="_a2_out")
    ap.add_argument("--max-confirm", type=int, default=0,
                    help="0 = 全量复核（默认）；>0 = 分批上限，审计将标记为 incomplete")
    a = ap.parse_args()

    cfg = load_config(pathlib.Path(a.config))   # 缺 ssim_th 会直接抛错
    ts = TransformSet(cfg["angle_step"], cfg["use_flip"])
    ssim_th = float(cfg["ssim_th"])
    ths = cfg.get("sens_ths", [2, 3, 4, 6, 8])

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def collect(root):
        root = pathlib.Path(root)
        if not root.exists():
            raise SystemExit(f"目录不存在：{root}。检查 --train/--test。")
        ps = sorted(p for p in root.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        H = np.vstack([ts.hashes(Image.open(p).convert("L"))[None, :] for p in ps])
        return ps, H

    tr_p, tr_h = collect(a.train)
    te_p, te_h = collect(a.test)
    print(f"train={len(tr_p)} test={len(te_p)} |G|={len(ts)} "
          f"mode={cfg.get('distance_mode','two-sided')} ssim_th={ssim_th}")

    D = hamming_min_over_T(te_h, tr_h, mode=cfg.get("distance_mode", "two-sided"))

    curve, confirmed, unreviewed = [], [], []
    for th in ths:
        ii, jj = np.nonzero(D <= th)
        cand = list(zip(ii.tolist(), jj.tolist()))
        truncated = False
        todo = cand
        if a.max_confirm and len(cand) > a.max_confirm:
            order = sorted(cand, key=lambda t: D[t[0], t[1]])
            todo = order[:a.max_confirm]
            for x, y in order[a.max_confirm:]:
                unreviewed.append(dict(threshold=th, test_file=te_p[x].name,
                                       train_file=tr_p[y].name, dist=int(D[x, y])))
            truncated = True
        hits = 0
        for x, y in todo:
            s, deg, flip, mode = best_aligned_ssim(te_p[x], tr_p[y], ts)
            if s >= ssim_th:
                hits += 1
                if th == cfg["primary_th"]:
                    confirmed.append(dict(test_file=te_p[x].name,
                                          train_file=tr_p[y].name,
                                          dist=int(D[x, y]), ssim=round(s, 4),
                                          angle=deg, flip=flip, geom_mode=mode))
        curve.append(dict(threshold=th, n_candidates=len(cand),
                          n_reviewed=len(todo), n_confirmed=hits,
                          truncated=truncated))
        print(f"  th={th}: 候选={len(cand)} 复核={len(todo)} 确认={hits}"
              + ("  [TRUNCATED]" if truncated else ""))

    Table(curve).to_csv(out / "near_dup_curve.csv", index=False,
                               encoding="utf-8-sig")
    Table(confirmed).to_csv(out / "confirmed_matches.csv", index=False,
                                   encoding="utf-8-sig")
    Table(unreviewed).to_csv(out / "unreviewed_candidates.csv", index=False,
                                    encoding="utf-8-sig")
    incomplete = bool(unreviewed)
    status = dict(complete=not incomplete,
                  n_confirmed=len(confirmed),
                  transform_set=f"{cfg['angle_step']}deg x flip={cfg['use_flip']}",
                  distance_mode=cfg.get("distance_mode", "two-sided"),
                  ssim_th=ssim_th,
                  geometric_alignment="rotation+flip search only "
                                      "(translation/crop/scale NOT implemented)")
    #: N6：统一盖加入口（原来这里没有 code_fingerprint）
    write_artifact(out / "audit_status.json", status,
                   cfg=cfg if isinstance(cfg, dict) else None)
    print(f"\n审计状态：{'INCOMPLETE（有未复核候选）' if incomplete else 'COMPLETE（全量复核）'}")
    print(f"输出：{out/'near_dup_curve.csv'}, {out/'confirmed_matches.csv'}, "
          f"{out/'unreviewed_candidates.csv'}, {out/'audit_status.json'}")


if __name__ == "__main__":
    main()
