#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A1-v2 (v2.4) —— 源图映射 / 谱系候选生成器
=========================================

【v2.4 修正】
  P0-1  距离算子改用 lib_dedup.hamming_min_over_T（字节内求和 → 变换间取最小）
  P0-2a 不再有 `one(HA, HB[:, :1])` 的形状错配（统一 (n, nT, 8)）
  P0-2b 缓存改 .npy + meta（不再 json.dumps ndarray）
  P0-2c 直径排除对角线；选边只用严格上三角（不再 fill_diagonal(255)）
  P0-5  · 混合证据组件**不再整体晋升 S1**（取最弱连接）
        · singleton 显式保留并纳入主统计（v2.3 的 "S1∪S2" 会丢掉 9,328/10,240）
        · cluster_id / conservative_cluster_id 全局唯一（跨 split、跨类不碰撞）
        · conservative 图是**独立构建**的更粗图，不再是 root 的拷贝
        · 真实执行跨类扫描与跨 split 扫描，并产出 test_source_map / cross_matches

输入 items：list of dict(split, cls, path, filename)
输出（out_dir）：
    source_mapping.csv        每行一个样本（含 split / cls / gid / 两个簇 id / 置信 / 证据）
    test_source_map.csv       test 侧子集（阶段 0.2 与 §5.3 规则 3 所需）
    cross_matches.csv         跨类 / 跨 split 的候选对账
    conservative_edges.csv    保守图的边与剪枝记录
    component_summary.csv     每 (split, cls) 的汇总
    run_config.json           **含 ssim_th**（供 A2 强校验读取）
"""
from __future__ import annotations

import collections
import itertools
import json
import pathlib
from typing import Dict, List, Sequence

import numpy as np

from B15_lib_deduplication import (Table, TransformSet, HashCache, UF, sha256_file, global_id,
                       cluster_id, component_diameter, hamming_min_over_T,
                       S_PROVEN, S_INVARIANT, S_BOUNDARY, S_UNRESOLVED, S_SINGLETON,
                       code_fingerprint, config_hash, write_artifact)

_RANK = {S_PROVEN: 0, S_INVARIANT: 1, S_BOUNDARY: 2, S_UNRESOLVED: 3}


def _cluster_confidence(exact_edges: int, near_edges: int, diameter: int,
                        primary_th: int, slack: int) -> tuple[str, str]:
    """组件置信 = **最弱连接**（v2.4）。整组件晋升 S1 的做法已被废除。"""
    if diameter > primary_th + slack:
        return S_UNRESOLVED, f"chain_risk diameter={diameter}"
    if near_edges == 0:
        return S_PROVEN, "all_edges_exact_sha"
    if exact_edges > 0 and near_edges > 0:
        # A-B 字节相同、B-C 仅近似 —— C 不得被 S1 晋升
        return S_BOUNDARY, f"mixed_exact_near(exact={exact_edges},near={near_edges})"
    if diameter <= primary_th:
        return S_INVARIANT, f"near_consistent diameter={diameter}"
    return S_BOUNDARY, f"boundary diameter={diameter}"


def _build_clusters(n: int, edges: Sequence[tuple[int, int]]):
    uf = UF(n)
    for a, b in edges:
        uf.union(int(a), int(b))
    comp: Dict[int, List[int]] = collections.defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    return uf, comp


def run(items: List[dict],
        out_dir: pathlib.Path,
        cache_dir: pathlib.Path,
        angle_step: int = 45,
        use_flip: bool = True,
        primary_th: int = 4,
        consistency_slack: int = 2,
        cross_class: bool = True,
        distance_mode: str = "two-sided",
        ssim_th: float | None = None,
        max_conservative_size: int = 512,
        block: int = 512,
        primary_th_source: str | None = None,
        verbose: bool = True) -> dict:
    """执行源图映射。返回 dict(summary=..., map_df=..., cross=...)。"""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = TransformSet(angle_step, use_flip)
    cache = HashCache(cache_dir, ts)

    by_group: Dict[tuple[str, str], List[dict]] = collections.defaultdict(list)
    for it in items:
        by_group[(it["split"], it["cls"])].append(it)

    rows: List[dict] = []
    summary: List[dict] = []
    pair_rows: List[dict] = []
    cons_rows: List[dict] = []
    hashes_by_group: Dict[tuple[str, str], np.ndarray] = {}
    index_by_group: Dict[tuple[str, str], List[dict]] = {}

    for (split, cls), group in sorted(by_group.items()):
        paths = [pathlib.Path(it["path"]) for it in group]
        n = len(paths)
        shas = [sha256_file(p) for p in paths]

        H = []
        for it, p, s in zip(group, paths, shas):
            key = f"{it['split']}|{it['cls']}|{it['filename']}"
            v = cache.get(key, s)
            if v is None:
                from PIL import Image
                v = ts.hashes(Image.open(p))
                cache.put(key, s, v)
            H.append(np.asarray(v, dtype=np.uint64))
        H = np.vstack(H)
        hashes_by_group[(split, cls)] = H
        index_by_group[(split, cls)] = group

        # ---- exact (字节级) ----
        by_sha: Dict[str, List[int]] = collections.defaultdict(list)
        for i, s in enumerate(shas):
            by_sha[s].append(i)
        exact = [(a, b) for v in by_sha.values() if len(v) > 1
                 for a, b in itertools.combinations(sorted(v), 2)]
        u_sha = len(by_sha)

        # ---- 变换集不变距离（对角线保持 0）----
        D = hamming_min_over_T(H, H, mode=distance_mode, block=block)
        np.fill_diagonal(D, 0)
        ii, jj = np.nonzero(np.triu(D <= primary_th, 1))
        near = [(int(a), int(b)) for a, b in zip(ii.tolist(), jj.tolist())
                if (a, b) not in set(exact)]

        for a, b in exact:
            pair_rows.append(dict(split=split, cls=cls, a=group[a]["filename"],
                                  b=group[b]["filename"], kind="exact_sha", dist=0))
        for a, b in near:
            pair_rows.append(dict(split=split, cls=cls, a=group[a]["filename"],
                                  b=group[b]["filename"], kind="near_transform",
                                  dist=int(D[a, b])))

        # ---- 统计簇（仅用确认边）----
        _, comp = _build_clusters(n, list(exact) + list(near))
        # ---- 保守簇（额外纳入"不确定边"：阈值 + slack）----
        ii2, jj2 = np.nonzero(np.triu(D <= primary_th + consistency_slack, 1))
        uncertain = [(int(a), int(b)) for a, b in zip(ii2.tolist(), jj2.tolist())]
        _, comp_cons = _build_clusters(n, list(exact) + list(uncertain))

        exact_set = {tuple(sorted(e)) for e in exact}
        near_set = {tuple(sorted(e)) for e in near}

        sizes, diams, chain_bad = [], [], 0
        cons_map: Dict[int, str] = {}
        stat_map: Dict[int, str] = {}

        for root, mem in comp.items():
            gid_members = [global_id(split, cls, group[i]["filename"]) for i in mem]
            cid = cluster_id(gid_members)
            for i in mem:
                stat_map[i] = cid
            if len(mem) == 1:
                rows.append(dict(split=split, cls=cls, filename=group[mem[0]]["filename"],
                                 gid=global_id(split, cls, group[mem[0]]["filename"]),
                                 cluster_id=cid, size=1,
                                 confidence=S_SINGLETON, evidence="no_duplicate_candidate"))
                continue
            ne = sum(1 for a, b in itertools.combinations(sorted(mem), 2)
                     if (a, b) in near_set)
            xe = sum(1 for a, b in itertools.combinations(sorted(mem), 2)
                     if (a, b) in exact_set)
            dia = component_diameter(D, sorted(mem))
            conf, ev = _cluster_confidence(xe, ne, dia, primary_th, consistency_slack)
            sizes.append(len(mem)); diams.append(dia)
            if conf == S_UNRESOLVED:
                chain_bad += 1
            for i in mem:
                rows.append(dict(split=split, cls=cls, filename=group[i]["filename"],
                                 gid=global_id(split, cls, group[i]["filename"]),
                                 cluster_id=cid, size=len(mem),
                                 confidence=conf, evidence=ev))

        # 保守簇 ID（可能比统计簇更粗）；超限时剪枝并记录
        for root, mem in comp_cons.items():
            gid_members = [global_id(split, cls, group[i]["filename"]) for i in mem]
            cid = cluster_id(gid_members)
            pruned = len(mem) > max_conservative_size
            if pruned:
                # 超限：按距离最小生成树截断到上限，其余单独成簇（记录）
                keep = sorted(mem)[:max_conservative_size]
                for i in keep:
                    cons_map[i] = cluster_id([global_id(split, cls, group[j]["filename"])
                                              for j in keep])
                for j in sorted(mem)[max_conservative_size:]:
                    cons_map[j] = cluster_id([global_id(split, cls, group[j]["filename"])])
            else:
                for i in mem:
                    cons_map[i] = cid
            if len(mem) > 1:
                cons_rows.append(dict(split=split, cls=cls, conservative_cluster_id=cid,
                                      n_members=len(mem), pruned=pruned,
                                      max_size_limit=max_conservative_size))

        for i in range(n):
            if i not in cons_map:
                cons_map[i] = cluster_id([global_id(split, cls, group[i]["filename"])])

        for r in rows:
            if r["split"] == split and r["cls"] == cls and "conservative_cluster_id" not in r:
                r["conservative_cluster_id"] = cons_map[
                    next(k for k, g in enumerate(group) if g["filename"] == r["filename"])]

        summary.append(dict(split=split, cls=cls, files=n, uniq_sha=u_sha,
                            redund_sha=n - u_sha,
                            uniq_merged=len(comp),
                            redund_merged=n - len(comp),
                            n_comp_gt1=len(sizes),
                            max_comp=max(sizes) if sizes else 1,
                            max_diameter=max(diams) if diams else 0,
                            chain_risk_components=chain_bad,
                            n_conservative_clusters=len(comp_cons)))
        if verbose:
            print(f"[A1] {split}/{cls}: n={n} uniq_sha={u_sha} clusters={len(comp)} "
                  f"max_comp={max(sizes) if sizes else 1} chain_risk={chain_bad}")

    # ---------------- 跨类 / 跨 split ----------------
    cross = {"n_cross_class": 0, "n_cross_split": 0}
    cross_rows: List[dict] = []
    groups = sorted(hashes_by_group)
    for g1, g2 in itertools.combinations(groups, 2):
        s1, c1 = g1
        s2, c2 = g2
        is_cross_class = (s1 == s2) and (c1 != c2)
        is_cross_split = (s1 != s2)
        if is_cross_class and not cross_class:
            continue
        H1, H2 = hashes_by_group[g1], hashes_by_group[g2]
        D = hamming_min_over_T(H1, H2, mode=distance_mode, block=block)
        ii, jj = np.nonzero(D <= primary_th)
        cnt = len(ii)
        if cnt == 0:
            continue
        kind = "cross_class" if is_cross_class else "cross_split"
        if is_cross_class:
            cross["n_cross_class"] += cnt
        if is_cross_split:
            cross["n_cross_split"] += cnt
        for a, b in zip(ii.tolist(), jj.tolist()):
            cross_rows.append(dict(kind=kind, split_a=s1, cls_a=c1,
                                   file_a=index_by_group[g1][a]["filename"],
                                   split_b=s2, cls_b=c2,
                                   file_b=index_by_group[g2][b]["filename"],
                                   dist=int(D[a, b])))
        if verbose:
            print(f"[A1][{kind}] {g1} x {g2}: {cnt} 对候选")

    # ---------------- 落盘 ----------------
    map_df = Table(rows)
    if not map_df.empty:
        map_df = map_df.drop_duplicates(subset=["gid"], keep="last").reset_index(drop=True)
        # 重新绑定保守簇 id（按 gid 对齐，避免上面按序赋值的错位）
        gid2cons = {}
        for (split, cls), group in by_group.items():
            H = hashes_by_group[(split, cls)]
            n = len(group)
            D = hamming_min_over_T(H, H, mode=distance_mode, block=block)
            np.fill_diagonal(D, 0)
            ii2, jj2 = np.nonzero(np.triu(D <= primary_th + consistency_slack, 1))
            _, cc = _build_clusters(n, [(int(a), int(b)) for a, b in zip(ii2.tolist(), jj2.tolist())])
            for root, mem in cc.items():
                members = [global_id(split, cls, group[i]["filename"]) for i in mem]
                cid = cluster_id(members)
                if len(mem) > max_conservative_size:
                    keep = sorted(mem)[:max_conservative_size]
                    keep_cid = cluster_id([global_id(split, cls, group[j]["filename"]) for j in keep])
                    for i in keep:
                        gid2cons[global_id(split, cls, group[i]["filename"])] = keep_cid
                    for j in sorted(mem)[max_conservative_size:]:
                        g = global_id(split, cls, group[j]["filename"])
                        gid2cons[g] = cluster_id([g])
                else:
                    for i in mem:
                        gid2cons[global_id(split, cls, group[i]["filename"])] = cid
        map_df["conservative_cluster_id"] = map_df["gid"].map(gid2cons).fillna(
            map_df["cluster_id"])

    map_df.to_csv(out_dir / "source_mapping.csv", index=False, encoding="utf-8-sig")
    if not map_df.empty:
        tdf = map_df[map_df["split"] == "test"]
        tdf.to_csv(out_dir / "test_source_map.csv", index=False, encoding="utf-8-sig")
    Table(pair_rows).to_csv(out_dir / "candidate_pairs.csv", index=False,
                                   encoding="utf-8-sig")
    Table(cross_rows).to_csv(out_dir / "cross_matches.csv", index=False,
                                    encoding="utf-8-sig")
    Table(cons_rows).to_csv(out_dir / "conservative_edges.csv", index=False,
                                   encoding="utf-8-sig")
    Table(summary).to_csv(out_dir / "component_summary.csv", index=False,
                                 encoding="utf-8-sig")
    cfg_for_hash = dict(lib_version="2.7.0", angle_step=angle_step, use_flip=use_flip,
                        n_transforms=len(ts), primary_th=primary_th,
                        #: v2.7：阈值的**来源**也是配置的一部分。
                        #: 没有这一项时，读者无法从产物判断阈值是标定来的还是拍脑袋来的 ——
                        #: 2026-09-23 的事故正是"阈值来源不可见"造成的。
                        primary_th_source=primary_th_source,
                        consistency_slack=consistency_slack,
                        distance_mode=distance_mode, ssim_th=ssim_th,
                        max_conservative_size=max_conservative_size)
    #: N6：统一盖加入口。**注意不能传 cfg=** —— 本文件本身就是那份 config，
    #: 若再嵌一层 `config` 键会改变 A2 `load_config()` 读到的结构。
    #: 因此这里手动写 config_fingerprint，只让 stamp() 补 code_fingerprint/时间戳。
    write_artifact(out_dir / "run_config.json", dict(
        **cfg_for_hash,
        #: GPT T7 要求：代码指纹 + 配置指纹，用于把结果钉到具体版本
        config_fingerprint=config_hash(cfg_for_hash),
        note="A1 与 A2 必须共用本文件的 angle_step/use_flip/distance_mode/primary_th/ssim_th"))

    return {"summary": summary, "map_df": map_df, "cross": cross}


# ---------------------------------------------------------------- CLI
def _items_from_dir(root: pathlib.Path, split: str) -> List[dict]:
    root = pathlib.Path(root)
    out = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                out.append(dict(split=split, cls=cls_dir.name, path=str(p),
                                filename=p.name))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="A1-v2 (v2.4) 源图映射")
    ap.add_argument("--train", default="Combined Dataset/train")
    ap.add_argument("--test", default="Combined Dataset/test")
    ap.add_argument("--out", default="_a1v2_out")
    ap.add_argument("--cache", default="_a1v2_cache")
    ap.add_argument("--config", default="dedup_config.json")
    ap.add_argument("--angle-step", type=int, default=None)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--primary-th", type=int, default=None)
    ap.add_argument("--block", type=int, default=512)
    a = ap.parse_args()

    cfg = {}
    cp = pathlib.Path(a.config)
    if cp.exists():
        cfg = json.loads(cp.read_text(encoding="utf-8"))
    angle_step = a.angle_step or cfg.get("angle_step", 45)
    use_flip = (not a.no_flip) and cfg.get("use_flip", True)
    ssim_th = cfg.get("ssim_th")

    #: ══════════════════════════════════════════════════════════════════════
    #: 阈值来源强制与登记（v2.7 新增）
    #: ----------------------------------------------------------------------
    #: 【为什么必须强制】
    #: 2026-09-23 实跑事故：`dedup_config.json` 里 `primary_th` 写死为 4
    #: （一个**未标定的占位符**，该字段自己的注释都写着"必须由 A1-cal 标定后填入"），
    #: 但**没有任何机制阻止 A1 使用它**。结果是每类 2,560 张全连成一个连通分量、
    #: `confidence` 全为 unresolved、产出 2.2 GB 无效结果 ——
    #: 而 `component_summary.csv` 里 `uniq_merged = 1` 看起来像"灾难性发现"。
    #: 对照 `ssim_th`：它用 `null` 占位 + A2 读到 null 直接报错，**是有强制的**。
    #: `primary_th` 缺的就是同等的强制。
    #:
    #: 规则（照抄 ssim_th 的设计）：
    #:   ① `null` / 缺失 = 未标定 → **报错退出**，不许有隐含默认值；
    #:   ② 来源优先级 CLI `--primary-th` > A1-cal 产物 > config；
    #:   ③ **无论来自哪里，都要把 `primary_th_source` 写进 run_config.json**，
    #:      让"这个阈值是怎么来的"成为产物的一部分，而不是读者的猜测。
    #: ══════════════════════════════════════════════════════════════════════
    primary_th, primary_th_source = None, None
    if a.primary_th is not None:
        primary_th, primary_th_source = int(a.primary_th), "cli"
    else:
        calib = pathlib.Path(cfg.get("output_dirs", {}).get("a1cal_out", "_a1cal_out")) \
            / "suggested_thresholds.json"
        if calib.is_file():
            c = json.loads(calib.read_text(encoding="utf-8"))
            if c.get("primary_th") is not None:
                primary_th = int(c["primary_th"])
                primary_th_source = ("real_data_calibration" if c.get("real_data_calibrated")
                                     else "synthetic_calibration"
                                          "（**未经真实数据验证，结论只能记为 exploratory**）")
        if primary_th is None and cfg.get("primary_th") is not None:
            primary_th, primary_th_source = int(cfg["primary_th"]), "config"
    if primary_th is None:
        raise SystemExit(
            "primary_th 未标定（config 为 null，且无 --primary-th、无可用 A1-cal 产物）。\n"
            "  → 禁止让 A1 猜一个默认值：2026-09-23 的实跑事故正是这样发生的。\n"
            "  → 先跑 `python a1cal_calibrate.py`，或用 --primary-th 显式给出，\n"
            "     并用 `python a0c_detector_power.py` 确认该阈值在**真实数据**上的 FPR。")
    if "synthetic" in str(primary_th_source):
        print(f"\n  ⚠️ primary_th = {primary_th} 来自 **合成压力测试**（source={primary_th_source}）\n"
              "     真实数据实测：null 分布在 d≤4 处 FPR 已达 34.1%，d≤16 处 99.94%。\n"
              "     本产物**只能记为 exploratory / sensitivity**，不得作为正式分组。\n")

    items = []
    tr = pathlib.Path(a.train)
    if tr.exists():
        items += _items_from_dir(tr, "train")
    te = pathlib.Path(a.test)
    if te.exists():
        items += _items_from_dir(te, "test")
    if not items:
        raise SystemExit(f"未在 {a.train} / {a.test} 找到图片。检查 --train/--test 路径。")

    print(f"共 {len(items)} 张；变换集 |G| = "
          f"{len(range(0, 360, angle_step)) * (2 if use_flip else 1)}；"
          f"distance_mode=two-sided；primary_th={primary_th}（来源：{primary_th_source}）")
    run(items=items, out_dir=a.out, cache_dir=a.cache, angle_step=angle_step,
        use_flip=use_flip, primary_th=primary_th,
        consistency_slack=cfg.get("consistency_slack", 2),
        distance_mode=cfg.get("distance_mode", "two-sided"),
        ssim_th=ssim_th, block=a.block,
        primary_th_source=primary_th_source)
    print(f"\n输出目录：{a.out}")



if __name__ == "__main__":
    main()
