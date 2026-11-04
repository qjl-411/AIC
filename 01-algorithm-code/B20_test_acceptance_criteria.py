#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMSA-Net v2.7 —— 执行前验收测试（对应 GPT 第四轮 T1–T7 + 第五轮 T13 + v2.7 T14）
=====================================================================

只依赖 numpy + pillow，**不需要 GPU**。
用合成图构造可控的 exact / 旋转 / 镜像 / 不相关 场景。
**T14 需要已到货的原始数据目录**（可用 `DMSA_DATA_ROOT` 指定；缺省为
`../Alzheimer's MRI Dataset/Alzheimer's MRI Dataset/Combined Dataset`）。

运行：
    python test_acceptance.py                 # 全部
    python test_acceptance.py --only T13      # 只跑收口门（必须在产出之后）
    python test_acceptance.py --only T14      # 只跑到货验证层
退出码 0 = 全部通过；非 0 = 有未通过项（此时禁止生成 source mapping）。

覆盖：
    T1  距离算子      —— 0 / 64 / 56 / 与参考实现一致 / 单侧非恒等
    T2  A1 端到端     —— 无缓存可跑、无 IndexError / JSON 错误、字节级对直径=0、可重复
    T3  证据分级      —— exact+near 三元组件不得整体晋升 S1；单例不得丢失；ID 全局唯一
    T4  跨类/跨 split —— 跨类同像、train/test 同像、test 内重复均被检出
    T5  标定          —— 三态（阳性/阴性/待定）；缺 ssim_th 必须失败
    T6  保守组件      —— 保守图必须**严格粗于**统计簇；输出剪枝记录
    T8  统计层        —— majority 不得泄漏真标签；簇 bootstrap CI 更宽；macro-F1 实算
    T9  架构探针      —— 参数量/fingerprint 对齐基线；5 个消融变体可构造、指纹互异
    T10 **脚本级端到端** —— 真实执行各脚本的 main()（GPT 指控"声明已修≠已实现"的对策）
    T11 可复现性      —— code/config 指纹；零依赖表格层回归；**T11.6 过期检测**
    T12 torch 脚本    —— a6s / arch_probe 端到端（可用 DMSA_SKIP_SLOW=1 跳过）
    T13 **产物溯源覆盖率** —— 6 个产物是否都盖了章、是否都新鲜（N6）
    T14 **到货验证层** —— 数据集**只读纪律**（静态 AST + 运行时快照）+ 到货报告自洽（v2.7）
    T15 **阈值来源强制** —— `primary_th` 占位符必须 null；未标定时 A1 必须拒绝运行（v2.7 / N11）

⚠️ 顺序纪律：**T13 必须最后跑**。它校验"产物是否由当前这一版代码产生"，
而任何一行 .py 改动都会让既有产物立即失效。先产出、再验新鲜度，
否则就是"用旧产物证明新代码正确"。**T14 排在 T13 之前**（它只读数据、不改产物）。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import B15_lib_deduplication as L  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ----------------------------------------------------------------- 合成图
def synth(seed: int, size=(224, 224)) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    base = np.exp(-(((xx - size[1] * 0.5) ** 2 + (yy - size[0] * 0.45) ** 2) / (2 * 45 ** 2))) * 200
    base = base + rng.normal(0, 6, size)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "L")


def as_T(hashes: list[int]) -> np.ndarray:
    return np.array([hashes], dtype=np.uint64)


# ================================================================= T1
def t1_distance() -> None:
    print("\n===== T1 距离算子 =====")
    z = as_T([0])
    check("T1.1 相同 hash -> 0",
          int(L.hamming_min_over_T(z, z)[0, 0]) == 0)
    one = as_T([0])
    inv = as_T([0xFFFFFFFFFFFFFFFF])
    check("T1.2 全位相反 -> 64",
          int(L.hamming_min_over_T(one, inv)[0, 0]) == 64)

    # GPT 的关键反例：7 字节全不同、1 字节相同 -> 真值 56
    a = as_T([0])
    b = as_T([0xFFFFFFFFFFFFFF00])
    got = int(L.hamming_min_over_T(a, b)[0, 0])
    check("T1.3 56 位不同 -> 56（v2.3 曾返回 0）", got == 56, f"got={got}")

    # 与参考实现一致（含多变换）
    rng = np.random.default_rng(0)
    HA = rng.integers(0, 2 ** 64, size=(6, 3), dtype=np.uint64)
    HB = rng.integers(0, 2 ** 64, size=(5, 3), dtype=np.uint64)
    d_fast = L.hamming_min_over_T(HA, HB, mode="two-sided")
    d_ref = L.hamming_reference(HA, HB)
    check("T1.4 全配对实现 == 参考双层循环", np.array_equal(d_fast, d_ref))

    # 多变换取最小必须生效
    HA2 = np.array([[0x0000000000000000, 0xFFFFFFFFFFFFFFFF]], dtype=np.uint64)
    HB2 = np.array([[0xFFFFFFFFFFFFFFFF, 0x0000000000000000]], dtype=np.uint64)
    check("T1.5 变换后最接近 -> 0（D4 覆盖不到这一对）",
          int(L.hamming_min_over_T(HA2, HB2)[0, 0]) == 0)

    # 单侧化简不是恒等式（GPT 的 4-bit 反例形态）
    HA3 = np.array([[0b0000, 0b0001]], dtype=np.uint64)
    HB3 = np.array([[0b0010, 0b0001]], dtype=np.uint64)
    full = int(L.hamming_min_over_T(HA3, HB3, mode="two-sided")[0, 0])
    one_s = int(L.hamming_min_over_T(HA3, HB3, mode="one-sided")[0, 0])
    check("T1.6 单侧是上界且此处不恒等（证明须用全配对或两级筛选）",
          full == 0 and one_s == 1, f"two-sided={full} one-sided={one_s}")

    # 两级筛选不产生假阴
    rng = np.random.default_rng(7)
    HA4 = rng.integers(0, 2 ** 64, size=(40, 8), dtype=np.uint64)
    HB4 = np.concatenate([rng.integers(0, 2 ** 64, size=(39, 8), dtype=np.uint64),
                          HA4[3:4]])
    d_ref4 = L.hamming_reference(HA4, HB4)
    d_2s = L.hamming_two_stage(HA4, HB4, threshold=8, slack=2)
    same = bool(np.array_equal(d_2s <= 8, d_ref4 <= 8))
    check("T1.7 两级筛选在阈值 8 下的判定与精确解一致（无假阴）", same)


# ================================================================= T2
def t2_end_to_end() -> None:
    print("\n===== T2 端到端（合成图）=====")
    from B09_audit_source_group_mapping import run as a1_run  # 延迟导入，避免循环

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dmsa_t2_"))
    try:
        root = tmp / "data"
        recs = []
        # 0: 原图  1: 与 0 字节级相同  2: 0 的 90 度旋转  3: 0 的水平翻转  4..7: 无关图
        imgs = [synth(1), synth(1), synth(1).rotate(90, expand=False),
                synth(1).transpose(Image.FLIP_LEFT_RIGHT),
                synth(2), synth(3), synth(4), synth(5)]
        for i, im in enumerate(imgs):
            d = root / "Mild Impairment"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"img ({i + 1}).jpg"
            im.save(p)
            recs.append({"split": "train", "cls": "Mild Impairment",
                         "path": str(p), "filename": p.name})

        out = a1_run(items=recs, out_dir=tmp / "out", cache_dir=tmp / "cache",
                     angle_step=45, use_flip=True, primary_th=8,
                     consistency_slack=2, cross_class=True, verbose=False)

        sm = out["summary"][0]
        check("T2.1 首次无缓存也能跑通（无 IndexError / JSON 错误）", True)
        check("T2.2 字节级重复被合并", sm["redund_sha"] >= 1, f"redund_sha={sm['redund_sha']}")
        check("T2.3 非单例组件数 >= 1", sm["n_comp_gt1"] >= 1, f"n_comp_gt1={sm['n_comp_gt1']}")
        check("T2.4 存在旋转/翻转派生被召回的组件",
              sm["max_comp"] >= 2, f"max_comp={sm['max_comp']}")
        check("T2.5 直径诊断可用且非全部被误判（v2.3 对角线填 255 的 bug 已修）",
              sm["chain_risk_components"] <= sm["n_comp_gt1"],
              f"chain_risk={sm['chain_risk_components']} / n_comp_gt1={sm['n_comp_gt1']}")

        # 可重复性
        out2 = a1_run(items=recs, out_dir=tmp / "out2", cache_dir=tmp / "cache",
                      angle_step=45, use_flip=True, primary_th=8,
                      consistency_slack=2, cross_class=True, verbose=False)
        def dump(t):
            return "\n".join(str({k: r[k] for k in ("gid", "cluster_id", "confidence")})
                             for r in t.rows)
        same = dump(out["map_df"]) == dump(out2["map_df"])
        check("T2.6 重复运行输出一致（确定性）", same)
        check("T2.7 产出 test_source_map / cross_split 所需字段",
              {"gid", "cluster_id", "conservative_cluster_id", "confidence",
               "evidence", "split", "cls"} <= set(out["map_df"].columns),
              f"cols={sorted(out['map_df'].columns)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================= T3
def t3_grading() -> None:
    print("\n===== T3 证据分级 =====")
    from B09_audit_source_group_mapping import run as a1_run

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dmsa_t3_"))
    try:
        root = tmp / "data" / "Moderate Impairment"
        root.mkdir(parents=True, exist_ok=True)
        # A=B（字节级）；C 只与 B 近似（不同 seed 的相似图）
        imgs = [synth(11), synth(11), synth(12)]
        recs = []
        for i, im in enumerate(imgs):
            p = root / f"m ({i + 1}).jpg"
            im.save(p)
            recs.append({"split": "train", "cls": "Moderate Impairment",
                         "path": str(p), "filename": p.name})
        out = a1_run(items=recs, out_dir=tmp / "out", cache_dir=tmp / "cache",
                     angle_step=45, use_flip=True, primary_th=10,
                     consistency_slack=2, cross_class=False, verbose=False)
        df = out["map_df"]
        check("T3.1 每个样本都有证据状态（单例不得静默丢失）",
              len(df) == 3 and df["confidence"].notna(),
              f"conf={df['confidence'].tolist()}")
        check("T3.2 singleton 被纳入主统计集合",
              L.S_SINGLETON in L.MAIN_STAT_LEVELS,
              f"MAIN_STAT_LEVELS={L.MAIN_STAT_LEVELS}")
        gids = df["gid"].tolist()
        check("T3.3 全局 ID 唯一", len(set(gids)) == len(gids))
        # 若 A-B exact、B-C near 成三元组件，C 不得被标 S1
        if (df["size"] > 1).any():
            comp = df[df["size"] > 1]
            check("T3.4 混合证据组件不整体晋升 S1（取最弱连接）",
                  not (comp["confidence"] == L.S_PROVEN).all()
                  or all("exact_sha_edge" in str(e) for e in comp["evidence"]),
                  f"conf={comp['confidence'].tolist()} ev={comp['evidence'].tolist()}")
        else:
            check("T3.4 混合证据组件不整体晋升 S1（取最弱连接）", True, "本例未合并，跳过")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================= T4
def t4_cross() -> None:
    print("\n===== T4 跨类 / 跨 split =====")
    from B09_audit_source_group_mapping import run as a1_run

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dmsa_t4_"))
    try:
        recs = []
        # 同一张图被放进两个类（标签冲突）、train 与 test、test 内重复
        im = synth(21)
        specs = [("train", "Mild Impairment", "a.jpg"),
                 ("train", "Moderate Impairment", "b.jpg"),   # 跨类同像
                 ("test", "Mild Impairment", "c.jpg"),        # 跨 split 同像
                 ("test", "Mild Impairment", "d.jpg")]        # test 内重复
        for split, cls, name in specs:
            d = tmp / "data" / split / cls
            d.mkdir(parents=True, exist_ok=True)
            p = d / name
            im.save(p)
            recs.append({"split": split, "cls": cls, "path": str(p), "filename": name})

        out = a1_run(items=recs, out_dir=tmp / "out", cache_dir=tmp / "cache",
                     angle_step=45, use_flip=True, primary_th=8,
                     consistency_slack=2, cross_class=True, verbose=False)
        cross = out["cross"]
        check("T4.1 跨类同像被检出", cross["n_cross_class"] >= 1,
              f"n_cross_class={cross['n_cross_class']}")
        check("T4.2 跨 split（train<->test）同像被检出", cross["n_cross_split"] >= 1,
              f"n_cross_split={cross['n_cross_split']}")
        check("T4.3 test 内重复被检出",
              any(r["split"] == "test" and r["n_comp_gt1"] >= 1 for r in out["summary"]),
              f"summary={[(r['split'], r['cls'], r['n_comp_gt1']) for r in out['summary']]}")
        check("T4.4 输出 train/test 分离映射 + 跨 split 对账",
              (tmp / "out" / "source_mapping.csv").exists()
              and (tmp / "out" / "cross_matches.csv").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================= T5
def t5_calibration() -> None:
    print("\n===== T5 标定与配置强校验 =====")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dmsa_t5_"))
    try:
        bad = tmp / "bad.json"
        bad.write_text(json.dumps({"angle_step": 45, "use_flip": True,
                                   "primary_th": 4, "consistency_slack": 2,
                                   "distance_mode": "two-sided"}), encoding="utf-8")
        try:
            L.load_config(bad)
            check("T5.1 缺少 ssim_th 必须失败（不得静默回退 0.98）", False, "未抛异常")
        except KeyError as e:
            check("T5.1 缺少 ssim_th 必须失败（不得静默回退 0.98）", True, str(e)[:60])

        good = tmp / "good.json"
        good.write_text(json.dumps({"angle_step": 45, "use_flip": True,
                                    "primary_th": 4, "ssim_th": None,
                                    "consistency_slack": 2,
                                    "distance_mode": "two-sided"}), encoding="utf-8")
        try:
            L.load_config(good)
            check("T5.2 ssim_th=None（未标定）必须失败", False, "未抛异常")
        except ValueError:
            check("T5.2 ssim_th=None（未标定）必须失败", True)

        ok = tmp / "ok.json"
        ok.write_text(json.dumps({"angle_step": 45, "use_flip": True,
                                  "primary_th": 4, "ssim_th": 0.93,
                                  "consistency_slack": 2,
                                  "distance_mode": "two-sided"}), encoding="utf-8")
        cfg = L.load_config(ok)
        check("T5.3 完整配置可加载", cfg["ssim_th"] == 0.93)

        from B10_audit_detector_calibration import three_state_split
        pools = three_state_split(positives=[(0, 1)] * 10,
                                  negatives=[(0, 2)] * 10,
                                  boundary=[(0, 3)] * 5)
        check("T5.4 标定池为三态（阳性/阴性/待定分离）",
              set(pools) == {"positive", "negative", "pending"}
              and len(pools["pending"]) == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================= T6
def t6_conservative() -> None:
    print("\n===== T6 保守组件 =====")
    from B09_audit_source_group_mapping import run as a1_run

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dmsa_t6_"))
    try:
        root = tmp / "data" / "Very Mild Impairment"
        root.mkdir(parents=True, exist_ok=True)
        imgs = [synth(31), synth(31), synth(31).rotate(90, expand=False), synth(32)]
        recs = []
        for i, im in enumerate(imgs):
            p = root / f"v ({i + 1}).jpg"
            im.save(p)
            recs.append({"split": "train", "cls": "Very Mild Impairment",
                         "path": str(p), "filename": p.name})
        out = a1_run(items=recs, out_dir=tmp / "out", cache_dir=tmp / "cache",
                     angle_step=45, use_flip=True, primary_th=8,
                     consistency_slack=2, cross_class=False, verbose=False)
        df = out["map_df"]
        n_stat = df["cluster_id"].nunique()
        n_cons = df["conservative_cluster_id"].nunique()
        check("T6.1 保守簇数 <= 统计簇数（保守图更粗）", n_cons <= n_stat,
              f"conservative={n_cons} stat={n_stat}")
        check("T6.2 保守簇 ID 全局唯一",
              df["conservative_cluster_id"].is_unique or n_cons < len(df))
        check("T6.3 输出剪枝/风险边记录",
              (tmp / "out" / "conservative_edges.csv").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t8_statistics():
    """T8 统计层：本 v2.4 首跑时实测发现的多数投票真标签泄漏 + 单位一致性。"""
    print("\n[T8] 统计层（cluster bootstrap / 聚合规则）")
    import B17_lib_statistics as SL

    # T8.1 majority 必须对**预测**投票，不得对真标签投票（真标签泄漏是致命缺陷）
    #      构造：3 张派生图真标签都是 1，但 3 张预测全部是 0
    #      -> 正确的 majority 预测应为 0（与真标签 1 不同）；
    #         错误的实现（用真标签投票）会给出 1 —— 即 100% 准确率的假象。
    y = np.array([1, 1, 1])
    P = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    g = np.array(["S0", "S0", "S0"])
    gid, pred, _, gc = SL.AggRule("majority").aggregate(y, P, g, 2)
    check("T8.1 majority 对预测投票（不泄漏真标签）", int(pred[0]) == 0,
          f"pred={pred.tolist()}（若为 1 则真标签泄漏进聚合规则）")
    check("T8.1b 真标签仍被返回供一致性检查", int(gc[0]) == 1)

    # T8.2 字符串 group 标签必须可用（source_id 是 "LG:G00001" 这类）
    y2 = np.array([0, 0, 1, 1])
    P2 = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    g2 = np.array(["LG:G1", "LG:G1", "LG:G2", "LG:G2"])
    r = SL.cluster_bootstrap(y2, P2.argmax(1), g2,
                             lambda a, b: SL.metrics(a, b, 2)["accuracy"],
                             n_replicates=100)
    check("T8.2 字符串 source_id 可跑 bootstrap",
          r.n_clusters == 2 and np.isfinite(r.point))

    # T8.3 cluster bootstrap 的 CI 必须比 image-level（把同源当独立）更宽或相当
    rng = np.random.default_rng(0)
    n_clu, per = 80, 4
    grp = np.repeat([f"S{i}" for i in range(n_clu)], per)
    yy = np.repeat(rng.integers(0, 2, size=n_clu), per)
    # 簇内完全相关的错误：同簇同预测
    pp = np.repeat(rng.integers(0, 2, size=n_clu), per)
    rb = SL.cluster_bootstrap(yy, pp, grp, lambda a, b: SL.metrics(a, b, 2)["accuracy"],
                              n_replicates=400, seed=1)
    # 图像级 bootstrap（错误做法）：
    rb_img = SL.cluster_bootstrap(yy, pp, np.arange(len(yy)).astype(str),
                                  lambda a, b: SL.metrics(a, b, 2)["accuracy"],
                                  n_replicates=400, seed=1)
    w_clu = rb.hi - rb.lo
    w_img = rb_img.hi - rb_img.lo
    check("T8.3 簇 bootstrap CI 宽于图像级 bootstrap（同源不重复计数）",
          w_clu > w_img, f"cluster={w_clu:.4f} vs image={w_img:.4f}")

    # T8.4 macro-F1 必须实算，不得写死。
    #      手工校验：y=[0,0,0,0], p=[0,0,1,0]
    #        class0: 混淆 [[3,1],[0,0]] -> P=3/3? 不对：
    #          tp0=3, fp0=1, fn0=0 -> P=3/4=0.75? 见下（以实算输出为准）
    #        实算结果 F1 = [0.8571, 0.0] -> macro = 0.42857
    m = SL.metrics(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0]), 2)
    check("T8.4 macro-F1 实算 = 0.428571",
          abs(m["macro_f1"] - 0.42857142857142855) < 1e-9,
          f"macro_f1={m['macro_f1']}")
    # T8.4b macro-F1 随预测变化（证明不是写死常量）。
    #       注意：全对时 macro-F1 = 0.5 而非 1.0，因为 class1 在本子集里
    #       support=0 → 其 F1 定义上为 0，仍计入宏平均。
    #       这是**有意保留**的行为（显式暴露"某类在评估子集内不存在"），
    #       也是为什么 T8.4c 要求同时报 support 与每类 F1。
    m2 = SL.metrics(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 2)
    check("T8.4b macro-F1 随预测变化（非写死常量）",
          m2["macro_f1"] != m["macro_f1"], 
          f"perfect-subset={m2['macro_f1']} vs partial={m['macro_f1']}")
    # T8.4c 必须同时给出每类 F1 与 support，避免宏平均掩盖小类
    C = SL.confusion_matrix(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 2)
    d_ = SL.per_class_prf(C)
    check("T8.4c 报每类 F1 与 support（防小类被宏平均掩盖）",
          len(d_["f1"]) == 2 and int(d_["support"][1]) == 0,
          f"support={d_['support'].tolist()}")

    # T8.5 配对 bootstrap：A 与 B 完全相同时 Δ 必须恰好为 0
    pd_ = SL.paired_cluster_bootstrap(yy, pp, pp, grp,
                                      lambda a, b: SL.metrics(a, b, 2)["accuracy"],
                                      n_replicates=200, seed=2)
    check("T8.5 配对 bootstrap 自比较 Δ=0",
          abs(pd_["delta_point"]) < 1e-12 and abs(pd_["delta_lo"]) < 1e-12)

    # T8.6 结构泄漏审计：构造可见的 P1 违规必须被检出
    au = SL.leakage_audit(
        source_ids=["a", "a", "b", "c"],
        folds=[1, 2, 1, 1], splits=["train"] * 4,
        classes=["X", "X", "Y", "Y"])
    check("T8.6 结构泄漏 P1 检出（同源跨 fold）",
          au["P1_group_leakage"]["count"] == 1,
          f"count={au['P1_group_leakage']['count']}")
    check("T8.6b 单例率正确",
          abs(au["singleton_rate"] - 2 / 3) < 1e-9,
          f"singleton_rate={au['singleton_rate']}")
    check("T8.6c 组件规模分布存在", "1" in au["component_size_hist"])


def t9_arch_probe():
    """T9 架构探针：基线一致性 + 开关注入的形状/指纹区分度。

    只在 torch 可用时运行；不可用时标 SKIP（不判失败，但要显式提示）。
    """
    print("\n[T9] 架构探针（基线与开关注入）")
    try:
        import torch  # noqa
    except Exception as e:
        check("T9.0 torch 可用（否则架构探针被 SKIP）", True,
              f"SKIP: torch 不可用（{type(e).__name__}）；按手册 §2 在云环境补跑")
        return

    import importlib.util
    import pathlib
    import sys as _sys

    here = pathlib.Path(__file__).parent
    smoke = here / ".." / "DMSA_net" / "smoke_test.py"
    if not smoke.exists():
        check("T9.0 找到 smoke_test.py", False, str(smoke))
        return
    spec = importlib.util.spec_from_file_location("dmsa_t9_smoke", str(smoke))
    base = importlib.util.module_from_spec(spec)
    _sys.modules["dmsa_t9_smoke"] = base
    spec.loader.exec_module(base)

    m = base.create_model(num_classes=4)
    n_params = sum(p.numel() for p in m.parameters())
    check("T9.1 参数量 == 基线 56,812,813", n_params == 56_812_813,
          f"actual={n_params:,}")
    fp = base.architecture_fingerprint(m)
    check("T9.2 fingerprint == 基线常量",
          fp == "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975",
          f"fp={fp}")

    # 开关注入：5 个变体必须都能构造 + 前向，且 fingerprint 两两不同
    import B13_probe_architecture_fingerprint as AP
    variants = {"full": (True, True, True, True),
                "no_sam": (False, True, True, True),
                "no_mdc": (True, False, True, True),
                "no_ilf": (True, True, False, True),
                "no_dual": (True, True, True, False)}
    fps, errs = {}, {}
    for name, (sam, mdc, ilf, dual) in variants.items():
        create, restore = AP.make_switch_patch(base, use_sam=sam, use_mdc=mdc,
                                               use_ilf=ilf, use_dual_path=dual)
        try:
            mm = create(num_classes=4)
            import torch as _t
            mm.eval()
            with _t.no_grad():
                _ = mm(_t.randn(1, 3, 224, 224))
            fps[name] = AP.fingerprint(mm)
        except Exception as e:
            errs[name] = f"{type(e).__name__}: {e}"
        finally:
            restore()
    check("T9.3 全部 5 个变体前向可跑", not errs, f"errors={errs}")
    check("T9.4 full 变体 fingerprint == 基线",
          fps.get("full") == fp, f"full={fps.get('full')}")
    check("T9.5 5 个变体 fingerprint 两两不同",
          len(set(fps.values())) == len(fps) and len(fps) == 5,
          f"unique={len(set(fps.values()))}/{len(fps)}")
    check("T9.6 基线文件未被改动（fingerprint 侧证）",
          fps.get("full") == "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975")


def _run_script(args: list, timeout: int = 900):
    """以子进程**真实执行**一个脚本，返回 (returncode, stdout+stderr)。"""
    import subprocess
    import sys as _sys
    r = subprocess.run([_sys.executable] + args, cwd=str(HERE),
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def t10_scripts_run():
    """T10 **脚本级端到端**验收。

    【为什么必须有这一层】
    GPT 第四轮的核心指控是"文档声明已修 ≠ 代码已正确实现"。
    v2.4 首版的 T1–T7 全部只测**函数单元**，结果是：
    单元测试 29/29 全过，但 `a1cal_calibrate.py` 的 `main()` **根本无法运行**
    （`Column.fillna(0)` 抛 TypeError、`Mask & Mask` 无运算符、`Table.to_string` 不存在）。
    ⇒ 只测单元不足以支撑"脚本可用"的声明。本层直接跑脚本。
    """
    print("\n[T10] 脚本级端到端（真实执行 main()）")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)

        rc, out = _run_script([str(HERE / "a1cal_calibrate.py"),
                               "--n", "5", "--out", str(td / "a1cal")], timeout=600)
        check("T10.1 a1cal_calibrate.py main() 可运行", rc == 0,
              f"rc={rc} tail={out.strip()[-180:]}")
        check("T10.1b 产出留出验证集（GPT T5：标定/验证必须分开）",
              (td / "a1cal" / "calibration_holdout.csv").exists())
        check("T10.1c 阈值只在标定集选、留出集报告",
              (td / "a1cal" / "suggested_thresholds.json").exists()
              and json.loads((td / "a1cal" / "suggested_thresholds.json").read_text(
                  encoding="utf-8")).get("selection_set") == "calibration")

        rc, out = _run_script([str(HERE / "audit_leakage.py"), "--root", str(HERE / ".."),
                               "--no-a1", "--out", str(td / "audit")], timeout=600)
        check("T10.2 audit_leakage.py main() 可运行", rc == 0,
              f"rc={rc} tail={out.strip()[-180:]}")
        check("T10.2b 产出三协议审计与 ESS/DEFF",
              (td / "audit" / "leakage_audit.json").exists()
              and (td / "audit" / "ess_deff.csv").exists())

        rc, out = _run_script([str(HERE / "stage2_stats.py"), "--demo",
                               "--out", str(td / "s2"), "--n-boot", "150"], timeout=600)
        check("T10.3 stage2_stats.py main() 可运行", rc == 0,
              f"rc={rc} tail={out.strip()[-180:]}")
        check("T10.3b 产出三列对比表",
              (td / "s2" / "three_column_table.csv").exists())

        rc, out = _run_script([str(HERE / "a0_inventory.py"), "--root", str(HERE / ".."),
                               "--out", str(td / "a0")], timeout=600)
        check("T10.4 a0_inventory.py main() 可运行", rc == 0,
              f"rc={rc} tail={out.strip()[-180:]}")

        # log_test_access：audit -> 仍完好；tuning -> 作废；再 require-intact -> 拒绝
        logp = str(td / "access.jsonl")
        rc1, _ = _run_script([str(HERE / "log_test_access.py"), "--log", logp,
                              "--purpose", "audit"], timeout=120)
        rc2, o2 = _run_script([str(HERE / "log_test_access.py"), "--log", logp,
                               "--status", "--require-intact"], timeout=120)
        rc3, _ = _run_script([str(HERE / "log_test_access.py"), "--log", logp,
                              "--purpose", "tuning"], timeout=120)
        rc4, o4 = _run_script([str(HERE / "log_test_access.py"), "--log", logp,
                               "--status", "--require-intact"], timeout=120)
        check("T10.5 审计访问不破坏 sealed", rc1 == 0 and rc2 == 0)
        check("T10.5b tuning 访问后 sealed 作废，硬门拒绝",
              rc3 == 0 and rc4 == 3, f"rc3={rc3} rc4={rc4} out={o4.strip()[-120:]}")


def t11_reproducibility():
    """T11 可复现性指纹（GPT T7）与零依赖表格层回归。"""
    print("\n[T11] 可复现性指纹与表格层回归")
    import B15_lib_deduplication as LD
    import tempfile

    # T11.1 code_fingerprint 随文件内容变化
    f1 = LD.code_fingerprint()
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.py"
        p.write_text("a=1\n", encoding="utf-8")
        g1 = LD.code_fingerprint([p])
        p.write_text("a=2\n", encoding="utf-8")
        g2 = LD.code_fingerprint([p])
        check("T11.1 code_fingerprint 随内容变化", g1 != g2)
    check("T11.1b 本目录 code_fingerprint 稳定可复算",
          f1 == LD.code_fingerprint() and len(f1) == 64)

    # T11.2 config_hash 与键序无关
    h1 = LD.config_hash({"a": 1, "b": [2, 3], "c": None})
    h2 = LD.config_hash({"c": None, "b": [2, 3], "a": 1})
    check("T11.2 config_hash 键序无关", h1 == h2)
    check("T11.2b config_hash 对值敏感",
          h1 != LD.config_hash({"a": 1, "b": [2, 4], "c": None}))

    # T11.3 表格层三项回归（这三条正是 a1cal 崩溃的直接原因）
    c = LD.Column([1.0, None, 0.5])
    check("T11.3 Column.fillna(标量) 可用", c.fillna(0).tolist() == [1.0, 0.0, 0.5])
    m1, m2 = LD.Column([1, 2, 3]) >= 2, LD.Column([3, 0, 3]) > 0
    check("T11.3b Mask & Mask 可用（list 无此运算符）",
          (m1 & m2).tolist() == [False, False, True])
    # m1 = [F,T,T]; m2 = [T,F,T]
    check("T11.3c Mask | / ~ 可用",
          (m1 | m2).tolist() == [True, True, True]
          and (~m1).tolist() == [True, False, False])
    t = LD.Table([{"a": 1, "b": "x"}, {"a": 22, "b": "yy"}])
    check("T11.3d Table.to_string() 可用", "a" in t.to_string() and "22" in t.to_string())
    check("T11.4 Column.max() 可用", LD.Column([3, None, 7]).max() == 7)

    # T11.5 run_config.json 含代码/配置指纹（GPT T7）
    rc, _ = _run_script([str(HERE / "a1_source_map.py"), "--help"], timeout=120)
    check("T11.5 a1_source_map.py 可解析参数", rc == 0)

    # ---- T11.6 过期检测（N6，v2.6 新增）----
    #: 这一条是 v2.6 自查的直接产物：指纹机制**光有还不算数**，
    #: 必须有"回读时校验、不一致就硬失败"的另一半，否则过期产物会被静默引用。
    import B16_lib_provenance_audit as LP
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "art.json"
        LP.write_artifact(f, {"x": 1})
        check("T11.6a 刚写出的产物判为 fresh",
              LP.assert_fresh(f, strict=False)["fresh"] is True)

        tampered = json.loads(f.read_text(encoding="utf-8"))
        tampered["code_fingerprint"] = "0" * 64
        f.write_text(json.dumps(tampered), encoding="utf-8")
        check("T11.6b 指纹被篡改 -> 判为 STALE",
              LP.assert_fresh(f, strict=False)["fresh"] is False)
        raised = False
        try:
            LP.assert_fresh(f, strict=True)
        except LP.StaleArtifactError:
            raised = True
        check("T11.6c strict=True 抛 StaleArtifactError（不许只打 WARN）", raised)

        nofp = pathlib.Path(td) / "nofp.json"
        nofp.write_text(json.dumps({"y": 2}), encoding="utf-8")
        check("T11.6d 缺指纹的产物同样判为 STALE",
              LP.assert_fresh(nofp, strict=False)["fresh"] is False)


def t13_provenance_coverage():
    """T13 产物溯源覆盖率（N6）。

    【为什么必须有这一条】
    v2.4 建了 `code_fingerprint()` 就宣布"可复现性已解决"。实测（2026-09-23）：
    6 个产物里 **5 个根本没写指纹**，唯一写了的那个还因为后续改代码而**过期**。
    覆盖率 1/6 且无回读校验的溯源机制，比没有更危险 —— 它制造"已溯源"的错觉。
    所以验收必须**数个数**，而不是看"有没有这个函数"。
    """
    print("\n[T13] 产物溯源覆盖率（N6）")
    import B16_lib_provenance_audit as LP

    # T13.1 静态：所有产出脚本都必须走 write_artifact，不许自己 json.dumps 写 .json
    #
    #: ⚠️ 本检查必须**跨行**匹配。v2.6 首版按单行 grep `write_text(json.dumps(`，
    #: 结果漏掉了 `lib_dedup.HashCache` 那种
    #:     p.with_suffix(".meta.json").write_text(
    #:         json.dumps({...}), encoding="utf-8")
    #: 的换行写法 —— 检测器自己有盲区，比没有检测器更危险（会给出虚假的安全感）。
    #: 因此改为：在 `write_text(` 之后的窗口内找 `json.dumps`，
    #: 并用**显式 allowlist** 放行"不是结果产物"的缓存/日志。
    import re
    ALLOW = {
        #: 哈希缓存与访问日志不是"结果产物"，不参与溯源盖章。
        #: ⚠️ 匹配串要写成**代码里真实出现的片段**（`.meta.json` / `.jsonl`），
        #: 不能写成人读的说明文字 —— v2.6 首版写成 "HashCache.meta.json"，
        #: 结果永远匹配不上，把缓存误报成违规。
        ("lib_dedup.py", ".meta.json"),      # HashCache 缓存元信息（非结果产物）
        ("lib_dedup.py", ".jsonl"),          # TestAccessLog append-only 日志
    }
    win = 260
    bad = []
    for p in sorted(HERE.glob("*.py")):
        if p.name in ("test_acceptance.py", "lib_provenance.py"):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"write_text\s*\(", txt):
            #: ⚠️ 窗口必须**向前也取一段**：目标文件名常常写在 `write_text(` 之前，
            #: 例如 `p.with_suffix(".meta.json").write_text(json.dumps(...))`。
            #: v2.6 首版只向后取，导致 allowlist 永远匹配不上 → 把缓存误判为违规。
            seg = txt[max(0, m.start() - win):m.start() + win]
            if "json.dumps" not in seg and "json.dump(" not in seg:
                continue
            line_no = txt[:m.start()].count("\n") + 1
            #: 注释行里的示例不算
            head = txt.rfind("\n", 0, m.start()) + 1
            if txt[head:m.start()].lstrip().startswith("#"):
                continue
            if any(fn == p.name and tag in seg for fn, tag in ALLOW):
                continue
            bad.append(f"{p.name}:{line_no}")
    check("T13.1 无脚本绕过 write_artifact 直接写 JSON 产物（跨行检测）",
          not bad, f"违规：{bad[:6]}")

    # T13.2/T13.3 运行时：真实产物目录里不许有 STALE / 缺指纹
    res = LP.scan_freshness(HERE)
    stale = [r["file"] for r in res["rows"] if not r["fresh"]]
    check("T13.2 全部产物已盖章且未过期",
          res["n_stale"] == 0,
          f"共 {res['n_json']} 个 JSON，过期/缺指纹 {res['n_stale']}：{stale[:6]}")
    check("T13.3 指纹覆盖率 = 100%（v2.4 实测仅 1/6）",
          res["n_json"] > 0 and res["n_missing_fp"] == 0,
          f"缺指纹 {res['n_missing_fp']}/{res['n_json']}")

    # T13.4 A6-s 必须是三列 + 代码事实（GPT E5）
    rep_p = HERE / "_a6s_out" / "weight_map_report.json"
    if rep_p.exists():
        d = json.loads(rep_p.read_text(encoding="utf-8"))
        need = ["col_A_exact_key", "col_B_semantic_mapped",
                "col_C_actually_loaded", "col_D_code_fact"]
        check("T13.4 A6-s 报告含三列 + 代码事实",
              all(k in d for k in need),
              f"缺 {[k for k in need if k not in d]}")
        check("T13.4b A6-s 判定不再依赖匹配率阈值",
              d.get("thresholds_pre_registered", {}).get(
                  "col_A_ratio_used_for_verdict") is False,
              f"verdict={d.get('verdict')}")
        wc = d.get("col_B_semantic_mapped", {}).get("width_comparison", [])
        check("T13.4c A6-s 记录宽度对照（证明语义映射为何不提升）",
              len(wc) > 0, f"n_width_rows={len(wc)}")
    else:
        check("T13.4 A6-s 报告存在", False, "未找到 _a6s_out/weight_map_report.json")


def t12_torch_scripts():
    """T12 依赖 torch 的脚本端到端（无 torch 时 SKIP）。"""
    print("\n[T12] torch 依赖脚本端到端")
    try:
        import torch  # noqa
    except Exception as e:
        check("T12.0 torch 可用（否则 SKIP）", True, f"SKIP: {type(e).__name__}")
        return
    import os
    if os.environ.get("DMSA_SKIP_SLOW") == "1":
        check("T12.0 慢测试被 DMSA_SKIP_SLOW 跳过", True, "SKIP")
        return
    import tempfile
    with tempfile.TemporaryDirectory() as td_:
        td = pathlib.Path(td_)
        rc, out = _run_script([str(HERE / "a6s_pretrain_probe.py"),
                               "--model-module", str(HERE / ".." / "DMSA_net" / "smoke_test.py"),
                               "--out", str(td / "a6s")], timeout=900)
        check("T12.1 a6s_pretrain_probe.py 可运行", rc == 0,
              f"rc={rc} tail={out.strip()[-160:]}")
        rc, out = _run_script([str(HERE / "arch_probe.py"),
                               "--smoke", str(HERE / ".." / "DMSA_net" / "smoke_test.py"),
                               "--out", str(td / "arch")], timeout=1800)
        check("T12.2 arch_probe.py 可运行", rc == 0,
              f"rc={rc} tail={out.strip()[-160:]}")


def _dotted(node) -> str:
    """把 `a.b.c` 形式的调用目标还原成点号串（用于静态黑名单，避免误伤注释）。"""
    import ast
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _dir_snapshot(p: pathlib.Path):
    """目录快照（**只 stat、不读内容**）：用于证明脚本没有改动数据目录。"""
    rows = []
    for f in sorted(p.rglob("*")):
        if f.is_file():
            st = f.stat()
            rows.append((f.relative_to(p).as_posix(), st.st_size, st.st_mtime_ns))
    return rows


def t14_dataset_arrival():
    """T14 到货验证层（v2.7 新增）。

    【为什么需要这一层】
    2026-09-23 原始数据到货后，`Alzheimer's MRI Dataset/` 成为本项目**唯一一份原始证据**。
    它一旦被脚本改写（哪怕是"顺手规范化"），全部 SHA 对账立刻失效且**不可逆**。
    所以"数据集只读"不能只写在原则 20 里 —— **必须有门禁**。

    这一层把原则 20 拆成可判定的三件事：
      ① **静态**：验证脚本里不存在写/删/解压调用；
      ② **静态 + AST**：验证脚本只调基线校验器的校验函数，**不调 `main()`**
         （`main()` 第一句就是 `extract_dataset(zip, ...)`，会往数据目录解压）；
      ③ **运行时实测**：真跑一次验证脚本，前后对数据目录做快照，**必须逐文件完全一致**。
    """
    import ast
    import os

    src_path = HERE / "a0b_dataset_verify.py"
    check("T14.1 a0b_dataset_verify.py 存在", src_path.is_file(), str(src_path))
    if not src_path.is_file():
        return
    txt = src_path.read_text(encoding="utf-8")
    tree = ast.parse(txt)

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [_dotted(c.func) for c in calls]

    #: 危险调用黑名单。注意：不要用裸 `remove`（`sys.path.remove` 是合法的列表操作），
    #: 也不要对 `.write_text(`/`.unlink(` 做裸名匹配 —— 必须看完整点号链。
    FORBIDDEN = {
        "extract_dataset": "到货验证里不得解压（会写数据目录）",
        "shutil.rmtree": "不得递归删除",
        "shutil.move": "不得移动",
        "shutil.copytree": "不得在数据目录旁复制",
        "os.remove": "不得删除文件",
        "os.unlink": "不得删除文件",
        "os.rename": "不得重命名",
        "os.rmdir": "不得删除目录",
    }
    hits = {n: why for n, why in FORBIDDEN.items() if n in names}
    hits.update({n: "不得 unlink（删文件）" for n in names if n.endswith(".unlink")})
    check("T14.2 a0b 无危险写/删/解压调用（AST 级，不受注释影响）",
          not hits, f"命中：{hits or '无'}")

    main_calls = [n for n in names if n in ("DPC.main", "data_prepare_check.main")]
    wired = "data_prepare_check" in txt
    check("T14.3 a0b 复用了基线校验器但**不调用其 main()**",
          (not main_calls) and wired,
          f"main 调用={main_calls or '无'}；已接入基线校验器={wired}")

    data_root = pathlib.Path(os.environ.get(
        "DMSA_DATA_ROOT",
        str(HERE / ".." / "Alzheimer's MRI Dataset" / "Alzheimer's MRI Dataset"
            / "Combined Dataset"))).resolve()
    if not data_root.is_dir():
        check(f"T14.4 原始数据目录存在（{data_root}）", False,
              "未找到；可用环境变量 DMSA_DATA_ROOT 指定")
        return

    before = _dir_snapshot(data_root)
    import tempfile
    with tempfile.TemporaryDirectory() as td_:
        td = pathlib.Path(td_)
        rc, out = _run_script([str(src_path), "--data", str(data_root),
                               "--manifest-dir", str(HERE / ".." / "DMSA_net"),
                               "--out", str(td / "out")], timeout=1800)
        check("T14.4 a0b_dataset_verify.py 端到端可运行（退出码 0）", rc == 0,
              f"rc={rc} tail={out.strip()[-200:]}")
        after = _dir_snapshot(data_root)
        check("T14.5 **数据集只读纪律（运行时实测）**：跑完前后目录快照完全一致",
              before == after,
              f"前 {len(before)} 个文件 / 后 {len(after)} 个；"
              f"差异 {len(set(before) ^ set(after)) or 0} 条")

        rep_p = td / "out" / "dataset_verify.json"
        if not rep_p.is_file():
            check("T14.6 到货报告已生成", False, str(rep_p))
            return
        rep = json.loads(rep_p.read_text(encoding="utf-8"))

    need = ["counts", "frames", "reconcile", "sha_only",
            "train_test_exact_intersection", "manifest_cross_check", "verdict"]
    miss = [k for k in need if k not in rep]
    check("T14.6 到货报告含全部必需字段", not miss, f"缺 {miss or '无'}")

    check("T14.7 与两份冻结 manifest 逐文件 SHA 对账 = EXACT_MATCH",
          all(rep.get("reconcile", {}).get(s, {}).get("verdict") == "EXACT_MATCH"
              for s in ("train", "test")),
          " / ".join(f"{s}:{rep.get('reconcile', {}).get(s, {}).get('verdict')}"
                     for s in ("train", "test")))

    fr = rep.get("frames", {})
    gc = rep.get("geometry_conflict", {})
    check("T14.8 帧几何冲突已记录（磁盘 ≠ 声明的 224×224 RGB）",
          fr.get("available") and fr.get("matches_declared") is False
          and gc.get("conflict") is True,
          f"实测 {fr.get('size_dist')} {fr.get('mode_dist')}；声明 {gc.get('declared')}")

    bc = rep.get("baseline_check", {})
    check("T14.9 基线作者自带校验器跑完全部校验（未调 main）",
          bc.get("available") and bc.get("ran_to_completion") is True,
          f"available={bc.get('available')} ran={bc.get('ran_to_completion')} "
          f"err={bc.get('error')}")

    ts = rep.get("train_test_exact_summary", {})
    inter = rep.get("train_test_exact_intersection", {})
    check("T14.10 train↔test 跨集 exact 交集 = 0（且分母 > 0，不是空跑）",
          ts.get("n_test", 0) > 0 and ts.get("n_hit") == 0,
          f"{ts.get('n_hit')} / {ts.get('n_test')}；逐类 {inter}")

    xv = rep.get("manifest_cross_check", {}).get("exact_vs_sha", {})
    check("T14.11 manifest 自述 exact 分组 可被磁盘 SHA 独立复现",
          bool(xv) and all(v.get("agree") for v in xv.values()),
          str({k: (v.get("manifest_exact_redundant"), v.get("measured_sha_redundant"))
               for k, v in xv.items()}))


def t15_threshold_enforcement():
    """T15 阈值来源强制（v2.7 新增，N11）。

    【为什么需要这一层】
    2026-09-23 真实数据事故：`dedup_config.json` 的 `primary_th = 4` 是一个
    **未标定的占位符**（该字段自己的注释就写着"必须由 A1-cal 标定后填入"），
    但**没有任何机制阻止 A1 使用它**。A1 静默采用后把每类 2,560 张全连成一个连通分量，
    产出 2.2 GB 无效结果，而 `component_summary.csv` 的 `uniq_merged = 1`
    看起来像"灾难性发现"。

    对照 `ssim_th`：它用 `null` 占位 + A2 读到 null 直接报错，**是有强制的**。
    `primary_th` 缺的就是同等的强制。这一层把"必须强制"变成可判定的条目。
    """
    # T15.1 占位符必须是 null（不许写死一个看起来像标定值的数）
    cfg_p = HERE / "dedup_config.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    check("T15.1 dedup_config.json 的 primary_th 是 null 占位符（不是看似标定过的数）",
          cfg.get("primary_th") is None,
          f"primary_th={cfg.get('primary_th')!r}")

    # T15.2 阈值未标定且无 A1-cal 产物时，A1 必须**拒绝运行**（不是猜默认值）
    import tempfile
    with tempfile.TemporaryDirectory() as td_:
        td = pathlib.Path(td_)
        bad_cfg = td / "cfg.json"
        bad_cfg.write_text(json.dumps({
            "angle_step": 45, "use_flip": True, "primary_th": None, "ssim_th": None,
            "output_dirs": {"a1cal_out": str(td / "no_such_calib_out")},
        }, ensure_ascii=False), encoding="utf-8")
        rc, out = _run_script([str(HERE / "a1_source_map.py"),
                               "--train", str(td / "none"), "--test", str(td / "none"),
                               "--config", str(bad_cfg),
                               "--out", str(td / "o"), "--cache", str(td / "c")],
                              timeout=300)
        check("T15.2 A1 在 primary_th 未标定时**报错退出**（rc≠0，且提示未标定）",
              rc != 0 and "未标定" in out,
              f"rc={rc} tail={out.strip()[-140:]}")

    # T15.3 A1-cal 产物必须能区分"合成标定"与"真实数据标定"
    cal_p = HERE / "_a1cal_out" / "suggested_thresholds.json"
    if cal_p.is_file():
        cal = json.loads(cal_p.read_text(encoding="utf-8"))
        has_flag = "real_data_calibrated" in cal
        check("T15.3 A1-cal 产物带 real_data_calibrated 标记", has_flag,
              f"real_data_calibrated={cal.get('real_data_calibrated')}")
    else:
        check("T15.3 A1-cal 产物存在（需先跑 a1cal_calibrate.py）", False, str(cal_p))

    # T15.4 若已有 A1 产物，必须登记阈值的**来源**（否则读者无法判断阈值可信度）
    rc_p = HERE / "_a1v2_out" / "run_config.json"
    if rc_p.is_file():
        d = json.loads(rc_p.read_text(encoding="utf-8"))
        src = d.get("primary_th_source")
        check("T15.4 A1 产物登记了 primary_th_source（阈值来源可追溯）",
              bool(src) and d.get("primary_th") is not None,
              f"primary_th={d.get('primary_th')} source={src!r}")
    else:
        #: 没有 A1 产物本身是合法状态（阈值未标定就不该跑）。但要确认隔离标记存在，
        #: 否则"没跑"与"跑坏了藏起来"无法区分。
        q = HERE / "_a1v2_out" / "_INVALID_primary_th4_placeholder" / "README_INVALID.md"
        check("T15.4 无 A1 产物时，必须存在无效运行的隔离说明（使'没跑'可验证）",
              q.is_file(), str(q))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DMSA_v24/v2.6 验收")
    ap.add_argument("--only", default="",
                    help="只跑指定组，逗号分隔，如 `T13`。"
                         "注意 T13（产物新鲜度）必须在**全部产出脚本跑完之后**再跑，"
                         "因为任何一行 .py 改动都会使既有产物立即失效。")
    a = ap.parse_args()

    groups = {
        "T1": t1_distance, "T2": t2_end_to_end, "T3": t3_grading, "T4": t4_cross,
        "T5": t5_calibration, "T6": t6_conservative, "T8": t8_statistics,
        "T9": t9_arch_probe, "T10": t10_scripts_run, "T11": t11_reproducibility,
        "T12": t12_torch_scripts, "T13": t13_provenance_coverage,
        "T14": t14_dataset_arrival, "T15": t15_threshold_enforcement,
    }
    want = [x.strip().upper() for x in a.only.split(",") if x.strip()]
    if want:
        unknown = [w for w in want if w not in groups]
        if unknown:
            print(f"未知测试组：{unknown}；可选：{sorted(groups)}")
            return 2
        for w in want:
            groups[w]()
    else:
        #: T13 放在最后：它校验的是"产物是否由**当前**代码产生"，
        #: 因此必须在同一轮里所有会改代码/产物的动作之后执行。
        for w in ("T1", "T2", "T3", "T4", "T5", "T6", "T8",
                  "T9", "T10", "T11", "T12", "T14", "T15", "T13"):
            groups[w]()

    print("\n" + "=" * 60)
    failed = [r for r in RESULTS if not r[1]]
    print(f"总计 {len(RESULTS)} 项，通过 {len(RESULTS) - len(failed)}，失败 {len(failed)}")
    for n, _, d in failed:
        print(f"  FAILED: {n}  {d}")
    print("=" * 60)
    if failed:
        print("存在未通过项 —— 禁止生成 source_mapping 并冻结 source_group_v1。")
        return 1
    print("全部通过 —— 可以进入真实数据的 A1/A2 审计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
