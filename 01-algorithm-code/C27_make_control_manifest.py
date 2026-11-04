#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C0 · 生成「忽略 group_id」的图像级随机划分清单（对照五折专用，**事前冻结**）

为什么需要它
------------
P-CV v2 的分组五折（`pcv_v2_folds_seed42.csv`）保证同一 `group_id` 不跨折，
从而封锁了「同源派生副本同时出现在 train 与 val」这类泄漏。
但「封掉了多少」一直是**定性**的。要把它变成**定量**，唯一办法是造一份
**只看图像、不看 group** 的随机划分，让泄漏按自然概率重新出现，
再在**同一套代码、同一台 GPU、同一超参**下比精度：

    Δ = acc(图像级随机划分) − acc(P-CV v2 分组划分)

Δ > 0 的那部分，就是「分组协议压掉的乐观偏差」。

做法（为什么这样设计）
----------------------
1. **类别分层**：每类每折恰好 512 张 val（与基线协议同构：2048 val / 8192 train）。
   这样两臂之间**唯一的差别只有"看不看 group_id"**，不会多引入"类别失衡"这个混淆。
2. 每类内部**先按 relative_path 排序再洗牌**（保证结果不受文件行序影响），
   用固定 seed 洗牌后按 512 一块切 5 折。
3. **除 `val_fold` 外所有列逐字节沿用基线 manifest**，schema 完全一致 ——
   因此基线的 `read_fold_manifest()` 一个字符都不用改，也不存在"列对不上"的风险。
4. 生成后**立即**做 6 项自检（见 `_self_check`），任何一项不过就 exit 2。

⚠️ 纪律
-------
* 本清单**必须在跑实验前冻结并记录 SHA-256**（同时写出 `<out>.sha256`）。
  跑实验时由 `c1_cv_matrix.py` 用**硬编码常量**回校，SHA 不符直接拒绝开跑。
* 本脚本**只读**基线 manifest；不碰数据集、不碰 sealed test、不需要 GPU。
* 只新增一个文件，不改任何基线文件。

用法
----
    python c0_make_control_manifest.py                       # 默认就地生成 + 冻结
    python c0_make_control_manifest.py --only-verify         # 只复验已冻结的那份
    python c0_make_control_manifest.py --out <路径> --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# 冻结事实（与 gpu_lib.py 同源；此处自带一份，保证本脚本零依赖可跑）
# ---------------------------------------------------------------------------
FOLDS = 5
EXPECTED_ROWS = 10240
PER_CLASS_TOTAL = 2560
PER_CLASS_VAL = 512
PER_CLASS_TRAIN = 2048
VAL_PER_FOLD = 2048
TRAIN_PER_FOLD = 8192
EXPECTED_CLASSES = (
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
)

BASELINE_MANIFEST_NAME = "pcv_v2_folds_seed42.csv"
BASELINE_MANIFEST_SHA256 = (
    "51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd"
)
DEFAULT_OUT_NAME = "pcv_v2_folds_imgsplit_seed42_FROZEN.csv"

# 这些列必须逐字节沿用基线（唯一允许改的是 val_fold）
FROZEN_COLUMNS = (
    "relative_path",
    "class_name",
    "filename",
    "sha256",
    "group_id",
    "group_size",
    "group_reason",
    "v1_val_fold",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_manifest(path: Path) -> Tuple[List[str], List[dict]]:
    """读 manifest，返回 (表头顺序, 行)。utf-8-sig：吃掉可能的 BOM。"""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    if "val_fold" not in fields:
        raise ValueError(f"manifest 缺少 val_fold 列：{path}")
    return fields, rows


def leak_descriptors(rows: Sequence[dict]) -> dict:
    """量化「同源泄漏」的三个描述量。

    - group_cross_fold：有多少个 group_id 跨了 ≥2 折（基线应为 0，图像级应 >0）
    - sha_cross_fold  ：有多少个 sha256  跨了 ≥2 折
    - val_tainted_per_fold：每折里，有多少张 val 图，其 group 同样出现在该折 train 中
    """
    group_folds: Dict[str, set] = defaultdict(set)
    sha_folds: Dict[str, set] = defaultdict(set)
    for r in rows:
        fold = int(r["val_fold"])
        group_folds[r["group_id"]].add(fold)
        sha_folds[r["sha256"]].add(fold)

    tainted: Dict[str, int] = {}
    for fold in range(1, FOLDS + 1):
        val_rows = [r for r in rows if int(r["val_fold"]) == fold]
        train_groups = {r["group_id"] for r in rows
                        if int(r["val_fold"]) != fold}
        tainted[str(fold)] = sum(1 for r in val_rows if r["group_id"] in train_groups)

    return {
        "group_cross_fold": sum(1 for v in group_folds.values() if len(v) > 1),
        "sha_cross_fold": sum(1 for v in sha_folds.values() if len(v) > 1),
        "n_groups": len(group_folds),
        "val_tainted_per_fold": tainted,
        "val_tainted_total": sum(tainted.values()),
    }


def build_control_rows(rows: Sequence[dict], seed: int) -> List[dict]:
    """类别分层 + 每类内部定序洗牌 → 切成 5 折；完全不看 group_id。"""
    out: List[dict] = []
    for cls in EXPECTED_CLASSES:
        members = sorted((r for r in rows if r["class_name"] == cls),
                         key=lambda r: r["relative_path"])
        if len(members) != PER_CLASS_TOTAL:
            raise ValueError(
                f"类别 {cls} 共 {len(members)} 行 != {PER_CLASS_TOTAL}"
            )
        # 固定 seed 洗牌；输入已排序 ⇒ 与文件行序无关，可复现
        random.Random(seed).shuffle(members)
        for i, r in enumerate(members):
            fold = i // PER_CLASS_VAL + 1
            if not 1 <= fold <= FOLDS:
                raise AssertionError(f"折号越界：{fold}")
            nr = dict(r)
            nr["val_fold"] = str(fold)
            out.append(nr)
    # 输出顺序 = 按类分组；再按原表顺序重排，便于肉眼 diff
    order = {r["relative_path"]: i for i, r in enumerate(rows)}
    out.sort(key=lambda r: order[r["relative_path"]])
    return out


def _self_check(base_rows: Sequence[dict], out_rows: Sequence[dict]) -> dict:
    """6 项自检；任何一项不过 → 抛异常（调用方 exit 2）。"""
    report: Dict[str, object] = {}

    # 1. 行数与类别总数
    if len(out_rows) != EXPECTED_ROWS:
        raise ValueError(f"[自检1] 行数 {len(out_rows)} != {EXPECTED_ROWS}")
    pc = Counter(r["class_name"] for r in out_rows)
    for c in EXPECTED_CLASSES:
        if pc[c] != PER_CLASS_TOTAL:
            raise ValueError(f"[自检1] 类别 {c} 共 {pc[c]} != {PER_CLASS_TOTAL}")
    report["rows"] = len(out_rows)
    report["per_class_total"] = dict(pc)

    # 2. 每折规模 + 每类每折构成
    for fold in range(1, FOLDS + 1):
        val = [r for r in out_rows if int(r["val_fold"]) == fold]
        trn = [r for r in out_rows if int(r["val_fold"]) != fold]
        if len(val) != VAL_PER_FOLD or len(trn) != TRAIN_PER_FOLD:
            raise ValueError(
                f"[自检2] fold {fold}: val={len(val)} train={len(trn)}"
                f"（应为 {VAL_PER_FOLD}/{TRAIN_PER_FOLD}）"
            )
        vc = Counter(r["class_name"] for r in val)
        tc = Counter(r["class_name"] for r in trn)
        for c in EXPECTED_CLASSES:
            if vc[c] != PER_CLASS_VAL or tc[c] != PER_CLASS_TRAIN:
                raise ValueError(
                    f"[自检2] fold {fold} 类 {c}: val={vc[c]} train={tc[c]}"
                    f"（应为 {PER_CLASS_VAL}/{PER_CLASS_TRAIN}）"
                )
    report["per_fold_sizes"] = "2048 val / 8192 train（每类 512 / 2048）"

    # 3. relative_path 唯一
    rel = [r["relative_path"] for r in out_rows]
    if len(set(rel)) != len(rel):
        raise ValueError("[自检3] relative_path 有重复")

    # 4. ★关键：确实"忽略"了 group_id（否则这份对照没有意义）
    d_out = leak_descriptors(out_rows)
    if d_out["group_cross_fold"] <= 0:
        raise ValueError(
            "[自检4] 图像级划分居然没有任何 group 跨折 —— 说明根本没忽略 group_id，"
            "这份对照无效。"
        )
    report["control_leak"] = d_out

    # 5. 除 val_fold 外，所有列逐行沿用基线
    base_by_rel = {r["relative_path"]: r for r in base_rows}
    n_diff = 0
    for r in out_rows:
        b = base_by_rel.get(r["relative_path"])
        if b is None:
            raise ValueError(f"[自检5] 输出里出现了基线没有的路径：{r['relative_path']}")
        for k in FROZEN_COLUMNS:
            if b[k] != r[k]:
                n_diff += 1
                if n_diff <= 3:
                    print(f"    ✗ 列 {k} 不一致：{r['relative_path']}")
    if n_diff:
        raise ValueError(f"[自检5] 除 val_fold 外有 {n_diff} 处列值被改动，拒绝冻结")
    if len(base_by_rel) != len(out_rows):
        raise ValueError("[自检5] 输出行集与基线行集不一致")

    # 6. 与基线泄漏描述对照
    report["baseline_leak"] = leak_descriptors(base_rows)
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="生成并冻结「忽略 group_id」的图像级划分清单")
    ap.add_argument("--manifest", default=str(here / BASELINE_MANIFEST_NAME),
                    help=f"基线冻结 manifest（默认 {BASELINE_MANIFEST_NAME}）")
    ap.add_argument("--out", default=str(here / DEFAULT_OUT_NAME),
                    help=f"输出 CSV 路径（默认 {DEFAULT_OUT_NAME}）")
    ap.add_argument("--seed", type=int, default=42,
                    help="洗牌 seed（默认 42；清单一旦冻结就不要再改）")
    ap.add_argument("--only-verify", action="store_true",
                    help="只复验已存在的输出清单，不重新生成")
    a = ap.parse_args()

    man = Path(a.manifest).expanduser()
    out = Path(a.out).expanduser()

    print("=" * 78)
    print("C0 · 图像级随机划分清单（对照五折用，事前冻结）")
    print("=" * 78)
    print(f"基线 manifest : {man}")

    if not man.is_file():
        print(f"❌ 找不到基线 manifest：{man}")
        return 2
    man_sha = sha256_file(man)
    print(f"基线 SHA-256  : {man_sha}")
    if man_sha != BASELINE_MANIFEST_SHA256:
        print("❌ 基线 manifest SHA-256 与冻结常量不符 —— 拒绝基于它生成对照清单。")
        print(f"   期望：{BASELINE_MANIFEST_SHA256}")
        return 2
    print("基线 SHA 校验 : PASS（与冻结常量一致）")

    fields, base_rows = read_manifest(man)
    print(f"基线行数      : {len(base_rows):,}  列={fields}")
    d_base = leak_descriptors(base_rows)
    print(f"基线泄漏描述  : group 跨折={d_base['group_cross_fold']} "
          f"sha 跨折={d_base['sha_cross_fold']} "
          f"被同源污染的 val 总数={d_base['val_tainted_total']}")

    if a.only_verify:
        if not out.is_file():
            print(f"❌ --only-verify 指定但文件不存在：{out}")
            return 2
        _vfields, out_rows = read_manifest(out)
        rep = _self_check(base_rows, out_rows)
        d_out = rep["control_leak"]  # type: ignore[index]
        print("\n复验结果")
        print(f"  行数        : {rep['rows']:,}")
        print(f"  对照泄漏描述: group 跨折={d_out['group_cross_fold']} "
              f"sha 跨折={d_out['sha_cross_fold']} "
              f"被同源污染的 val 总数={d_out['val_tainted_total']}")
        print(f"  文件 SHA-256: {sha256_file(out)}")
        print("\n✅ 复验通过（6/6 自检）")
        return 0

    print(f"\n生成中 …（seed={a.seed}，类别分层，每类每折 {PER_CLASS_VAL} 张 val）")
    out_rows = build_control_rows(base_rows, a.seed)

    rep = _self_check(base_rows, out_rows)
    d_out = rep["control_leak"]  # type: ignore[index]
    print("自检          : 6/6 PASS")
    print(f"  对照泄漏描述: group 跨折={d_out['group_cross_fold']} "
          f"（基线 {d_base['group_cross_fold']}）")
    print(f"                sha   跨折={d_out['sha_cross_fold']} "
          f"（基线 {d_base['sha_cross_fold']}）")
    print(f"                被同源污染的 val 总数={d_out['val_tainted_total']} "
          f"（基线 {d_base['val_tainted_total']}）")
    print(f"                各折污染数={d_out['val_tainted_per_fold']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    out_sha = sha256_file(out)
    sidecar = out.with_suffix(out.suffix + ".sha256")
    sidecar.write_text(f"{out_sha}  {out.name}\n", encoding="utf-8")

    report = {
        "script": Path(__file__).name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_manifest": str(man),
        "baseline_manifest_sha256": man_sha,
        "control_manifest": str(out),
        "control_manifest_sha256": out_sha,
        "shuffle_seed": a.seed,
        "design": "类别分层（每类每折 512 val）+ 忽略 group_id；除 val_fold 外逐列沿用基线",
        "self_check": rep,
        "frozen_note": (
            "SHA-256 必须原样写进 c1_cv_matrix.py 的 CONTROL_MANIFEST_SHA256 常量；"
            "跑实验前后都不得再改本文件。"
        ),
    }
    (out.parent / (out.stem + ".report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n已写出        : {out}")
    print(f"  SHA-256      : {out_sha}")
    print(f"  侧车文件      : {sidecar}")
    print(f"  自检报告      : {out.parent / (out.stem + '.report.json')}")
    print("\n★ 下一步：把上面这行 SHA-256 写进 c1_cv_matrix.py 的 CONTROL_MANIFEST_SHA256。")
    print(f"★ 别忘了一并上传：{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
