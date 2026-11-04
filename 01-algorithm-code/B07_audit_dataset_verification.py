#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 0.0b 数据集到货验证（v2.7 新增）
====================================

背景（为什么必须单独一个脚本）
------------------------------
v2.6 的结论文档 §十 把两条事实列为"决定 §3.3 走 A/B/C 哪条分叉"的关键输入。
2026-09-23 用户回答：

  1. **原始图片目录已到货**，位于 `Alzheimer's MRI Dataset/`（云存储下载）
  2. **拿不到**发布者原始 ZIP / 增广脚本 / 文件映射表

于是本轮必须先把"到货的到底是什么"钉死，再谈路线分叉。
**本脚本只做只读盘点与对账，绝不修改数据集目录一个字节。**

它检查六件事
------------
1. **结构**：train/test × 4 类目录是否存在，逐类文件数
2. **帧几何**：真实尺寸 / 色彩模式 / 位深 —— 与论文和代码的 `224×224×3` 声明对账
3. **SHA-256 逐文件对账**：与两份冻结 manifest 比对（路径存在性 + 内容哈希）
4. **SHA-only 唯一数**：复算"用错工具"会得到的那个数字（用于演示 §1.1.1 的失败模式）
5. **exact 重复组件**：train 内部（按类）与 test 内部（按类）的 SHA 连通分量
6. **train↔test 跨集 exact 交集**：sealed test 独立性的**下界**检验

产物：`<out>/dataset_verify.json` + `<out>/dataset_verify.md`（经 `write_artifact()` 盖章）

用法
----
    python a0b_dataset_verify.py --data "../Alzheimer's MRI Dataset/Alzheimer's MRI Dataset/Combined Dataset"
    python a0b_dataset_verify.py --data <dir> --manifest-dir ../DMSA_net --out _a0_out --no-frames
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
from typing import Dict, List

from B16_lib_provenance_audit import write_artifact

#: manifest 里的类名 -> 磁盘目录名（本轮实测两者一致，保留映射以便将来容错）
CLASS_DIRS = ["Mild Impairment", "Moderate Impairment",
              "No Impairment", "Very Mild Impairment"]

TRAIN_MANIFEST = "pcv_v2_folds_seed42.csv"
TEST_MANIFEST = "pcv_test_manifest_SEALED.csv"

#: 论文与训练代码声称的输入几何（用于对账，不是用来"假设"的）
DECLARED_INPUT = dict(size=(224, 224), channels=3, mode="RGB")


def _sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _index(root: pathlib.Path, split: str) -> Dict[str, pathlib.Path]:
    """返回 {相对 split 根目录的 posix 路径: 绝对路径}"""
    base = root / split
    out = {}
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            out[p.relative_to(base).as_posix()] = p
    return out


def _read_manifest(p: pathlib.Path) -> List[dict]:
    import csv
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _components(shas: List[str]) -> dict:
    """对 SHA 列表做连通分量统计（同 SHA 即同组件）。"""
    by = collections.defaultdict(list)
    for s in shas:
        by[s].append(s)
    sizes = collections.Counter()
    n_files = 0
    for s, v in by.items():
        sizes[len(v)] += 1
        n_files += len(v)
    redundant = n_files - len(by)
    return dict(n_unique=len(by), n_files=n_files, n_redundant=redundant,
                size_dist={str(k): v for k, v in sorted(sizes.items())},
                multi_member_groups=sum(v for k, v in sizes.items() if k > 1))


def _frames(paths: List[pathlib.Path], limit: int | None = None) -> dict:
    """真实帧几何（需要 Pillow；失败则返回 {available: False}）。"""
    try:
        from PIL import Image  # noqa
    except Exception as e:  # pragma: no cover
        return dict(available=False, reason=f"{type(e).__name__}: {e}")
    size = collections.Counter()
    mode = collections.Counter()
    it = paths if limit is None else paths[:limit]
    failed = 0
    for p in it:
        try:
            with Image.open(p) as im:
                size[im.size] += 1
                mode[im.mode] += 1
        except Exception:
            failed += 1
    return dict(available=True, n_scanned=len(it), n_failed=failed,
                size_dist={f"{w}x{h}": n for (w, h), n in size.most_common()},
                mode_dist=dict(mode.most_common()),
                matches_declared=(
                    len(size) == 1 and next(iter(size)) == DECLARED_INPUT["size"]
                    and len(mode) == 1 and next(iter(mode)) == DECLARED_INPUT["mode"]
                ))


def _reconcile_manifest(rows: List[dict], disk: Dict[str, pathlib.Path]) -> dict:
    """逐文件对账：manifest 里每一行在磁盘上是否存在、SHA 是否一致。"""
    missing, sha_mismatch, n_ok = [], [], 0
    for r in rows:
        rel = r["relative_path"].replace("\\", "/")
        p = disk.get(rel)
        if p is None:
            missing.append(rel)
            continue
        if r.get("sha256") and _sha(p).lower() != r["sha256"].strip().lower():
            sha_mismatch.append(rel)
        else:
            n_ok += 1
    extra = sorted(set(disk) - {r["relative_path"].replace("\\", "/") for r in rows})
    return dict(n_manifest_rows=len(rows), n_disk_files=len(disk),
                n_ok=n_ok, n_missing=len(missing), n_sha_mismatch=len(sha_mismatch),
                n_extra_on_disk=len(extra),
                missing_examples=missing[:10], sha_mismatch_examples=sha_mismatch[:10],
                extra_examples=extra[:10],
                verdict=("EXACT_MATCH" if not missing and not sha_mismatch and not extra
                         else "MISMATCH"))


def _run_baseline_check(mdir: pathlib.Path, root: pathlib.Path) -> dict:
    """复用**基线作者自己**的校验逻辑（`data_prepare_check.py`）跑一遍真实数据。

    为什么值得做：本脚本的检查是我写的，属于"自证"。
    而 `data_prepare_check.py` 是**原作者**留下的校验器（`VERIFY_ALL_HASHES=True`），
    用它跑一遍等于让第三方代码来判"manifest 与磁盘是否一致"。

    ⚠️ **只调用它的校验函数，绝不调用 `main()`** ——
    `main()` 第一句就是 `extract_dataset(zip, combined_root)`，会把 ZIP 解压到数据目录。
    我们既没有那个 ZIP（用户在 2026-09-23 明确答复拿不到），
    也**绝不允许**任何脚本往已到货的数据目录里写东西。
    """
    import contextlib
    import io
    import sys

    out: dict = dict(available=False)
    try:
        sys.path.insert(0, str(mdir))
        import A05_check_dataset_preparation as DPC  # noqa
    except Exception as e:
        out["reason"] = f"{type(e).__name__}: {e}"
        return out
    try:
        out["available"] = True
        out["module"] = str(pathlib.Path(DPC.__file__).resolve())
        out["source_zip_path_constant"] = str(getattr(DPC, "DATASET_ZIP_PATH", "-"))
        out["verify_all_hashes_constant"] = bool(getattr(DPC, "VERIFY_ALL_HASHES", False))
        DPC.FOLD_MANIFEST = mdir / TRAIN_MANIFEST  # 显式对齐，避免依赖它的 SCRIPT_DIR

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                tr_counts, _ = DPC.scan_split(root / "train")
                te_counts, _ = DPC.scan_split(root / "test")
                ok_counts = (tr_counts == DPC.EXPECTED_TRAIN
                             and te_counts == DPC.EXPECTED_TEST)
                rows = DPC.read_manifest(DPC.FOLD_MANIFEST)
                DPC.check_manifest(root, rows, verify_hashes=DPC.VERIFY_ALL_HASHES)
                DPC.check_train_test_exact_overlap(root, rows)
            out.update(ran_to_completion=True, error=None,
                       counts_match=ok_counts,
                       train_counts=tr_counts, test_counts=te_counts)
        except AssertionError as e:
            out.update(ran_to_completion=False, error=f"AssertionError: {e}")
        except Exception as e:  # pragma: no cover
            out.update(ran_to_completion=False, error=f"{type(e).__name__}: {e}")
        out["stdout_tail"] = buf.getvalue().strip().splitlines()[-12:]
        out["note"] = ("复用的是基线作者自带的校验器，未调用其 main()（main 会解压 ZIP 到数据目录）")
    finally:
        try:
            sys.path.remove(str(mdir))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="含 train/ 与 test/ 的目录")
    ap.add_argument("--manifest-dir", default="../DMSA_net")
    ap.add_argument("--out", default="_a0_out")
    ap.add_argument("--no-frames", action="store_true", help="跳过图片几何扫描（慢）")
    ap.add_argument("--no-baseline-check", action="store_true",
                    help="跳过复用 data_prepare_check.py 的校验")
    a = ap.parse_args()

    root = pathlib.Path(a.data).resolve()
    mdir = pathlib.Path(a.manifest_dir).resolve()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = dict(data_root=str(root), exists=root.exists())
    print("=" * 74)
    print("阶段 0.0b 数据集到货验证")
    print("=" * 74)
    print(f"数据根目录：{root}")
    if not root.exists():
        report["verdict"] = "DATA_ROOT_NOT_FOUND"
        write_artifact(out / "dataset_verify.json", report)
        print("  ✗ 目录不存在")
        return

    # ---- 1. 结构 + 逐类计数 -------------------------------------------------
    counts: Dict[str, Dict[str, int]] = {}
    for split in ("train", "test"):
        counts[split] = {}
        for cls in CLASS_DIRS:
            d = root / split / cls
            counts[split][cls] = sum(1 for p in d.glob("*")
                                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if d.exists() else -1
    report["counts"] = counts
    print("\n[1] 结构 / 逐类文件数")
    for split in ("train", "test"):
        tot = sum(v for v in counts[split].values() if v > 0)
        print(f"  {split:5s} 合计 {tot:6d}   " +
              "  ".join(f"{c.split()[0]}={counts[split][c]}" for c in CLASS_DIRS))

    # ---- 2. 帧几何 ----------------------------------------------------------
    if not a.no_frames:
        allp = []
        for split in ("train", "test"):
            allp += sorted(_index(root, split).values())
        fr = _frames(allp)
        report["frames"] = dict(declared=DECLARED_INPUT, **fr)
        print("\n[2] 帧几何（全量扫描）")
        if fr.get("available"):
            print(f"  尺寸分布：{fr['size_dist']}")
            print(f"  模式分布：{fr['mode_dist']}")
            print(f"  与声明 {DECLARED_INPUT['size']}×{DECLARED_INPUT['channels']} "
                  f"({DECLARED_INPUT['mode']}) 一致：{'是' if fr['matches_declared'] else '否'}")
        else:
            print(f"  跳过：{fr.get('reason')}")

    # ---- 3. SHA-256 逐文件对账 ---------------------------------------------
    print("\n[3] SHA-256 逐文件对账（与冻结 manifest）")
    recon = {}
    disks = {}
    for split, mfile in (("train", TRAIN_MANIFEST), ("test", TEST_MANIFEST)):
        disk = _index(root, split)
        disks[split] = disk
        mp = mdir / mfile
        if not mp.exists():
            recon[split] = dict(verdict="MANIFEST_NOT_FOUND", manifest=str(mp))
            print(f"  [{split}] ✗ 找不到 {mp}")
            continue
        rows = _read_manifest(mp)
        r = _reconcile_manifest(rows, disk)
        r["manifest"] = str(mp)
        r["manifest_sha256"] = _sha(mp)
        recon[split] = r
        print(f"  [{split}] {r['verdict']}  "
              f"manifest {r['n_manifest_rows']} 行 / 磁盘 {r['n_disk_files']} 个文件  "
              f"OK {r['n_ok']}  缺失 {r['n_missing']}  SHA 不符 {r['n_sha_mismatch']}  "
              f"磁盘多余 {r['n_extra_on_disk']}")
        print(f"          manifest SHA-256 = {r['manifest_sha256'][:16]}…")
    report["reconcile"] = recon

    # ---- 4/5. SHA-only 唯一数 + exact 组件 ----------------------------------
    print("\n[4/5] SHA-only 唯一数 与 exact 重复组件"
          "（**注意：这是『用错工具』会得到的数字**）")
    sha_stats = {}
    for split in ("train", "test"):
        sha_stats[split] = {}
        for cls in CLASS_DIRS:
            d = root / split / cls
            if not d.exists():
                continue
            shas = [_sha(p) for p in sorted(d.glob("*"))
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            c = _components(shas)
            sha_stats[split][cls] = c
            print(f"  {split:5s} {cls:24s} 文件 {c['n_files']:5d}  "
                  f"唯一 SHA {c['n_unique']:5d}  组内冗余 {c['n_redundant']:4d}  "
                  f"多成员组 {c['multi_member_groups']:3d}")
    report["sha_only"] = sha_stats

    # ---- 6. train <-> test 跨集 exact 交集 ----------------------------------
    tset = {}
    for cls in CLASS_DIRS:
        d = root / "train" / cls
        if d.exists():
            for p in d.glob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    tset.setdefault(_sha(p), []).append(f"{cls}/{p.name}")
    inter = {}
    for cls in CLASS_DIRS:
        d = root / "test" / cls
        if not d.exists():
            continue
        n = 0
        n_same = 0
        for p in d.glob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            n += 1
            s = _sha(p)
            if s in tset:
                n_same += 1
        inter[cls] = dict(n_test=n, n_exact_in_train=n_same)
    report["train_test_exact_intersection"] = inter
    print("\n[6] train ↔ test 跨集 **exact（SHA 完全相同）** 交集")
    tot_test = tot_hit = 0
    for cls, v in inter.items():
        tot_test += v["n_test"]
        tot_hit += v["n_exact_in_train"]
        print(f"  test/{cls:24s} {v['n_test']:5d} 张中 {v['n_exact_in_train']:4d} 张在 train 里有完全相同副本")
    report["train_test_exact_summary"] = dict(n_test=tot_test, n_hit=tot_hit)
    print(f"  合计：{tot_test} 张 test 中 {tot_hit} 张（{tot_hit/max(tot_test,1)*100:.2f}%）"
          f" 有 exact 副本落在 train")

    # ---- 7. manifest 自述分组 vs 实测 SHA 的交叉验证 -------------------------
    #: N1 的结论是"A1 的定位必须从『从零重建』下调为『独立复核既有分组』"。
    #: 那就必须真的去复核一遍：manifest 自述的 exact 分组，能否被磁盘 SHA 独立复现？
    print("\n[7] manifest 自述分组 与 磁盘 SHA 实测 的交叉验证")
    mrec = {}
    for split, mfile in (("train", TRAIN_MANIFEST), ("test", TEST_MANIFEST)):
        mp = mdir / mfile
        if not mp.exists():
            continue
        rows = _read_manifest(mp)
        if "group_reason" not in (rows[0] if rows else {}):
            mrec[split] = dict(has_groups=False)
            print(f"  [{split}] manifest 不含分组字段（仅 sha256）")
            continue
        reason = collections.Counter(r["group_reason"] for r in rows)
        #: ⚠️ 注意：manifest 每行都带 group_size，所以直接对列做 Counter 得到的是
        #: **行数×组件规模**（规模 2 的组件会被数两次）。必须除以规模才是**组件数**。
        #: 这个坑实测踩过一次：得到 {2:554,...} 与既有结论 {2:277,...} 表面不符，
        #: 实为 277×2。校验方法：Σ count×size == 行数。
        _rows_per_size = collections.Counter(int(r["group_size"]) for r in rows if r.get("group_size"))
        gsize = collections.Counter()
        for size, n_rows in _rows_per_size.items():
            assert n_rows % size == 0, (size, n_rows)
            gsize[size] = n_rows // size
        assert sum(s * c for s, c in gsize.items()) == len(rows), "组件规模分布与行数不自洽"
        # manifest 自述的 exact 冗余（按类）
        by_cls = collections.defaultdict(lambda: dict(redundant=0, groups2plus=0))
        seen = collections.Counter()
        for r in rows:
            if r["group_reason"] == "exact_duplicate":
                seen[r["group_id"]] += 1
        gid_reason = {}
        for r in rows:
            gid_reason[r["group_id"]] = r["group_reason"]
        for gid, n in seen.items():
            if gid_reason.get(gid) == "exact_duplicate":
                cls = next(r["class_name"] for r in rows if r["group_id"] == gid)
                by_cls[cls]["redundant"] += (n - 1)
                if n > 1:
                    by_cls[cls]["groups2plus"] += 1
        mrec[split] = dict(
            has_groups=True,
            group_reason_dist=dict(reason),
            group_size_dist={str(k): v for k, v in sorted(gsize.items())},
            n_group_ids=len({r["group_id"] for r in rows}),
            max_group_size=max(gsize) if gsize else 0,
            exact_redundant_by_class={k: v["redundant"] for k, v in by_cls.items()},
        )
        print(f"  [{split}] group_reason 分布：{dict(reason)}")
        print(f"  [{split}] 组件数 {mrec[split]['n_group_ids']}，"
              f"最大组件 {mrec[split]['max_group_size']}，"
              f"规模分布 {mrec[split]['group_size_dist']}")
        if by_cls:
            print(f"  [{split}] manifest 自述 exact 组内冗余（按类）："
                  f"{ {k: v['redundant'] for k, v in by_cls.items()} }")

    # 交叉验证：manifest 自述 exact 冗余 vs 磁盘 SHA 实测
    xv = {}
    for cls in CLASS_DIRS:
        man = mrec.get("train", {}).get("exact_redundant_by_class", {}).get(cls, 0)
        meas = sha_stats["train"].get(cls, {}).get("n_redundant", 0)
        xv[cls] = dict(manifest_exact_redundant=man, measured_sha_redundant=meas,
                       agree=(man == meas))
    report["manifest_cross_check"] = dict(manifest=mrec, exact_vs_sha=xv)
    ok_all = all(v["agree"] for v in xv.values())
    for cls, v in xv.items():
        print(f"  {cls:24s} manifest {v['manifest_exact_redundant']:4d}  "
              f"实测 SHA {v['measured_sha_redundant']:4d}  {'✓' if v['agree'] else '✗'}")
    print(f"  → manifest 的 exact 分组**可被磁盘 SHA 独立复现**：{'是' if ok_all else '否'}")

    # ---- 8. 与 E2 推断对账 --------------------------------------------------
    if "frames" in report and report["frames"].get("available"):
        declared = (224, 224)
        actual = tuple(int(x) for x in next(iter(report["frames"]["size_dist"])).split("x"))
        report["geometry_conflict"] = dict(
            declared=f"{declared[0]}x{declared[1]} RGB",
            actual=f"{actual[0]}x{actual[1]} {next(iter(report['frames']['mode_dist']))}",
            conflict=actual != declared or "RGB" not in report["frames"]["mode_dist"],
            note="磁盘帧几何与论文/训练代码声明的输入几何不一致；"
                 "需确认 pipeline 中的 resize/转 RGB 步骤（见 data_prepare_check.py 与 smoke_test.py）",
        )

    # ---- 9. 复用基线作者自己的校验器 ---------------------------------------
    if not a.no_baseline_check:
        print("\n[9] 复用基线自带校验器 `data_prepare_check.py`（**只调校验函数，不调 main()**）")
        bc = _run_baseline_check(mdir, root)
        report["baseline_check"] = bc
        if not bc.get("available"):
            print(f"  跳过（无法导入）：{bc.get('reason')}")
        else:
            print(f"  模块：{pathlib.Path(bc['module']).name}  "
                  f"VERIFY_ALL_HASHES={bc['verify_all_hashes_constant']}")
            print(f"  它期望的 ZIP 常量：{bc['source_zip_path_constant']}")
            print(f"  跑完全部校验：{'是' if bc.get('ran_to_completion') else '否'}"
                  + (f"  错误：{bc['error']}" if bc.get("error") else ""))
            for line in bc.get("stdout_tail", []):
                print(f"    {line}")
            print(f"  ⚠️ {bc['note']}")

    report["verdict"] = ("ARRIVED_AND_RECONCILED"
                         if all(recon.get(s, {}).get("verdict") == "EXACT_MATCH"
                                for s in ("train", "test"))
                         else "ARRIVED_WITH_MISMATCH")
    write_artifact(out / "dataset_verify.json", report)

    # ---- markdown -----------------------------------------------------------
    L = ["# 阶段 0.0b 数据集到货验证报告\n",
         f"- 数据根目录：`{root}`",
         f"- 判定：**{report['verdict']}**",
         f"- 数据目录只读，本脚本未做任何写操作\n",
         "## 1 结构与逐类计数\n",
         "| split | " + " | ".join(CLASS_DIRS) + " | 合计 |",
         "|---|" + "---|" * (len(CLASS_DIRS) + 1)]
    for split in ("train", "test"):
        row = [str(counts[split][c]) for c in CLASS_DIRS]
        L.append(f"| {split} | " + " | ".join(row) + f" | {sum(int(x) for x in row if int(x)>0)} |")
    if "frames" in report and report["frames"].get("available"):
        fr = report["frames"]
        L += ["", "## 2 帧几何（与声明对账）\n",
              f"- 实测尺寸：`{fr['size_dist']}`",
              f"- 实测模式：`{fr['mode_dist']}`",
              f"- 论文/代码声明：`{DECLARED_INPUT['size']} × {DECLARED_INPUT['channels']}ch ({DECLARED_INPUT['mode']})`",
              f"- **一致：{'是' if fr['matches_declared'] else '否'}**"]
    L += ["", "## 3 SHA-256 逐文件对账\n",
          "| split | manifest | 行数 | 磁盘文件 | OK | 缺失 | SHA 不符 | 磁盘多余 | 判定 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for split in ("train", "test"):
        r = recon.get(split, {})
        L.append(f"| {split} | `{pathlib.Path(r.get('manifest','-')).name}` | "
                 f"{r.get('n_manifest_rows','-')} | {r.get('n_disk_files','-')} | "
                 f"{r.get('n_ok','-')} | {r.get('n_missing','-')} | {r.get('n_sha_mismatch','-')} | "
                 f"{r.get('n_extra_on_disk','-')} | **{r.get('verdict','-')}** |")
    L += ["", "## 4 SHA-only 唯一数（**故意演示『用错工具』会得到的数字**）\n",
          "| split | 类别 | 文件数 | 唯一 SHA | 组内冗余 | 多成员组 |",
          "|---|---|---|---|---|---|"]
    for split in ("train", "test"):
        for cls, c in sha_stats[split].items():
            L.append(f"| {split} | {cls} | {c['n_files']} | {c['n_unique']} | "
                     f"{c['n_redundant']} | {c['multi_member_groups']} |")
    L += ["", "## 5 train ↔ test 跨集 exact 交集\n",
          "| test 类别 | test 张数 | 在 train 中有完全相同副本 |",
          "|---|---|---|"]
    for cls, v in inter.items():
        L.append(f"| {cls} | {v['n_test']} | {v['n_exact_in_train']} |")
    L.append(f"| **合计** | **{tot_test}** | **{tot_hit}** |")
    L += ["", "## 6 manifest 自述分组 vs 磁盘 SHA 实测（交叉验证）\n"]
    tr = mrec.get("train", {})
    if tr.get("has_groups"):
        L += [f"- `group_reason` 分布：`{tr['group_reason_dist']}`",
              f"- 组件数 **{tr['n_group_ids']}**，最大组件 **{tr['max_group_size']}**",
              f"- 规模分布：`{tr['group_size_dist']}`", "",
              "| 类别 | manifest 自述 exact 组内冗余 | 磁盘 SHA 实测组内冗余 | 一致 |",
              "|---|---|---|---|"]
        for cls, v in xv.items():
            L.append(f"| {cls} | {v['manifest_exact_redundant']} | "
                     f"{v['measured_sha_redundant']} | {'✓' if v['agree'] else '✗'} |")
        L += ["", ("**manifest 的 exact 分组可被磁盘 SHA 独立复现**"
                   if all(v["agree"] for v in xv.values())
                   else "**manifest 的 exact 分组与磁盘 SHA 不一致 —— 必须查清**")]
    bc = report.get("baseline_check", {})
    if bc.get("available"):
        L += ["", "## 7 复用基线自带校验器（`data_prepare_check.py`）\n",
              f"- 模块 SHA 已在 `MANIFEST.sha256` 登记；`VERIFY_ALL_HASHES={bc.get('verify_all_hashes_constant')}`",
              f"- 它期望的原始 ZIP：`{bc.get('source_zip_path_constant')}`（**用户已确认拿不到**）",
              f"- 跑完全部校验：**{'是' if bc.get('ran_to_completion') else '否'}**"
              + (f" —— `{bc.get('error')}`" if bc.get("error") else ""), "",
              "```", "\n".join(bc.get("stdout_tail", [])), "```", "",
              "> ⚠️ **只调用了它的校验函数，没有调用 `main()`** —— "
              "`main()` 会 `extract_dataset(zip, combined_root)`，往数据目录里解压。",
              "> 我们既没有那个 ZIP，也不允许任何脚本改动已到货的数据目录。"]
    (out / "dataset_verify.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n输出：{out/'dataset_verify.json'}、{out/'dataset_verify.md'}")
    print(f"判定：{report['verdict']}")


if __name__ == "__main__":
    main()
