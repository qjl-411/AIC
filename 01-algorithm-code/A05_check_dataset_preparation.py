#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DMSA-Net / PyCharm 第 1 个文件：数据解压 + 数据划分检查
===============================================================================

【你在 PyCharm 中只需要改下面“用户配置区”】

这个文件负责：
1. 从你原始的 MRI 数据集 ZIP 解压出 train/ 和 test/；
2. 检查 train=10240、test=1279；
3. 检查固定的 P-CV v2 manifest；
4. 检查五折都是 8192 train / 2048 validation；
5. 检查 exact/near-duplicate group 不跨 fold；
6. 可选：重新计算所有 MRI 的 SHA-256，并检查 train/test 没有完全相同图片。

它不会训练模型，也不会运行 smoke test。

---------------------------------------------------------------------------
需要安装什么包？
---------------------------------------------------------------------------

这个文件本身只用 Python 标准库，不需要额外 pip 包。

后面的 smoke_test.py / train_5fold.py 需要：
    torch
    torchvision
    numpy
    scikit-learn
    matplotlib
    tqdm
    Pillow

清华镜像安装普通依赖：
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scikit-learn matplotlib tqdm Pillow

本地 CPU 试运行如果还没有 PyTorch，可以尝试：
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision

正式云 GPU：
    优先使用云平台已经安装且 torch.cuda.is_available() == True 的 PyTorch。
    不要为了统一清华源而随便覆盖 GPU 版 PyTorch；CUDA wheel 要和云端环境匹配。

---------------------------------------------------------------------------
运行顺序
---------------------------------------------------------------------------

先在 PyCharm 直接运行本文件。
看到最后：
    DATA PREPARATION AND P-CV V2 CHECK: PASS
以后，再运行 smoke_test.py。
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# 用户配置区 —— PyCharm 运行前只改这里
# =============================================================================

# 1) 改成你手上的原始 MRI 数据集 ZIP
# Windows 示例：
# DATASET_ZIP_PATH = r"D:/DMSA/Alzheimer's MRI Dataset.zip"
#
# 云端 Linux 示例：
# DATASET_ZIP_PATH = r"/home/featurize/work/Alzheimer_MRI.zip"
DATASET_ZIP_PATH = r"/home/featurize/work/DMSA_PCV2_20260824/input/Alzheimer_MRI.zip"

# 2) 数据解压到哪里
# 程序会自动创建：
#   WORK_DIR/Combined Dataset/train
#   WORK_DIR/Combined Dataset/test
WORK_DIR = r"/home/featurize/data/dmsa_pcv2"

# 3) 第一次正式实验前建议 True。
# 会逐张重新计算 SHA-256，启动稍慢，但不会训练 epoch。
VERIFY_ALL_HASHES = True

# 4) 如果 Combined Dataset 已经存在：
# False = 不覆盖，直接检查现有数据；
# True  = 删除后重新从 ZIP 解压。
REEXTRACT_DATASET = False

# =============================================================================
# 用户配置区结束 —— 下面一般不要改
# =============================================================================


SCRIPT_DIR = Path(__file__).resolve().parent
FOLD_MANIFEST = SCRIPT_DIR / "pcv_v2_folds_seed42.csv"

EXPECTED_TRAIN = {
    "Mild Impairment": 2560,
    "Moderate Impairment": 2560,
    "No Impairment": 2560,
    "Very Mild Impairment": 2560,
}
EXPECTED_TEST = {
    "Mild Impairment": 179,
    "Moderate Impairment": 12,
    "No Impairment": 640,
    "Very Mild Impairment": 448,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parse_zip_member(member_name: str):
    normalized = member_name.replace("\\", "/")
    for marker, split in [
        ("/Combined Dataset/train/", "train"),
        ("/Combined Dataset/test/", "test"),
    ]:
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            parts = rel.split("/", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return split, parts[0], parts[1]
    return None


def extract_dataset(dataset_zip: Path, combined_root: Path) -> None:
    if combined_root.exists():
        if REEXTRACT_DATASET:
            print(f"[INFO] 删除旧数据目录：{combined_root}")
            shutil.rmtree(combined_root)
        else:
            print(f"[INFO] 已存在数据目录，跳过解压：{combined_root}")
            return

    if not dataset_zip.is_file():
        raise FileNotFoundError(
            f"找不到 MRI ZIP：{dataset_zip}\n"
            "请修改文件顶部 DATASET_ZIP_PATH。"
        )

    combined_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(dataset_zip) as zf:
        extracted = 0
        for info in zf.infolist():
            parsed = parse_zip_member(info.filename)
            if parsed is None:
                continue

            split, class_name, filename = parsed
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue

            # 不允许 ZIP 里的相对路径穿越目标目录。
            safe_name = Path(filename).name
            if safe_name != filename:
                raise RuntimeError(f"Unexpected nested filename: {info.filename}")

            dst = combined_root / split / class_name / safe_name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                raise RuntimeError(f"解压时出现重复目标文件：{dst}")

            with zf.open(info, "r") as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1

    if extracted == 0:
        raise RuntimeError(
            "ZIP 中没有找到预期的 Combined Dataset/train 和 test 结构。"
        )
    print(f"[PASS] ZIP 解压完成，共解压 {extracted} 张图像。")


def scan_split(split_root: Path):
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)

    rows = []
    counts = {}
    for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        counts[class_dir.name] = len(images)
        for p in images:
            rows.append((class_dir.name, p))
    return counts, rows


def read_manifest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到固定 fold 文件：{path}\n"
            "请确保 pcv_v2_folds_seed42.csv 和三个 .py 文件放在同一个目录。"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_manifest(combined_root: Path, manifest_rows, verify_hashes: bool) -> None:
    train_root = combined_root / "train"

    if len(manifest_rows) != 10240:
        raise AssertionError(f"manifest 应有 10240 行，实际 {len(manifest_rows)}")

    required = {
        "relative_path", "class_name", "sha256", "group_id", "val_fold"
    }
    missing = required - set(manifest_rows[0])
    if missing:
        raise AssertionError(f"manifest 缺少字段：{sorted(missing)}")

    # 每张 train 图片只出现一次。
    rels = [r["relative_path"] for r in manifest_rows]
    if len(set(rels)) != len(rels):
        raise AssertionError("manifest 中存在重复 relative_path")

    # group 必须只属于一个 validation fold。
    group_to_folds = defaultdict(set)
    sha_to_folds = defaultdict(set)
    overall = Counter()

    for r in manifest_rows:
        fold = int(r["val_fold"])
        if fold not in {1, 2, 3, 4, 5}:
            raise AssertionError(f"非法 fold：{fold}")

        p = train_root / r["relative_path"]
        if not p.is_file():
            raise FileNotFoundError(f"manifest 中的图片不存在：{p}")
        if p.parent.name != r["class_name"]:
            raise AssertionError(f"类别目录和 manifest 不一致：{p}")

        if verify_hashes:
            actual = sha256_file(p)
            if actual != r["sha256"]:
                raise AssertionError(f"SHA-256 不一致：{p}")

        overall[r["class_name"]] += 1
        group_to_folds[r["group_id"]].add(fold)
        sha_to_folds[r["sha256"]].add(fold)

    if dict(sorted(overall.items())) != EXPECTED_TRAIN:
        raise AssertionError(
            f"manifest 类别计数不一致：{dict(sorted(overall.items()))}"
        )

    bad_groups = {g: fs for g, fs in group_to_folds.items() if len(fs) != 1}
    if bad_groups:
        raise AssertionError(
            f"发现 group 跨 fold：{list(bad_groups.items())[:5]}"
        )

    bad_sha = {h: fs for h, fs in sha_to_folds.items() if len(fs) != 1}
    if bad_sha:
        raise AssertionError(
            f"发现 exact duplicate SHA 跨 fold：{list(bad_sha.items())[:5]}"
        )

    # 每一折数量和类别数。
    for fold in range(1, 6):
        val_rows = [r for r in manifest_rows if int(r["val_fold"]) == fold]
        tr_rows = [r for r in manifest_rows if int(r["val_fold"]) != fold]

        tr_counts = Counter(r["class_name"] for r in tr_rows)
        va_counts = Counter(r["class_name"] for r in val_rows)

        if len(tr_rows) != 8192 or len(val_rows) != 2048:
            raise AssertionError(
                f"Fold {fold} 数量错误：train={len(tr_rows)} val={len(val_rows)}"
            )
        if set(tr_counts.values()) != {2048}:
            raise AssertionError(f"Fold {fold} train 类别不平衡：{tr_counts}")
        if set(va_counts.values()) != {512}:
            raise AssertionError(f"Fold {fold} val 类别不平衡：{va_counts}")

        tr_groups = {r["group_id"] for r in tr_rows}
        va_groups = {r["group_id"] for r in val_rows}
        if tr_groups & va_groups:
            raise AssertionError(f"Fold {fold} group 泄漏")

        print(
            f"[PASS] Fold {fold}: train=8192, val=2048, "
            "train/class=2048, val/class=512, group_overlap=0"
        )


def check_train_test_exact_overlap(combined_root: Path, manifest_rows) -> None:
    # 当 VERIFY_ALL_HASHES=True 时，train manifest 的 SHA 已经逐张复核。
    train_hashes = {r["sha256"] for r in manifest_rows}

    test_root = combined_root / "test"
    _, test_rows = scan_split(test_root)
    test_hashes = set()

    for _, p in test_rows:
        test_hashes.add(sha256_file(p))

    overlap = train_hashes & test_hashes
    if overlap:
        raise AssertionError(
            f"发现 train/test 完全相同图片 SHA-256 overlap：{len(overlap)}"
        )
    print("[PASS] train/test exact SHA-256 overlap = 0")


def main():
    dataset_zip = Path(DATASET_ZIP_PATH).expanduser()
    work_dir = Path(WORK_DIR).expanduser()
    combined_root = work_dir / "Combined Dataset"

    print("=" * 78)
    print("DMSA-Net | 数据解压 + P-CV v2 检查")
    print("=" * 78)
    print(f"MRI ZIP:    {dataset_zip}")
    print(f"DATA_ROOT:  {combined_root}")
    print(f"Manifest:   {FOLD_MANIFEST}")
    print()

    extract_dataset(dataset_zip, combined_root)

    train_counts, _ = scan_split(combined_root / "train")
    test_counts, _ = scan_split(combined_root / "test")

    if train_counts != EXPECTED_TRAIN:
        raise AssertionError(
            f"train 数量不符合预期：\nExpected={EXPECTED_TRAIN}\nActual={train_counts}"
        )
    if test_counts != EXPECTED_TEST:
        raise AssertionError(
            f"test 数量不符合预期：\nExpected={EXPECTED_TEST}\nActual={test_counts}"
        )

    print(f"[PASS] train = {sum(train_counts.values())}")
    print(f"[PASS] test  = {sum(test_counts.values())}")

    manifest_rows = read_manifest(FOLD_MANIFEST)
    check_manifest(
        combined_root=combined_root,
        manifest_rows=manifest_rows,
        verify_hashes=VERIFY_ALL_HASHES,
    )

    if VERIFY_ALL_HASHES:
        check_train_test_exact_overlap(combined_root, manifest_rows)
    else:
        print(
            "[INFO] VERIFY_ALL_HASHES=False：本次没有重新计算全部 SHA-256。"
        )

    print()
    print("=" * 78)
    print("DATA PREPARATION AND P-CV V2 CHECK: PASS")
    print("=" * 78)
    print(f"后面两个脚本中的 DATA_ROOT 请填写：")
    print(combined_root)


if __name__ == "__main__":
    main()
