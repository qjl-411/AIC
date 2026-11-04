#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DMSA-Net 最终实验 / 第 2 步：sealed test 单次正式评价
===============================================================================

先完成：
    final_train.py

本文件只做一件事：
    加载固定训练 126 epochs 得到的 final model，
    在原始封存 test 1,279 张 MRI 上做一次正式评价。

【绝对不要】
- 根据 test Accuracy 改 epoch；
- 根据 test confusion matrix 改 augmentation；
- 根据 test 某一类别表现改模型；
- 改完后把同一个 test 当成“新的独立测试”继续反复挑最好结果。

如果以后要做新的模型版本，当前 test 结果应被视为已经看过，
不能再声称它对新版本仍是完全 untouched 的 test。

本代码会：
1. 验证 final checkpoint 是 epoch 126；
2. 验证 DMSA-Net architecture fingerprint；
3. 验证 sealed test manifest SHA-256；
4. 验证 1,279 张 test 的类别数量；
5. 默认逐张复核 test 文件 SHA-256；
6. 再次确认 train/test 没有 exact SHA-256 overlap；
7. model.eval() + torch.no_grad() 进行测试；
8. 保存 Accuracy / macro Precision / Recall / F1 / Kappa；
9. 保存每类别指标、confusion matrix、逐样本 prediction/probability；
10. 若同一 final run 已经有 test_metrics.json，则拒绝再次评价，避免误操作。
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from A01_train_final_model import (
    EXPECTED_ARCHITECTURE_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
    ManifestImageDataset,
    architecture_fingerprint,
    build_transforms,
    compute_metrics,
    create_model,
    environment_info,
    model_parameter_stats,
    read_fold_manifest,
    resolve_device,
    save_confusion_matrix,
    save_json,
    seed_everything,
    seed_worker,
    sha256_file,
)


# =============================================================================
# 固定协议
# =============================================================================

EXPECTED_TEST_MANIFEST_SHA256 = "bc8e53e24722167fb3bf0ab468863ed8d0bcaec1341bb52007021802c31126a6"
EXPECTED_FINAL_EPOCH = 126
EXPECTED_TEST_IMAGES = 1279
EXPECTED_TEST_COUNTS = {
    "Mild Impairment": 179,
    "Moderate Impairment": 12,
    "No Impairment": 640,
    "Very Mild Impairment": 448,
}
TEST_BATCH_SIZE = 16
SEED = 42

# =============================================================================
# 用户配置区 —— 正式 sealed test 前只改这里
# =============================================================================

# 1) 与 final_train.py 相同的 Combined Dataset
DATA_ROOT = r"/home/featurize/data/dmsa_pcv2/Combined Dataset"

# 2) final_train.py 生成的运行目录
FINAL_RUN_DIR = r"/home/featurize/work/dmsa_final_outputs/dmsa_final_train_seed42_epoch126"

# 3) 正式 GPU
DEVICE = "cuda:0"

# 4) DataLoader workers。与正式五折/final train 保持 6。
NUM_WORKERS = 6

# 5) 推荐保持 True：正式评价前逐张复核 1,279 test SHA-256。
VERIFY_TEST_HASHES = True

# =============================================================================
# 用户配置区结束
# =============================================================================


def _read_test_manifest(path: Path) -> List[dict]:
    required = {"relative_path", "class_name", "filename", "sha256"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"sealed test 应有 {EXPECTED_TEST_IMAGES} 行，实际 {len(rows)}"
        )
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"test manifest 缺少字段：{sorted(missing)}")
    return rows


def _validate_test_files(
    rows: Sequence[dict],
    test_root: Path,
    verify_hashes: bool,
) -> None:
    counts = Counter(r["class_name"] for r in rows)
    if dict(sorted(counts.items())) != dict(sorted(EXPECTED_TEST_COUNTS.items())):
        raise RuntimeError(
            f"sealed test 类别数不一致。\nExpected={EXPECTED_TEST_COUNTS}\nActual={dict(counts)}"
        )

    rels = [r["relative_path"] for r in rows]
    if len(set(rels)) != len(rels):
        raise RuntimeError("sealed test manifest 中存在重复 relative_path。")

    for r in rows:
        rel = Path(r["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Unsafe test relative path: {r['relative_path']}")
        p = test_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"缺少 sealed test MRI：{p}")
        if p.parent.name != r["class_name"]:
            raise RuntimeError(f"test path/class 不一致：{p}")
        if verify_hashes and sha256_file(p) != r["sha256"]:
            raise RuntimeError(f"sealed test SHA-256 不一致：{p}")


def _per_class_from_cm(cm: np.ndarray, class_names: Sequence[str]) -> List[dict]:
    rows = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append({
            "class_name": name,
            "support": support,
            "predicted_count": predicted,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


@torch.no_grad()
def _evaluate_once(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    dataset_rows: Sequence[dict],
):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []
    all_probs = []

    bar = tqdm(loader, desc="SEALED TEST", file=None)
    for images, labels in bar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss during sealed test.")

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        n = labels.size(0)
        total_loss += float(loss.item()) * n
        total_samples += n
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    if total_samples != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"sealed test 实际处理 {total_samples} 张，预期 {EXPECTED_TEST_IMAGES}"
        )

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    probabilities = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_pred, num_classes=len(class_names))
    metrics["test_loss"] = total_loss / total_samples
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

    prediction_rows = []
    for i, row in enumerate(dataset_rows):
        item = {
            "relative_path": row["relative_path"],
            "true_class": class_names[int(y_true[i])],
            "predicted_class": class_names[int(y_pred[i])],
            "true_index": int(y_true[i]),
            "predicted_index": int(y_pred[i]),
            "correct": int(y_true[i] == y_pred[i]),
        }
        for cls_idx, cls_name in enumerate(class_names):
            safe = cls_name.lower().replace(" ", "_")
            item[f"prob_{safe}"] = float(probabilities[i, cls_idx])
        prediction_rows.append(item)

    return metrics, cm, prediction_rows


def main():
    script_dir = Path(__file__).resolve().parent
    train_manifest_path = script_dir / "pcv_v2_folds_seed42.csv"
    test_manifest_path = script_dir / "pcv_test_manifest_SEALED.csv"

    if not train_manifest_path.is_file():
        raise FileNotFoundError(train_manifest_path)
    if not test_manifest_path.is_file():
        raise FileNotFoundError(test_manifest_path)

    train_manifest_sha = sha256_file(train_manifest_path)
    test_manifest_sha = sha256_file(test_manifest_path)

    if train_manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("train manifest 版本与正式五折不一致。")
    if test_manifest_sha != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError(
            "sealed test manifest SHA-256 不一致。\n"
            f"Expected: {EXPECTED_TEST_MANIFEST_SHA256}\n"
            f"Actual:   {test_manifest_sha}"
        )

    data_root = Path(DATA_ROOT).expanduser().resolve()
    test_root = data_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(
            f"找不到 test：{test_root}\n"
            "请让 DATA_ROOT 指向 Combined Dataset。"
        )

    run_dir = Path(FINAL_RUN_DIR).expanduser().resolve()
    final_model_path = run_dir / f"final_model_epoch{EXPECTED_FINAL_EPOCH}.pth"
    if not final_model_path.is_file():
        raise FileNotFoundError(
            f"找不到 final model：{final_model_path}\n"
            "请先完整运行 final_train.py。"
        )

    test_out = run_dir / "sealed_test_once"
    if (test_out / "test_metrics.json").is_file():
        raise RuntimeError(
            "这个 final run 已经存在正式 sealed test 结果：\n"
            f"{test_out / 'test_metrics.json'}\n"
            "为防止无意中反复查看 test 并据此调参，本脚本拒绝再次评价。"
        )
    if test_out.exists() and any(test_out.iterdir()):
        raise RuntimeError(
            f"sealed_test_once 目录已存在但不完整：{test_out}\n"
            "如果确认上一次只是程序崩溃且从未得到完整 test 结果，"
            "请人工检查后删除该目录，再重新运行。"
        )
    test_out.mkdir(parents=True, exist_ok=True)

    # Snapshot code/manifests BEFORE evaluation.
    shutil.copy2(Path(__file__).resolve(), test_out / "sealed_test_code_used.py")
    shutil.copy2(test_manifest_path, test_out / "sealed_test_manifest_used.csv")
    shutil.copy2(train_manifest_path, test_out / "train_manifest_reference.csv")

    test_rows = _read_test_manifest(test_manifest_path)
    _validate_test_files(
        test_rows,
        test_root=test_root,
        verify_hashes=VERIFY_TEST_HASHES,
    )

    # Reconfirm no byte-identical train/test overlap using frozen manifests.
    train_rows = read_fold_manifest(train_manifest_path)
    train_hashes = {r["sha256"] for r in train_rows}
    test_hashes = {r["sha256"] for r in test_rows}
    overlap = train_hashes & test_hashes
    if overlap:
        raise RuntimeError(
            f"发现 train/test exact SHA-256 overlap：{len(overlap)}"
        )

    checkpoint = torch.load(
        final_model_path,
        map_location="cpu",
    )

    if checkpoint.get("checkpoint_type") != "final-model-for-sealed-test":
        raise RuntimeError("加载的 checkpoint 不是 final sealed-test model。")
    if int(checkpoint.get("epoch", -1)) != EXPECTED_FINAL_EPOCH:
        raise RuntimeError(
            f"Final checkpoint epoch 应为 {EXPECTED_FINAL_EPOCH}，"
            f"实际 {checkpoint.get('epoch')}"
        )
    if checkpoint.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Final checkpoint 使用的 train manifest 不一致。")
    if checkpoint.get("architecture_fingerprint") != EXPECTED_ARCHITECTURE_FINGERPRINT:
        raise RuntimeError("Final checkpoint architecture fingerprint 不一致。")

    class_to_idx: Dict[str, int] = checkpoint["class_to_idx"]
    class_names = [
        name for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1])
    ]
    if len(class_names) != 4:
        raise RuntimeError(f"Unexpected class mapping: {class_to_idx}")

    seed_everything(SEED, deterministic=False)
    device = resolve_device(DEVICE)

    model = create_model(num_classes=4)
    if architecture_fingerprint(model) != EXPECTED_ARCHITECTURE_FINGERPRINT:
        raise RuntimeError("本地 create_model() 结构与 final checkpoint 不一致。")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)

    _, test_transform = build_transforms()
    dataset = ManifestImageDataset(
        train_root=test_root,
        rows=test_rows,
        class_to_idx=class_to_idx,
        transform=test_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    save_json(
        test_out / "test_environment.json",
        environment_info(device),
    )
    save_json(
        test_out / "pre_evaluation_integrity.json",
        {
            "timestamp": datetime.now().isoformat(),
            "final_model_path": str(final_model_path),
            "final_model_sha256": sha256_file(final_model_path),
            "train_manifest_sha256": train_manifest_sha,
            "test_manifest_sha256": test_manifest_sha,
            "test_images": len(test_rows),
            "test_class_counts": dict(Counter(r["class_name"] for r in test_rows)),
            "train_test_exact_sha256_overlap": 0,
            "expected_final_epoch": EXPECTED_FINAL_EPOCH,
            "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
        },
    )

    print("=" * 78)
    print("SEALED TEST — FINAL EVALUATION")
    print("=" * 78)
    print(f"Final model : {final_model_path}")
    print(f"Epoch       : {EXPECTED_FINAL_EPOCH}")
    print(f"Test images : {len(dataset):,}")
    print(f"Device      : {device}")
    print("注意：以下结果是正式 held-out test 结果，不用于继续调参。")
    print("=" * 78)

    metrics, cm, prediction_rows = _evaluate_once(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
        dataset_rows=test_rows,
    )

    per_class = _per_class_from_cm(cm, class_names)

    # Save predictions.
    pred_path = test_out / "test_predictions.csv"
    pred_fields = list(prediction_rows[0].keys())
    with pred_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields)
        w.writeheader()
        w.writerows(prediction_rows)

    # Save per-class metrics.
    pc_path = test_out / "test_per_class_metrics.csv"
    with pc_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "class_name", "support", "predicted_count",
                "true_positive", "precision", "recall", "f1",
            ],
        )
        w.writeheader()
        w.writerows(per_class)

    # Save confusion matrix.
    cm_csv = test_out / "test_confusion_matrix.csv"
    with cm_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, cm):
            w.writerow([name, *[int(x) for x in row]])

    save_confusion_matrix(
        cm=cm,
        class_labels=class_names,
        save_path=test_out / "test_confusion_matrix.png",
        title="DMSA-Net Final Sealed Test Confusion Matrix",
    )

    result = {
        **metrics,
        "test_images": EXPECTED_TEST_IMAGES,
        "correct": int(np.trace(cm)),
        "incorrect": int(cm.sum() - np.trace(cm)),
        "per_class": per_class,
        "checkpoint_epoch": EXPECTED_FINAL_EPOCH,
        "final_model_sha256": sha256_file(final_model_path),
        "test_manifest_sha256": test_manifest_sha,
        "evaluated_at": datetime.now().isoformat(),
        "interpretation_rule": (
            "This held-out test result is final evaluation only and must not be used "
            "to retune epochs, architecture, augmentation, or other hyperparameters."
        ),
    }
    save_json(test_out / "test_metrics.json", result)

    (test_out / "SEALED_TEST_EVALUATION_COMPLETE.txt").write_text(
        "SEALED TEST EVALUATION COMPLETE\n"
        f"evaluated_at={result['evaluated_at']}\n"
        f"accuracy={result['accuracy']:.12f}\n"
        "Do not tune the model on this result and then reuse the same test set "
        "as if it were untouched.\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("SEALED TEST COMPLETE")
    print("=" * 78)
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")
    print(f"F1        : {metrics['f1']:.6f}")
    print(f"Kappa     : {metrics['kappa']:.6f}")
    print(f"Test loss : {metrics['test_loss']:.6f}")
    print(f"Correct   : {int(np.trace(cm))}/{int(cm.sum())}")
    print(f"Results   : {test_out}")
    print()
    print("从现在开始，这个 test 结果只能用于报告/分析，不能再用于修改模型后重新挑最好结果。")
    print("=" * 78)


if __name__ == "__main__":
    main()
