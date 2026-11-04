#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DMSA-Net / PyCharm 第 3 个文件：正式 P-CV v2 五折训练
===============================================================================

【先不要直接运行本文件】

正确顺序：
    1. data_prepare_check.py  -> PASS
    2. smoke_test.py          -> PASS
    3. train_5fold.py         -> 正式实验

本文件不复制 DMSA-Net 网络代码。
它从 smoke_test.py import 同一套已经 smoke 通过的核心实现，
这样可以保证：

    Smoke Test 使用的模型 == 正式五折使用的模型

不会出现两份核心代码分别维护造成的版本漂移。

---------------------------------------------------------------------------
PyCharm 运行前需要安装
---------------------------------------------------------------------------

    torch
    torchvision
    numpy
    scikit-learn
    matplotlib
    tqdm
    Pillow

清华镜像安装普通依赖：
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scikit-learn matplotlib tqdm Pillow

正式云 GPU 上的 torch/torchvision：
    优先使用云平台已经正确配置 CUDA 的版本。
    不要盲目用普通 PyPI/清华源覆盖 GPU PyTorch。

检查：
    python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

---------------------------------------------------------------------------
正式基线参数
---------------------------------------------------------------------------

第一轮正式实验保持：
    Batch size = 16
    Epochs = 200
    Adam lr = 0.001
    ExponentialLR gamma = 0.96
    Seed = 42

STRICT_BASELINE_CONFIG=True 时，如果误改这些基线参数，本文件会直接报错，
避免无意中跑出另一套实验。

P-CV v2：
    每折 Train = 8192
    每折 Val   = 2048
    原 test/ 在本阶段继续封存。
"""

from pathlib import Path

from A04_smoke_test_pipeline import run_configured_experiment


# =============================================================================
# 用户配置区 —— PyCharm 正式训练前只改这里
# =============================================================================

# 1) 必须和 smoke_test.py 使用同一个 Combined Dataset
DATA_ROOT = r"/home/featurize/data/dmsa_pcv2/Combined Dataset"

# 2) 正式实验输出目录
OUTPUT_DIR = r"/home/featurize/work/dmsa_outputs"

# 3) 云 GPU 通常填 cuda:0
DEVICE = "cuda:0"

# 4) 运行哪一折：
#    0 = 一次顺序跑 Fold 1~5
#    1~5 = 只跑某一折
#
# 如果云平台单次任务时间有限，建议分别跑 1、2、3、4、5。
FOLD_TO_RUN = 5

# 5) 每次运行使用新的名字。
# 如果 FOLD_TO_RUN=0：
RUN_NAME = "pcv_v2_seed42_fold5"
#
# 如果分折运行，可以改成：
# RUN_NAME = "pcv_v2_seed42_fold1"

# 6) DataLoader workers。云 GPU 可先用 8；若平台 CPU 核较少可降低。
NUM_WORKERS = 6

# 7) 第一次正式云训练前，建议 True 再核一次 train 文件 SHA-256。
# 后续同一份已验证数据分折运行时可设 False，减少启动时间。
VERIFY_HASHES = False


# -----------------------------------------------------------------------------
# 正式基线超参数：第一轮不要改
# -----------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
LR_GAMMA = 0.96
SEED = 42

# True = 防止手滑改了上面正式基线参数还继续训练。
STRICT_BASELINE_CONFIG = True

# 严格 deterministic 可能降低性能，第一轮保持 False。
DETERMINISTIC = False

# 0 = 不额外周期保存巨大模型；best + final 仍会保存。
PERIODIC_SAVE_EVERY = 0

# =============================================================================
# 用户配置区结束
# =============================================================================


def check_baseline_config():
    if not STRICT_BASELINE_CONFIG:
        return

    expected = {
        "BATCH_SIZE": 16,
        "EPOCHS": 200,
        "LEARNING_RATE": 0.001,
        "LR_GAMMA": 0.96,
        "SEED": 42,
    }
    actual = {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "LR_GAMMA": LR_GAMMA,
        "SEED": SEED,
    }
    if actual != expected:
        raise RuntimeError(
            "STRICT_BASELINE_CONFIG=True，但正式基线超参数被修改。\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "如果以后确实要做新的超参数实验，请单独记录，并主动把 "
            "STRICT_BASELINE_CONFIG 改为 False。"
        )


def main():
    check_baseline_config()

    if FOLD_TO_RUN not in {0, 1, 2, 3, 4, 5}:
        raise ValueError("FOLD_TO_RUN 必须是 0~5")

    print("=" * 78)
    print("DMSA-Net | 正式 P-CV v2 训练")
    print("=" * 78)
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"DEVICE    = {DEVICE}")
    print(f"FOLD      = {FOLD_TO_RUN}")
    print(f"EPOCHS    = {EPOCHS}")
    print(f"BATCH     = {BATCH_SIZE}")
    print()
    print("确认：本阶段不会用原 test/ 做模型选择。")
    print()

    run_configured_experiment(
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        device_arg=DEVICE,
        fold=FOLD_TO_RUN,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_gamma=LR_GAMMA,
        seed=SEED,
        num_workers=NUM_WORKERS,
        deterministic=DETERMINISTIC,
        verify_hashes=VERIFY_HASHES,
        periodic_save_every=PERIODIC_SAVE_EVERY,
        smoke_test=False,
        smoke_train_batches=1,  # formal 模式不会使用
        smoke_val_batches=1,    # formal 模式不会使用
        caller_script=__file__,
    )


if __name__ == "__main__":
    main()
