#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DMSA-Net / PyCharm 第 2 个文件：Smoke Test + 共享核心实现
===============================================================================

【为什么正式五折代码会 import 这个文件？】

为了避免出现“smoke test 用一套模型代码，正式实验又复制了一套模型代码”
导致两份核心代码以后悄悄不一致，本文件同时保存：

    1. 已审计的 DMSA-Net 网络实现；
    2. 数据加载 / train / validation / metrics 公共函数；
    3. Smoke Test 的 PyCharm 直接运行入口。

正式的 train_5fold.py 会 import 本文件中的同一套核心函数。
因此：
    smoke 通过的模型代码 == 正式五折使用的模型代码。

这不是把正式训练“混在 smoke 里执行”：
    直接运行 smoke_test.py 只会跑少量 batch；
    直接运行 train_5fold.py 才会跑正式 epoch。

---------------------------------------------------------------------------
PyCharm 运行前需要安装
---------------------------------------------------------------------------

第三方包：
    torch
    torchvision
    numpy
    scikit-learn
    matplotlib
    tqdm
    Pillow

清华镜像安装普通依赖：
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scikit-learn matplotlib tqdm Pillow

本地 CPU 如果还没有 PyTorch，可尝试：
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision

云 GPU：
    优先使用云平台已安装且 CUDA 可用的 PyTorch。
    先检查：
    python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

---------------------------------------------------------------------------
运行顺序
---------------------------------------------------------------------------

先确保 data_prepare_check.py 已经 PASS。

然后在下面“用户配置区”填写 DATA_ROOT / OUTPUT_DIR / DEVICE，
直接点击 PyCharm 的 Run。

本地 CPU：
    DEVICE = "cpu"
    SMOKE_BATCH_SIZE = 1
    NUM_WORKERS = 0

云 GPU：
    DEVICE = "cuda:0"
    SMOKE_BATCH_SIZE = 16
    NUM_WORKERS = 8

Smoke 指标没有科研意义，只检查：
    数据、CUDA、显存、forward/backward、loss、metrics、模型保存、日志。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm


# ==================== 第一部分:动态激活函数（创新点 3）====================

class Swish(nn.Module):
    """
    Swish 激活函数

    创新点 3：使用 Swish 替代 ReLU
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# ==================== 第二部分:空间注意力模块（创新点 4）====================

class SpatialAttentionModule(nn.Module):
    """
    空间注意力模块

    创新点 4：捕捉空间位置的重要性
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        # 通道压缩：降维到1个通道，降低计算量
        # 空间注意力卷积：大核捕捉全局空间信息
        padding = kernel_size // 2
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            x: [B, C, H, W] 应用空间注意力后的特征
        """
        # 通道维度平均池化
        x_mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 应用空间注意力卷积
        spatial_map = self.spatial_conv(x_mean)
        spatial_map = self.bn(spatial_map)
        spatial_map = self.sigmoid(spatial_map)

        # 广播并应用注意力权重
        return x * spatial_map


# ==================== 第三部分:多尺度空洞卷积（创新点 5）【修复版】====================

class MultiScaleDilatedConv(nn.Module):
    """
    多尺度空洞卷积模块（修复版）

    创新点 5：使用不同空洞率并行提取多尺度特征

    【关键修复】自动调整通道数，确保可被 3 整除
    - 如果 out_channels 不能被 3 整除，自动添加 padding
    - 确保三个分支的输出通道数一致

    为什么有效？
    - AD的病理变化出现在不同尺度（从细胞级到组织级）
    - 单一感受野无法同时捕捉所有尺度的变化
    - 多尺度并行卷积能够全面覆盖不同尺度的特征
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 【修复】计算每个分支的输出通道数，确保可被 3 整除
        num_branches = 3
        channels_per_branch = out_channels // num_branches
        padding_channels = out_channels % num_branches

        # 如果有余数，调整每个分支的通道数
        if padding_channels > 0:
            # 方式1：将余数分配到前几个分支
            channels_list = [channels_per_branch + 1 if i < padding_channels else channels_per_branch
                             for i in range(num_branches)]
        else:
            channels_list = [channels_per_branch] * num_branches

        # 三个并行分支：空洞率分别为1、3、5
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3,
                      padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels_list[0]),
            Swish(beta=1.0)
        )

        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[1], kernel_size=3,
                      padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(channels_list[1]),
            Swish(beta=1.0)
        )

        self.dilated_conv5 = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[2], kernel_size=3,
                      padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(channels_list[2]),
            Swish(beta=1.0)
        )

        # 记录实际总输出通道数
        self.actual_out_channels = sum(channels_list)

        # 融合层：将所有分支输出融合到目标通道数
        self.fusion_conv = nn.Conv2d(self.actual_out_channels, out_channels, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.swish = Swish(beta=1.0)

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            x: [B, out_channels, H, W] 多尺度特征融合后的输出
        """
        # 并行提取多尺度特征
        out1 = self.dilated_conv1(x)
        out3 = self.dilated_conv3(x)
        out5 = self.dilated_conv5(x)

        # 通道拼接
        out = torch.cat([out1, out3, out5], dim=1)

        # 融合到目标通道数
        out = self.fusion_conv(out)
        out = self.fusion_bn(out)
        out = self.swish(out)

        return out


# ==================== 第四部分:增强版双路径瓶颈结构（创新点 1 + 4 + 5）====================

class EnhancedDualPathBottleneck(nn.Module):
    """
    增强版双路径瓶颈块（修复版）

    创新点整合：
    1. 双路径瓶颈结构（保留原有优势）
    4. 空间注意力增强（新增）
    5. 多尺度空洞卷积（新增，已修复通道整除问题）
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32,
                 base_width=4, use_spatial_attention=True, use_multi_scale=True):
        super(EnhancedDualPathBottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        # 路径1：标准 ResNeXt（输出 planes * expansion / 2 通道）
        self.path1_conv1 = nn.Conv2d(inplanes, width // 2, kernel_size=1, stride=stride, bias=False)
        self.path1_bn1 = nn.BatchNorm2d(width // 2)
        self.path1_conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, groups=groups // 2, bias=False)
        self.path1_bn2 = nn.BatchNorm2d(width // 2)
        self.path1_conv3 = nn.Conv2d(width // 2, planes * self.expansion // 2, kernel_size=1, bias=False)
        self.path1_bn3 = nn.BatchNorm2d(planes * self.expansion // 2)

        # 路径2：增强型路径（集成多尺度空洞卷积）
        self.path2_conv1 = nn.Conv2d(inplanes, width // 2, kernel_size=1, stride=stride, bias=False)
        self.path2_bn1 = nn.BatchNorm2d(width // 2)

        # 创新点 5：多尺度空洞卷积（仅在路径2中使用，已修复）
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.path2_conv2 = MultiScaleDilatedConv(width // 2, width // 2)
        else:
            self.path2_conv2 = nn.Sequential(
                nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1,
                          groups=width // 2, bias=False),
                nn.BatchNorm2d(width // 2),
                Swish(beta=1.0)
            )

        self.path2_conv3 = nn.Conv2d(width // 2, planes * self.expansion // 2, kernel_size=1, bias=False)
        self.path2_bn3 = nn.BatchNorm2d(planes * self.expansion // 2)

        # 融合层：将两条路径的输出融合
        self.fusion_conv = nn.Conv2d(planes * self.expansion, planes * self.expansion, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(planes * self.expansion)

        # 创新点 4：空间注意力模块
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = SpatialAttentionModule(kernel_size=7)

        # 创新点 3：使用 Swish 激活函数
        self.swish = Swish(beta=1.0)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 路径1：标准 ResNeXt
        out1 = self.path1_conv1(x)
        out1 = self.path1_bn1(out1)
        out1 = self.swish(out1)

        out1 = self.path1_conv2(out1)
        out1 = self.path1_bn2(out1)
        out1 = self.swish(out1)

        out1 = self.path1_conv3(out1)
        out1 = self.path1_bn3(out1)

        # 路径2：增强型路径（集成多尺度空洞卷积）
        out2 = self.path2_conv1(x)
        out2 = self.path2_bn1(out2)
        out2 = self.swish(out2)

        out2 = self.path2_conv2(out2)  # 多尺度空洞卷积（已修复）
        out2 = self.path2_conv3(out2)
        out2 = self.path2_bn3(out2)

        # 通道拼接
        out = torch.cat([out1, out2], dim=1)

        # 融合层
        out = self.fusion_conv(out)
        out = self.fusion_bn(out)
        out = self.swish(out)

        # 创新点 4：空间注意力
        if self.use_spatial_attention:
            out = self.spatial_attention(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.swish(out)

        return out


# ==================== 第五部分:层间特征融合（创新点 2）====================

class InterLayerFusion(nn.Module):
    """
    层间特征融合模块

    创新点 2：在不同层之间添加特征融合连接

    【关键修复】确保通道数正确匹配：
    - layer2 输出：512 通道（128 * 4）
    - layer3 输出：1024 通道（256 * 4）
    - layer4 输出：2048 通道（512 * 4）
    """

    def __init__(self, low_channels, high_channels):
        super().__init__()

        # 通道对齐
        self.align_conv = nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(high_channels)

        # 注意力加权
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels * 2, high_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_channels // 4, high_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.swish = Swish(beta=1.0)

    def forward(self, low_feat, high_feat):
        """
        Args:
            low_feat: [B, low_channels, H*2, W*2] 浅层特征（尺寸是高层的2倍）
            high_feat: [B, high_channels, H, W] 深层特征

        Returns:
            fused_feat: [B, high_channels, H, W] 融合后的特征
        """
        # 上采样浅层特征（从 [B, low_channels, H*2, W*2] 到 [B, low_channels, H, W]）
        low_feat_down = nn.functional.interpolate(low_feat, scale_factor=0.5, mode='bilinear', align_corners=False)

        # 通道对齐
        low_feat_aligned = self.align_conv(low_feat_down)
        low_feat_aligned = self.align_bn(low_feat_aligned)
        low_feat_aligned = self.swish(low_feat_aligned)

        # 拼接
        concat_feat = torch.cat([low_feat_aligned, high_feat], dim=1)

        # 注意力加权
        attention_weights = self.attention(concat_feat)
        fused_feat = high_feat * attention_weights + low_feat_aligned * (1 - attention_weights)

        return fused_feat


# ==================== 第六部分:增强版 ResNeXt50 ====================

class EnhancedResNeXt50(nn.Module):
    """
    增强版 ResNeXt50：融合五个架构创新（修复版）

    创新点配置：
    - 创新点 1：双路径瓶颈结构（保留）
    - 创新点 2：层间特征融合（保留）
    - 创新点 3：动态激活函数 Swish（保留）
    - 创新点 4：空间注意力增强（新增）
    - 创新点 5：多尺度空洞卷积（新增，已修复通道整除问题）

    【关键修复】层间特征融合的通道配置：
    - layer2 输出：512 通道
    - layer3 输出：1024 通道
    - layer4 输出：2048 通道
    """

    def __init__(self, num_classes=4):
        super(EnhancedResNeXt50, self).__init__()

        # 层配置
        layers_config = [
            (64, 64, 3),  # layer[0]: conv1_x -> 输出 256 通道
            (256, 128, 4),  # layer[1]: conv2_x -> 输出 512 通道
            (512, 256, 6),  # layer[2]: conv3_x -> 输出 1024 通道
            (1024, 512, 3)  # layer[3]: conv4_x -> 输出 2048 通道
        ]

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.swish = Swish(beta=1.0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建层（使用增强版双路径瓶颈块）
        # 在深层（layer3, layer4）启用空间注意力和多尺度卷积
        self.layer1 = self._make_layer(layers_config[0], block_idx=0,
                                       use_spatial_attention=False, use_multi_scale=False)
        self.layer2 = self._make_layer(layers_config[1], block_idx=1, stride=2,
                                       use_spatial_attention=False, use_multi_scale=False)
        self.layer3 = self._make_layer(layers_config[2], block_idx=2, stride=2,
                                       use_spatial_attention=True, use_multi_scale=True)
        self.layer4 = self._make_layer(layers_config[3], block_idx=3, stride=2,
                                       use_spatial_attention=True, use_multi_scale=True)

        # 创新点 2：层间特征融合（修复通道配置）
        self.fusion23 = InterLayerFusion(512, 1024)  # layer2 (512) -> layer3 (1024)
        self.fusion34 = InterLayerFusion(1024, 2048)  # layer3 (1024) -> layer4 (2048)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类头
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _make_layer(self, config, block_idx, stride=1, use_spatial_attention=False, use_multi_scale=False):
        """构建层"""
        inplanes, planes, blocks = config

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(EnhancedDualPathBottleneck(inplanes, planes, stride, downsample,
                                                 use_spatial_attention=use_spatial_attention,
                                                 use_multi_scale=use_multi_scale))

        for _ in range(1, blocks):
            layers.append(EnhancedDualPathBottleneck(planes * 4, planes,
                                                     use_spatial_attention=use_spatial_attention,
                                                     use_multi_scale=use_multi_scale))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish(x)
        x = self.maxpool(x)

        # Layer 1
        x1 = self.layer1(x)

        # Layer 2
        x2 = self.layer2(x1)

        # Layer 3（与 Layer 2 融合）
        x3 = self.layer3(x2)
        x3_fused = self.fusion23(x2, x3)

        # Layer 4（与 Layer 3 融合）
        x4 = self.layer4(x3_fused)
        x4_fused = self.fusion34(x3, x4)

        # 全局平均池化
        x = self.avgpool(x4_fused)
        x = torch.flatten(x, 1)

        # 分类头
        x = self.fc(x)

        return x



# ==================== P-CV v2 reproduction pipeline ====================

EXPECTED_CLASSES = (
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
)
EXPECTED_TRAIN_IMAGES = 10240
EXPECTED_FOLDS = 5
EXPECTED_VAL_PER_FOLD = 2048
EXPECTED_TRAIN_PER_FOLD = 8192
EXPECTED_PER_CLASS_TOTAL = 2560
EXPECTED_PER_CLASS_VAL = 512
EXPECTED_PER_CLASS_TRAIN = 2048


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed major RNGs.

    We always seed Python, NumPy and PyTorch. Strict deterministic kernels are
    opt-in because they can reduce performance and may reject unsupported ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep cuDNN benchmarking disabled (the PyTorch default) so algorithm
    # selection is stable rather than benchmark-dependent.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)

    torch.use_deterministic_algorithms(bool(deterministic))


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy inside DataLoader workers from PyTorch worker seed."""
    del worker_id  # worker identity is already encoded in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return legacy training augmentation and deterministic validation preprocessing."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_transform, val_transform


def read_fold_manifest(path: Path) -> List[dict]:
    required = {
        "relative_path",
        "class_name",
        "filename",
        "sha256",
        "group_id",
        "group_size",
        "group_reason",
        "val_fold",
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Fold manifest is empty: {path}")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Fold manifest is missing columns: {sorted(missing)}")
    return rows


def _safe_relative_path(text: str) -> Path:
    rel = Path(text)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe relative path in manifest: {text!r}")
    return rel


def validate_frozen_manifest(
    rows: Sequence[dict],
    train_root: Path,
    verify_files: bool = True,
    verify_hashes: bool = False,
) -> Dict[str, int]:
    """Validate the frozen P-CV v2 manifest before training.

    This function never creates or changes folds. It only verifies the frozen
    assignment and, optionally, the backing image files.
    """
    if len(rows) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_IMAGES} train rows, got {len(rows)}."
        )

    classes = sorted({r["class_name"] for r in rows})
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected classes: {classes}")

    rel_paths = [r["relative_path"] for r in rows]
    if len(set(rel_paths)) != len(rel_paths):
        raise ValueError("Duplicate relative_path entries in fold manifest.")

    overall = Counter(r["class_name"] for r in rows)
    if any(overall[c] != EXPECTED_PER_CLASS_TOTAL for c in EXPECTED_CLASSES):
        raise ValueError(f"Unexpected overall class counts: {dict(overall)}")

    group_folds: Dict[str, set] = defaultdict(set)
    sha_folds: Dict[str, set] = defaultdict(set)

    for r in rows:
        fold = int(r["val_fold"])
        if not 1 <= fold <= EXPECTED_FOLDS:
            raise ValueError(f"Invalid val_fold={fold}: {r['relative_path']}")
        group_folds[r["group_id"]].add(fold)
        sha_folds[r["sha256"]].add(fold)

        rel = _safe_relative_path(r["relative_path"])
        if rel.parts[0] != r["class_name"]:
            raise ValueError(
                f"Manifest path/class mismatch: {r['relative_path']} vs {r['class_name']}"
            )

        if verify_files or verify_hashes:
            image_path = train_root / rel
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing train image: {image_path}")
            if verify_hashes and sha256_file(image_path) != r["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {image_path}")

    bad_groups = {g: fs for g, fs in group_folds.items() if len(fs) != 1}
    if bad_groups:
        raise ValueError(
            f"P-CV group leakage in manifest: {list(bad_groups.items())[:5]}"
        )

    bad_sha = {h: fs for h, fs in sha_folds.items() if len(fs) != 1}
    if bad_sha:
        raise ValueError(
            f"Exact duplicate SHA-256 crosses folds: {list(bad_sha.items())[:5]}"
        )

    for fold in range(1, EXPECTED_FOLDS + 1):
        val_rows = [r for r in rows if int(r["val_fold"]) == fold]
        train_rows = [r for r in rows if int(r["val_fold"]) != fold]

        if len(val_rows) != EXPECTED_VAL_PER_FOLD:
            raise ValueError(f"Fold {fold}: unexpected val size {len(val_rows)}")
        if len(train_rows) != EXPECTED_TRAIN_PER_FOLD:
            raise ValueError(f"Fold {fold}: unexpected train size {len(train_rows)}")

        val_counts = Counter(r["class_name"] for r in val_rows)
        train_counts = Counter(r["class_name"] for r in train_rows)
        if any(val_counts[c] != EXPECTED_PER_CLASS_VAL for c in EXPECTED_CLASSES):
            raise ValueError(f"Fold {fold}: unexpected val class counts {dict(val_counts)}")
        if any(train_counts[c] != EXPECTED_PER_CLASS_TRAIN for c in EXPECTED_CLASSES):
            raise ValueError(
                f"Fold {fold}: unexpected train class counts {dict(train_counts)}"
            )

        train_groups = {r["group_id"] for r in train_rows}
        val_groups = {r["group_id"] for r in val_rows}
        if train_groups & val_groups:
            raise ValueError(f"Fold {fold}: group leakage between train and validation.")

        train_sha = {r["sha256"] for r in train_rows}
        val_sha = {r["sha256"] for r in val_rows}
        if train_sha & val_sha:
            raise ValueError(
                f"Fold {fold}: exact duplicate leakage between train and validation."
            )

    return {c: overall[c] for c in EXPECTED_CLASSES}


class ManifestImageDataset(Dataset):
    """Image dataset driven exclusively by rows from the frozen fold manifest."""

    def __init__(
        self,
        train_root: Path,
        rows: Sequence[dict],
        class_to_idx: Dict[str, int],
        transform,
    ) -> None:
        self.train_root = Path(train_root)
        self.rows = list(rows)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.train_root / _safe_relative_path(row["relative_path"])
        # torchvision ImageFolder's default PIL loader also converts to RGB.
        # Reusing default_loader preserves the legacy 3-channel model input.
        image = default_loader(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        target = self.class_to_idx[row["class_name"]]
        return image, target


def create_model(num_classes: int) -> EnhancedResNeXt50:
    """Single model factory used by every fold."""
    return EnhancedResNeXt50(num_classes=num_classes)


def model_parameter_stats(model: nn.Module) -> dict:
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "spatial_attention_params": int(
            sum(
                p.numel()
                for name, p in model.named_parameters()
                if "spatial_attention" in name
            )
        ),
        "multi_scale_dilated_params": int(
            sum(
                p.numel()
                for name, p in model.named_parameters()
                if "dilated_conv" in name
            )
        ),
    }


def architecture_fingerprint(model: nn.Module) -> str:
    """Hash parameter/buffer names, shapes and dtypes (not trained values)."""
    parts = []
    for name, tensor in model.state_dict().items():
        parts.append(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def build_dataloaders(
    train_root: Path,
    manifest_rows: Sequence[dict],
    fold: int,
    class_to_idx: Dict[str, int],
    batch_size: int,
    num_workers: int,
    loader_seed: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, List[dict], List[dict]]:
    train_transform, val_transform = build_transforms()

    train_rows = [r for r in manifest_rows if int(r["val_fold"]) != fold]
    val_rows = [r for r in manifest_rows if int(r["val_fold"]) == fold]

    train_dataset = ManifestImageDataset(
        train_root=train_root,
        rows=train_rows,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )
    val_dataset = ManifestImageDataset(
        train_root=train_root,
        rows=val_rows,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(loader_seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    # Validation is intentionally not shuffled and uses no random transforms.
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, train_rows, val_rows


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty validation set.")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    labels = list(range(num_classes))
    accuracy = float(np.mean(y_true == y_pred))
    precision = float(
        precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
    )
    recall = float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
    }


def save_confusion_matrix(
    cm: np.ndarray,
    class_labels: Sequence[str],
    save_path: Path,
    title: str,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_split_snapshot(path: Path, train_rows: Sequence[dict], val_rows: Sequence[dict]) -> None:
    fields = [
        "role",
        "relative_path",
        "class_name",
        "sha256",
        "group_id",
        "group_reason",
        "val_fold",
    ]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for role, rows in (("train", train_rows), ("validation", val_rows)):
            for r in rows:
                writer.writerow({
                    "role": role,
                    "relative_path": r["relative_path"],
                    "class_name": r["class_name"],
                    "sha256": r["sha256"],
                    "group_id": r["group_id"],
                    "group_reason": r["group_reason"],
                    "val_fold": r["val_fold"],
                })


def save_json(path: Path, obj) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def environment_info(device: torch.device) -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device_requested_or_selected": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["gpu_total_memory_bytes"] = int(props.total_memory)
    return info


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device_arg}) but CUDA is unavailable.")
    return device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    bar = tqdm(loader, file=sys.stdout, desc=f"train epoch[{epoch}/{epochs}]")
    for batch_idx, (images, labels) in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Preserve legacy optimizer semantics rather than introducing
        # set_to_none=True as an unrelated optimization change.
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_idx}.")
        loss.backward()
        optimizer.step()

        batch_size_actual = labels.size(0)
        total_loss += float(loss.item()) * batch_size_actual
        total_samples += batch_size_actual
        bar.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise RuntimeError("Training loop processed zero samples.")
    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int] = None,
) -> Tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_pred: List[int] = []
    all_true: List[int] = []

    bar = tqdm(loader, file=sys.stdout, desc="validation")
    for batch_idx, (images, labels) in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss at batch {batch_idx}.")

        pred = torch.argmax(outputs, dim=1)
        batch_size_actual = labels.size(0)
        total_loss += float(loss.item()) * batch_size_actual
        total_samples += batch_size_actual
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("Validation loop processed zero samples.")

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return total_loss / total_samples, metrics, cm, y_true, y_pred



def run_fold(
    fold: int,
    args: argparse.Namespace,
    manifest_rows: Sequence[dict],
    class_to_idx: Dict[str, int],
    run_root: Path,
    device: torch.device,
    manifest_sha256: str,
) -> dict:
    if not 1 <= fold <= EXPECTED_FOLDS:
        raise ValueError(f"Invalid fold: {fold}")

    # CRITICAL: every fold starts from a fresh model and fresh optimizer/scheduler.
    # The model seed is reset to the same base seed so fold differences are not
    # confounded by different initial weights.
    seed_everything(args.seed, deterministic=args.deterministic)
    loader_seed = args.seed + fold

    fold_dir = run_root / f"fold_{fold}"
    if fold_dir.exists() and any(fold_dir.iterdir()):
        raise FileExistsError(
            f"Fold output directory is not empty: {fold_dir}. "
            "Use a new --run-name to avoid accidental overwrite."
        )
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_rows, val_rows = build_dataloaders(
        train_root=Path(args.data_root) / "train",
        manifest_rows=manifest_rows,
        fold=fold,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loader_seed=loader_seed,
        pin_memory=(device.type == "cuda"),
    )
    write_split_snapshot(fold_dir / "split_snapshot.csv", train_rows, val_rows)

    model = create_model(num_classes=len(class_to_idx)).to(device)
    params = model_parameter_stats(model)
    arch_fp = architecture_fingerprint(model)
    save_json(fold_dir / "model_info.json", {**params, "architecture_fingerprint": arch_fp})

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    loss_function = nn.CrossEntropyLoss()

    print("=" * 78)
    print(f"Fold {fold}/{EXPECTED_FOLDS}")
    print(f"Train samples: {len(train_rows):,} | Validation samples: {len(val_rows):,}")
    print(f"Fresh model parameters: {params['total_params']:,}")
    print(f"Device: {device}")
    print("=" * 78)

    best_state = {
        "accuracy": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "kappa": 0.0,
        "val_loss": None,
        "epoch": 0,
    }

    metrics_path = fold_dir / "metrics.csv"
    metric_fields = [
        "epoch",
        "lr_used",
        "next_lr",
        "train_loss",
        "val_loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "kappa",
        "best_accuracy_so_far",
        "best_epoch_so_far",
    ]

    effective_epochs = 1 if args.smoke_test else args.epochs
    max_train_batches = args.smoke_train_batches if args.smoke_test else None
    max_val_batches = args.smoke_val_batches if args.smoke_test else None

    with metrics_path.open("w", newline="", encoding="utf-8-sig") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        writer.writeheader()

        for epoch in range(1, effective_epochs + 1):
            lr_used = float(optimizer.param_groups[0]["lr"])

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                device=device,
                epoch=epoch,
                epochs=effective_epochs,
                max_batches=max_train_batches,
            )

            val_loss, metrics, cm, _, _ = validate_one_epoch(
                model=model,
                loader=val_loader,
                loss_function=loss_function,
                device=device,
                num_classes=len(class_to_idx),
                max_batches=max_val_batches,
            )

            # Preserve the legacy schedule: one ExponentialLR step per epoch.
            scheduler.step()
            next_lr = float(optimizer.param_groups[0]["lr"])

            is_best = metrics["accuracy"] > best_state["accuracy"]
            if is_best:
                best_state = {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "kappa": metrics["kappa"],
                    "val_loss": val_loss,
                    "epoch": epoch,
                }
                # Save only model weights for the best epoch. Keeping optimizer
                # moments in every checkpoint would roughly triple checkpoint size
                # for this ~56.8M-parameter Adam model and add substantial I/O time.
                torch.save(model.state_dict(), fold_dir / "best_model_state.pth")
                save_confusion_matrix(
                    cm,
                    class_labels=list(class_to_idx.keys()),
                    save_path=fold_dir / "best_confusion_matrix.png",
                    title=f"Fold {fold} Best Confusion Matrix (epoch {epoch})",
                )

            # Optional lightweight periodic model-state snapshot. This does not
            # change training and is disabled by default to avoid large per-epoch I/O.
            if args.periodic_save_every > 0 and epoch % args.periodic_save_every == 0:
                torch.save(model.state_dict(), fold_dir / "periodic_model_state.pth")

            row = {
                "epoch": epoch,
                "lr_used": lr_used,
                "next_lr": next_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision"],
                "recall_macro": metrics["recall"],
                "f1_macro": metrics["f1"],
                "kappa": metrics["kappa"],
                "best_accuracy_so_far": best_state["accuracy"],
                "best_epoch_so_far": best_state["epoch"],
            }
            writer.writerow(row)
            metrics_file.flush()

            print(
                f"[Fold {fold} epoch {epoch}/{effective_epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
                f"kappa={metrics['kappa']:.4f} "
                f"best={best_state['accuracy']:.4f}@{best_state['epoch']}"
            )

    # Final weights are saved once after the fold completes.
    torch.save(model.state_dict(), fold_dir / "final_model_state.pth")
    save_json(fold_dir / "best_metrics.json", best_state)
    return {"fold": fold, **best_state}


def summarize_folds(results: Sequence[dict]) -> dict:
    if not results:
        raise ValueError("No fold results to summarize.")
    metric_names = ["accuracy", "precision", "recall", "f1", "kappa"]
    summary = {"n_folds": len(results), "fold_results": list(results)}
    for name in metric_names:
        values = np.asarray([r[name] for r in results], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values, ddof=0)),
            "values": [float(x) for x in values],
        }
    return summary


def create_run_root(base_output_dir: Path, run_name: Optional[str], smoke_test: bool) -> Path:
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        prefix = "smoke" if smoke_test else "run"
        run_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = base_output_dir / run_name
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_root}. "
            "Choose a different --run-name."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


# =============================================================================
# 共享运行入口：smoke_test.py 和 train_5fold.py 都调用这个函数
# =============================================================================

def run_configured_experiment(
    *,
    data_root,
    output_dir,
    run_name,
    device_arg,
    fold,
    batch_size,
    epochs,
    learning_rate,
    lr_gamma,
    seed,
    num_workers,
    deterministic,
    verify_hashes,
    periodic_save_every,
    smoke_test,
    smoke_train_batches,
    smoke_val_batches,
    caller_script=None,
):
    data_root = Path(data_root).expanduser().resolve()
    train_root = data_root / "train"
    test_root = data_root / "test"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"找不到 train 目录：{train_root}\n"
            "请先运行 data_prepare_check.py，并检查 DATA_ROOT。"
        )
    if not test_root.is_dir():
        raise FileNotFoundError(
            f"找不到 test 目录：{test_root}\n"
            "请先运行 data_prepare_check.py。"
        )

    manifest_path = Path(__file__).resolve().parent / "pcv_v2_folds_seed42.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到固定 fold 文件：{manifest_path}\n"
            "请确保 pcv_v2_folds_seed42.csv 和三个 .py 文件在同一目录。"
        )

    if batch_size <= 0 or epochs <= 0:
        raise ValueError("batch_size 和 epochs 必须 > 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须 > 0")
    if not (0.0 < lr_gamma <= 1.0):
        raise ValueError("lr_gamma 必须在 (0, 1]")
    if fold not in {0, 1, 2, 3, 4, 5}:
        raise ValueError("fold 必须是 0~5；0 表示依次运行全部五折")
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, batch_size if batch_size > 1 else 0, 8)
    if num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")

    args = SimpleNamespace(
        data_root=str(data_root),
        batch_size=int(batch_size),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        lr_gamma=float(lr_gamma),
        seed=int(seed),
        num_workers=int(num_workers),
        device=str(device_arg),
        deterministic=bool(deterministic),
        verify_hashes=bool(verify_hashes),
        periodic_save_every=int(periodic_save_every),
        smoke_test=bool(smoke_test),
        smoke_train_batches=int(smoke_train_batches),
        smoke_val_batches=int(smoke_val_batches),
    )

    device = resolve_device(args.device)
    manifest_rows = read_fold_manifest(manifest_path)
    validate_frozen_manifest(
        manifest_rows,
        train_root=train_root,
        verify_files=True,
        verify_hashes=args.verify_hashes,
    )

    class_names = sorted({r["class_name"] for r in manifest_rows})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    if len(class_to_idx) != 4:
        raise ValueError(f"Expected 4 classes, got {class_to_idx}")

    manifest_sha = sha256_file(manifest_path)
    run_root = create_run_root(
        Path(output_dir).expanduser(),
        run_name,
        smoke_test,
    )

    # 留存本次真正使用的代码/manifest，方便以后排查。
    shutil.copy2(manifest_path, run_root / "fold_manifest_used.csv")
    shutil.copy2(Path(__file__).resolve(), run_root / "dmsa_core_used.py")
    if caller_script is not None:
        caller_path = Path(caller_script).resolve()
        if caller_path.is_file() and caller_path != Path(__file__).resolve():
            shutil.copy2(caller_path, run_root / "entry_script_used.py")

    save_json(run_root / "class_to_idx.json", class_to_idx)
    save_json(run_root / "environment.json", environment_info(device))
    save_json(
        run_root / "run_config.json",
        {
            **vars(args),
            "fold_requested": fold,
            "data_root": str(data_root),
            "fold_manifest": str(manifest_path),
            "fold_manifest_sha256": manifest_sha,
            "device_resolved": str(device),
            "protocol": "P-CV v2 FINAL",
            "test_policy": "sealed; not loaded for model training/selection",
            "entry_script": str(Path(caller_script).resolve()) if caller_script else str(Path(__file__).resolve()),
        },
    )

    print("P-CV v2 manifest validation: PASS")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"Run directory: {run_root}")
    print("Original provided test/ remains sealed and is not used for CV training.")

    folds = list(range(1, EXPECTED_FOLDS + 1)) if fold == 0 else [fold]
    results = []
    for current_fold in folds:
        results.append(
            run_fold(
                fold=current_fold,
                args=args,
                manifest_rows=manifest_rows,
                class_to_idx=class_to_idx,
                run_root=run_root,
                device=device,
                manifest_sha256=manifest_sha,
            )
        )

    summary = summarize_folds(results)
    save_json(run_root / "cv_summary.json", summary)

    summary_csv = run_root / "cv_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "fold", "best_epoch", "accuracy", "precision",
            "recall", "f1", "kappa", "val_loss",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "fold": r["fold"],
                "best_epoch": r["epoch"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "kappa": r["kappa"],
                "val_loss": r["val_loss"],
            })

    print("\nCompleted requested fold(s).")
    for metric in ("accuracy", "precision", "recall", "f1", "kappa"):
        print(
            f"{metric}: {summary[metric]['mean']:.4f} "
            f"± {summary[metric]['std_population']:.4f}"
        )

    if smoke_test:
        print("SMOKE TEST ONLY: 上面的指标没有科研意义。")

    return run_root, summary


# =============================================================================
# 用户配置区 —— PyCharm 运行 smoke_test.py 前只改这里
# =============================================================================

# 必须指向 data_prepare_check.py 最后打印的 Combined Dataset
DATA_ROOT = r"/home/featurize/data/dmsa_pcv2/Combined Dataset"

# Smoke 输出保存位置
OUTPUT_DIR = r"/home/featurize/work/DMSA_PCV2_20260824/output"

# 本地 CPU： "cpu"
# 云 GPU：   "cuda:0"
DEVICE = "cuda:0"

# 本地 CPU 建议 1；云 GPU 正式检查建议 16
SMOKE_BATCH_SIZE = 1

# 本地 CPU 建议 0；云 GPU 可设 8（也可按机器调整）
NUM_WORKERS = 0

# Smoke 只跑 Fold 1
FOLD = 1

# 只跑很少 batch，不看准确率
SMOKE_TRAIN_BATCHES = 1
SMOKE_VAL_BATCHES = 1

# 每次换一个名字，避免覆盖
RUN_NAME = "smoke_fold1"

# 第 1 个脚本如果已做过全量 rehash，这里通常 False 即可
VERIFY_HASHES_AGAIN = False

# =============================================================================
# 用户配置区结束
# =============================================================================


def main():
    print("=" * 78)
    print("DMSA-Net | Smoke Test")
    print("=" * 78)
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"DEVICE    = {DEVICE}")
    print(f"BATCH     = {SMOKE_BATCH_SIZE}")
    print()

    run_configured_experiment(
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        device_arg=DEVICE,
        fold=FOLD,
        batch_size=SMOKE_BATCH_SIZE,
        epochs=1,  # smoke 固定只跑 1 个 epoch 框架
        learning_rate=0.001,
        lr_gamma=0.96,
        seed=42,
        num_workers=NUM_WORKERS,
        deterministic=False,
        verify_hashes=VERIFY_HASHES_AGAIN,
        periodic_save_every=0,
        smoke_test=True,
        smoke_train_batches=SMOKE_TRAIN_BATCHES,
        smoke_val_batches=SMOKE_VAL_BATCHES,
        caller_script=__file__,
    )


if __name__ == "__main__":
    main()
