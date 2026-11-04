#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DMSA-Net 最终实验 / 第 1 步：完整 10,240 train 固定训练 126 epochs
===============================================================================

【重要：这是五折 P-CV v2 完成后的 final-model training，不是新的调参实验。】

五折结果已经在 sealed test 完全未参与的情况下冻结。
根据五个 folds 在“相同 epoch”上的平均 validation accuracy：
    epoch 126 与 epoch 150 并列最高；
并且五折各自 best epoch 的均值为 124.8。

因此本最终实验预先固定：
    FINAL_EPOCHS = 126

这个数字在运行 sealed test 之前已经确定。
后面不允许因为 test 结果好/坏，再把 126 改成 120、150、200 后重新测试。

本文件只读取：
    Combined Dataset/train/
    pcv_v2_folds_seed42.csv

本文件【不会读取 test/，也不会读取 sealed test manifest】。
训练完成后只生成 final model，不计算 test 指标。

随后单独运行：
    sealed_test_once.py

这样把“最终训练”和“最终测试”物理分开，避免在训练阶段误碰 test。

===============================================================================
PyCharm / 云 GPU：你只需要改下面的“用户配置区”
===============================================================================

需要安装：
    torch
    torchvision
    numpy
    scikit-learn
    matplotlib
    tqdm
    Pillow

普通依赖可用清华镜像：
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scikit-learn matplotlib tqdm Pillow

云 GPU 的 torch/torchvision：
    优先使用云平台已经配置正确、torch.cuda.is_available()==True 的版本，
    不要为了统一清华源而盲目覆盖 GPU PyTorch。

检查：
    python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

===============================================================================
中断恢复
===============================================================================

你之前担心长时间训练中途断掉，所以本代码额外提供【epoch 边界恢复】。

默认：
    每 5 epochs 覆盖保存一次 resume_checkpoint.pth

如果中断：
1. 不要删除原运行目录；
2. 把 RESUME_FROM 改成该目录下 resume_checkpoint.pth；
3. 再点 Run；
4. 程序会恢复 model / Adam / scheduler / RNG / DataLoader generator 状态，
   从已保存 epoch 的下一轮继续。

注意：
    resume checkpoint 包含 Adam 状态，会比较大。
    它只用于容灾，不参与模型选择。
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


# =============================================================================
# 最终实验固定常量（不要因为 test 结果修改）
# =============================================================================

EXPECTED_MANIFEST_SHA256 = "51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd"
EXPECTED_ARCHITECTURE_FINGERPRINT = "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975"

FINAL_EPOCHS = 126
BATCH_SIZE = 16
LEARNING_RATE = 0.001
LR_GAMMA = 0.96
SEED = 44
NUM_CLASSES = 4

# =============================================================================
# 用户配置区 —— 正式云 GPU 运行前主要改这里
# =============================================================================

# 1) 指向 data_prepare_check.py 生成的 Combined Dataset。
# 目录下面必须有 train/ 和 test/，但本文件只读取 train/。
DATA_ROOT = r"/home/featurize/work/Combined_Dataset"

# 2) 最终实验输出根目录。
OUTPUT_DIR = r"/home/featurize/work/AIC_train_again/g2"

# 3) 每次正式 final training 使用唯一名称。
RUN_NAME = "dmsa_final_train_seed44_epoch126"

# 4) 正式云 GPU 通常为 cuda:0。
DEVICE = "cuda:0"

# 5) 与已完成正式五折保持一致，五折实际 run_config 使用 6 workers。
NUM_WORKERS = 6

# 6) 第一次 final training 建议 True：重新核对 10,240 train 的 SHA-256。
VERIFY_TRAIN_HASHES = True

# 7) 容灾 checkpoint：每多少个 epoch 覆盖保存一次。
#    5 表示最多损失约 4 个 epoch 的训练时间。
RESUME_SAVE_EVERY = 5

# 8) 新训练保持空字符串。
#    如果训练中断，把它改成：
#    r"/你的输出目录/dmsa_final_train_seed42_epoch126/resume_checkpoint.pth"
RESUME_FROM = ""

# 9) 与五折一致，严格 deterministic kernels 保持 False。
DETERMINISTIC = False

# =============================================================================
# 用户配置区结束
# =============================================================================


def _check_final_protocol() -> None:
    expected = {
        "FINAL_EPOCHS": 126,
        "BATCH_SIZE": 16,
        "LEARNING_RATE": 0.001,
        "LR_GAMMA": 0.96,
        "SEED": 44,
        "NUM_CLASSES": 4,
    }
    actual = {
        "FINAL_EPOCHS": FINAL_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "LR_GAMMA": LR_GAMMA,
        "SEED": SEED,
        "NUM_CLASSES": NUM_CLASSES,
    }
    if actual != expected:
        raise RuntimeError(
            "最终协议固定参数被修改。\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            "如果未来要做新实验，请另存代码，不要覆盖本 final protocol。"
        )
    if RESUME_SAVE_EVERY <= 0:
        raise ValueError("RESUME_SAVE_EVERY 必须 > 0。")


def _atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer tensor states to the active model device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rewrite_history_to_epoch(history_path: Path, max_epoch: int) -> None:
    """On resume, remove CSV rows newer than the last durable checkpoint."""
    history_path = Path(history_path)
    if not history_path.is_file():
        return

    with history_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r["epoch"]) <= max_epoch]

    fields = ["epoch", "lr_used", "next_lr", "train_loss"]
    with history_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _build_full_train_loader(
    train_root: Path,
    manifest_rows: Sequence[dict],
    class_to_idx: Dict[str, int],
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
    pin_memory: bool,
) -> DataLoader:
    train_transform, _ = build_transforms()

    dataset = ManifestImageDataset(
        train_root=train_root,
        rows=manifest_rows,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )


def _save_train_manifest_snapshot(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "relative_path",
        "class_name",
        "sha256",
        "group_id",
        "group_reason",
        "val_fold",
    ]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main():
    _check_final_protocol()

    script_dir = Path(__file__).resolve().parent
    manifest_path = script_dir / "pcv_v2_folds_seed42.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到 {manifest_path.name}。请把它和本脚本放在同一目录。"
        )

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "P-CV v2 manifest SHA-256 与五折正式实验使用的版本不一致。\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual:   {manifest_sha}"
        )

    data_root = Path(DATA_ROOT).expanduser().resolve()
    train_root = data_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"找不到 train：{train_root}\n"
            "请修改 DATA_ROOT，使其指向 Combined Dataset。"
        )

    # IMPORTANT: final training deliberately does not construct or read data_root/test.
    manifest_rows = read_fold_manifest(manifest_path)
    validate_frozen_manifest(
        manifest_rows,
        train_root=train_root,
        verify_files=True,
        verify_hashes=VERIFY_TRAIN_HASHES,
    )

    class_names = sorted({r["class_name"] for r in manifest_rows})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    if len(class_to_idx) != NUM_CLASSES:
        raise RuntimeError(f"Expected 4 classes, got {class_to_idx}")

    device = resolve_device(DEVICE)
    run_dir = Path(OUTPUT_DIR).expanduser().resolve() / RUN_NAME
    resume_path = Path(RESUME_FROM).expanduser().resolve() if RESUME_FROM.strip() else None

    if resume_path is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"最终训练输出目录已存在且非空：{run_dir}\n"
                "为了避免覆盖正式结果，请换 RUN_NAME；如果是中断恢复，请填写 RESUME_FROM。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not resume_path.is_file():
            raise FileNotFoundError(f"找不到 resume checkpoint：{resume_path}")
        if resume_path.parent != run_dir:
            raise RuntimeError(
                "RESUME_FROM 必须位于当前 RUN_NAME 对应的 run_dir 内，"
                "避免把不同实验的 checkpoint 混用。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED, deterministic=DETERMINISTIC)

    model = create_model(NUM_CLASSES).to(device)
    model_info = {
        **model_parameter_stats(model),
        "architecture_fingerprint": architecture_fingerprint(model),
    }
    if model_info["architecture_fingerprint"] != EXPECTED_ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            "DMSA-Net architecture fingerprint 与正式五折不一致。\n"
            f"Expected: {EXPECTED_ARCHITECTURE_FINGERPRINT}\n"
            f"Actual:   {model_info['architecture_fingerprint']}"
        )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    loss_function = nn.CrossEntropyLoss()

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    train_loader = _build_full_train_loader(
        train_root=train_root,
        manifest_rows=manifest_rows,
        class_to_idx=class_to_idx,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        generator=loader_generator,
        pin_memory=(device.type == "cuda"),
    )

    if len(train_loader.dataset) != 10240:
        raise RuntimeError(f"Final train dataset 应为 10240，实际 {len(train_loader.dataset)}")

    history_path = run_dir / "train_history.csv"
    resume_checkpoint_path = run_dir / "resume_checkpoint.pth"
    final_model_path = run_dir / f"final_model_epoch{FINAL_EPOCHS}.pth"

    start_epoch = 1

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")

        required = {
            "epoch_completed", "final_epochs", "manifest_sha256",
            "architecture_fingerprint", "class_to_idx", "model_state_dict",
            "optimizer_state_dict", "scheduler_state_dict",
            "rng_state", "loader_generator_state",
        }
        missing = required - set(checkpoint)
        if missing:
            raise RuntimeError(f"Resume checkpoint 缺少字段：{sorted(missing)}")

        if int(checkpoint["final_epochs"]) != FINAL_EPOCHS:
            raise RuntimeError("Resume checkpoint 的 FINAL_EPOCHS 与当前协议不一致。")
        if checkpoint["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Resume checkpoint 的 manifest 与当前正式 manifest 不一致。")
        if checkpoint["architecture_fingerprint"] != EXPECTED_ARCHITECTURE_FINGERPRINT:
            raise RuntimeError("Resume checkpoint 的模型结构 fingerprint 不一致。")
        if checkpoint["class_to_idx"] != class_to_idx:
            raise RuntimeError("Resume checkpoint 的 class_to_idx 不一致。")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        _restore_rng_state(checkpoint["rng_state"])

        completed = int(checkpoint["epoch_completed"])
        if not (0 <= completed < FINAL_EPOCHS):
            raise RuntimeError(f"非法 epoch_completed={completed}")
        start_epoch = completed + 1
        _rewrite_history_to_epoch(history_path, completed)

        print(f"[RESUME] 已恢复到 epoch {completed}，将从 epoch {start_epoch} 继续。")

    else:
        # Fresh formal run snapshot.
        shutil.copy2(Path(__file__).resolve(), run_dir / "final_train_code_used.py")
        shutil.copy2(manifest_path, run_dir / "pcv_v2_folds_used.csv")
        _save_train_manifest_snapshot(run_dir / "full_train_snapshot.csv", manifest_rows)

        save_json(run_dir / "class_to_idx.json", class_to_idx)
        save_json(run_dir / "environment.json", environment_info(device))
        save_json(run_dir / "model_info.json", model_info)
        save_json(
            run_dir / "final_training_protocol.json",
            {
                "protocol": "P-CV v2 -> final full-train model",
                "train_images": 10240,
                "test_policy": "sealed; not loaded by final_train.py",
                "fixed_epoch_source": "5-fold aligned-epoch validation analysis; fixed before sealed test",
                "final_epochs": FINAL_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "lr_gamma": LR_GAMMA,
                "seed": SEED,
                "num_workers": NUM_WORKERS,
                "deterministic": DETERMINISTIC,
                "manifest_sha256": manifest_sha,
                "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
                "resume_save_every": RESUME_SAVE_EVERY,
            },
        )

        with history_path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(
                f, fieldnames=["epoch", "lr_used", "next_lr", "train_loss"]
            ).writeheader()

    print("=" * 78)
    print("DMSA-Net FINAL TRAINING")
    print("=" * 78)
    print(f"Train images : {len(train_loader.dataset):,}")
    print(f"Epochs       : {FINAL_EPOCHS}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Device       : {device}")
    print(f"Parameters   : {model_info['total_params']:,}")
    print(f"Run dir      : {run_dir}")
    print("Sealed test  : NOT LOADED")
    print("=" * 78)

    for epoch in range(start_epoch, FINAL_EPOCHS + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            epoch=epoch,
            epochs=FINAL_EPOCHS,
            max_batches=None,
        )

        scheduler.step()
        next_lr = float(optimizer.param_groups[0]["lr"])

        with history_path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["epoch", "lr_used", "next_lr", "train_loss"]
            )
            w.writerow({
                "epoch": epoch,
                "lr_used": lr_used,
                "next_lr": next_lr,
                "train_loss": train_loss,
            })

        print(
            f"[Final epoch {epoch}/{FINAL_EPOCHS}] "
            f"train_loss={train_loss:.6f} lr={lr_used:.10g} -> {next_lr:.10g}"
        )

        should_save_resume = (
            epoch % RESUME_SAVE_EVERY == 0 or epoch == FINAL_EPOCHS
        )
        if should_save_resume:
            _atomic_torch_save(
                {
                    "checkpoint_type": "epoch-boundary-resume",
                    "epoch_completed": epoch,
                    "final_epochs": FINAL_EPOCHS,
                    "manifest_sha256": manifest_sha,
                    "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
                    "class_to_idx": class_to_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": _capture_rng_state(),
                    "loader_generator_state": loader_generator.get_state(),
                },
                resume_checkpoint_path,
            )
            print(f"[CHECKPOINT] resume 已保存：epoch {epoch}")

    # Save the only model used for sealed test. No validation/best selection here.
    final_payload = {
        "checkpoint_type": "final-model-for-sealed-test",
        "epoch": FINAL_EPOCHS,
        "num_classes": NUM_CLASSES,
        "class_to_idx": class_to_idx,
        "manifest_sha256": manifest_sha,
        "architecture_fingerprint": EXPECTED_ARCHITECTURE_FINGERPRINT,
        "model_parameter_stats": model_info,
        "training_config": {
            "train_images": 10240,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lr_gamma": LR_GAMMA,
            "seed": SEED,
            "epochs": FINAL_EPOCHS,
            "num_workers": NUM_WORKERS,
            "deterministic": DETERMINISTIC,
        },
        "model_state_dict": model.state_dict(),
    }
    _atomic_torch_save(final_payload, final_model_path)

    final_sha = sha256_file(final_model_path)
    save_json(
        run_dir / "FINAL_MODEL_READY.json",
        {
            "status": "ready_for_sealed_test",
            "model_path": str(final_model_path),
            "model_sha256": final_sha,
            "epoch": FINAL_EPOCHS,
            "sealed_test_has_been_evaluated_by_this_script": False,
        },
    )

    print()
    print("=" * 78)
    print("FINAL TRAINING COMPLETE")
    print("=" * 78)
    print(f"Final model : {final_model_path}")
    print(f"SHA-256     : {final_sha}")
    print("下一步：不要改模型/epoch/超参数，运行 sealed_test_once.py。")
    print("=" * 78)


if __name__ == "__main__":
    main()
