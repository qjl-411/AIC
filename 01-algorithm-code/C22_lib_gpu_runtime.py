#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC 提分实验 —— 共享库
===============================================================================

本文件只做一件事：把「安全地启动一次训练」所需的公共逻辑收拢到一处，
供 g1 / g2 / g3 三个脚本复用。

设计铁律（与项目 MEMORY.md 一致）：

  L1. **基线文件零改动**。本库只 import 基线的 `smoke_test.py`（核心实现真源）
      与 `final_train.py`（final protocol），**从不写入、从不修改**它们。

  L2. **sealed test 不参与任何选择**。本库的入口函数**物理上不接收 test 相关参数**，
      也**不会构造 data_root/test**。所有实验只用 manifest 里 val_fold 指定的折内验证集。

  L3. **协议一致性硬校验**。任何实验开跑前必须先通过：
      - train manifest SHA-256 == 51f5b251…edfd
      - 全量模型 architecture fingerprint == 8e542321…69d7
      - 10,240 行 / 四类各 2,560 / 五折各 2,048 val
      任一不符即抛异常退出，绝不"带着疑问继续跑"。

  L4. **一份产物只许经一个写入口落盘**。本库只提供 `write_json()` / `write_csv()`，
      且自动附 `provenance`（脚本 SHA、时间、Git 状态）。禁止绕过它直接 open().write()。

  L5. **结果里必须记录"这次到底跑的是什么"**。每次 run 落盘
      `run_config.json`，含 seed / epochs / 消融开关 / manifest SHA / 代码 SHA。

---------------------------------------------------------------------------
本机（无 GPU）能做什么
---------------------------------------------------------------------------

本机是 torch CPU-only，**不能训练**。因此本库提供 `--dry-run`：
在 CPU 上把「数据 → manifest 校验 → 模型构造 → 前向 → 反传 → 指标」整条链路
跑 1~2 个 batch，用来证明脚本本身没有接线错误。

    python gpu_lib.py --smoke           # CPU 上自检本库
    python g2_multi_seed.py --dry-run   # CPU 上自检多种子脚本

真正的训练必须在 GPU 机器上跑。
===============================================================================
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 0. 常量：全部来自项目冻结事实（MEMORY.md 第二节）
# =============================================================================

# ─────────────────────────────────────────────────────────── 固定路径（云机器）
# 用户要求把全部结果固定存在 featurize 工作区的 work/AIC_train_again 下。
# 与项目原有 DMSA_net 代码一致（它用的是 /home/featurize/work/...）。
FIXED_OUT_ROOT = "/home/featurize/work/AIC_train_again"

# 云端数据集位置（用户实测确认；作为 --data-root 的默认值，找不到会明确报错并给提示）
# ⚠️ 基线 `final_train.py:1115` / `smoke_test.py:1480` 里硬编码的 `DATA_ROOT` 指向
#    另一条路径（`.../data/dmsa_pcv2/...`），**与这里不一致**；本包一律以本常量为准。
FIXED_DATA_ROOT = "/home/featurize/work/Combined_Dataset"

# 包版本：每次改动请递增。各脚本启动时会打印这一行，
# 用来一眼判断「云端跑的到底是不是最新包」（见 RUNBOOK §1）。
PKG_VERSION = "2026-09-24.4"

TRAIN_MANIFEST_NAME = "pcv_v2_folds_seed42.csv"
TRAIN_MANIFEST_SHA256 = (
    "51f5b251f9e7cce6c557798ae8fa8e263a6baf58c125524be4e1a1df4bd9edfd"
)
SEALED_TEST_MANIFEST_NAME = "pcv_test_manifest_SEALED.csv"
SEALED_TEST_MANIFEST_SHA256 = (
    "bc8e53e24722167fb3bf0ab468863ed8d0bcaec1341bb52007021802c31126a6"
)
ARCHITECTURE_FINGERPRINT = (
    "8e542321b2886db4db0986dc19d8d8a0984a39c472c086e55ac1fd1db3d76975"
)

EXPECTED_TRAIN_IMAGES = 10240
EXPECTED_FOLDS = 5
EXPECTED_VAL_PER_FOLD = 2048
EXPECTED_TRAIN_PER_FOLD = 8192
EXPECTED_PER_CLASS_TOTAL = 2560
EXPECTED_PER_CLASS_VAL = 512
EXPECTED_PER_CLASS_TRAIN = 2048
EXPECTED_CLASSES = (
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
)

# 基线超参（不可动）
BASE_BATCH_SIZE = 16
BASE_EPOCHS = 200          # 五折 CV 上限，折内按 val accuracy 选 best
FINAL_EPOCHS = 126         # final 全量训练固定轮数
BASE_LR = 0.001
BASE_GAMMA = 0.96
BASE_SEED = 42

# final 全量训练的每轮迭代数：final_train.py 断言 train 恰好 10,240 张，
# batch=16 ⇒ 640 iter/epoch（整除，无余数）。进度显示的"总 step"用它换算，
# 属于**可推导量**，不是新引入的假设。
STEPS_PER_EPOCH_FINAL = EXPECTED_TRAIN_IMAGES // BASE_BATCH_SIZE   # 640
TOTAL_STEPS_FINAL = FINAL_EPOCHS * STEPS_PER_EPOCH_FINAL           # 80,640


# =============================================================================
# 1. 路径解析：输出目录 + 基线代码 + manifest
# =============================================================================

IS_POSIX = (os.name == "posix")


def default_out_root() -> Path:
    """固定输出根目录。

    云机器（Linux）→ `/home/featurize/work/AIC_train_again`（已定死）。
    本机 Windows 自检 → 退回脚本所在目录下的 `_out_local`，
    避免在 C:/E: 盘根目录下真建出 `\\home\\featurize\\...` 这种垃圾目录。
    """
    if IS_POSIX:
        return Path(FIXED_OUT_ROOT)
    return Path(__file__).resolve().parent / "_out_local"


def resolve_out_dir(explicit: Optional[str], tag: str) -> Path:
    """决定某个脚本的输出目录，并确保它存在。

    优先级：
        --out-dir 显式指定  >  固定根目录 / <tag>

    返回**已 resolve 的绝对路径**，调用方必须把它打印出来，
    这样"结果到底存哪"永远是可见的，不靠猜。
    """
    if explicit:
        p = Path(explicit).expanduser()
    else:
        p = default_out_root() / tag
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"无法创建输出目录：{p}\n原因：{e}\n"
            "请检查磁盘是否挂载、是否有写权限，或用 --out-dir 换一个位置。"
        ) from e
    return p.resolve()


def default_data_root() -> str:
    """`--data-root` 的默认值（数据集路径）。"""
    return FIXED_DATA_ROOT


def resolve_data_root(explicit: Optional[str]) -> str:
    """返回数据根目录字符串。显式参数优先。"""
    return explicit if explicit else default_data_root()


def find_baseline_dir(explicit: Optional[str] = None) -> Path:
    """定位含 smoke_test.py + final_train.py + 冻结 manifest 的基线目录。"""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    here = Path(__file__).resolve()
    # 依次向上找，兼容「脚本被拷到云机任意目录」的情况
    for up in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidates.append(up / "DMSA_net")
        candidates.append(up)
        candidates.append(up / "NeuroEvo_AD" / "DMSA_net")
        candidates.append(up.parent / "NeuroEvo_AD" / "DMSA_net")

    candidates.append(Path(r"E:/Code/NeuroEvo/NeuroEvo_AD/DMSA_net"))

    need = ["smoke_test.py", "final_train.py", TRAIN_MANIFEST_NAME]
    tried: List[str] = []
    for c in candidates:
        tried.append(str(c))
        if all((c / n).is_file() for n in need):
            return c

    raise FileNotFoundError(
        "找不到基线目录（需同时包含 smoke_test.py / final_train.py / "
        f"{TRAIN_MANIFEST_NAME}）。\n已尝试：\n  " + "\n  ".join(tried) + "\n"
        "请用 --baseline-dir 显式指定。"
    )


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_module(path: Path, name: str):
    """以 importlib 方式加载模块（避免依赖 sys.path 顺序）。

    注意：`arch_probe.py` 会 `from lib_provenance import ...`，
    因此**必须**先把该文件所在目录插入 sys.path，否则会 ModuleNotFoundError。
    """
    path = Path(path).resolve()
    d = str(path.parent)
    if d not in sys.path:
        sys.path.insert(0, d)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# 2. 协议守卫：任何实验开跑前的硬校验
# =============================================================================

def read_manifest(path: Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def assert_train_manifest(rows: Sequence[dict], *, allow_group_relax: bool = False) -> dict:
    """校验 train manifest 的结构不变量。

    allow_group_relax=True 时**跳过** group 泄漏检查，
    仅供「对照五折」（`c1_cv_matrix.py`）在构造"忽略 group_id 的 NEW 划分"时自检用。
    注意：① 即便如此，**仍强制**要求 group_id 列存在，便于事后对账；
          ② 调用方 `c1_cv_matrix.py` 还会**正向断言** group 跨折数 > 0，
             否则说明这份"对照"其实还是分组的，属假对照。
    """
    if len(rows) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(f"train manifest 行数 {len(rows)} != {EXPECTED_TRAIN_IMAGES}")

    if "group_id" not in rows[0]:
        raise ValueError("train manifest 缺少 group_id 列（对账必需）")

    overall = Counter(r["class_name"] for r in rows)
    for c in EXPECTED_CLASSES:
        if overall[c] != EXPECTED_PER_CLASS_TOTAL:
            raise ValueError(f"类别 {c} 总数 {overall[c]} != {EXPECTED_PER_CLASS_TOTAL}")

    rel = [r["relative_path"] for r in rows]
    if len(set(rel)) != len(rel):
        raise ValueError("train manifest 存在重复 relative_path")

    report = {"rows": len(rows), "per_class": dict(overall), "group_relaxed": bool(allow_group_relax)}

    if not allow_group_relax:
        group_folds: Dict[str, set] = defaultdict(set)
        sha_folds: Dict[str, set] = defaultdict(set)
        for r in rows:
            group_folds[r["group_id"]].add(int(r["val_fold"]))
            sha_folds[r["sha256"]].add(int(r["val_fold"]))
        bad_g = {g: sorted(fs) for g, fs in group_folds.items() if len(fs) != 1}
        if bad_g:
            raise ValueError(
                f"P-CV group 跨折泄漏：{list(bad_g.items())[:5]}"
            )
        bad_s = {h: sorted(fs) for h, fs in sha_folds.items() if len(fs) != 1}
        if bad_s:
            raise ValueError(f"exact 重复跨折：{list(bad_s.items())[:5]}")

    for fold in range(1, EXPECTED_FOLDS + 1):
        val = [r for r in rows if int(r["val_fold"]) == fold]
        trn = [r for r in rows if int(r["val_fold"]) != fold]
        if len(val) != EXPECTED_VAL_PER_FOLD:
            raise ValueError(f"fold {fold} val={len(val)} != {EXPECTED_VAL_PER_FOLD}")
        if len(trn) != EXPECTED_TRAIN_PER_FOLD:
            raise ValueError(f"fold {fold} train={len(trn)} != {EXPECTED_TRAIN_PER_FOLD}")
        vc = Counter(r["class_name"] for r in val)
        tc = Counter(r["class_name"] for r in trn)
        for c in EXPECTED_CLASSES:
            if vc[c] != EXPECTED_PER_CLASS_VAL:
                raise ValueError(f"fold {fold} val 类 {c}={vc[c]} != {EXPECTED_PER_CLASS_VAL}")
            if tc[c] != EXPECTED_PER_CLASS_TRAIN:
                raise ValueError(f"fold {fold} train 类 {c}={tc[c]} != {EXPECTED_PER_CLASS_TRAIN}")
        if not allow_group_relax:
            if {r["group_id"] for r in trn} & {r["group_id"] for r in val}:
                raise ValueError(f"fold {fold} train/val group 泄漏")
            if {r["sha256"] for r in trn} & {r["sha256"] for r in val}:
                raise ValueError(f"fold {fold} train/val exact 泄漏")

    return report


def assert_architecture(model, fp_func) -> str:
    fp = fp_func(model)
    if fp != ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            "architecture fingerprint 与正式基线不一致——禁止用这个模型做对比实验。\n"
            f"Expected: {ARCHITECTURE_FINGERPRINT}\nActual:   {fp}"
        )
    return fp


# =============================================================================
# 3. 溯源与落盘：唯一写入口
# =============================================================================

def _git_state(cwd: Path) -> dict:
    def _run(args: Sequence[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(cwd), capture_output=True,
                text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "dirty": bool(_run(["status", "--porcelain"])),
    }


def provenance(script_path: Path) -> dict:
    return {
        "script": Path(script_path).name,
        "script_sha256": sha256_file(script_path),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cwd": str(Path.cwd()),
    }


def write_json(path: Path, obj, script_path: Path) -> None:
    """唯一的 JSON 写入口；自动注入 provenance。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, dict) and "provenance" not in obj:
        obj = {**obj, "provenance": provenance(script_path)}
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  [write_json] {path}")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """写 CSV。多余字段自动裁掉、缺失字段补空——避免"字典有额外键就崩"。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  [write_csv ] {path}")


# =============================================================================
# 4. 设备探测
# =============================================================================

def probe_device(requested: str) -> Tuple[object, dict]:
    """返回 (torch.device, 能力字典)。CPU 机器上不会静默降级。"""
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "device_name": None,
        "device_total_memory_bytes": None,
        "device_capability": None,
    }
    if torch.cuda.is_available():
        idx = 0
        if ":" in requested:
            idx = int(requested.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        info["device_name"] = props.name
        info["device_total_memory_bytes"] = int(props.total_memory)
        info["device_capability"] = f"{props.major}.{props.minor}"
        return torch.device(f"cuda:{idx}"), info

    if requested.startswith("cuda"):
        raise RuntimeError(
            f"请求了 {requested}，但本机 torch.cuda.is_available()=False。\n"
            "真正训练必须在 GPU 机器上跑；本机请加 --dry-run 只做接线自检。"
        )
    return torch.device("cpu"), info


def reproducible_seed(seed: int, torch_mod, np_mod, random_mod) -> None:
    random_mod.seed(seed)
    np_mod.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)
    if hasattr(torch_mod.backends, "cudnn"):
        torch_mod.backends.cudnn.benchmark = False
        torch_mod.backends.cudnn.deterministic = False


# =============================================================================
# 5. CPU 自检（--dry-run）
# =============================================================================

def cpu_smoke(base_dir: Path, max_batches: int = 1) -> dict:
    """在 CPU 上跑通 数据→前向→反传→指标 整条链路。

    目的只有一个：证明脚本接线正确。**不产生任何科研结论。**
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print("=" * 78)
    print("CPU DRY-RUN（仅验证接线，指标无科研意义）")
    print("=" * 78)

    smoke = load_module(base_dir / "smoke_test.py", "dmsa_baseline_smoke")

    man = base_dir / TRAIN_MANIFEST_NAME
    rep = assert_train_manifest(read_manifest(man))
    print(f"[1/5] manifest 校验 PASS  rows={rep['rows']:,}  "
          f"per_class={rep['per_class']}")

    model = smoke.create_model(num_classes=4)
    fp = assert_architecture(model, smoke.architecture_fingerprint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2/5] 架构校验 PASS  params={n_params:,}  fp={fp[:12]}…")

    # 用 manifest 前若干行构造一个极小 loader（文件真实存在性由 manifest 保证）
    rows = read_manifest(man)[: max(2, max_batches)]
    class_to_idx = {n: i for i, n in enumerate(sorted({r["class_name"] for r in read_manifest(man)}))}
    train_root = None
    _cands = [
        Path(FIXED_DATA_ROOT) / "train",   # 云端实测位置（首选）
        Path(r"E:/Code/NeuroEvo/DMSA_netV2/Alzheimer's MRI Dataset/"
             r"Alzheimer's MRI Dataset/Combined Dataset/train"),   # 本机 Windows
    ]
    for cand in _cands:
        if cand.is_dir():
            train_root = cand
            break
    if train_root is None:
        print("[3/5] 跳过真实数据加载（未找到 train/）—— 已试过：")
        for _c in _cands:
            print(f"        {_c}")
        print("      请对照 RUNBOOK §1 核对数据根目录；本项跳过不影响后面步骤。")
        data_ok = False
    else:
        ds = smoke.ManifestImageDataset(
            train_root=train_root, rows=rows,
            class_to_idx=class_to_idx, transform=smoke.build_transforms()[0],
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        imgs, labels = next(iter(loader))
        print(f"[3/5] 真实数据加载 PASS  batch={tuple(imgs.shape)}  labels={labels.tolist()}")
        data_ok = True

    if not data_ok:
        imgs = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])

    # 前向
    model.train()
    out = model(imgs)
    print(f"[4/5] 前向 PASS  logits={tuple(out.shape)}")

    # 反传
    opt = optim.Adam(model.parameters(), lr=BASE_LR)
    lossf = nn.CrossEntropyLoss()
    opt.zero_grad()
    loss = lossf(out, labels)
    loss.backward()
    opt.step()
    print(f"[5/5] 反传+更新 PASS  loss={float(loss.item()):.6f}")

    print("=" * 78)
    print("CPU DRY-RUN 完成：脚本接线正常。真实训练请在 GPU 机器上运行。")
    print("=" * 78)
    return {"ok": True, "params": n_params, "fingerprint": fp, "loss": float(loss.item())}


# =============================================================================
# 6. 命令行
# =============================================================================

def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--baseline-dir", default=None,
                   help="含 smoke_test.py / final_train.py / 冻结 manifest 的目录")
    p.add_argument("--dry-run", action="store_true",
                   help="仅在 CPU 上验证接线，不训练")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None,
                   help=f"输出目录（默认已固定为 {FIXED_OUT_ROOT}/<tag>）")
    return p


# =============================================================================
# 7. 进度显示：epoch / step / 百分比 / 已用时 / ETA
# =============================================================================

def fmt_hms(seconds: Optional[float]) -> str:
    """秒 → HH:MM:SS。None（无法估计）显示为 --:--:--。"""
    if seconds is None:
        return "--:--:--"
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_tee_sink(log_path):
    """返回一个 sink：既打印到屏幕，也**即时追加**写入日志文件。

    为什么用"追加 + 每行开关文件"：进度行的价值就在于"随时 tail 都能看到最新的"，
    因此宁可牺牲一点 IO，也要保证落盘即时可见（不依赖缓冲区被 flush）。
    """
    lp = Path(log_path)
    lp.parent.mkdir(parents=True, exist_ok=True)

    def _sink(line: str) -> None:
        print(line)
        try:
            with lp.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
        except Exception:
            pass  # 日志写不进去不能把训练搞崩

    return _sink


class ProgressReporter:
    """统一的进度出口：step/总 step、百分比、已用时、ETA。

    - `sink` 是唯一出口（可以是 print，也可以是"打印 + 写日志"的复合函数），
      保证**屏幕上看到的**和**日志里的**是同一份内容，不存在两处口径不一致。
    - 节流：默认最少间隔 `min_interval` 秒才输出一行，避免把日志刷爆
      （tqdm 每个 batch 都会吐一帧，126×640 = 8 万帧，必须节流）。
    - ETA 由**实测速率**外推：`已用时 / 已完成量 × 剩余量`。
      刚开始（完成量=0）无法外推，显示 `--:--:--`，不给假数字。
    """

    def __init__(self, *, label: str = "", total_epochs: int = 0,
                 steps_per_epoch: int = 0, total_units: Optional[int] = None,
                 unit: str = "step", sink=None, min_interval: float = 5.0,
                 base_units: int = 0):
        self.label = label
        self.total_epochs = int(total_epochs or 0)
        self.steps_per_epoch = int(steps_per_epoch or 0)
        if total_units is not None:
            self.total_units = max(1, int(total_units))
        else:
            self.total_units = max(1, self.total_epochs * max(self.steps_per_epoch, 1))
        self.unit = unit
        self.sink = sink if sink is not None else (lambda line: print(line))
        self.min_interval = float(min_interval)
        self.t0 = time.time()
        # 已完成的"上一阶段"单位数：用于把多个连续 run（G1 的 变体×折）
        # 折算成**同一个全局进度**，否则每个 run 都会从 0% 重新数一遍。
        self.base_units = int(base_units or 0)
        self._last_emit = 0.0
        self.epoch = 0
        self.step = 0
        self.loss = None
        self.extra = ""

    def set_base(self, n: int) -> None:
        """设置已完成基数（跨 run 累加用）。"""
        self.base_units = int(n)

    def set(self, *, epoch=None, step=None, loss=None, extra=None) -> None:
        if epoch is not None:
            self.epoch = int(epoch)
        if step is not None:
            self.step = int(step)
        if loss is not None:
            self.loss = loss
        if extra is not None:
            self.extra = str(extra)

    def done_units(self) -> int:
        """已完成的最小工作单位数（用于算百分比与 ETA）。

        = 已完成阶段基数 + 本 run 内已完成数。
        """
        prev_epochs = max(self.epoch - 1, 0)
        return self.base_units + prev_epochs * self.steps_per_epoch + max(self.step, 0)

    def line(self) -> str:
        done = self.done_units()
        total = self.total_units
        pct = 100.0 * done / total
        elapsed = time.time() - self.t0
        if done <= 0:
            eta = None
        elif done >= total:
            eta = 0.0
        else:
            eta = elapsed * (total - done) / done
        parts = ["[进度]"]
        if self.label:
            parts.append(self.label)
        if self.total_epochs:
            parts.append(f"epoch {self.epoch}/{self.total_epochs}")
        if self.steps_per_epoch:
            parts.append(f"本轮 step {self.step}/{self.steps_per_epoch}")
        parts.append(f"{self.unit} {done}/{total}")
        parts.append(f"{pct:5.1f}%")
        parts.append(f"用时 {fmt_hms(elapsed)}")
        parts.append(f"剩余≈{fmt_hms(eta)}")
        if self.loss is not None:
            try:
                parts.append(f"loss={float(self.loss):.6f}")
            except Exception:
                pass
        if self.extra:
            parts.append(self.extra)
        return "  ".join(parts)

    def emit(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_emit) < self.min_interval:
            return
        self._last_emit = now
        self.sink(self.line())


_FRAME_SEP = re.compile(r"[\r\n]")


def stream_run(cmd, log_path, on_frame=None, *, cwd=None, env=None):
    """流式执行子进程：输出**实时**落盘，并逐帧回调。

    ⚠️ 为什么不能用 `subprocess.run(stdout=PIPE)`：
    它把子进程输出**全量缓存在内存**里，等进程退出才拿到 ⇒
      (a) 训练期间日志文件恒为 0 字节，`tail -f` 看不到任何东西（用户会以为卡死）；
      (b) 进程被抢占 / OOM kill 时，缓存里的输出**全部丢失**。
    这里改用 Popen + `read1()` 增量读（真·流式），并逐帧写盘 + flush。

    返回 `(returncode, n_frames)`。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    dec = codecs.getincrementaldecoder("utf-8")("replace")

    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None, env=env,
        )
        buf = ""
        try:
            while True:
                chunk = proc.stdout.read1(8192)   # 有数据就返回，不阻塞等满
                if not chunk:
                    tail = dec.decode(b"", True)
                    if tail:
                        buf += tail
                    break
                buf += dec.decode(chunk)
                while True:
                    m = _FRAME_SEP.search(buf)
                    if not m:
                        break
                    frame = buf[: m.start()]
                    buf = buf[m.end():]
                    if not frame.strip():
                        continue
                    n_frames += 1
                    logf.write(frame + "\n")
                    logf.flush()
                    if on_frame is not None:
                        try:
                            on_frame(frame)
                        except Exception:
                            pass   # 解析进度失败绝不能影响训练
        finally:
            if buf.strip():
                logf.write(buf + "\n")
                logf.flush()
                if on_frame is not None:
                    try:
                        on_frame(buf)
                    except Exception:
                        pass
            try:
                proc.stdout.close()
            except Exception:
                pass
        rc = proc.wait()
    return rc, n_frames


# =============================================================================
# 8. 安全覆盖：重跑即覆盖上一次产物
# =============================================================================

def safe_rmtree(path, *, must_be_under, dry: bool = False) -> dict:
    """只删除"明确位于 `must_be_under` **之下**"的目录，其余一律拒绝。

    为什么必须有这道硬校验：本函数是"用户不确认就删"的（需求要求重跑即覆盖）。
    一旦 `--out-dir` 拼错、或误把数据根目录传进来，后果**不可逆**。

    拒绝条件（任一命中即抛异常）：
      - 路径是符号链接（可能指向目录之外）
      - 路径 == `must_be_under` 本身，或不在它之下
      - 路径等于所在盘符/根
      - 路径是数据集目录本身、或其祖先/后代
    """
    # ⚠️ 顺序很重要：**先做全部拒绝判定，再看它存不存在**。
    # 若先判 exists()，那么"路径不存在"会提前 return，把下面的校验全部绕过——
    # 一旦 --out-dir 被拼错成别的目录，就相当于什么都没检查。
    p = Path(path)
    root = Path(must_be_under).resolve()

    # 链接判定要同时查 symlink 与 junction：
    # Windows 上目录联接(junction)的 `is_symlink()` 可能是 False，
    # 而 rmtree 顺着联接删下去会删到**目标目录的内容**——必须挡住。
    if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
        raise RuntimeError(f"拒绝删除符号链接/目录联接（可能指向目录之外）：{p}")

    pr = p.resolve()
    if root not in pr.parents:
        raise RuntimeError(
            f"拒绝删除：{pr}\n它不在 {root} 之下（或就是它本身）。\n"
            "这是防误删的硬校验。若确需删除，请人工确认路径后手动操作。"
        )
    if pr == Path(pr.anchor):
        raise RuntimeError(f"拒绝删除盘符/根目录：{pr}")

    try:
        data_root = Path(FIXED_DATA_ROOT).resolve()
    except Exception:
        data_root = None
    if data_root is not None:
        if pr == data_root or data_root in pr.parents or pr in data_root.parents:
            raise RuntimeError(f"拒绝删除数据集目录（或其祖先/后代）：{pr}")

    if not p.exists():
        return {"path": str(pr), "existed": False, "removed": False}

    n_files = 0
    n_bytes = 0
    for f in pr.rglob("*"):
        if f.is_file():
            n_files += 1
            try:
                n_bytes += f.stat().st_size
            except OSError:
                pass
    info = {"path": str(pr), "existed": True, "files": n_files, "bytes": n_bytes,
            "removed": not dry}
    if dry:
        return info
    shutil.rmtree(pr)
    return info


def safe_unlink(path, *, must_be_under) -> bool:
    """删除 `must_be_under` 之下的单个文件（同上，带包含性校验）。"""
    p = Path(path)
    root = Path(must_be_under).resolve()
    if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
        raise RuntimeError(f"拒绝删除符号链接/目录联接：{p}")
    pr = p.resolve()
    if pr == root or root not in pr.parents:
        raise RuntimeError(f"拒绝删除（不在 {root} 之下）：{pr}")
    if not p.exists():
        return False
    pr.unlink()
    return True


def human_mb(n_bytes: float) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    ap = base_parser("DMSA-Net 提分实验共享库 · 自检")
    ap.add_argument("--smoke", action="store_true", help="跑 CPU dry-run 自检")
    a = ap.parse_args()

    if not a.smoke:
        ap.print_help()
        sys.exit(0)

    bdir = find_baseline_dir(a.baseline_dir)
    print(f"baseline dir = {bdir}")
    print(f"train manifest SHA = {sha256_file(bdir / TRAIN_MANIFEST_NAME)}")
    res = cpu_smoke(bdir)
    sys.exit(0 if res.get("ok") else 1)
