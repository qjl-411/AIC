#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC 提分实验 G1：模块消融矩阵（★★）
===============================================================================

对应需求文档《04_需用户本人操作事项反馈.md》§5「★★ 模块消融矩阵」。

---------------------------------------------------------------------------
它解决什么问题
---------------------------------------------------------------------------

方案里 SAM / MDC / ILF / 双路径 四个模块目前只有**结构证据**（参数量、指纹），
**没有精度证据**。评委一定会问：「你说 MDC 有用，关掉会掉多少？」
本脚本给出这张表的**实测部分**。

---------------------------------------------------------------------------
关键设计：容量不匹配必须同时报告
---------------------------------------------------------------------------

关掉模块**会改变模型容量**（实测）：

    full          56,812,813   （= 基线，指纹一致 ✅）
    no_sam        56,812,354   Δ −459        ← 容量几乎不变
    no_mdc        45,037,801   Δ −11,775,012 ← 掉 20.7%
    no_ilf        55,495,947   Δ −1,316,866
    no_dual_path  46,753,037   Δ −10,059,776 ← 掉 17.7%

⇒ **`no_mdc` / `no_dual_path` 的精度差，不能直接归因给"这个模块"**，
   因为模型同时也变小了。方案里必须写成
   「移除 X 后的整体效果（含容量变化）」，而不是「X 的边际贡献」。

   若要严格拆出"模块语义"的贡献，需要**容量匹配对照**——
   即把去掉模块后省下的参数，用等量的普通卷积补回去。
   本脚本提供 `--capacity-matched` 选项生成这类变体的描述，
   但**不自动构造**（构造规则依赖具体设计选择，必须人工确认后写进方案）。

---------------------------------------------------------------------------
sealed test 铁律
---------------------------------------------------------------------------

**本脚本只报告折内验证集（val）指标，绝不触碰 sealed test。**

原因：sealed test 已在 seed=42 的 full 模型上正式评价过一次。
若现在拿消融变体去评 sealed test，就变成"在同一测试集上比较 5 个模型"——
这是典型的测试集调参，评委会直接扣分。

---------------------------------------------------------------------------
为什么要跑 2 折而不是 1 折
---------------------------------------------------------------------------

单折 val 只有 2,048 张（每类 512）。消融之间的差异往往小于 1%，
单折的噪声可能盖过真实差异。跑 2 折（建议 fold 1 与 fold 5，
两者在已完成的五折中一个偏高一个偏低）能给出**折间差异**这一最小可信度信号。

成本：4 变体 × 2 折 × 200 epoch ≈ 16–32 GPU h。

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------

    # 本机（无 GPU）自检接线：
    python g1_ablation.py --dry-run

    # GPU 机器正式跑：
    python g1_ablation.py \
        --data-root "/home/featurize/work/Combined_Dataset" \
        --out-dir   "/home/featurize/work/dmsa_ablation" \
        --variants no_sam,no_mdc,no_ilf \
        --folds 1,5 \
        --epochs 200

    # 先只跑一个变体一折，确认无误再全量：
    python g1_ablation.py --variants no_sam --folds 1 --epochs 200 ...
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import csv as _csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_GAMMA,
    BASE_LR,
    BASE_SEED,
    EXPECTED_CLASSES,
    FIXED_DATA_ROOT,
    PKG_VERSION,
    ProgressReporter,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    assert_train_manifest,
    base_parser,
    cpu_smoke,
    find_baseline_dir,
    fmt_hms,
    load_module,
    make_tee_sink,
    read_manifest,
    reproducible_seed,
    resolve_data_root,
    resolve_out_dir,
    safe_rmtree,
    safe_unlink,
    sha256_file,
    write_csv,
    write_json,
)

# 消融开关定义（与 DMSA_v24/arch_probe.py 的 variants 完全一致）
VARIANT_SWITCHES: Dict[str, Dict[str, bool]] = {
    "full":         dict(use_sam=True,  use_mdc=True,  use_ilf=True,  use_dual_path=True),
    "no_sam":       dict(use_sam=False, use_mdc=True,  use_ilf=True,  use_dual_path=True),
    "no_mdc":       dict(use_sam=True,  use_mdc=False, use_ilf=True,  use_dual_path=True),
    "no_ilf":       dict(use_sam=True,  use_mdc=True,  use_ilf=False, use_dual_path=True),
    "no_dual_path": dict(use_sam=True,  use_mdc=True,  use_ilf=True,  use_dual_path=False),
}

DEFAULT_VARIANTS = ["no_sam", "no_mdc", "no_ilf"]


# =============================================================================
# 1.5 进度：把基线 tqdm 的帧接出来（只解析，不改造基线）
# =============================================================================
#
# 基线 `smoke_test.py:1018` 里本来就写死了
#     bar = tqdm(loader, file=sys.stdout, desc=f"train epoch[{epoch}/{epochs}]")
# 逐 batch 往 stdout 吐帧（形如 `train epoch[3/200]: 45%|████ | 288/512 [...]`）。
# 所以本脚本**不改基线**（铁律 2），只在调用期间临时替换 sys.stdout，
# 把每个字节原样转发给真 stdout，同时解析出 (epoch, step) 喂给 ProgressReporter。
#
# ⚠️ 解析失败不是灾难：每个 epoch 开头/结尾会**强制**各输出一行进度，
#    即便 step 级帧一个都没解析到，epoch 级进度依然完整。
_TQDM_RE = re.compile(
    r"epoch\[(\d+)/(\d+)\].*?([\d.]+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]"
)


class _TqdmTap:
    """替身 stdout：原样转发 + 解析 tqdm 帧。"""

    def __init__(self, real, rep: ProgressReporter):
        self._real = real
        self._rep = rep

    def write(self, s: str) -> int:
        try:
            self._real.write(s)
        except Exception:
            pass
        try:
            for frame in re.split(r"[\r\n]", s):
                m = _TQDM_RE.search(frame)
                if m:
                    self._rep.set(epoch=int(m.group(1)), step=int(m.group(4)))
                    self._rep.emit()      # 内部有节流，不会把日志刷爆
        except Exception:
            pass                          # 进度解析绝不能把训练搞崩
        return len(s)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False


# =============================================================================
# 1. 载入消融注入机制
# =============================================================================

def load_arch_probe(candidates: Sequence[Optional[str]]) -> object:
    """定位并加载 arch_probe.py（消融开关注入的真源）。

    ⚠️ 候选列表里**必须**含「脚本同目录」，否则「把所有文件平铺放一层」的布局
    在云机器上必然失败：那些 `DMSA_v24/` 子目录候选全部落空，而末尾的 `E:` 回退
    在 POSIX 下会退化成**相对路径**（`Path("E:/...")` 在 Linux 上 is_absolute()=False），
    同样落空。这个坑在本机测不出来——因为 E: 回退在 Windows 上真的存在。
    """
    tried: List[str] = []
    for c in list(candidates) + [None]:
        paths: List[Path] = []
        if c:
            p = Path(c)
            paths.append(p if p.is_file() else p / "arch_probe.py")
        here = Path(__file__).resolve()
        # ① 扁平布局：arch_probe.py 与脚本同目录（lib_provenance.py 也应在同目录）
        paths.append(here.parent / "arch_probe.py")
        # ② 分层布局：<上层>/DMSA_v24/arch_probe.py
        for up in [here.parent, here.parent.parent, here.parent.parent.parent]:
            paths.append(up / "DMSA_v24" / "arch_probe.py")
            paths.append(up / "DMSA_netV2" / "DMSA_v24" / "arch_probe.py")
        # ③ 本机开发回退（POSIX 下退化为相对路径，自然失效，不会误命中）
        paths.append(Path(r"E:/Code/NeuroEvo/DMSA_netV2/DMSA_v24/arch_probe.py"))
        for p in paths:
            tried.append(str(p))
            if p.is_file():
                print(f"arch_probe   : {p}")
                return load_module(p, "dmsa_arch_probe")
    raise FileNotFoundError(
        "找不到 arch_probe.py（消融注入机制）。\n"
        "请把它与 g1_ablation.py 放在同一目录（推荐），"
        "或放在 <上层目录>/DMSA_v24/ 下，或用 --arch-probe 显式指定。\n"
        "已尝试：\n  " + "\n  ".join(tried)
    )


# =============================================================================
# 2. 单变体 × 单折 训练
# =============================================================================

def train_variant_fold(
    *,
    smoke_mod,
    arch_probe,
    variant: str,
    fold: int,
    rows: List[dict],
    class_to_idx: Dict[str, int],
    train_root: Path,
    device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    out_fold_dir: Path,
    max_batches: Optional[int] = None,
    rep: Optional[ProgressReporter] = None,
    sink=None,
    run_index: int = 0,
    run_total: int = 0,
) -> dict:
    """训练一个 (variant, fold) 组合，返回折内指标。

    `rep` / `sink` 只影响**进度展示**，不参与任何训练逻辑：
    传 None 时行为与改动前完全一致（保证自检脚本仍可跑）。
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    emit_line = sink if sink is not None else print

    cfg = VARIANT_SWITCHES[variant]

    out_fold_dir.mkdir(parents=True, exist_ok=True)

    # ---- 构造带开关的模型 ----
    create, restore = arch_probe.make_switch_patch(smoke_mod, **cfg)
    try:
        reproducible_seed(BASE_SEED, torch, __import__("numpy"), __import__("random"))
        model = create(num_classes=len(class_to_idx))
    finally:
        restore()

    fp = arch_probe.fingerprint(model)
    n_params = sum(p.numel() for p in model.parameters())

    # hard gate：full 必须与基线同构
    if variant == "full" and fp != ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            f"`full` 变体指纹 {fp} 与基线 {ARCHITECTURE_FINGERPRINT} 不一致——"
            "注入机制已失效，禁止用其结果做任何对比。"
        )

    model = model.to(device)

    train_rows = [r for r in rows if int(r["val_fold"]) != fold]
    val_rows = [r for r in rows if int(r["val_fold"]) == fold]
    emit_line(f"\n  [{variant} · fold {fold}] "
              f"train={len(train_rows):,} val={len(val_rows):,} "
              f"params={n_params:,} fp={fp[:10]}…")

    # ---- 数据 ----
    train_tf, val_tf = smoke_mod.build_transforms()
    import torch as _t
    gen = _t.Generator()
    gen.manual_seed(BASE_SEED + fold)

    train_ds = smoke_mod.ManifestImageDataset(
        train_root=train_root, rows=train_rows,
        class_to_idx=class_to_idx, transform=train_tf,
    )
    val_ds = smoke_mod.ManifestImageDataset(
        train_root=train_root, rows=val_rows,
        class_to_idx=class_to_idx, transform=val_tf,
    )
    pin = (device.type == "cuda")
    train_loader = _t.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gen,
        num_workers=num_workers, pin_memory=pin,
        worker_init_fn=smoke_mod.seed_worker,
    )
    val_loader = _t.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        worker_init_fn=smoke_mod.seed_worker,
    )

    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=BASE_GAMMA)
    lossf = nn.CrossEntropyLoss()

    # ---- 进度器：step 总数以真实 DataLoader 长度为准 ----
    # ⚠️ G1 是**折内**训练（8,192 train / 2,048 val），与 G2 的 final 全量训练
    # （10,240 张 → 640 step/epoch）不同：这里每 epoch 只有 512 step。
    if rep is not None:
        n_batches = len(train_loader)
        if max_batches:
            n_batches = min(n_batches, int(max_batches))
        rep.total_epochs = int(epochs)
        rep.steps_per_epoch = max(1, int(n_batches))
        # ⚠️ 不覆盖 total_units：它是**整个 G1** 的全局总 step（跨 变体×折），
        #    本 run 的完成量通过 `base_units` 折算进同一个百分比。
        rep.set(epoch=0, step=0, loss=None,
                extra=(f"run {run_index}/{run_total}" if run_total else ""))
        emit_line(f"  进度口径     : 本折 {epochs} epoch × {rep.steps_per_epoch} step"
                  f" = {epochs * rep.steps_per_epoch:,} step（val 前向不计入）；"
                  f"全局总 step ≈ {rep.total_units:,}")

    history: List[dict] = []
    best = {"accuracy": -1.0, "epoch": 0}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if rep is not None:
            rep.set(epoch=epoch, step=0)
            rep.emit(force=True)
        if rep is not None:
            with contextlib.redirect_stdout(_TqdmTap(sys.stdout, rep)):
                train_loss = smoke_mod.train_one_epoch(
                    model=model, loader=train_loader, optimizer=optimizer,
                    loss_function=lossf, device=device, epoch=epoch,
                    epochs=epochs, max_batches=max_batches,
                )
        else:
            train_loss = smoke_mod.train_one_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                loss_function=lossf, device=device, epoch=epoch,
                epochs=epochs, max_batches=max_batches,
            )
        scheduler.step()
        val_loss, metrics, cm, y_true, y_pred = smoke_mod.validate_one_epoch(
            model=model, loader=val_loader, loss_function=lossf,
            device=device, num_classes=len(class_to_idx), max_batches=max_batches,
        )
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "kappa")},
        }
        history.append(row)
        if metrics["accuracy"] > best["accuracy"]:
            best = {"accuracy": metrics["accuracy"], "epoch": epoch,
                    **{k: metrics[k] for k in ("precision", "recall", "f1", "kappa")},
                    "val_loss": val_loss}
        if rep is not None:
            rep.set(step=rep.steps_per_epoch, loss=train_loss,
                    extra=("val_acc=%.4f" % metrics["accuracy"])
                          + (f"  run {run_index}/{run_total}" if run_total else ""))
            rep.emit(force=True)
        if epoch % 10 == 0 or epoch == epochs:
            emit_line(f"    ep{epoch:>4}/{epochs} train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} acc={metrics['accuracy']:.4f} "
                      f"best={best['accuracy']:.4f}@{best['epoch']}")
        del cm, y_true, y_pred

    elapsed = time.time() - t0

    # ---- 进度：本折已全部跑完 ⇒ 把本折的量计入全局基数 ----
    # ⚠️ 必须在**每折结束时**累加，绝不能放到 main 的 variant 循环末尾：
    #   (1) 一个变体 = len(folds) 折 = len(folds) 个 run，末尾只加 1 份会**少记**；
    #   (2) 折与折之间 `rep.set(epoch=0, step=0)` 会重置本折计数，基数若不跟着走，
    #       第 2 折起进度会**倒退**回本变体起点；
    #   (3) 最后一折的累加若发生在 main 的 `set_base()` 调用之后，则永远进不了显示。
    if rep is not None:
        rep.set_base(rep.base_units + rep.total_epochs * rep.steps_per_epoch)

    write_csv(
        out_fold_dir / "metrics.csv", history,
        ["epoch", "train_loss", "val_loss", "accuracy",
         "precision", "recall", "f1", "kappa"],
    )
    write_json(
        out_fold_dir / "best_metrics.json",
        {
            "variant": variant, "fold": fold, "switches": cfg,
            "params": n_params, "fingerprint": fp,
            "fingerprint_equals_baseline": fp == ARCHITECTURE_FINGERPRINT,
            "best": best, "epochs_run": epochs,
            "elapsed_seconds": elapsed,
            "val_rows": len(val_rows),
            "val_class_counts": dict(Counter(r["class_name"] for r in val_rows)),
        },
        Path(__file__),
    )

    return {
        "variant": variant, "fold": fold,
        "params": n_params, "fingerprint": fp,
        "epochs_run": epochs,
        "best_epoch": best["epoch"],
        "accuracy": best["accuracy"],
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "f1": best.get("f1"),
        "kappa": best.get("kappa"),
        "val_loss": best.get("val_loss"),
        "elapsed_seconds": elapsed,
    }


def train_variant(
    *, smoke_mod, arch_probe, variant, folds, rows, class_to_idx,
    train_root, device, epochs, batch_size, num_workers, out_dir, max_batches=None,
    rep=None, sink=None, run_total: int = 0, run_counter=None,
) -> List[dict]:
    """训练一个变体的所有折。`run_counter` 是可变容器，用于跨变体累计 run 序号。"""
    out: List[dict] = []
    counter = run_counter if run_counter is not None else [0]
    for fold in folds:
        counter[0] += 1
        out.append(train_variant_fold(
            smoke_mod=smoke_mod, arch_probe=arch_probe, variant=variant, fold=fold,
            rows=rows, class_to_idx=class_to_idx, train_root=train_root,
            device=device, epochs=epochs, batch_size=batch_size,
            num_workers=num_workers,
            out_fold_dir=out_dir / variant / f"fold_{fold}",
            max_batches=max_batches,
            rep=rep, sink=sink, run_index=counter[0], run_total=run_total,
        ))
    return out


# =============================================================================
# 3. 参考基线：已完成的五折 val 指标（seed=42, full 模型）
# =============================================================================

REFERENCE_CV_LOCAL = Path(
    r"E:/Code/NeuroEvo/NeuroEvo_AD/复现版实验数据"
)


def load_reference_cv(
    explicit: Optional[str] = None,
    extra_dirs: Sequence[Path] = (),
    *,
    required_folds: int = 5,
) -> Optional[dict]:
    """读已完成的五折 best_metrics.json 作为 `full` 的参照。

    ⚠️ 云机器上没有 `E:` 盘，所以这里**不能**只认硬编码路径，否则 `full` 行会
    静默消失、`delta_vs_full` 全变 None，表格看起来"跑完了"但其实是残的。

    语义（显式即显式）：
      - `--reference-cv` 给出时，**只用它**；找不到或折数不足 → **直接报错**，
        绝不悄悄回退到别的目录（否则用户以为用了 A，实际用了 B）。
      - 未给出时按候选列表依次找；都没有 → 返回 None（由调用方决定是否实跑 full）。
    """

    def _try(c: Path) -> Optional[dict]:
        c = Path(c)
        if not c.is_dir():
            return None
        found = {}
        for i in range(1, required_folds + 1):
            p = c / f"Fold{i}" / "best_metrics.json"
            if p.is_file():
                found[i] = json.loads(p.read_text(encoding="utf-8-sig"))
        if len(found) == required_folds:
            return found
        if found:
            print(
                f"⚠️ 参照目录 {c} 只找到 {len(found)}/{required_folds} 折"
                f"（{sorted(found)}）→ 视为不完整。"
            )
        return None

    if explicit:
        c = Path(explicit).expanduser()
        if not c.is_dir():
            raise FileNotFoundError(f"--reference-cv 指向的目录不存在：{c}")
        r = _try(c)
        if r is None:
            raise FileNotFoundError(
                f"--reference-cv 目录里没有完整的 {required_folds} 折 "
                f"（需 Fold1..Fold{required_folds}/best_metrics.json）：{c}"
            )
        print(f"full 参照目录 : {c}（{len(r)} 折，显式指定）")
        return r

    cands: List[Path] = [REFERENCE_CV_LOCAL]
    for d in extra_dirs:
        cands += [Path(d) / "复现版实验数据", Path(d)]
    for c in cands:
        r = _try(c)
        if r is not None:
            print(f"full 参照目录 : {c}（{len(r)} 折，自动搜索）")
            return r
    return None


# =============================================================================
# 4. 汇总
# =============================================================================

def summarize(results: List[dict], folds: Sequence[int], out_dir: Path,
              ref: Optional[dict], epochs: int = BASE_EPOCHS) -> dict:
    by_var: Dict[str, List[dict]] = {}
    for r in results:
        by_var.setdefault(r["variant"], []).append(r)

    table: List[dict] = []
    for var, rs in by_var.items():
        accs = [r["accuracy"] for r in rs if r["accuracy"] is not None]
        mean = sum(accs) / len(accs) if accs else None
        spread = (max(accs) - min(accs)) if len(accs) >= 2 else None
        table.append({
            "variant": var,
            "params": rs[0]["params"],
            "n_folds": len(rs),
            "accuracy_mean": mean,
            "accuracy_min": min(accs) if accs else None,
            "accuracy_max": max(accs) if accs else None,
            "fold_spread": spread,
            "per_fold": {str(r["fold"]): r["accuracy"] for r in rs},
            "best_epochs": [r["best_epoch"] for r in rs],
        })

    full_row = next((t for t in table if t["variant"] == "full"), None)
    ref_note = None
    if full_row is None and ref:
        # 用已完成五折作为 full 参照
        accs = [ref[f]["accuracy"] for f in ref]
        full_row = {
            "variant": "full",
            "params": 56812813,
            "n_folds": len(accs),
            "accuracy_mean": sum(accs) / len(accs),
            "accuracy_min": min(accs),
            "accuracy_max": max(accs),
            "fold_spread": max(accs) - min(accs),
            "per_fold": {str(f): ref[f]["accuracy"] for f in ref},
            "best_epochs": [ref[f]["epoch"] for f in ref],
        }
        table.insert(0, full_row)
        ref_note = (
            "full 行直接取自已完成的五折实验（复现版实验数据/Fold1..5/best_metrics.json，"
            "seed=42，epochs=200，折内按 val accuracy 选 best），**未重跑**。"
        )

    for t in table:
        t["delta_vs_full"] = (
            (t["accuracy_mean"] - full_row["accuracy_mean"])
            if (full_row and t["accuracy_mean"] is not None) else None
        )

    summary = {
        "experiment": "G1 module ablation matrix",
        "folds_requested": list(folds),
        "epochs": int(epochs),
        "batch_size": BASE_BATCH_SIZE,
        "seed": BASE_SEED,
        "selection_rule": "折内按 val accuracy 选 best epoch（与基线一致）",
        "validation_only": True,
        "sealed_test_untouched": True,
        "full_reference_note": ref_note,
        "table": table,
        "capacity_warning": (
            "关掉模块会同时改变模型容量（no_mdc −11,775,012 / no_dual_path −10,059,776）。"
            "因此这些变体的精度差 **不能直接归因给单个模块**，只能表述为"
            "「移除 X 后的整体效果（含容量变化）」。"
            "严格拆出模块语义贡献需要容量匹配对照。"
        ),
        "limitation_note": (
            "只跑 val（折内验证集），不评 sealed test。"
            "若只跑 1 折，差异可能被单折噪声淹没；fold_spread 字段给出折间离散度，"
            "请只在 |Δ| > fold_spread 时才声称差异显著。"
        ),
    }
    write_json(out_dir / "g1_ablation_summary.json", summary, Path(__file__))
    return summary


def write_markdown(summary: dict, ref: Optional[dict]) -> None:
    lines: List[str] = []
    lines.append("# G1 模块消融矩阵（实测）\n")
    lines.append("> 评估集：**折内验证集（val）**，sealed test **未被触碰**。")
    lines.append(f"> 选模规则：折内按 val accuracy 选 best epoch（与基线一致）。")
    lines.append(f"> 训练轮数：{summary['epochs']} epochs；seed={summary['seed']}。\n")

    lines.append("| 变体 | 参数量 | 折数 | val 准确率(均) | Δ vs full | 折间跨度 |"
                 " 各折 best epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for t in summary["table"]:
        d = t.get("delta_vs_full")
        lines.append(
            f"| `{t['variant']}` | {t['params']:,} | {t['n_folds']} | "
            f"{t['accuracy_mean']*100:.3f}% | "
            f"{('%+.3f pp' % (d*100)) if d is not None else '—'} | "
            f"{(('%.3f pp' % (t['fold_spread']*100)) if t['fold_spread'] is not None else '—')} | "
            f"{t['best_epochs']} |"
        )
    lines.append("")
    if summary.get("full_reference_note"):
        lines.append(f"> {summary['full_reference_note']}\n")

    lines.append("## 必须写进方案的诚实说明\n")
    lines.append(f"1. **容量不匹配**：{summary['capacity_warning']}\n")
    lines.append(f"2. **统计效力**：{summary['limitation_note']}\n")

    p = Path(_OUT / "g1_ablation_table.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [write_md   ] {p}")


# =============================================================================
# 5. main
# =============================================================================

_OUT: Path = Path(".")


def main() -> int:
    global _OUT

    ap = base_parser("DMSA-Net 模块消融矩阵（★★）")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help=f"逗号分隔，可选：{','.join(VARIANT_SWITCHES)}")
    ap.add_argument("--folds", default="1,5", help="逗号分隔的折号（默认 1,5）")
    ap.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BASE_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--data-root", default=None,
                    help=f"Combined Dataset 路径（默认 {FIXED_DATA_ROOT}）")
    ap.add_argument("--arch-probe", default=None, help="arch_probe.py 路径")
    ap.add_argument("--reference-cv", default=None,
                    help="已完成的五折参照目录（内含 Fold1..5/best_metrics.json）；"
                         "不给则自动搜索，找不到就让 full 实跑（若 full 在 --variants 里）")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="每 epoch 最多几个 batch（调试用；正式跑请留空）")
    ap.add_argument("--progress-log", default=None,
                    help="进度日志路径（默认 <out-dir>/g1_progress.log，实时追加写入）")
    ap.add_argument("--progress-interval", type=float, default=5.0,
                    help="进度行最小输出间隔（秒）；每 epoch 首尾各有一行强制输出")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="遇到已存在的 (variant, fold) 产物就报错退出"
                         "（默认：重跑即覆盖，路径不变）")
    ap.add_argument("--overwrite-dry-run", action="store_true",
                    help="只预演覆盖会删掉什么，不真删、也不训练")
    a = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = resolve_out_dir(a.out_dir, "g1")
    _OUT = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bdir = find_baseline_dir(a.baseline_dir)
    print("=" * 78)
    print("G1 · 模块消融矩阵")
    print("=" * 78)
    print(f"包版本      : {PKG_VERSION}")
    print(f"baseline dir : {bdir}")
    print(f"out dir      : {out_dir}")
    print(f"覆盖策略     : {'重跑即覆盖（路径不变）' if a.overwrite else 'NO-OVERWRITE（遇到已有产物即报错）'}"
          + ("  ⚠ 本次为覆盖预演，不删除、不训练" if a.overwrite_dry_run else ""))

    # ---- 协议守卫 ----
    man = bdir / TRAIN_MANIFEST_NAME
    man_sha = sha256_file(man)
    if man_sha != TRAIN_MANIFEST_SHA256:
        raise RuntimeError(
            f"train manifest SHA 不符！\nExpected: {TRAIN_MANIFEST_SHA256}\nActual: {man_sha}"
        )
    rows = read_manifest(man)
    assert_train_manifest(rows)
    class_to_idx = {n: i for i, n in enumerate(sorted({r["class_name"] for r in rows}))}
    print(f"协议校验     : PASS  rows={len(rows):,}  classes={class_to_idx}")

    smoke_mod = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke")
    arch_probe = load_arch_probe([a.arch_probe])
    print(f"smoke_test   : {bdir / 'smoke_test.py'}")
    print(f"manifest SHA : {man_sha}")

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    folds = [int(f.strip()) for f in a.folds.split(",") if f.strip()]
    for v in variants:
        if v not in VARIANT_SWITCHES:
            raise ValueError(f"未知变体 {v}；可选：{list(VARIANT_SWITCHES)}")
    for f in folds:
        if not 1 <= f <= 5:
            raise ValueError(f"非法折号 {f}")

    # ---- dry-run ----
    if a.dry_run:
        res = cpu_smoke(bdir)
        print("\n-- dry-run：验证 4 个变体都能构造 + 前向 ------------------")
        import torch
        probe = []
        for v in ["full"] + variants:
            create, restore = arch_probe.make_switch_patch(smoke_mod, **VARIANT_SWITCHES[v])
            try:
                m = create(num_classes=4)
                fp = arch_probe.fingerprint(m)
                pr = sum(p.numel() for p in m.parameters())
                m.eval()
                with torch.no_grad():
                    y = m(torch.randn(1, 3, 224, 224))
                ok = tuple(y.shape) == (1, 4)
                probe.append({"variant": v, "params": pr, "fingerprint": fp,
                              "forward_ok": ok,
                              "equals_baseline": fp == ARCHITECTURE_FINGERPRINT,
                              "params_delta": pr - 56812813})
                print(f"    {v:14s} params={pr:>12,} Δ{pr-56812813:>+11,} "
                      f"fwd={'OK' if ok else 'FAIL'} "
                      f"{'[=baseline]' if fp == ARCHITECTURE_FINGERPRINT else ''}")
            except Exception as e:
                probe.append({"variant": v, "error": f"{type(e).__name__}: {e}"})
                print(f"    {v:14s} ❌ {type(e).__name__}: {e}")
            finally:
                restore()
        gate = next((p for p in probe if p["variant"] == "full"), {})
        if not gate.get("equals_baseline"):
            print("\n❌ 硬门失败：`full` 指纹与基线不一致，禁止用这套注入做消融。")
            write_json(out_dir / "g1_dryrun.json",
                       {"cpu_smoke": res, "variants": probe, "gate_passed": False},
                       script_path)
            return 2
        print("\n✅ 硬门通过：`full` 与基线同构；4 个变体均可构造并前向。")
        write_json(out_dir / "g1_dryrun.json",
                   {"cpu_smoke": res, "variants": probe, "gate_passed": True},
                   script_path)
        return 0

    # ---- 正式训练 ----
    # 数据根目录默认取 gpu_lib.FIXED_DATA_ROOT（用户实测确认的云端位置），
    # --data-root 可覆盖。⚠️ 基线 `final_train.py` / `smoke_test.py` 里硬编码的
    # `DATA_ROOT` 指向 `.../data/dmsa_pcv2/...`，**与实测位置不一致**，
    # 所以下面这条 `train_root` 存在性校验必须保留 —— 它就是用来挡住这种不一致的。
    data_root = resolve_data_root(a.data_root)
    train_root = Path(data_root).expanduser() / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"找不到 train 目录：{train_root}\n"
            f"（当前数据根目录 = {data_root}，来自默认值或 --data-root）\n"
            "请确认数据已上传到该位置，或用 --data-root 显式指定 Combined Dataset 路径。"
        )
    print(f"data root    : {data_root}")

    from C22_lib_gpu_runtime import probe_device
    device, dev_info = probe_device(a.device)
    print(f"device       : {device}  {dev_info.get('device_name')}")

    # ---- `full` 基准：优先复用已完成的五折参照；找不到就实跑 ----
    # 放在写 run_config 之前：--reference-cv 给错时应当**先失败**，不留半套产物。
    ref = load_reference_cv(
        a.reference_cv, extra_dirs=[bdir, bdir.parent, Path.home() / "work"]
    )
    if ref is None:
        print("\n⚠️ 未找到完整的「已完成五折参照」（复现版实验数据/Fold1..5/"
              "best_metrics.json）。")
        if "full" in variants:
            print("   → `full` 将在本次**实跑生成**，表格仍会有 Δ 基准。")
        else:
            print("   → `--variants` 里没有 full：本次表格将**没有 full 行、"
                  "也没有 Δ vs full**。")
            print("     要拿到 Δ，请把 full 加进 --variants（多 1 个变体的成本），"
                  "或用 --reference-cv 指向已有的五折结果目录。")

    write_json(
        out_dir / "g1_run_config.json",
        {
            "variants": variants, "folds": folds, "epochs": a.epochs,
            "batch_size": a.batch_size, "seed": BASE_SEED,
            "lr": BASE_LR, "lr_gamma": BASE_GAMMA,
            "num_workers": a.num_workers,
            "manifest_sha256": man_sha,
            "train_root": str(train_root),
            "smoke_test_sha256": sha256_file(bdir / "smoke_test.py"),
            "arch_probe_sha256": sha256_file(Path(arch_probe.__file__)),
            "device": dev_info,
            "sealed_test_policy": "not touched; val only",
            "max_batches": a.max_batches,
        },
        script_path,
    )

    # ---- 本次实际要跑的 (variant, fold) 清单 ----
    planned: List[Tuple[str, int]] = [
        (v, f) for v in variants if not (v == "full" and ref is not None)
        for f in folds
    ]

    # ---- 覆盖：重跑即覆盖，路径不变（与 G2 同口径）----
    # 删除只针对 `out_dir/<variant>/fold_<f>`，且经 `safe_rmtree` 硬校验：
    # 不在 out_dir 之下 / 数据集 / 盘符根 / 符号链接 一律拒绝。
    verb = "将删除" if a.overwrite_dry_run else "已清理"
    cleared = []
    for v, f in planned:
        d = out_dir / v / f"fold_{f}"
        if not d.exists():
            continue
        if not a.overwrite:
            raise FileExistsError(
                f"产物已存在：{d}\n"
                "当前是 --no-overwrite 模式。\n"
                "  · 想保留它 → 换一个 --out-dir，或从 --variants/--folds 去掉这一项\n"
                "  · 想重跑覆盖 → 去掉 --no-overwrite（默认就是覆盖）"
            )
        cleared.append(safe_rmtree(d, must_be_under=out_dir, dry=a.overwrite_dry_run))
    if cleared:
        tot_f = sum(c.get("files", 0) for c in cleared)
        tot_b = sum(c.get("bytes", 0) for c in cleared)
        print(f"覆盖清理     : {verb} {len(cleared)} 个旧目录、{tot_f} 个文件、"
              f"{tot_b / (1024 * 1024):.1f} MB")
        for c in cleared:
            print(f"    · {c['path']}  ({c.get('files', 0)} 文件, "
                  f"{c.get('bytes', 0) / (1024 * 1024):.1f} MB)")
    leftover = sorted(
        {p.name for p in out_dir.iterdir() if p.is_dir() and p.name in VARIANT_SWITCHES}
        - {v for v, _ in planned}
    )
    if leftover:
        print(f"⚠️ 残留目录（本次不跑 → 不覆盖、也不会进汇总表）：{leftover}")

    if a.overwrite_dry_run:
        print("\n[--overwrite-dry-run] 覆盖预演结束：上面就是将被删除的内容。")
        print("确认无误后，去掉 --overwrite-dry-run 即可真正重跑覆盖。")
        return 0

    # ---- 进度出口：屏幕与 progress.log 是同一份 ----
    progress_log = Path(a.progress_log) if a.progress_log else out_dir / "g1_progress.log"
    if a.overwrite:
        safe_unlink(progress_log, must_be_under=progress_log.parent)
    sink = make_tee_sink(progress_log)
    print(f"进度日志     : {progress_log}（实时追加，可随时 tail -f）")

    n_train_first = (len([r for r in rows if int(r["val_fold"]) != folds[0]])
                     if folds else len(rows))
    steps_est = max(1, -(-n_train_first // a.batch_size))     # ceil
    if a.max_batches:
        steps_est = min(steps_est, int(a.max_batches))
    rep = ProgressReporter(
        label="G1", total_epochs=a.epochs, steps_per_epoch=steps_est,
        total_units=max(1, len(planned) * a.epochs * steps_est),
        unit="step", sink=sink, min_interval=a.progress_interval,
    )

    results: List[dict] = []
    t_start = time.time()
    carry = [0.0]          # 已完成 run 的累计耗时 → 让"用时/ETA"是**整个 G1** 的口径
    counter = [0]
    run_total = len(planned)
    for v in variants:
        if v == "full" and ref is not None:
            sink("\n[skip] `full` 不重跑：直接引用已完成五折结果。")
            continue
        rep.label = v
        # ⚠️ 基数由 `train_variant_fold` 在**每折结束时**累加（见该函数内注释）。
        #    这里**不能**用局部变量 set_base 覆盖，否则会把前几折的累加抹掉。
        rep.t0 = time.time() - carry[0]
        t_run = time.time()
        results.extend(train_variant(
            smoke_mod=smoke_mod, arch_probe=arch_probe, variant=v, folds=folds,
            rows=rows, class_to_idx=class_to_idx, train_root=train_root,
            device=device, epochs=a.epochs, batch_size=a.batch_size,
            num_workers=a.num_workers, out_dir=out_dir, max_batches=a.max_batches,
            rep=rep, sink=sink, run_total=run_total, run_counter=counter,
        ))
        carry[0] += time.time() - t_run

    summary = summarize(results, folds, out_dir, ref, epochs=a.epochs)
    write_markdown(summary, ref)
    write_csv(
        out_dir / "g1_ablation_table.csv",
        [{k: v for k, v in r.items() if k != "per_fold"} for r in summary["table"]],
        ["variant", "params", "n_folds", "accuracy_mean", "accuracy_min",
         "accuracy_max", "fold_spread", "delta_vs_full"],
    )
    write_csv(
        out_dir / "g1_ablation_per_fold.csv", results,
        ["variant", "fold", "params", "best_epoch", "accuracy",
         "precision", "recall", "f1", "kappa", "val_loss",
         "elapsed_seconds", "epochs_run"],
    )

    # ⚠️ 收尾行：本折的量已在 `train_variant_fold` 折末计入 base，
    #    这里**不能**再给 `step=steps_per_epoch`，否则最后一个 run 会被算两遍 → >100%。
    rep.total_epochs = 0          # 收尾行不必再显示 "epoch x/x"
    rep.set(epoch=0, step=0, loss=None, extra="G1 全部 run 完成")
    rep.emit(force=True)

    hours = (time.time() - t_start) / 3600.0
    print("\n" + "=" * 78)
    print(f"G1 完成，总耗时 {hours:.2f} h（{fmt_hms(hours * 3600)}）")
    print(f"  进度日志：{progress_log}")
    print(f"  {out_dir / 'g1_ablation_summary.json'}")
    print(f"  {out_dir / 'g1_ablation_table.md'}")
    for t in summary["table"]:
        d = t.get("delta_vs_full")
        print(f"    {t['variant']:14s} val_acc={t['accuracy_mean']*100:.3f}%  "
              f"Δ={('%+.3f pp' % (d*100)) if d is not None else '—'}")
    print("提醒：sealed test 未被本实验触碰。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
