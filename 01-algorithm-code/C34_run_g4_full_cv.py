#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G4 · 一站式实验：对照五折（双腕）＋ 跨种子 ＋ 完整消融矩阵

一次运行解决三件事，**全部用同一份代码、同一张卡、同一个 data_root、同一个 seed 口径**：

  阶段 1（A1）基线臂 —— 分组协议 `full` × 5 折
      作用：① 拿到基线协议在本脚本下的五折结果；
            ② 同时充当**阶段 2 消融的 `full` 参照**（这是关键——Δ 从此可归因）。
  阶段 2（B） 消融矩阵 —— `no_sam` / `no_mdc` / `no_ilf` / `no_dual_path` × 5 折
      作用：把创新点的「结构性证据」升级为「因果证据」，撤销 L3。
      ⚠️ G1 那次的 Δ 是「参考性对照」，因为参照来自**另一个脚本、另一张卡、另一个 data_root**
         （entry_script=train_5fold.py≠smoke_test.py、RTX 3060≠4090、空格路径≠下划线路径）。
         本次参照 = 阶段 1 自己跑出来的 `full`，**只剩「容量变化」这一个残留混淆**。
      📌 折数口径（2026-09-26 邱总定）：走真源 §3.3 的「**完整方案**」= 4 变体 × **5 折**，
         而不是「最小方案」的 2 折。理由：5 折更稳，且能正面回应「为什么只做 2 折」的质疑。
         若想退回 2 折，加 `--folds 1,5` 即可（会同时作用于 A* 阶段）。
  阶段 3（A2）对照臂 —— 图像级随机划分（忽略 group_id）`full` × 5 折
      作用：Δ = acc(对照) − acc(基线) = **分组协议压掉的乐观偏差**（量化协议价值）。
      两臂唯一变量是划分清单 ⇒ Δ 可归因。
  阶段 4（A3/A4）跨种子 —— 分组协议 `full` × seed 43/44 × 5 折
      作用：给出**跨种子 val 准确率离散度**，补上 G2(mode=final) 只落 train_loss 的缺口。

它回答什么
----------
1. **我们的防泄漏协议到底值多少？** → 阶段3 − 阶段1（定量，不是一个形容词）
2. **每个模块到底贡献多少？**     → 阶段2 vs 阶段1 的 `full`
3. **换个种子会不会崩？**         → seed 42/43/44 在同一折上的极差

三条纪律（写死在代码里，不靠自觉）
----------------------------------
* **不碰 sealed test**：全程只评折内 val，代码里根本不引用 `pcv_test_manifest_SEALED.csv`。
* **不改基线**：`smoke_test.py` / `final_train.py` / 基线 manifest 一个字节都不动；
  消融靠 `arch_probe.make_switch_patch()` 在**调用层**注入，且 `full` 必须过指纹硬门
  （== `8e542321…d76975`），否则直接报错退出。
* **事前冻结**：图像级清单由 `c0_make_control_manifest.py` 生成并冻结，
  本文件用**硬编码 SHA-256** 回校；清单改一个字节就拒绝开跑。

进度怎么看（不需要另开终端）
----------------------------
* 直接在终端前台运行本脚本，进度就刷在**同一个终端**里，无需 tail、无需第二条命令。
* 每行都显式 `flush=True`，所以即使重定向到文件也是即时可见的。
* 屏幕上有三层信息：
    ① 基线 tqdm 的**逐 batch 进度条**（原样转发，实时）；
    ② 每个 epoch 结束时的 `ep  N/200 train_loss=… val_loss=… acc=… best=…`；
    ③ 全局进度行 `[进度] G4 epoch i/200 … 45.2% 用时 12:34:56 剩余≈14:20:10`
       —— 百分比与 ETA 是**跨全部 40 个 run** 的口径。
* 同时会把同一份内容 tee 到 `<out>/g4_progress.log`（断线后想回看用）。

用法
----
    python g4_run.py --plan                       # 只预检 + 报总时长，不训练
    python g4_run.py --dry-run --device cpu       # CPU 上验证接线（含 5 个变体构造）
    python g4_run.py                              # 正式跑（默认 40 个 run）
    python g4_run.py --parts A1,B                 # 只跑指定阶段
    python g4_run.py --resume                     # 断点续跑：已完成的 run 自动跳过

依赖（都已在云端基线目录里，无需新增）
--------------------------------------
    gpu_lib.py（共享库） / arch_probe.py（消融注入） / smoke_test.py（训练与评估真源）
    pcv_v2_folds_seed42.csv（基线冻结清单）
    pcv_v2_folds_imgsplit_seed42_FROZEN.csv（图像级冻结清单，需一并上传）
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_GAMMA,
    BASE_LR,
    BASE_SEED,
    EXPECTED_CLASSES,
    EXPECTED_FOLDS,
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
    human_mb,
    load_module,
    make_tee_sink,
    probe_device,
    read_manifest,
    reproducible_seed,
    resolve_data_root,
    resolve_out_dir,
    sha256_file,
    write_csv,
    write_json,
)

# =============================================================================
# 0. 冻结常量
# =============================================================================

CONTROL_MANIFEST_NAME = "pcv_v2_folds_imgsplit_seed42_FROZEN.csv"
CONTROL_MANIFEST_SHA256 = (
    "d823021990e317fa997fa660ac36ada1abbb3ed2b29cd01712f06e3707c168b8"
)

# 消融开关（与 DMSA_v24/arch_probe.py 的 variants 完全一致）
VARIANT_SWITCHES: Dict[str, Dict[str, bool]] = {
    "full":         dict(use_sam=True,  use_mdc=True,  use_ilf=True,  use_dual_path=True),
    "no_sam":       dict(use_sam=False, use_mdc=True,  use_ilf=True,  use_dual_path=True),
    "no_mdc":       dict(use_sam=True,  use_mdc=False, use_ilf=True,  use_dual_path=True),
    "no_ilf":       dict(use_sam=True,  use_mdc=True,  use_ilf=False, use_dual_path=True),
    "no_dual_path": dict(use_sam=True,  use_mdc=True,  use_ilf=True,  use_dual_path=False),
}

# 期望参数量（来自 arch_probe 未训练时的实测；dry-run 会逐个比对，对不上直接报错）
EXPECTED_VARIANT_PARAMS: Dict[str, int] = {
    "full":         56_812_813,
    "no_sam":       56_812_354,      # Δ −459（= spatial_attention 参数量）
    "no_mdc":       45_037_801,      # Δ −11,775,012
    "no_ilf":       55_495_947,      # Δ −1,316,866
    "no_dual_path": 46_753_037,      # Δ −10,059,776
}

# 阶段定义。键的顺序 = 实际执行顺序。
PART_SPECS: Dict[str, dict] = {
    "A1": dict(
        title="阶段1 · 基线臂（分组协议 · full）",
        arms=["grouped"], variants=["full"], seeds=[BASE_SEED],
        folds=(1, 2, 3, 4, 5),
    ),
    "B": dict(
        # 2026-09-26 邱总定：消融走真源 §3.3 的「完整方案」= 4 变体 × 5 折
        # （参照 A1 的 `full` × 5 折逐折对齐 ⇒ Δ 可归因）。
        title="阶段2 · 消融矩阵（4 变体 × 5 折）",
        arms=["grouped"],
        variants=["no_sam", "no_mdc", "no_ilf", "no_dual_path"],
        seeds=[BASE_SEED], folds=(1, 2, 3, 4, 5),
    ),
    "A2": dict(
        title="阶段3 · 对照臂（图像级随机 · full）",
        arms=["control"], variants=["full"], seeds=[BASE_SEED],
        folds=(1, 2, 3, 4, 5),
    ),
    "A3": dict(
        title="阶段4a · 跨种子（分组 · full · seed 43）",
        arms=["grouped"], variants=["full"], seeds=[43],
        folds=(1, 2, 3, 4, 5),
    ),
    "A4": dict(
        title="阶段4b · 跨种子（分组 · full · seed 44）",
        arms=["grouped"], variants=["full"], seeds=[44],
        folds=(1, 2, 3, 4, 5),
    ),
}
PART_ORDER = ["A1", "B", "A2", "A3", "A4"]
DEFAULT_PARTS = ",".join(PART_ORDER)

# 单 run 实测时长：G1 六个 run 的 elapsed_seconds = 9390/10014/8353/9341/10358/9252 s
# → 均值 9451 s = 2.63 h（200 epoch × 512 step，每折 8192 train / 2048 val，RTX 4090）
MEASURED_SECONDS_PER_RUN = 9451.0

OUT_TAG = "g4"


# =============================================================================
# 1. 进度：把基线 tqdm 的帧接出来（只解析、不改造基线）
# =============================================================================
# 基线 `smoke_test.py` 里 `train_one_epoch()` 本来就往 stdout 吐 tqdm 帧：
#     train epoch[3/200]:  45%|████▌     | 288/512 [01:23<01:45,  3.34it/s]
# 这里在调用期间临时替换 sys.stdout：**每个字节原样转发给真终端**
# （所以逐 batch 的实时进度条照常显示），同时把 (epoch, step) 解析出来喂给
# ProgressReporter，用于算**跨 run 的全局百分比与 ETA**。
_TQDM_RE = re.compile(
    r"epoch\[(\d+)/(\d+)\].*?([\d.]+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]"
)


class _TqdmTap:
    """替身 stdout：原样转发 + 解析 tqdm 帧。解析失败绝不影响训练。

    `passthrough=False`（CLI `--quiet`）时**只解析、不转发**：用于 `nohup` 后台跑、
    stdout 被重定向到文件的场景 —— 否则 40 run × 200 epoch × 512 step = **409.6 万帧**
    tqdm 会把日志写成几百 MB 的单行。此时进度仍可从 `g4_progress.log` 与
    `[进度]` 行读到，信息不丢。
    """

    def __init__(self, real, rep: "ProgressReporter", passthrough: bool = True):
        self._real = real
        self._rep = rep
        self._passthrough = passthrough

    def write(self, s: str) -> int:
        if self._passthrough:
            try:
                self._real.write(s)
            except Exception:
                pass
        try:
            for frame in re.split(r"[\r\n]", s):
                m = _TQDM_RE.search(frame)
                if m:
                    self._rep.set(epoch=int(m.group(1)), step=int(m.group(4)))
                    self._rep.emit()          # 内部自带节流
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False


# =============================================================================
# 2. 载入消融注入机制
# =============================================================================
def load_arch_probe(candidates: Sequence[Optional[str]]) -> object:
    """定位并加载 arch_probe.py。

    ⚠️ 候选里**必须**含「脚本同目录」：云端是扁平布局（所有 .py 平铺一层），
    只有 `DMSA_v24/xxx` 这类分层候选会全部落空；而末尾的 `E:` 回退在 POSIX 下
    会退化成**相对路径**（`Path("E:/…").is_absolute()` 在 Linux 上为 False），
    同样落空——这个坑在 Windows 本机测不出来。
    """
    tried: List[str] = []
    for c in list(candidates) + [None]:
        paths: List[Path] = []
        if c:
            p = Path(c)
            paths.append(p if p.is_file() else p / "arch_probe.py")
        here = Path(__file__).resolve()
        paths.append(here.parent / "arch_probe.py")                 # 扁平布局（云端）
        for up in [here.parent, here.parent.parent, here.parent.parent.parent]:
            paths.append(up / "DMSA_v24" / "arch_probe.py")          # 分层布局
            paths.append(up / "DMSA_netV2" / "DMSA_v24" / "arch_probe.py")
        paths.append(Path(r"E:/Code/NeuroEvo/DMSA_netV2/DMSA_v24/arch_probe.py"))
        for p in paths:
            tried.append(str(p))
            if p.is_file():
                print(f"arch_probe   : {p}", flush=True)
                return load_module(p, "dmsa_arch_probe")
    raise FileNotFoundError(
        "找不到 arch_probe.py。请把它与 g4_run.py 放同一目录，"
        "或放在 <上层>/DMSA_v24/ 下，或用 --arch-probe 指定。\n已尝试：\n  "
        + "\n  ".join(tried)
    )


# =============================================================================
# 3. 单 run 训练（一个 (arm, variant, seed, fold) 组合）
# =============================================================================
def train_one_run(
    *,
    smoke_mod,
    arch_probe,
    arm: str,
    variant: str,
    seed: int,
    fold: int,
    rows: List[dict],
    class_to_idx: Dict[str, int],
    train_root: Path,
    device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    out_run_dir: Path,
    max_batches: Optional[int] = None,
    rep: Optional[ProgressReporter] = None,
    sink=None,
    run_index: int = 0,
    run_total: int = 0,
    tqdm_passthrough: bool = True,
) -> dict:
    """训练一个组合并返回折内指标。`rep`/`sink` 只影响展示，不参与训练逻辑。"""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    emit = sink if sink is not None else (lambda s: print(s, flush=True))
    cfg = VARIANT_SWITCHES[variant]

    out_run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 构造带开关的模型 ----
    create, restore = arch_probe.make_switch_patch(smoke_mod, **cfg)
    try:
        reproducible_seed(seed, torch, __import__("numpy"), __import__("random"))
        model = create(num_classes=len(class_to_idx))
    finally:
        restore()

    fp = arch_probe.fingerprint(model)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- 硬门 1：`full` 必须与基线同构，否则注入机制已失效 ----
    if variant == "full" and fp != ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            f"`full` 指纹 {fp} != 基线 {ARCHITECTURE_FINGERPRINT} —— 注入机制失效，"
            "禁止用任何结果做对比。"
        )
    # ---- 硬门 2：参数量必须等于文档值（防止开关"接上了但接错位置"）----
    exp = EXPECTED_VARIANT_PARAMS.get(variant)
    if exp is not None and n_params != exp:
        raise RuntimeError(
            f"变体 `{variant}` 参数量 {n_params:,} != 期望 {exp:,}"
            f"（Δ {n_params - exp:+,}）—— 注入接线与文档不符，拒绝继续。"
        )

    model = model.to(device)

    train_rows = [r for r in rows if int(r["val_fold"]) != fold]
    val_rows = [r for r in rows if int(r["val_fold"]) == fold]
    emit(f"\n  [{arm} · {variant} · seed{seed} · fold {fold}] "
         f"train={len(train_rows):,} val={len(val_rows):,} "
         f"params={n_params:,} fp={fp[:10]}…")

    # ---- 数据 ----
    train_tf, val_tf = smoke_mod.build_transforms()
    import torch as _t
    gen = _t.Generator()
    gen.manual_seed(seed + fold)

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

    if rep is not None:
        n_batches = len(train_loader)
        if max_batches:
            n_batches = min(n_batches, int(max_batches))
        rep.total_epochs = int(epochs)
        rep.steps_per_epoch = max(1, int(n_batches))
        rep.set(epoch=0, step=0, loss=None,
                extra=(f"run {run_index}/{run_total}" if run_total else ""))
        emit(f"  本 run 口径   : {epochs} epoch × {rep.steps_per_epoch} step"
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
            with contextlib.redirect_stdout(
                _TqdmTap(sys.stdout, rep, tqdm_passthrough)
            ):
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
        # ⚠️ validation 也会吐 tqdm（`smoke_test.py` 里 `file=sys.stdout`），
        #    若只包住 train，`--quiet` 就漏掉这一半 —— 40 run 会有 102 万帧 val 输出。
        #    这里一并包住；val 帧不含 `epoch[i/N]`，不会被 `_TQDM_RE` 误解析。
        if rep is not None:
            with contextlib.redirect_stdout(
                _TqdmTap(sys.stdout, rep, tqdm_passthrough)
            ):
                val_loss, metrics, cm, y_true, y_pred = smoke_mod.validate_one_epoch(
                    model=model, loader=val_loader, loss_function=lossf,
                    device=device, num_classes=len(class_to_idx),
                    max_batches=max_batches,
                )
        else:
            val_loss, metrics, cm, y_true, y_pred = smoke_mod.validate_one_epoch(
                model=model, loader=val_loader, loss_function=lossf,
                device=device, num_classes=len(class_to_idx), max_batches=max_batches,
            )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "kappa")},
        })
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
            emit(f"    ep{epoch:>4}/{epochs} train_loss={train_loss:.4f} "
                 f"val_loss={val_loss:.4f} acc={metrics['accuracy']:.4f} "
                 f"best={best['accuracy']:.4f}@{best['epoch']}")
        del cm, y_true, y_pred

    elapsed = time.time() - t0

    # ---- 本 run 跑完 ⇒ 把本 run 的量计入全局基数 ----
    # ⚠️ 必须在**每个 run 结束时**累加：
    #   (1) 一个变体 × N 折 = N 个 run，末尾只加 1 份会**少记**；
    #   (2) 下一个 run 会 `rep.set(epoch=0, step=0)` 重置本 run 计数，
    #       基数若不跟着走，进度会**倒退**回本 run 起点；
    #   (3) 最后一个 run 的累加若发生在收尾之后，则永远进不了显示。
    if rep is not None:
        rep.set_base(rep.base_units + rep.total_epochs * rep.steps_per_epoch)

    write_csv(out_run_dir / "metrics.csv", history,
              ["epoch", "train_loss", "val_loss", "accuracy",
               "precision", "recall", "f1", "kappa"])
    write_json(out_run_dir / "best_metrics.json", {
        "arm": arm, "variant": variant, "seed": int(seed), "fold": int(fold),
        "switches": cfg, "params": n_params, "fingerprint": fp,
        "fingerprint_equals_baseline": fp == ARCHITECTURE_FINGERPRINT,
        "params_expected": exp, "params_matches_doc": (exp is None or n_params == exp),
        "best": best, "epochs_run": epochs, "elapsed_seconds": elapsed,
        "val_rows": len(val_rows),
        "val_class_counts": dict(Counter(r["class_name"] for r in val_rows)),
        "sealed_test_untouched": True,
    }, Path(__file__))
    print(f"  → {out_run_dir.name} 完成：best={best['accuracy']:.4f}@{best['epoch']} "
          f"用时 {fmt_hms(elapsed)}", flush=True)

    return {
        "arm": arm, "variant": variant, "seed": int(seed), "fold": int(fold),
        "params": n_params, "fingerprint": fp,
        "params_matches_doc": (exp is None or n_params == exp),
        "epochs_run": epochs, "best_epoch": best["epoch"],
        "accuracy": best["accuracy"],
        "precision": best.get("precision"), "recall": best.get("recall"),
        "f1": best.get("f1"), "kappa": best.get("kappa"),
        "val_loss": best.get("val_loss"), "elapsed_seconds": elapsed,
        "skipped": False,
    }


# =============================================================================
# 4. 任务矩阵
# =============================================================================
def build_jobs(parts: Sequence[str], folds_override: Optional[Sequence[int]],
               epochs: int) -> List[dict]:
    jobs: List[dict] = []
    for key in PART_ORDER:
        if key not in parts:
            continue
        spec = PART_SPECS[key]
        folds = tuple(folds_override) if folds_override else tuple(spec["folds"])
        for arm in spec["arms"]:
            for variant in spec["variants"]:
                for seed in spec["seeds"]:
                    for fold in folds:
                        jobs.append({
                            "part": key, "part_title": spec["title"],
                            "arm": arm, "variant": variant,
                            "seed": int(seed), "fold": int(fold),
                            "label": f"{arm}__{variant}__seed{seed}__fold{fold}",
                        })
    # 去重守卫：同一个 (arm,variant,seed,fold) 不能出现两次
    seen = set()
    for j in jobs:
        k = (j["arm"], j["variant"], j["seed"], j["fold"])
        if k in seen:
            raise ValueError(f"任务矩阵有重复项：{k}")
        seen.add(k)
    return jobs


def leak_descriptors(rows: Sequence[dict]) -> dict:
    """量化「同源泄漏」：group 跨折数 / sha 跨折数 / 被同源污染的 val 数。"""
    gf: Dict[str, set] = defaultdict(set)
    sf: Dict[str, set] = defaultdict(set)
    for r in rows:
        f = int(r["val_fold"])
        gf[r["group_id"]].add(f)
        sf[r["sha256"]].add(f)
    tainted: Dict[str, int] = {}
    for fold in range(1, EXPECTED_FOLDS + 1):
        tg = {r["group_id"] for r in rows if int(r["val_fold"]) != fold}
        tainted[str(fold)] = sum(
            1 for r in rows
            if int(r["val_fold"]) == fold and r["group_id"] in tg
        )
    return {
        "group_cross_fold": sum(1 for v in gf.values() if len(v) > 1),
        "sha_cross_fold": sum(1 for v in sf.values() if len(v) > 1),
        "n_groups": len(gf),
        "val_tainted_per_fold": tainted,
        "val_tainted_total": sum(tainted.values()),
    }


# =============================================================================
# 5. 预检
# =============================================================================
def precheck(*, bdir: Path, bdir_rows: List[dict], control_rows: Optional[List[dict]],
             control_path: Optional[Path], data_root: str, out_dir: Path,
             jobs: List[dict], device_str: str) -> Tuple[bool, dict]:
    checks: List[dict] = []
    hard_fail = False

    def add(item: str, status: str, detail: str) -> None:
        nonlocal hard_fail
        checks.append({"item": item, "status": status, "detail": detail})
        if status == "FAIL":
            hard_fail = True

    # 1 基线 manifest
    man = bdir / TRAIN_MANIFEST_NAME
    if not man.is_file():
        add("基线 manifest", "FAIL", f"不存在：{man}")
    elif sha256_file(man) != TRAIN_MANIFEST_SHA256:
        add("基线 manifest", "FAIL", f"SHA 不符：{sha256_file(man)}")
    else:
        add("基线 manifest", "OK", f"{TRAIN_MANIFEST_NAME} SHA 与冻结常量一致")

    # 2 图像级对照 manifest（只在需要 control 臂时才是硬条件）
    need_control = any(j["arm"] == "control" for j in jobs)
    if control_path is None or not control_path.is_file():
        add("对照 manifest", "FAIL" if need_control else "SKIP",
            f"不存在：{control_path}（需要时用 c0_make_control_manifest.py 生成并上传）")
    elif sha256_file(control_path) != CONTROL_MANIFEST_SHA256:
        add("对照 manifest", "FAIL",
            f"SHA 不符！期望 {CONTROL_MANIFEST_SHA256[:16]}… "
            f"实际 {sha256_file(control_path)[:16]}…")
    else:
        add("对照 manifest", "OK", f"{control_path.name} SHA 与冻结常量一致")

    # 3 基线清单结构不变量（严格）
    try:
        assert_train_manifest(bdir_rows)
        d = leak_descriptors(bdir_rows)
        if d["group_cross_fold"] or d["sha_cross_fold"]:
            add("基线清单守卫", "FAIL",
                f"基线清单竟有跨折同源：group={d['group_cross_fold']} sha={d['sha_cross_fold']}")
        else:
            add("基线清单守卫", "OK",
                f"{len(bdir_rows):,} 行 / 每折 8192+2048 / 同源跨折=0 / 每类每折 512 val")
    except Exception as e:  # noqa: BLE001
        add("基线清单守卫", "FAIL", f"{type(e).__name__}: {e}")

    # 4 对照清单守卫（放宽同源检查 + 正向断言"确实忽略了 group"）
    if control_rows is not None:
        try:
            assert_train_manifest(control_rows, allow_group_relax=True)
            d = leak_descriptors(control_rows)
            if d["group_cross_fold"] <= 0:
                add("对照清单守卫", "FAIL",
                    "对照清单没有任何 group 跨折 ⇒ 它其实还是分组的，对照无效")
            else:
                add("对照清单守卫", "OK",
                    f"结构不变量 PASS；group 跨折={d['group_cross_fold']}、"
                    f"sha 跨折={d['sha_cross_fold']}、"
                    f"被同源污染的 val={d['val_tainted_total']}")
        except Exception as e:  # noqa: BLE001
            add("对照清单守卫", "FAIL", f"{type(e).__name__}: {e}")
    else:
        add("对照清单守卫", "SKIP", "本次任务里没有 control 臂")

    # 5 参数期望值守卫（变体名必须都在表里）
    bad = [j["variant"] for j in jobs if j["variant"] not in EXPECTED_VARIANT_PARAMS]
    if bad:
        add("变体名守卫", "FAIL", f"未知变体：{sorted(set(bad))}")
    else:
        add("变体名守卫", "OK",
            f"{sorted({j['variant'] for j in jobs})} 均有文档参数量可校验")

    # 6 数据集
    train_root = Path(data_root).expanduser() / "train"
    if not train_root.is_dir():
        add("数据目录", "FAIL", f"找不到 {train_root}（--data-root 可覆盖）")
    else:
        missing = [c for c in EXPECTED_CLASSES if not (train_root / c).is_dir()]
        if missing:
            add("数据目录", "FAIL", f"缺类别目录：{missing}")
        else:
            add("数据目录", "OK", f"{train_root} 四类目录齐全（只读，不写入）")

    # 7 输出目录与磁盘
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil as _sh
        free_gb = _sh.disk_usage(str(out_dir)).free / 1024 ** 3
        if free_gb < 2:
            add("磁盘空间", "FAIL", f"可用 {free_gb:.1f} GB < 2 GB")
        elif free_gb < 8:
            add("磁盘空间", "WARN", f"可用 {free_gb:.1f} GB（本实验只写 csv/json，够用）")
        else:
            add("磁盘空间", "OK", f"可用 {free_gb:.1f} GB")
    except Exception as e:  # noqa: BLE001
        add("磁盘空间", "FAIL", f"{type(e).__name__}: {e}")

    # 8 GPU
    if device_str.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                add("GPU 显存", "OK",
                    f"{torch.cuda.get_device_name(0)}：空闲 {free_b/1024**3:.1f} / "
                    f"总 {total_b/1024**3:.1f} GB（单 run 训练峰值约 5.1 GB）")
            else:
                add("GPU 显存", "FAIL", "指定了 cuda 但 torch.cuda.is_available()=False")
        except Exception as e:  # noqa: BLE001
            add("GPU 显存", "FAIL", f"{type(e).__name__}: {e}")
    else:
        add("GPU 显存", "OK", f"device={device_str}，无需检查")

    return (not hard_fail), {"checks": checks}


def variant_probe(bdir: Path, variants: Sequence[str]) -> List[dict]:
    """逐个构造变体模型，比对参数量与指纹（CPU 上也能跑，用于 --dry-run）。"""
    import torch
    smoke_mod = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke_probe")
    ap = load_arch_probe([None])
    out: List[dict] = []
    for v in variants:
        rec: Dict[str, object] = {"variant": v}
        try:
            create, restore = ap.make_switch_patch(smoke_mod, **VARIANT_SWITCHES[v])
            try:
                m = create(num_classes=len(EXPECTED_CLASSES))
                fp = ap.fingerprint(m)
                pr = sum(p.numel() for p in m.parameters())
                m.eval()
                with torch.no_grad():
                    y = m(torch.randn(1, 3, 224, 224))
                fwd_ok = tuple(y.shape) == (1, len(EXPECTED_CLASSES))
            finally:
                restore()
            exp = EXPECTED_VARIANT_PARAMS[v]
            rec.update({
                "params": pr, "params_expected": exp, "params_delta": pr - exp,
                "params_matches_doc": pr == exp,
                "fingerprint": fp, "forward_ok": fwd_ok,
                "equals_baseline": fp == ARCHITECTURE_FINGERPRINT,
                "ok": fwd_ok and (pr == exp) and (fp == ARCHITECTURE_FINGERPRINT
                                                  if v == "full" else True),
            })
        except Exception as e:  # noqa: BLE001
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        out.append(rec)
    return out


# =============================================================================
# 6. 汇总与报告
# =============================================================================
def _find(results: Sequence[dict], **kw) -> Optional[dict]:
    for r in results:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def _acc(results: Sequence[dict], **kw) -> Optional[float]:
    r = _find(results, **kw)
    return None if r is None else r.get("accuracy")


def summarize(results: List[dict], parts: Sequence[str],
              ctrl_diag: Optional[dict] = None,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None) -> dict:
    # ⚠️ 不要写死 BASE_EPOCHS/BASE_BATCH_SIZE：正式跑恰好等于默认值，但一旦用
    #    --epochs/--batch-size 覆盖，写死就会让报告头部与事实不符（"报告说谎"）。
    #    这里优先取调用方传入的实参，其次取结果里记录的 epochs_run，最后才回退默认值。
    run_epochs = epochs
    if run_epochs is None:
        _rs = [r.get("epochs_run") for r in results if r.get("epochs_run")]
        run_epochs = max(_rs) if _rs else BASE_EPOCHS
    summary: Dict[str, object] = {
        "experiment": "G4 · 对照五折 + 跨种子 + 完整消融矩阵",
        "epochs": int(run_epochs),
        "batch_size": int(batch_size if batch_size is not None else BASE_BATCH_SIZE),
        "selection_rule": "折内按 val accuracy 选 best epoch（与基线一致）",
        "sealed_test_untouched": True,
        "parts": list(parts),
        "n_runs": len(results),
    }

    # ---------- 表1：对照五折（分组 vs 图像级） ----------
    folds5 = [1, 2, 3, 4, 5]
    g_acc = {f: _acc(results, arm="grouped", variant="full", seed=BASE_SEED, fold=f)
             for f in folds5}
    c_acc = {f: _acc(results, arm="control", variant="full", seed=BASE_SEED, fold=f)
             for f in folds5}
    gv = [v for v in g_acc.values() if v is not None]
    cv = [v for v in c_acc.values() if v is not None]

    def stat(vals: Sequence[float]) -> dict:
        if not vals:
            return {"n_folds": 0}
        d = {
            "n_folds": len(vals),
            "accuracy_mean": statistics.fmean(vals),
            "accuracy_min": min(vals), "accuracy_max": max(vals),
            "fold_spread_pp": (max(vals) - min(vals)) * 100.0,
            "params": EXPECTED_VARIANT_PARAMS["full"],
        }
        if len(vals) > 1:
            d["accuracy_stdev"] = statistics.stdev(vals)
        return d

    summary["protocol_arms"] = {
        "grouped": {**stat(gv), "per_fold": {str(f): g_acc[f] for f in folds5}},
        "control": {**stat(cv), "per_fold": {str(f): c_acc[f] for f in folds5}},
    }

    deltas = {f: (c_acc[f] - g_acc[f]) for f in folds5
              if c_acc.get(f) is not None and g_acc.get(f) is not None}
    if deltas:
        dv = list(deltas.values())
        summary["protocol_delta"] = {
            "definition": "Δ = acc(图像级随机划分) − acc(P-CV v2 分组划分)；正值 = 分组协议压掉的乐观偏差",
            "per_fold_pp": {str(k): v * 100.0 for k, v in deltas.items()},
            "mean_pp": statistics.fmean(dv) * 100.0,
            "n_folds": len(dv),
            "attributable": True,
            "confounders": "两臂同脚本、同 GPU、同 data_root、同 seed、同超参；唯一变量 = 划分清单",
            "caveat": "n≤5 折，无 CI，禁止声称显著性；只在 |Δ| > 折间跨度时才有资格谈差异。",
        }
    else:
        summary["protocol_delta"] = None

    # ---------- 表2：消融矩阵（Δ 对阶段1 的 full，完全对齐） ----------
    abl_folds = sorted({r["fold"] for r in results if r["variant"] != "full"})
    ablation = []
    for v in ["no_sam", "no_mdc", "no_ilf", "no_dual_path"]:
        recs = sorted((r for r in results
                       if r["variant"] == v and r["arm"] == "grouped"
                       and r["seed"] == BASE_SEED),
                      key=lambda r: r["fold"])
        if not recs:
            continue
        accs = [r["accuracy"] for r in recs]
        d = {
            "variant": v,
            "params": recs[0]["params"],
            "params_delta_vs_full": recs[0]["params"] - EXPECTED_VARIANT_PARAMS["full"],
            "params_matches_doc": recs[0].get("params_matches_doc"),
            "n_folds": len(accs),
            "accuracy_mean": statistics.fmean(accs),
            "per_fold": {str(r["fold"]): r["accuracy"] for r in recs},
            "best_epochs": [r["best_epoch"] for r in recs],
            "fold_spread_pp": (max(accs) - min(accs)) * 100.0 if len(accs) > 1 else 0.0,
        }
        # Δ vs full：只用**同折**都有数的折
        pd: Dict[str, float] = {}
        for r in recs:
            f = r["fold"]
            fa = g_acc.get(f)
            if fa is not None and r["accuracy"] is not None:
                pd[str(f)] = (r["accuracy"] - fa) * 100.0
        if pd:
            d["delta_vs_full_pp"] = pd
            d["delta_vs_full_mean_pp"] = statistics.fmean(pd.values())
            d["reference"] = ("阶段1 同脚本/同卡/同 data_root/同 seed 的 `full` 折内最优 val 准确率"
                              "（唯一残留混淆 = 容量变化本身）")
        ablation.append(d)
    summary["ablation"] = ablation
    full_accs = [g_acc[f] for f in abl_folds if g_acc.get(f) is not None]
    if full_accs:
        summary["ablation_reference_full"] = {
            "per_fold": {str(f): g_acc[f] for f in abl_folds},
            "accuracy_mean": statistics.fmean(full_accs),
            "n_folds": len(full_accs),
            "source": "阶段1（A1）本脚本自跑",
        }

    # ---------- 表3：跨种子离散度 ----------
    seeds = [j for j in (BASE_SEED, 43, 44)]
    seed_per_fold: Dict[str, dict] = {}
    for f in folds5:
        vals = {str(s): _acc(results, arm="grouped", variant="full", seed=s, fold=f)
                for s in seeds}
        got = [v for v in vals.values() if v is not None]
        if len(got) < 2:
            if got:
                seed_per_fold[str(f)] = {"values": vals, "note": "可用种子 <2，无法给离散度"}
            continue
        seed_per_fold[str(f)] = {
            "values": vals,
            "mean": statistics.fmean(got),
            "stdev": statistics.stdev(got),
            "range_pp": (max(got) - min(got)) * 100.0,
        }
    ranges = [v["range_pp"] for v in seed_per_fold.values() if "range_pp" in v]
    summary["seed_dispersion"] = {
        "seeds": seeds,
        "per_fold": seed_per_fold,
        "max_range_pp": max(ranges) if ranges else None,
        "mean_range_pp": statistics.fmean(ranges) if ranges else None,
        "caveat": ("只评折内 val，**不碰 sealed test**。这正是 G2(mode=final) 缺的那一维——"
                   "G2 只落了 train_loss。n = 种子数 × 折数，样本量小，只能报离散度。"),
    } if seed_per_fold else None

    summary["runs"] = results
    summary["control_manifest_diagnostics"] = ctrl_diag
    return summary


def write_report(summary: dict, out_path: Path) -> None:
    L: List[str] = []
    L.append("# G4 · 对照五折 + 跨种子 + 完整消融矩阵（实测）")
    L.append("")
    L.append("> 评估集：**折内验证集（val）**；`sealed test` **未被触碰**（代码里根本不引用它）。")
    L.append(f"> 选模规则：折内按 val accuracy 选 best epoch（与基线一致）；"
             f"训练 {summary['epochs']} epoch、batch={summary['batch_size']}。")
    L.append("> 全部 run 同脚本、同 GPU、同 data_root、同 seed 口径 —— 因此各表里的 Δ 可归因。")
    L.append("")

    arms = summary.get("protocol_arms") or {}
    L.append("## 1. 对照五折：P-CV v2 分组协议到底压掉了多少乐观偏差")
    L.append("")
    L.append("| 臂 | 划分依据 | 折数 | val 准确率(均) | min | max | 折间跨度 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    nm = {"grouped": "P-CV v2 分组（基线清单）", "control": "图像级随机（忽略 group_id）"}
    for arm in ("grouped", "control"):
        a = arms.get(arm)
        if not a or not a.get("n_folds"):
            continue
        L.append(f"| `{arm}` | {nm[arm]} | {a['n_folds']} | {a['accuracy_mean']*100:.3f}% | "
                 f"{a['accuracy_min']*100:.3f}% | {a['accuracy_max']*100:.3f}% | "
                 f"{a['fold_spread_pp']:.3f} pp |")
    L.append("")
    d = summary.get("protocol_delta")
    if d:
        L.append("| 折 | Δ (pp) |")
        L.append("|---:|---:|")
        for f, v in d["per_fold_pp"].items():
            L.append(f"| {f} | {v:+.3f} |")
        L.append(f"| **均值** | **{d['mean_pp']:+.3f}** |")
        L.append("")
        L.append(f"✅ 可归因：{d['confounders']}。")
        L.append(f"⚠️ {d['caveat']}")
    else:
        L.append("（未同时跑两臂，无法给 Δ）")
    L.append("")

    ref = summary.get("ablation_reference_full")
    L.append("## 2. 模块消融矩阵（Δ 对同脚本自跑的 `full`，完全对齐）")
    L.append("")
    if ref:
        L.append(f"参照 `full`（阶段1 自跑）：{ref['accuracy_mean']*100:.3f}%"
                 f"（{ref['n_folds']} 折；各折 " +
                 "、".join(f"{k}:{v*100:.3f}%" for k, v in ref["per_fold"].items()) + "）")
        L.append("")
    L.append("| 变体 | 参数量 | ΔParams vs full | 折数 | val 准确率(均) | Δ vs full (pp) | 各折 best epoch |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    _abl = summary.get("ablation") or []
    for a in _abl:
        dvs = a.get("delta_vs_full_mean_pp")
        L.append(f"| `{a['variant']}` | {a['params']:,} | {a['params_delta_vs_full']:+,} | "
                 f"{a['n_folds']} | {a['accuracy_mean']*100:.3f}% | "
                 f"{'—' if dvs is None else f'{dvs:+.3f}'} | {a['best_epochs']} |")
    L.append("")
    L.append("⚠️ **容量不匹配**：关掉模块会同时改变参数量（`no_mdc` −1,177 万、"
             "`no_dual_path` −1,006 万、`no_ilf` −132 万）。只有 `no_sam`（ΔParams −459）"
             "是容量匹配的，**只有它**的 Δ 能直接归因给 SAM 本身；其余只能表述为"
             "「移除 X 后的整体效果（含容量变化）」。")
    _n_abl = max((a.get("n_folds") or 0) for a in _abl) if _abl else 0
    if _n_abl >= EXPECTED_FOLDS:
        L.append(f"⚠️ 折数 {_n_abl}（完整五折）；仍**不给 CI**，禁止声称显著性。")
    else:
        L.append(f"⚠️ 折数仅 {_n_abl}，无 CI，禁止声称显著性。")
    L.append("")

    s = summary.get("seed_dispersion")
    L.append("## 3. 跨种子离散度（补 G2 缺的 val/acc 那一维）")
    L.append("")
    if s:
        seeds = s["seeds"]
        L.append("| 折 | " + " | ".join(f"seed {x}" for x in seeds) + " | 均值 | range (pp) |")
        L.append("|---:|" + "---:|" * (len(seeds) + 2))
        for f, v in s["per_fold"].items():
            cells = " | ".join(
                "—" if v["values"].get(str(x)) is None else f"{v['values'][str(x)]*100:.3f}%"
                for x in seeds)
            if "range_pp" in v:
                L.append(f"| {f} | {cells} | {v['mean']*100:.3f}% | {v['range_pp']:.3f} |")
            else:
                L.append(f"| {f} | {cells} | — | — |")
        L.append("")
        if s.get("max_range_pp") is not None:
            L.append(f"最大跨种子极差：**{s['max_range_pp']:.3f} pp**"
                     f"（均值 {s['mean_range_pp']:.3f} pp）")
        L.append("")
        L.append(f"⚠️ {s['caveat']}")
    else:
        L.append("（未跑额外种子）")
    L.append("")

    cd = summary.get("control_manifest_diagnostics")
    L.append("## 4. 对照清单的泄漏描述（证明对照确实忽略了 group）")
    L.append("")
    if cd:
        L.append(f"- group 跨折数：**{cd['group_cross_fold']}**（基线为 0）")
        L.append(f"- exact-SHA 跨折数：**{cd['sha_cross_fold']}**（基线为 0）")
        L.append(f"- 被同源污染的 val 总数：**{cd['val_tainted_total']}**"
                 f"（各折 {cd['val_tainted_per_fold']}）")
    else:
        L.append("（本次未记录）")
    L.append("")
    L.append("## 5. 必须随数字一起写的诚实说明")
    L.append("")
    L.append("1. **不碰 sealed test**：全程只评折内 val；代码里根本不引用 `pcv_test_manifest_SEALED.csv`。")
    L.append("2. **不改基线**：`smoke_test.py` / `final_train.py` / 基线 manifest 一字节未动；"
             "消融靠调用层 monkeypatch，`full` 过指纹硬门（== `8e542321…d76975`）。")
    L.append("3. **样本量小**：对照 5 折、消融 2 折、跨种子 3 个 —— 均不给 CI，不得声称显著性。")
    L.append("4. **Δ 的方向解释**：对照表里 Δ>0 表示图像级随机划分把同源副本漏进 val、抬高精度，"
             "这部分就是分组协议封掉的乐观偏差。")
    L.append("5. **消融的残留混淆**：本次参照已与消融同脚本/同卡/同 data_root/同 seed，"
             "残留混淆只剩「容量变化」，故比 G1 的「参考性对照」强；但 `no_mdc`/`no_ilf`/"
             "`no_dual_path` 仍不可归因到单模块。")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# =============================================================================
# 7. main
# =============================================================================
def main() -> int:
    ap = base_parser("G4 · 对照五折 + 跨种子 + 完整消融矩阵（一次跑完）")
    ap.add_argument("--parts", default=DEFAULT_PARTS,
                    help=f"要跑的阶段，逗号分隔（默认 {DEFAULT_PARTS}）；可选 {PART_ORDER}")
    ap.add_argument("--folds", default=None,
                    help="覆盖各阶段的折号（默认按阶段自身定义：所有阶段均为 1-5；"
                         "加 --folds 1,5 可退回消融 2 折）")
    ap.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BASE_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="只跑前 N 个 batch（自检用；正式跑留空）")
    ap.add_argument("--progress-interval", type=float, default=5.0,
                    help="全局进度行最小间隔（秒）")
    ap.add_argument("--quiet", action="store_true",
                    help="不转发基线 tqdm 的逐 batch 帧（后台/nohup 跑时用，"
                         "否则 409.6 万帧会把重定向日志写成几百 MB）；"
                         "`[进度]` 行与 epoch 汇总行仍照常输出，信息不丢")
    ap.add_argument("--arch-probe", default=None, help="arch_probe.py 路径（默认自动找）")
    ap.add_argument("--data-root", default=None,
                    help=f"Combined Dataset 路径（默认 {FIXED_DATA_ROOT}）")
    ap.add_argument("--plan", action="store_true", help="只预检 + 报总时长，不训练")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：已有 best_metrics.json 的 run 自动跳过")
    ap.add_argument("--hours-per-run", type=float, default=None,
                    help=f"ETA 用的单 run 时长（默认 {MEASURED_SECONDS_PER_RUN/3600:.2f} h，G1 实测）")
    a = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = resolve_out_dir(a.out_dir, OUT_TAG)
    bdir = find_baseline_dir(a.baseline_dir)

    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PART_SPECS]
    if unknown:
        print(f"❌ 未知阶段 {unknown}；可选 {PART_ORDER}", flush=True)
        return 2
    folds_override = ([int(x) for x in a.folds.split(",") if x.strip()]
                      if a.folds else None)
    for f in (folds_override or (1, 2, 3, 4, 5)):
        if not 1 <= f <= EXPECTED_FOLDS:
            print(f"❌ 非法折号 {f}", flush=True)
            return 2

    jobs = build_jobs(parts, folds_override, a.epochs)
    if not jobs:
        print("❌ 没有生成任何任务", flush=True)
        return 2

    print("=" * 78, flush=True)
    print("G4 · 对照五折 + 跨种子 + 完整消融矩阵", flush=True)
    print("=" * 78, flush=True)
    print(f"包版本      : {PKG_VERSION}", flush=True)
    print(f"baseline dir: {bdir}", flush=True)
    print(f"out dir     : {out_dir}", flush=True)
    print(f"阶段        : {parts}", flush=True)
    print(f"任务        : {len(jobs)} 个 run", flush=True)
    for key in PART_ORDER:
        if key not in parts:
            continue
        n = sum(1 for j in jobs if j["part"] == key)
        print(f"   {key:<3} {PART_SPECS[key]['title']:<40} {n:>2} run", flush=True)

    # Windows 本地自检提示（云端 Linux 用 fork，不受影响）
    if os.name == "nt" and a.num_workers > 0 and not a.plan:
        print("\n⚠️  Windows 本地自检提示：本机 DataLoader 走 spawn，动态加载的 smoke_test "
              "在子进程里 import 不到（ModuleNotFoundError: dmsa_baseline_smoke）。\n"
              "      本机自检请加 --num-workers 0；云端保持默认 6。", flush=True)

    # ---- 载入两臂清单 ----
    man = bdir / TRAIN_MANIFEST_NAME
    bdir_rows: List[dict] = []
    if man.is_file():
        try:
            bdir_rows = read_manifest(man)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 基线 manifest 读不动：{type(e).__name__}: {e}", flush=True)
    control_path = bdir / CONTROL_MANIFEST_NAME
    if not control_path.is_file():
        control_path = HERE / CONTROL_MANIFEST_NAME
    control_rows: Optional[List[dict]] = None
    if control_path.is_file():
        try:
            control_rows = read_manifest(control_path)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 对照 manifest 读不动：{type(e).__name__}: {e}", flush=True)

    data_root = resolve_data_root(a.data_root)

    # ---- 预检 ----
    print("\n" + "-" * 78, flush=True)
    print("预检查", flush=True)
    print("-" * 78, flush=True)
    ok, pr = precheck(bdir=bdir, bdir_rows=bdir_rows, control_rows=control_rows,
                      control_path=control_path, data_root=data_root,
                      out_dir=out_dir, jobs=jobs, device_str=a.device)
    for c in pr["checks"]:
        icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "· "}[c["status"]]
        print(f"  {icon} [{c['status']:<4}] {c['item']:<14} {c['detail']}", flush=True)
    if not ok:
        print("\n❌ 预检未通过，拒绝开跑（不做任何训练）。", flush=True)
        write_json(out_dir / "g4_precheck.json", pr, script_path)
        return 2

    ctrl_diag = leak_descriptors(control_rows) if control_rows is not None else None
    if ctrl_diag:
        print(f"  对照清单泄漏：group 跨折={ctrl_diag['group_cross_fold']}、"
              f"sha 跨折={ctrl_diag['sha_cross_fold']}、"
              f"被同源污染的 val={ctrl_diag['val_tainted_total']}", flush=True)

    hpr = a.hours_per_run or (MEASURED_SECONDS_PER_RUN / 3600.0)
    total_h = len(jobs) * hpr
    print("-" * 78, flush=True)
    print(f"  结论：预检通过 ｜ {len(jobs)} 个 run × {hpr:.2f} h ≈ **{total_h:.1f} h**"
          f"（按 G1 实测单 run 外推，非估算）", flush=True)
    print("-" * 78, flush=True)

    # ---- dry-run：CPU 上验证 5 个变体 + 指纹/参数量硬门 ----
    if a.dry_run:
        all_variants = [v for v in EXPECTED_VARIANT_PARAMS
                        if any(j["variant"] == v for j in jobs)] or list(EXPECTED_VARIANT_PARAMS)
        print(f"\n-- dry-run：逐个构造 {len(all_variants)} 个变体并校验 --------------", flush=True)
        probe = variant_probe(bdir, all_variants)
        for p in probe:
            if p.get("ok"):
                d_full = p["params"] - EXPECTED_VARIANT_PARAMS["full"]
                print(f"    {p['variant']:<14} params={p['params']:>12,} "
                      f"Δvs full={d_full:>+12,} fwd=OK 参数=文档值 "
                      f"{'[= 基线指纹]' if p['equals_baseline'] else ''}", flush=True)
            else:
                print(f"    {p['variant']:<14} ❌ {p.get('error') or '参数量/前向不符'}"
                      f"  params={p.get('params')} 期望={p.get('params_expected')}", flush=True)
        gate_ok = all(p.get("ok") for p in probe)
        print(f"\n{'✅' if gate_ok else '❌'} dry-run "
              f"{'通过' if gate_ok else '未通过'}："
              f"{len(probe)}/{len(all_variants)} 个变体参数与指纹均符合文档值。", flush=True)
        write_json(out_dir / "g4_dryrun.json",
                   {"precheck": pr, "variants": probe, "gate_passed": gate_ok,
                    "n_jobs": len(jobs)}, script_path)
        return 0 if gate_ok else 2

    if a.plan:
        print("\n（--plan：只预检，不训练。去掉 --plan 即正式开跑。）", flush=True)
        write_json(out_dir / "g4_precheck.json",
                   {**pr, "n_jobs": len(jobs),
                    "jobs": [j["label"] for j in jobs],
                    "estimated_hours": total_h, "hours_per_run": hpr,
                    "control_diag": ctrl_diag}, script_path)
        return 0

    # ---- 正式训练 ----
    cpu_smoke(bdir)
    device, dev_info = probe_device(a.device)
    print(f"device      : {dev_info}", flush=True)

    runs_root = out_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    sink = make_tee_sink(out_dir / "g4_progress.log")

    smoke_mod = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke")
    arch_probe = load_arch_probe([a.arch_probe])
    class_to_idx = {n: i for i, n in enumerate(sorted(EXPECTED_CLASSES))}
    train_root = Path(data_root).expanduser() / "train"

    # ---- 断点续跑：先扫一遍已完成的 run ----
    done_flags: List[bool] = []
    for j in jobs:
        done_flags.append((runs_root / j["label"] / "best_metrics.json").is_file())
    if a.resume:
        n_done = sum(done_flags)
        print(f"\n断点续跑：{n_done}/{len(jobs)} 个 run 已完成，将跳过。", flush=True)

    epochs = a.epochs
    steps_per_epoch = (8192 + a.batch_size - 1) // a.batch_size
    if a.max_batches:
        steps_per_epoch = min(steps_per_epoch, a.max_batches)
    units_per_run = epochs * steps_per_epoch
    pre_done = sum(units_per_run for d, j in zip(done_flags, jobs) if d and a.resume)

    rep = ProgressReporter(
        label="G4", total_epochs=epochs, steps_per_epoch=steps_per_epoch,
        total_units=len(jobs) * units_per_run, sink=sink,
        min_interval=a.progress_interval, base_units=pre_done,
    )

    results: List[dict] = []
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        label = j["label"]
        run_dir = runs_root / label

        if a.resume and (run_dir / "best_metrics.json").is_file():
            bm = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))
            best = bm.get("best") or {}
            results.append({
                "part": j["part"], "arm": j["arm"], "variant": j["variant"],
                "seed": j["seed"], "fold": j["fold"], "label": label,
                "params": bm.get("params"), "fingerprint": bm.get("fingerprint"),
                "params_matches_doc": bm.get("params_matches_doc", True),
                "epochs_run": bm.get("epochs_run"),
                "best_epoch": best.get("epoch"), "accuracy": best.get("accuracy"),
                "precision": best.get("precision"), "recall": best.get("recall"),
                "f1": best.get("f1"), "kappa": best.get("kappa"),
                "val_loss": best.get("val_loss"),
                "elapsed_seconds": bm.get("elapsed_seconds"), "skipped": True,
            })
            print(f"\n[跳过 {i}/{len(jobs)}] {label}（已有产物）", flush=True)
            continue

        print("\n" + "=" * 78, flush=True)
        print(f"[{i}/{len(jobs)}]  {j['part']} · {j['part_title']}", flush=True)
        print(f"  run = {label}", flush=True)
        print(f"  arm={j['arm']}  variant={j['variant']}  seed={j['seed']}  fold={j['fold']}",
              flush=True)
        print("=" * 78, flush=True)

        rows = bdir_rows if j["arm"] == "grouped" else control_rows
        if rows is None:
            raise RuntimeError(f"arm={j['arm']} 但没有对应清单（对照清单缺失？）")

        rec = train_one_run(
            smoke_mod=smoke_mod, arch_probe=arch_probe,
            arm=j["arm"], variant=j["variant"], seed=j["seed"], fold=j["fold"],
            rows=rows, class_to_idx=class_to_idx, train_root=train_root,
            device=device, epochs=epochs, batch_size=a.batch_size,
            num_workers=a.num_workers, out_run_dir=run_dir,
            max_batches=a.max_batches, rep=rep, sink=sink,
            run_index=i, run_total=len(jobs),
            tqdm_passthrough=not a.quiet,
        )
        rec.update({"part": j["part"], "label": label})
        results.append(rec)

    elapsed = time.time() - t0

    # ---- 收尾 ----
    rep.total_epochs = 0
    rep.set_base(0)
    rep.set(epoch=0, step=0, loss=None, extra="G4 全部 run 完成")
    rep.emit(force=True)

    summary = summarize(results, parts, ctrl_diag,
                        epochs=a.epochs, batch_size=a.batch_size)
    summary["elapsed_seconds"] = elapsed
    summary["control_manifest"] = {"name": CONTROL_MANIFEST_NAME,
                                   "sha256": CONTROL_MANIFEST_SHA256}
    write_json(out_dir / "g4_summary.json", summary, script_path)
    write_csv(out_dir / "g4_per_run.csv", results,
              ["part", "arm", "variant", "seed", "fold", "label", "params",
               "fingerprint", "params_matches_doc", "best_epoch", "accuracy",
               "precision", "recall", "f1", "kappa", "val_loss",
               "elapsed_seconds", "epochs_run", "skipped"])
    write_report(summary, out_dir / "g4_report.md")

    print("\n" + "=" * 78, flush=True)
    print(f"G4 完成，总耗时 {fmt_hms(elapsed)}（{len(jobs)} 个 run）", flush=True)
    arms = summary.get("protocol_arms") or {}
    for arm in ("grouped", "control"):
        a_ = arms.get(arm)
        if a_ and a_.get("n_folds"):
            print(f"  {arm:<8} val_acc={a_['accuracy_mean']*100:.3f}%"
                  f"（{a_['n_folds']} 折）", flush=True)
    d = summary.get("protocol_delta")
    if d:
        print(f"  Δ(图像级−分组) = {d['mean_pp']:+.3f} pp", flush=True)
    ref = summary.get("ablation_reference_full")
    if ref:
        print(f"  full 参照 = {ref['accuracy_mean']*100:.3f}%（{ref['n_folds']} 折）", flush=True)
    for a_ in summary.get("ablation") or []:
        dv = a_.get("delta_vs_full_mean_pp")
        print(f"  {a_['variant']:<14} val_acc={a_['accuracy_mean']*100:.3f}%  "
              f"Δvs full={'—' if dv is None else f'{dv:+.3f} pp'}", flush=True)
    s = summary.get("seed_dispersion")
    if s and s.get("max_range_pp") is not None:
        print(f"  跨种子最大极差 = {s['max_range_pp']:.3f} pp", flush=True)
    print(f"  产物：{out_dir}", flush=True)
    print("    g4_summary.json / g4_per_run.csv / g4_report.md / g4_progress.log", flush=True)
    print("  提醒：sealed test 未被本实验触碰。", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
