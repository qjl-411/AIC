#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1 · 对照五折（协议价值量化）＋ 跨种子 CV（补 L6）—— 一次跑完的统一矩阵

它回答两个问题，共用同一套代码、同一台 GPU、同一超参：

问题 1（★★★ 对照五折）：**P-CV v2 的分组协议到底压掉了多少乐观偏差？**
    两臂，只有划分清单不同：
      arm=grouped  → 基线冻结清单 `pcv_v2_folds_seed42.csv`（同源不跨折）
      arm=control  → C0 冻结的图像级清单 `*_imgsplit_seed42_FROZEN.csv`（忽略 group_id）
      Δ = acc(control) − acc(grouped)   ⇒ Δ > 0 的部分就是分组协议压掉的乐观偏差

问题 2（★★ 跨种子，补 L6）：**换个随机种子，val 准确率稳不稳？**
     同一 arm=grouped，额外种子 43/44 跑 `--seed-folds` 指定的折；
     再与 seed=42 的同一折配对 → 得到「跨种子离散度」。
     ⚠️ 这正是 G2（mode=final）没给的东西：G2 只落了 train_loss，没有 val/acc。
     ⚠️ 全程**不碰 sealed test**（只评折内 val）。

为什么这样设计
--------------
* **两臂同源**：所有 run 都走 `g1_ablation.train_variant_fold(variant="full")`，
  所以不存在 G1 那种「参照脚本/GPU/data_root 三处未对齐」的问题，
  Δ 是**可归因**的（唯一变量 = 划分清单）。
* **不改基线**：一个基线文件都不动。图像级清单由 C0 事前冻结、此处用**硬编码
  SHA-256 常量**回校；指纹硬门（`full` 必须 == `8e542321…`）在训练前强制。
* **对照组是真的"忽略 group"**：不是跳过检查就完事，而是**正向断言**
  `group 跨折数 > 0`，否则直接报错——防止拿到一份"其实还是分组"的假对照。

用法
----
    python c1_cv_matrix.py --plan                  # 只预检 + 报时长，不训练
    python c1_cv_matrix.py --dry-run               # CPU 上验证接线（不训练）
    python c1_cv_matrix.py                         # 全跑（默认 14 个 run）
    python c1_cv_matrix.py --arms control          # 只跑对照臂（5 个 run）
    python c1_cv_matrix.py --no-seed-part          # 只跑对照五折双腕
    python c1_cv_matrix.py --folds 1,2             # 先跑 2 折试点
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import C31_run_g1_ablation as g1  # noqa: E402

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_SEED,
    EXPECTED_CLASSES,
    FIXED_DATA_ROOT,
    PKG_VERSION,
    ProgressReporter,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    assert_train_manifest,
    base_parser,
    find_baseline_dir,
    fmt_hms,
    load_module,
    make_tee_sink,
    probe_device,
    read_manifest,
    resolve_data_root,
    resolve_out_dir,
    sha256_file,
    write_csv,
    write_json,
)

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------
# C0 生成的图像级清单（事前冻结）。改动它就等于事后改划分 ⇒ 禁止。
CONTROL_MANIFEST_NAME = "pcv_v2_folds_imgsplit_seed42_FROZEN.csv"
CONTROL_MANIFEST_SHA256 = (
    "d823021990e317fa997fa660ac36ada1abbb3ed2b29cd01712f06e3707c168b8"
)

# 单 run 实测时长（证据：cloud_results/20260925_G1 里 6 个 run 的 elapsed_seconds
# = 9390/10014/8353/9341/10358/9252 s，均值 9451 s = 2.63 h；200 epoch × 512 step，
# 每折 8192 train / 2048 val，RTX 4090）。ETA 用它外推，不靠猜。
MEASURED_SECONDS_PER_RUN = 9451.0

FOLDS_ALL = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# 泄漏描述量（与 c0_make_control_manifest.py 同口径，独立实现以免互相掩盖 bug）
# ---------------------------------------------------------------------------
def leak_descriptors(rows: Sequence[dict]) -> dict:
    group_folds: Dict[str, set] = defaultdict(set)
    sha_folds: Dict[str, set] = defaultdict(set)
    for r in rows:
        f = int(r["val_fold"])
        group_folds[r["group_id"]].add(f)
        sha_folds[r["sha256"]].add(f)

    tainted: Dict[str, int] = {}
    for fold in FOLDS_ALL:
        train_groups = {r["group_id"] for r in rows if int(r["val_fold"]) != fold}
        tainted[str(fold)] = sum(
            1 for r in rows
            if int(r["val_fold"]) == fold and r["group_id"] in train_groups
        )

    return {
        "group_cross_fold": sum(1 for v in group_folds.values() if len(v) > 1),
        "sha_cross_fold": sum(1 for v in sha_folds.values() if len(v) > 1),
        "n_groups": len(group_folds),
        "val_tainted_per_fold": tainted,
        "val_tainted_total": sum(tainted.values()),
    }


# ---------------------------------------------------------------------------
# 任务矩阵
# ---------------------------------------------------------------------------
def build_jobs(arms: str, folds: Sequence[int], extra_seeds: Sequence[int],
               seed_folds: Sequence[int], no_seed_part: bool) -> List[dict]:
    jobs: List[dict] = []
    if arms in ("both", "grouped"):
        for f in folds:
            jobs.append({"arm": "grouped", "seed": BASE_SEED, "fold": f})
    if arms in ("both", "control"):
        for f in folds:
            jobs.append({"arm": "control", "seed": BASE_SEED, "fold": f})
    if not no_seed_part:
        for s in extra_seeds:
            for f in seed_folds:
                jobs.append({"arm": "grouped", "seed": s, "fold": f})
    return jobs


def job_label(j: dict) -> str:
    return f"{j['arm']}_seed{j['seed']}_fold{j['fold']}"


def est_hours(n_runs: int) -> float:
    return n_runs * MEASURED_SECONDS_PER_RUN / 3600.0


# ---------------------------------------------------------------------------
# 预检
# ---------------------------------------------------------------------------
def precheck(*, bdir: Path, bdir_rows: List[dict], control_rows: Optional[List[dict]],
             control_path: Optional[Path], data_root: str, out_dir: Path,
             depth: str) -> Tuple[bool, dict]:
    """所有硬条件一次查完；任何 FAIL 直接 return False（调用方 exit 2）。"""
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
    else:
        s = sha256_file(man)
        if s != TRAIN_MANIFEST_SHA256:
            add("基线 manifest", "FAIL", f"SHA 不符：{s}")
        else:
            add("基线 manifest", "OK", f"{TRAIN_MANIFEST_NAME} SHA 与冻结常量一致")

    # 2 对照 manifest
    if control_path is None or not control_path.is_file():
        add("对照 manifest", "FAIL",
            f"不存在：{control_path}（请先跑 c0_make_control_manifest.py 并上传）")
    else:
        s = sha256_file(control_path)
        if s != CONTROL_MANIFEST_SHA256:
            add("对照 manifest", "FAIL",
                f"SHA 不符！期望 {CONTROL_MANIFEST_SHA256[:16]}… 实际 {s[:16]}…")
        else:
            add("对照 manifest", "OK", f"{control_path.name} SHA 与冻结常量一致")

    # 3 grouped 臂结构不变量（严格：同源不得跨折）
    try:
        assert_train_manifest(bdir_rows)
        d = leak_descriptors(bdir_rows)
        if d["group_cross_fold"] != 0 or d["sha_cross_fold"] != 0:
            add("grouped 臂守卫", "FAIL",
                f"基线清单竟有跨折同源：group={d['group_cross_fold']} sha={d['sha_cross_fold']}")
        else:
            add("grouped 臂守卫", "OK",
                f"10240 行 / 每折 8192+2048 / 同源跨折=0 / 每类每折 512 val")
    except Exception as e:  # noqa: BLE001
        add("grouped 臂守卫", "FAIL", f"{type(e).__name__}: {e}")

    # 4 control 臂结构不变量（放宽同源检查，但必须"确实忽略了 group"）
    if control_rows is not None:
        try:
            assert_train_manifest(control_rows, allow_group_relax=True)
            d = leak_descriptors(control_rows)
            if d["group_cross_fold"] <= 0:
                add("control 臂守卫", "FAIL",
                    "对照清单没有任何 group 跨折 ⇒ 它其实还是分组的，对照无效")
            else:
                add("control 臂守卫", "OK",
                    f"结构不变量 PASS；group 跨折={d['group_cross_fold']}、"
                    f"sha 跨折={d['sha_cross_fold']}、被同源污染的 val={d['val_tainted_total']}")
        except Exception as e:  # noqa: BLE001
            add("control 臂守卫", "FAIL", f"{type(e).__name__}: {e}")
    else:
        add("control 臂守卫", "SKIP", "本次没有 control 臂任务")

    # 5 数据集
    train_root = Path(data_root).expanduser() / "train"
    if not train_root.is_dir():
        add("数据目录", "FAIL", f"找不到 {train_root}（--data-root 可覆盖）")
    else:
        missing = [c for c in EXPECTED_CLASSES if not (train_root / c).is_dir()]
        if missing:
            add("数据目录", "FAIL", f"缺类别目录：{missing}")
        else:
            add("数据目录", "OK", f"{train_root} 四类目录齐全（只读，不写入）")

    # 6 输出目录与磁盘
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil as _sh
        free_gb = _sh.disk_usage(str(out_dir)).free / 1024**3
        if free_gb < 2:
            add("磁盘空间", "FAIL", f"可用 {free_gb:.1f} GB < 2 GB")
        elif free_gb < 8:
            add("磁盘空间", "WARN", f"可用 {free_gb:.1f} GB（本实验只写 csv/json，够用）")
        else:
            add("磁盘空间", "OK", f"可用 {free_gb:.1f} GB")
    except Exception as e:  # noqa: BLE001
        add("磁盘空间", "FAIL", f"{type(e).__name__}: {e}")

    # 7 GPU
    try:
        import torch
        if depth == "gpu" and torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            add("GPU 显存", "OK",
                f"{torch.cuda.get_device_name(0)}：空闲 {free_b/1024**3:.1f} / "
                f"总 {total_b/1024**3:.1f} GB（单 run 训练峰值约 5.1 GB）")
        else:
            add("GPU 显存", "OK", f"device={depth}，无需检查")
    except Exception as e:  # noqa: BLE001
        add("GPU 显存", "SKIP", f"{type(e).__name__}: {e}")

    return (not hard_fail), {"checks": checks}


def fingerprint_gate(bdir: Path) -> dict:
    """构建 `full` 模型，确认与基线同构（硬门）。"""
    import torch
    smoke_mod = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke_gate")
    ap = g1.load_arch_probe([None])
    create, restore = ap.make_switch_patch(smoke_mod, **g1.VARIANT_SWITCHES["full"])
    try:
        m = create(num_classes=len(EXPECTED_CLASSES))
        fp = ap.fingerprint(m)
        params = sum(p.numel() for p in m.parameters())
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(1, 3, 224, 224))
        fwd_ok = tuple(y.shape) == (1, len(EXPECTED_CLASSES))
    finally:
        restore()
    return {
        "fingerprint": fp, "params": params, "forward_ok": fwd_ok,
        "equals_baseline": fp == ARCHITECTURE_FINGERPRINT,
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def run_matrix(*, jobs: List[dict], bdir: Path, bdir_rows: List[dict],
               control_rows: Optional[List[dict]], data_root: str, out_dir: Path,
               device, epochs: int, batch_size: int, num_workers: int,
               max_batches: Optional[int], rep_interval: float,
               resume: bool) -> Tuple[List[dict], Optional[ProgressReporter]]:
    smoke_mod = load_module(bdir / "smoke_test.py", "dmsa_baseline_smoke")
    arch_probe = g1.load_arch_probe([None])
    class_to_idx = {n: i for i, n in enumerate(sorted(EXPECTED_CLASSES))}
    train_root = Path(data_root).expanduser() / "train"
    runs_root = out_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    sink = make_tee_sink(out_dir / "c1_progress.log")
    steps_per_epoch = (8192 + batch_size - 1) // batch_size
    total_units = len(jobs) * epochs * steps_per_epoch
    rep = ProgressReporter(
        label="C1", total_epochs=epochs, steps_per_epoch=steps_per_epoch,
        total_units=total_units, sink=sink, min_interval=rep_interval,
    )

    results: List[dict] = []
    for i, job in enumerate(jobs, 1):
        label = job_label(job)
        out_fold_dir = runs_root / label
        done_file = out_fold_dir / "best_metrics.json"

        if resume and done_file.is_file():
            print(f"\n[skip] {label} 已完成（--resume）")
            bm = json.loads(done_file.read_text(encoding="utf-8"))
            rep.set_base(rep.base_units + epochs * steps_per_epoch)
            results.append({
                "arm": job["arm"], "seed": job["seed"], "fold": job["fold"],
                "label": label, "skipped": True,
                "params": bm.get("params"), "fingerprint": bm.get("fingerprint"),
                "best_epoch": (bm.get("best") or {}).get("epoch"),
                "accuracy": (bm.get("best") or {}).get("accuracy"),
                "elapsed_seconds": bm.get("elapsed_seconds"),
            })
            continue

        rows = bdir_rows if job["arm"] == "grouped" else control_rows
        assert rows is not None, f"arm={job['arm']} 但没有对应清单"

        # ⚠️ train_variant_fold 在调用时读**模块全局** BASE_SEED
        #    （控制初始化种子 + DataLoader shuffle），所以换种子就是改这个全局。
        g1.BASE_SEED = int(job["seed"])

        print(f"\n{'='*78}\n[{i}/{len(jobs)}] {label}"
              f"  arm={job['arm']}  seed={job['seed']}  fold={job['fold']}\n{'='*78}")
        rec = g1.train_variant_fold(
            smoke_mod=smoke_mod, arch_probe=arch_probe, variant="full",
            fold=int(job["fold"]), rows=rows, class_to_idx=class_to_idx,
            train_root=train_root, device=device, epochs=epochs,
            batch_size=batch_size, num_workers=num_workers,
            out_fold_dir=out_fold_dir, max_batches=max_batches,
            rep=rep, sink=print, run_index=i, run_total=len(jobs),
        )
        cn = Counter(r["class_name"] for r in rows if int(r["val_fold"]) == int(job["fold"]))
        rec.update({"arm": job["arm"], "seed": job["seed"], "fold": job["fold"],
                    "label": label, "skipped": False,
                    "val_class_counts": dict(cn)})
        results.append(rec)

    rep.total_epochs = 0
    rep.set_base(0)
    rep.set(epoch=0, step=0, loss=None, extra="C1 全部 run 完成")
    rep.emit(force=True)
    return results, rep


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(results: List[dict], folds: Sequence[int], extra_seeds: Sequence[int],
              seed_folds: Sequence[int], control_diag: Optional[dict]) -> dict:
    def acc_of(arm: str, seed: int, fold: int) -> Optional[float]:
        for r in results:
            if r["arm"] == arm and r["seed"] == seed and r["fold"] == fold:
                return r.get("accuracy")
        return None

    per_arm: Dict[str, dict] = {}
    for arm in ("grouped", "control"):
        accs = [acc_of(arm, BASE_SEED, f) for f in folds]
        accs = [a for a in accs if a is not None]
        if not accs:
            continue
        per_arm[arm] = {
            "n_folds": len(accs),
            "accuracy_mean": statistics.fmean(accs),
            "accuracy_min": min(accs),
            "accuracy_max": max(accs),
            "fold_spread": (max(accs) - min(accs)) if len(accs) > 1 else 0.0,
            "per_fold": {str(f): acc_of(arm, BASE_SEED, f) for f in folds},
            "params": next((r.get("params") for r in results
                            if r["arm"] == arm and r["seed"] == BASE_SEED), None),
        }
        if len(accs) > 1:
            per_arm[arm]["accuracy_stdev"] = statistics.stdev(accs)

    delta = None
    if "grouped" in per_arm and "control" in per_arm:
        per_fold_delta = {}
        for f in folds:
            a, b = acc_of("grouped", BASE_SEED, f), acc_of("control", BASE_SEED, f)
            per_fold_delta[str(f)] = (None if (a is None or b is None) else (b - a))
        vals = [v for v in per_fold_delta.values() if v is not None]
        delta = {
            "definition": "Δ = acc(control) − acc(grouped)；正值 = 分组协议压掉的乐观偏差",
            "per_fold_pp": {k: (None if v is None else v * 100.0)
                            for k, v in per_fold_delta.items()},
            "mean_pp": (statistics.fmean(vals) * 100.0) if vals else None,
            "n_folds": len(vals),
            "same_script_same_gpu": True,
            "caveat": (
                "n≤5 折，无 CI；折间本身有噪声，只在 |Δ| > fold_spread 时才有资格谈差异。"
                "本对照两臂同代码、同 GPU、同超参，唯一变量是划分清单，故 Δ **可归因**；"
                "但样本量小 ⇒ 只能报点估计 + 折间范围，不得声称显著性。"
            ),
        }

    seed_disp = None
    if extra_seeds and seed_folds and any(r["arm"] == "grouped"
                                          for r in results if r["seed"] != BASE_SEED):
        per_fold: Dict[str, dict] = {}
        for f in seed_folds:
            vals = {str(s): acc_of("grouped", s, f) for s in (BASE_SEED, *extra_seeds)}
            got = [v for v in vals.values() if v is not None]
            if len(got) < 2:
                per_fold[str(f)] = {"values": vals, "note": "可用种子 <2，无法给离散度"}
                continue
            per_fold[str(f)] = {
                "values": vals,
                "mean": statistics.fmean(got),
                "stdev": statistics.stdev(got),
                "range_pp": (max(got) - min(got)) * 100.0,
            }
        ranges = [v["range_pp"] for v in per_fold.values() if "range_pp" in v]
        seed_disp = {
            "seeds": [BASE_SEED, *extra_seeds],
            "folds": list(seed_folds),
            "per_fold": per_fold,
            "max_range_pp": max(ranges) if ranges else None,
            "caveat": (
                "只评折内 val，**不碰 sealed test**。这补的是 G2 的缺口——"
                "G2(mode=final) 只落了 train_loss，没有 val/acc。"
                "n = 种子数 × 折数 很小，只能报离散度，不得声称显著。"
            ),
        }

    return {
        "experiment": "C1 对照五折 + 跨种子 CV",
        "epochs": BASE_EPOCHS,
        "batch_size": BASE_BATCH_SIZE,
        "selection_rule": "折内按 val accuracy 选 best epoch（与基线一致）",
        "sealed_test_untouched": True,
        "arms": per_arm,
        "delta_control_minus_grouped": delta,
        "seed_dispersion": seed_disp,
        "control_manifest_diagnostics": control_diag,
        "capacity_note": (
            "两臂都是 `full`（56,812,813 参数、指纹 == 基线），容量完全一致，"
            "因此 Δ 不含容量混淆。"
        ),
        "runs": results,
    }


def write_markdown(summary: dict, out_path: Path) -> None:
    L: List[str] = []
    L.append("# C1 · 对照五折 + 跨种子 CV（实测）")
    L.append("")
    L.append("> 评估集：**折内验证集（val）**；`sealed test` **未被触碰**。")
    L.append("> 选模规则：折内按 val accuracy 选 best epoch（与基线一致）。")
    L.append(f"> 训练轮数：{summary['epochs']} epoch；batch={summary['batch_size']}。")
    L.append("> 两臂均为 `full`（参数量 56,812,813、指纹 == 基线）⇒ 容量一致，Δ 不含容量混淆。")
    L.append("")

    arms = summary.get("arms") or {}
    L.append("## 1. 两臂精度")
    L.append("")
    L.append("| 臂 | 划分依据 | 折数 | val 准确率(均) | min | max | 折间跨度 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    name_map = {"grouped": "P-CV v2 分组（基线清单）",
                "control": "图像级随机（忽略 group_id）"}
    for arm in ("grouped", "control"):
        if arm not in arms:
            continue
        a = arms[arm]
        L.append(f"| `{arm}` | {name_map[arm]} | {a['n_folds']} | "
                 f"{a['accuracy_mean']*100:.3f}% | {a['accuracy_min']*100:.3f}% | "
                 f"{a['accuracy_max']*100:.3f}% | {a['fold_spread']*100:.3f} pp |")
    L.append("")

    d = summary.get("delta_control_minus_grouped")
    L.append("## 2. Δ = 对照 − 分组（协议压掉的乐观偏差）")
    L.append("")
    if not d:
        L.append("（本次没有同时跑两臂，无法给 Δ）")
    else:
        L.append("| 折 | Δ (pp) |")
        L.append("|---:|---:|")
        for f, v in d["per_fold_pp"].items():
            L.append(f"| {f} | {'—' if v is None else f'{v:+.3f}'} |")
        L.append(f"| **均值** | **{d['mean_pp']:+.3f}** |" if d["mean_pp"] is not None
                 else "| **均值** | — |")
        L.append("")
        L.append(f"⚠️ {d['caveat']}")
    L.append("")

    s = summary.get("seed_dispersion")
    L.append("## 3. 跨种子离散度（补 L6）")
    L.append("")
    if not s:
        L.append("（本次没跑额外种子）")
    else:
        seeds = s["seeds"]
        L.append("| 折 | " + " | ".join(f"seed {x}" for x in seeds) +
                 " | 均值 | range (pp) |")
        L.append("|---:|" + "---:|" * (len(seeds) + 2))
        for f, v in s["per_fold"].items():
            if "range_pp" not in v:
                L.append(f"| {f} | " + " | ".join(
                    "—" if v["values"].get(str(x)) is None
                    else f"{v['values'][str(x)]*100:.3f}%" for x in seeds) + " | — | — |")
                continue
            cells = " | ".join(
                "—" if v["values"].get(str(x)) is None
                else f"{v['values'][str(x)]*100:.3f}%" for x in seeds)
            L.append(f"| {f} | {cells} | {v['mean']*100:.3f}% | {v['range_pp']:.3f} |")
        L.append("")
        L.append(f"最大跨种子极差：**{s['max_range_pp']:.3f} pp**" if s["max_range_pp"]
                 is not None else "最大跨种子极差：—")
        L.append("")
        L.append(f"⚠️ {s['caveat']}")
    L.append("")

    cd = summary.get("control_manifest_diagnostics")
    L.append("## 4. 对照清单的泄漏描述（证明这份对照确实忽略了 group）")
    L.append("")
    if cd:
        L.append(f"- group 跨折数：**{cd['group_cross_fold']}**（基线为 0）")
        L.append(f"- exact-SHA 跨折数：**{cd['sha_cross_fold']}**（基线为 0）")
        L.append(f"- 被同源污染的 val 总数：**{cd['val_tainted_total']}**"
                 f"（各折 {cd['val_tainted_per_fold']}）")
    else:
        L.append("（未记录）")
    L.append("")
    L.append("## 5. 必须随数字一起写的诚实说明")
    L.append("")
    L.append(f"1. **容量一致**：{summary['capacity_note']}")
    L.append("2. **不碰 sealed test**：全程只评折内 val，sealed test 一次都没有被读过。")
    L.append("3. **样本量小**：折数 ≤5、种子数 ≤3，均不给 CI，不得声称显著性。")
    L.append("4. **Δ 的方向解释**：Δ > 0 表示图像级随机划分让同源副本跨进 val，"
             "抬高了精度 —— 这部分就是分组协议压掉的乐观偏差。")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = base_parser("C1 · 对照五折 + 跨种子 CV（一次跑完）")
    ap.add_argument("--arms", choices=["both", "grouped", "control"], default="both",
                    help="跑哪些臂（默认 both：分组 + 图像级）")
    ap.add_argument("--folds", default="1,2,3,4,5", help="主两臂的折（默认全部 5 折）")
    ap.add_argument("--seed-extra", default="43,44",
                    help="额外种子，逗号分隔（默认 43,44；不要含 42）")
    ap.add_argument("--seed-folds", default="1,5",
                    help="额外种子只跑哪些折（默认 1,5 —— 与 G1 的折一致，便于互引）")
    ap.add_argument("--no-seed-part", action="store_true", help="跳过跨种子部分")
    ap.add_argument("--data-root", default=None,
                    help=f"Combined Dataset 路径（默认 {FIXED_DATA_ROOT}）")
    ap.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BASE_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--progress-interval", type=float, default=5.0)
    ap.add_argument("--plan", action="store_true", help="只预检 + 报时长，不训练")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已完成（已有 best_metrics.json）的 run")
    ap.add_argument("--hours-per-run", type=float, default=None,
                    help=f"覆盖 ETA 用的单 run 时长（默认 {MEASURED_SECONDS_PER_RUN/3600:.2f} h，实测）")
    a = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = resolve_out_dir(a.out_dir, "c1")
    bdir = find_baseline_dir(a.baseline_dir)

    folds = [int(x) for x in a.folds.split(",") if x.strip()]
    seed_folds = [int(x) for x in a.seed_folds.split(",") if x.strip()]
    extra_seeds = [int(x) for x in a.seed_extra.split(",") if x.strip()]
    for f in folds + seed_folds:
        if not 1 <= f <= 5:
            raise ValueError(f"非法折号 {f}")
    if BASE_SEED in extra_seeds:
        raise ValueError(f"--seed-extra 里不要含基线种子 {BASE_SEED}")
    if a.no_seed_part:
        extra_seeds = []

    jobs = build_jobs(a.arms, folds, extra_seeds, seed_folds, a.no_seed_part)
    if not jobs:
        print("❌ 没有生成任何任务，检查 --arms / --folds / --no-seed-part")
        return 2

    # ⚠️ 本机（Windows）自检专用提示。云端 Linux 不受影响（fork 不重导入模块）。
    if os.name == "nt" and a.num_workers > 0:
        print("⚠️  Windows 本地自检提示：本机 DataLoader 用 spawn 起 worker，"
              "动态加载的 smoke_test 模块在子进程里 import 不到，会报 "
              "ModuleNotFoundError: No module named 'dmsa_baseline_smoke'。")
        print("      本机自检请加 `--num-workers 0`；云端 Linux 用 fork，保持默认 6 即可。")
        print()

    print("=" * 78)
    print("C1 · 对照五折 + 跨种子 CV")
    print("=" * 78)
    print(f"包版本      : {PKG_VERSION}")
    print(f"baseline dir: {bdir}")
    print(f"out dir     : {out_dir}")
    print(f"任务        : {len(jobs)} 个 run ｜ arms={a.arms} "
          f"folds={folds} 额外种子={extra_seeds or '无'} seed_folds={seed_folds}")

    # ---- 载入两臂清单 ----
    man = bdir / TRAIN_MANIFEST_NAME
    bdir_rows: List[dict] = []
    if man.is_file():
        try:
            bdir_rows = read_manifest(man)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 基线 manifest 读不动：{type(e).__name__}: {e}")
    control_path = bdir / CONTROL_MANIFEST_NAME
    if not control_path.is_file():
        control_path = HERE / CONTROL_MANIFEST_NAME
    control_rows: Optional[List[dict]] = None
    if control_path.is_file():
        try:
            control_rows = read_manifest(control_path)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 对照 manifest 读不动：{type(e).__name__}: {e}")

    data_root = resolve_data_root(a.data_root)

    # ---- 预检 ----
    print("\n" + "-" * 78)
    print("预检查")
    print("-" * 78)
    ok, pr = precheck(bdir=bdir, bdir_rows=bdir_rows, control_rows=control_rows,
                      control_path=control_path, data_root=data_root,
                      out_dir=out_dir, depth=a.device.split(":")[0])
    for c in pr["checks"]:
        icon = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "· "}[c["status"]]
        print(f"  {icon} [{c['status']:<4}] {c['item']:<16} {c['detail']}")

    if not ok:
        print("\n❌ 预检未通过，拒绝开跑（不做任何训练）。")
        write_json(out_dir / "c1_precheck.json", pr, script_path)
        return 2

    hpr = a.hours_per_run or (MEASURED_SECONDS_PER_RUN / 3600.0)
    total_h = len(jobs) * hpr
    print("-" * 78)
    print(f"  结论：预检通过 ｜ {len(jobs)} 个 run × {hpr:.2f} h ≈ **{total_h:.1f} h**"
          f"（按 G1 实测单 run 外推，非估算）")
    print("-" * 78)

    ctrl_diag = leak_descriptors(control_rows) if control_rows is not None else None
    if ctrl_diag:
        print(f"  对照清单泄漏：group 跨折={ctrl_diag['group_cross_fold']}、"
              f"sha 跨折={ctrl_diag['sha_cross_fold']}、"
              f"被同源污染的 val={ctrl_diag['val_tainted_total']}")

    # ---- dry-run：CPU 上验证接线 ----
    if a.dry_run:
        print("\n-- dry-run：验证指纹硬门 + 两臂守卫 -------------")
        gate = fingerprint_gate(bdir)
        print(f"   full 模型: params={gate['params']:,} fp={gate['fingerprint'][:12]}… "
              f"前向={'OK' if gate['forward_ok'] else 'FAIL'} "
              f"{'[= 基线]' if gate['equals_baseline'] else '[≠ 基线!]'}")
        if not gate["equals_baseline"]:
            print("\n❌ 硬门失败：`full` 指纹与基线不一致，禁止用这套注入做任何对比。")
            write_json(out_dir / "c1_dryrun.json",
                       {"precheck": pr, "gate": gate, "gate_passed": False}, script_path)
            return 2
        jobs_probe = []
        for j in jobs[:3]:
            rows = bdir_rows if j["arm"] == "grouped" else control_rows
            n_val = sum(1 for r in (rows or []) if int(r["val_fold"]) == j["fold"])
            jobs_probe.append({**j, "label": job_label(j),
                               "rows": len(rows or []), "val": n_val})
            print(f"   {job_label(j):<28} rows={len(rows or []):,} val={n_val:,}")
        print(f"\n✅ 硬门通过；{len(jobs)} 个 run 的任务表可构造。")
        write_json(out_dir / "c1_dryrun.json",
                   {"precheck": pr, "gate": gate, "gate_passed": True,
                    "jobs": jobs_probe, "n_jobs": len(jobs)}, script_path)
        return 0

    # ---- --plan：到此为止 ----
    if a.plan:
        print("\n（--plan：只预检，不训练。去掉 --plan 即正式开跑。）")
        write_json(out_dir / "c1_precheck.json",
                   {**pr, "n_jobs": len(jobs), "jobs": [job_label(j) for j in jobs],
                    "estimated_hours": total_h, "hours_per_run": hpr,
                    "control_diag": ctrl_diag}, script_path)
        return 0

    # ---- 正式训练 ----
    from C22_lib_gpu_runtime import cpu_smoke
    cpu_smoke(bdir)

    device, dev_info = probe_device(a.device)
    print(f"device      : {dev_info}")

    t0 = time.time()
    results, rep = run_matrix(
        jobs=jobs, bdir=bdir, bdir_rows=bdir_rows, control_rows=control_rows,
        data_root=data_root, out_dir=out_dir, device=device, epochs=a.epochs,
        batch_size=a.batch_size, num_workers=a.num_workers,
        max_batches=a.max_batches, rep_interval=a.progress_interval,
        resume=a.resume,
    )
    elapsed = time.time() - t0

    summary = summarize(results, folds, extra_seeds, seed_folds, ctrl_diag)
    summary["elapsed_seconds"] = elapsed
    summary["n_jobs"] = len(jobs)
    summary["control_manifest"] = {"name": CONTROL_MANIFEST_NAME,
                                   "sha256": CONTROL_MANIFEST_SHA256}
    write_json(out_dir / "c1_summary.json", summary, script_path)
    write_csv(out_dir / "c1_per_fold.csv", results,
              ["arm", "seed", "fold", "label", "params", "fingerprint",
               "best_epoch", "accuracy", "precision", "recall", "f1", "kappa",
               "val_loss", "elapsed_seconds", "epochs_run", "skipped"])
    write_markdown(summary, out_dir / "c1_table.md")

    print("\n" + "=" * 78)
    print(f"C1 完成，总耗时 {fmt_hms(elapsed)}（{len(jobs)} 个 run）")
    for arm in ("grouped", "control"):
        a_ = summary["arms"].get(arm)
        if a_:
            print(f"  {arm:<8} val_acc={a_['accuracy_mean']*100:.3f}%  "
                  f"（{a_['n_folds']} 折，跨度 {a_['fold_spread']*100:.3f} pp）")
    d = summary.get("delta_control_minus_grouped")
    if d and d.get("mean_pp") is not None:
        print(f"  Δ(对照−分组) = {d['mean_pp']:+.3f} pp")
    s = summary.get("seed_dispersion")
    if s and s.get("max_range_pp") is not None:
        print(f"  跨种子最大极差 = {s['max_range_pp']:.3f} pp")
    print("  产物：c1_summary.json / c1_per_fold.csv / c1_table.md")
    print("  提醒：sealed test 未被本实验触碰。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
