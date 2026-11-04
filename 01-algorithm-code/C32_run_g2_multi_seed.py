#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC 提分实验 G2：多随机种子重复（★★）
===============================================================================

对应需求文档《04_需用户本人操作事项反馈.md》§5「★★ 多随机种子重复」。

---------------------------------------------------------------------------
它解决什么问题
---------------------------------------------------------------------------

现状：方案里所有数字都来自 **单一 seed=42**。评委可以合理质疑：
「你换一个种子，99.0% 的准确率会不会掉到 95%？」
这是「实验验证」维度里最容易补、也最容易被追问的一条。

本脚本在**不动任何超参**的前提下，只改 SEED，重复 final 全量训练 126 epochs，
得到跨种子的均值 ± 标准差。

---------------------------------------------------------------------------
铁律：为什么不直接改 final_train.py 的 SEED
---------------------------------------------------------------------------

`final_train_code_used.py` 里 `_check_final_protocol()`（约 1149 行）硬断言
SEED 必须等于 42，改了就 `RuntimeError`。这是**故意设计**的——防止有人
偷偷改种子去挑一个好看的 test 结果。

所以本脚本的做法是：

  1. **逐字节复制**一份 `final_train_code_used.py` 到临时目录；
  2. 在副本里**只替换 SEED 这一行**（用正则精确匹配，替换前先断言原文唯一）；
  3. 同时替换 OUTPUT_DIR / RUN_NAME；
  4. 子进程运行副本。

原文件**零改动**（有 SHA-256 前后比对为证）。

---------------------------------------------------------------------------
sealed test 怎么办（关键）
---------------------------------------------------------------------------

**本脚本不碰 sealed test，一次都不碰。**

理由：sealed test 已经在 seed=42 上正式评价过一次（accuracy 0.98984）。
如果现在拿另外 2 个种子再各评一次，然后挑最好的报出去，
那就是「在同一测试集上重复挑选」——评委会一眼看穿，反而扣分。

正确做法：
  - 多种子只报告 **五折 CV 的 val 指标** 与 **final 训练的 train loss 曲线**；
  - 若方案需要「多种子 test 指标」，必须**另找一批从未用过的测试数据**，
    或者明确写「test 仅 seed=42 单次评价，多种子差异由 CV 折间方差佐证」。

本脚本默认生成前者，并在产物里写清这条限制。

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------

    # 本机（无 GPU）只自检接线：
    python g2_multi_seed.py --dry-run

    # GPU 机器正式跑（2 个额外种子 + 可选复制 seed42 作为对照）：
    python g2_multi_seed.py \
        --data-root "/home/featurize/work/Combined_Dataset" \
        --out-dir   "/home/featurize/work/dmsa_multiseed" \
        --seeds 43,44

    # 只跑五折 val 的多种子（更快，但不含 final 全量训练）：
    python g2_multi_seed.py --mode cv --seeds 43,44 ...

预计成本（RTX 4090 级）：
    final 模式  : 每种子 ~4–8 h（126 epoch × 640 iter）
    cv 模式     : 每种子 5 折 × ~4–8 h，明显更贵，一般不需要

建议：**只跑 final 模式，2 个种子（43, 44）**，约 8–16 h。
===============================================================================
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from C22_lib_gpu_runtime import (  # noqa: E402
    ARCHITECTURE_FINGERPRINT,
    BASE_EPOCHS,
    BASE_SEED,
    FINAL_EPOCHS,
    FIXED_DATA_ROOT,
    PKG_VERSION,
    STEPS_PER_EPOCH_FINAL,
    TOTAL_STEPS_FINAL,
    TRAIN_MANIFEST_NAME,
    TRAIN_MANIFEST_SHA256,
    ProgressReporter,
    assert_train_manifest,
    base_parser,
    cpu_smoke,
    find_baseline_dir,
    human_mb,
    make_tee_sink,
    read_manifest,
    resolve_data_root,
    resolve_out_dir,
    safe_rmtree,
    safe_unlink,
    sha256_file,
    stream_run,
    write_csv,
    write_json,
)


# =============================================================================
# 1. 源码改写：只动 SEED / OUTPUT_DIR / RUN_NAME
# =============================================================================

def _replace_once(src: str, pattern: str, repl, *, what: str) -> str:
    """精确替换且**要求原文恰好出现一次**，否则报错——防止误改。

    ⚠️ 关键细节：`repl` 若是普通字符串，必须经 lambda 插入。
    否则 `re.sub` 会把它当"模板"解析——Windows 路径 `E:\\Code\\...` 里的
    `\\U` / `\\A` 会被当成非法转义，抛 `re.PatternError: bad escape \\U`。
    这个 bug 在 Linux 路径下不会触发（没有反斜杠），只在 Windows 上炸，
    所以必须显式防住。
    """
    hits = re.findall(pattern, src, flags=re.MULTILINE)
    if len(hits) != 1:
        raise RuntimeError(
            f"期望 {what} 在源码中恰好出现 1 次，实际 {len(hits)} 次。"
            f"模式：{pattern!r}\n"
            "基线代码可能已变动，请人工核对后再跑。"
        )
    if isinstance(repl, str):
        return re.sub(pattern, lambda m: repl, src, count=1, flags=re.MULTILINE)
    return re.sub(pattern, repl, src, count=1, flags=re.MULTILINE)


def make_seed_variant(
    base_script: Path,
    dest: Path,
    *,
    seed: int,
    output_dir: str,
    run_name: str,
    data_root: Optional[str] = None,
) -> Dict[str, object]:
    """把 final_train 副本改写成指定 seed 的版本。返回改写前后指纹对比。

    传入 `data_root` 时会把副本里的 `DATA_ROOT` 一并改写（`None` = 不改写，仅供对照测试）。

    ⚠️ 为什么调用方**总是**传：基线 `final_train.py:1115` 里硬编码的 `DATA_ROOT`
    指向 `.../data/dmsa_pcv2/Combined Dataset`，与云端实测位置**不一致**；
    不改写的话副本会去读一个不存在的目录。`DATA_ROOT` **不在**
    `_check_final_protocol()` 的 6 项断言里，改写不会触发硬门，且写完有回读校验。
    """
    src = base_script.read_text(encoding="utf-8")
    sha_before = sha256_file(base_script)

    # 去掉结尾分隔符：raw string 不能以反斜杠结尾（r"E:\foo\" 是语法错误）
    output_dir = str(output_dir).rstrip("\\/")
    if not output_dir:
        raise ValueError("output_dir 不能为空")

    dr: Optional[str] = None
    if data_root is not None:
        dr = str(data_root).rstrip("\\/")
        if not dr:
            raise ValueError("data_root 不能为空字符串")
        if '"' in dr or "\n" in dr or "\r" in dr:
            raise ValueError(f"data_root 含非法字符（引号/换行）：{dr!r}")

    src = _replace_once(
        src, r"^SEED = 42$", f"SEED = {int(seed)}",
        what="SEED = 42",
    )
    src = _replace_once(
        src, r'^OUTPUT_DIR = r".*"$', f'OUTPUT_DIR = r"{output_dir}"',
        what="OUTPUT_DIR",
    )
    src = _replace_once(
        src, r'^RUN_NAME = ".*"$', f'RUN_NAME = "{run_name}"',
        what="RUN_NAME",
    )

    # DATA_ROOT：仅在显式给出时改写（它不在协议断言里，改它不会触发硬门）
    if dr is not None:
        src = _replace_once(
            src, r'^DATA_ROOT = r".*"$', f'DATA_ROOT = r"{dr}"',
            what="DATA_ROOT",
        )

    # _check_final_protocol 里的期望值也必须同步，否则副本自己会报错。
    # 这是"改种子"的必然代价，所以才会明确记录"副本已偏离 final protocol"。
    src = _replace_once(
        src, r'^(\s*)"SEED": 42,$', lambda m: f'{m.group(1)}"SEED": {int(seed)},',
        what='_check_final_protocol 中的 "SEED": 42',
    )

    dest.write_text(src, encoding="utf-8")

    # 回读校验：确认改了就一定改了
    back = dest.read_text(encoding="utf-8")
    if f"SEED = {int(seed)}" not in back:
        raise RuntimeError(f"回读失败：副本 {dest} 里没有 SEED = {seed}")
    if re.search(r"^SEED = 42$", back, flags=re.MULTILINE):
        raise RuntimeError(f"回读失败：副本 {dest} 里仍残留 SEED = 42")
    if dr is not None and f'DATA_ROOT = r"{dr}"' not in back:
        raise RuntimeError(f"回读失败：副本 {dest} 里没有 DATA_ROOT = r\"{dr}\"")

    return {
        "source_script": str(base_script),
        "source_sha256": sha_before,
        "source_sha256_after": sha256_file(base_script),
        "variant_script": str(dest),
        "variant_sha256": sha256_file(dest),
        "seed": int(seed),
        "run_name": run_name,
        "output_dir": output_dir,
        "data_root_override": dr,
    }


# =============================================================================
# 2. 训练执行
# =============================================================================

# ---------------------------------------------------------------------------
# 进度从哪里来？（重要：不改动基线文件）
# ---------------------------------------------------------------------------
# 基线 `final_train.py` **禁止修改**（项目铁律），但它内部的 `train_one_epoch()`
# 本来就用了 tqdm，会逐 batch 往 stdout 吐：
#     train epoch[3/126]:  45%|████▌     | 288/640 [01:23<01:45,  3.34it/s]
# 每轮结束还会吐一行：
#     [Final epoch 3/126] train_loss=0.512345 lr=... -> ...
# 所以本脚本**只解析、不改造**：训练逻辑与结果含义 100% 不变。
_TQDM_RE = re.compile(
    r"epoch\[(\d+)/(\d+)\].*?([\d.]+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]"
)
_EPOCH_DONE_RE = re.compile(
    r"\[Final epoch\s+(\d+)/(\d+)\]\s+train_loss=([0-9eE+\-.]+)"
)
# 只回显这些关键行；tqdm 的海量帧由 ProgressReporter 节流后统一呈现，
# 否则 8 万帧会把屏幕和日志一起淹掉。
_ECHO_HINTS = (
    "Traceback", "Error", "error", "Exception",
    "[CHECKPOINT", "[RESUME", "FINAL TRAINING COMPLETE",
    "FINAL_MODEL_READY", "DMSA-Net FINAL TRAINING",
)


def run_variant(script: Path, *, dry: bool, log_path: Path,
                progress_path: Optional[Path] = None,
                seed: Optional[int] = None,
                progress_interval: float = 5.0) -> int:
    """流式执行一个 seed 的训练副本：日志实时落盘 + 进度实时可见。"""
    cmd = [sys.executable, str(script)]
    print(f"\n  $ {' '.join(cmd)}")
    if dry:
        print("  [DRY-RUN] 不实际执行")
        return 0

    sink = make_tee_sink(progress_path) if progress_path else None
    rep = ProgressReporter(
        label=f"seed{seed}" if seed is not None else "",
        total_epochs=FINAL_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH_FINAL,
        total_units=TOTAL_STEPS_FINAL,
        unit="step", sink=sink, min_interval=progress_interval,
    )
    print(f"  进度日志   : {progress_path}")
    print(f"  完整日志   : {log_path}（实时写入，可随时 tail -f）")

    def on_frame(frame: str) -> None:
        m = _EPOCH_DONE_RE.search(frame)
        if m:
            rep.set(epoch=int(m.group(1)), step=STEPS_PER_EPOCH_FINAL,
                    loss=float(m.group(3)))
            rep.emit(force=True)
            return
        m = _TQDM_RE.search(frame)
        if m:
            rep.set(epoch=int(m.group(1)), step=int(m.group(4)))
            rep.emit()
            return
        if any(h in frame for h in _ECHO_HINTS):
            print(f"    | {frame.strip()}")

    rc, n_frames = stream_run(cmd, log_path, on_frame=on_frame)
    rep.emit(force=True)
    print(f"  日志行数   : {n_frames} 行（已实时落盘）")
    if rc != 0:
        print(f"  ❌ 退出码 {rc}；日志尾部：")
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace"
                                      ).strip().splitlines()[-40:]
            for ln in tail:
                print(f"    | {ln}")
        except Exception:
            pass
    return rc


# =============================================================================
# 3. 结果汇总
# =============================================================================

def collect_history(run_dir: Path) -> Optional[List[dict]]:
    """读取该 seed 的 train_history.csv（epoch,lr_used,next_lr,train_loss）。"""
    import csv as _csv

    p = Path(run_dir) / "train_history.csv"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(_csv.DictReader(f))


def summarize(per_seed: List[dict], script_path: Path, out_dir: Path) -> dict:
    import statistics

    usable = [r for r in per_seed if r.get("final_train_loss") is not None]
    losses = [float(r["final_train_loss"]) for r in usable]

    summary = {
        "experiment": "G2 multi-seed reproducibility",
        "baseline_seed": BASE_SEED,
        "epochs": FINAL_EPOCHS,
        "per_seed": per_seed,
        "n_seeds_ok": len(usable),
    }
    if len(losses) >= 2:
        summary["final_train_loss"] = {
            "mean": float(statistics.fmean(losses)),
            "stdev": float(statistics.stdev(losses)),
            "min": float(min(losses)),
            "max": float(max(losses)),
            "values": losses,
        }
    else:
        summary["final_train_loss"] = {
            "note": "可用种子不足 2 个，无法给出标准差",
            "values": losses,
        }

    summary["sealed_test_untouched"] = True
    summary["limitation_note"] = (
        "本实验**不对 sealed test 做任何重复评价**。sealed test 仅在 seed=42 上做过"
        "一次正式评价（accuracy 0.989835809226）。多种子的稳健性证据来自"
        "train loss 曲线与（若--mode cv）五折 val 指标的离散程度。"
        "如果方案需要多种子的 test 指标，必须换一批从未使用过的测试数据，"
        "不可在原 sealed test 上重复挑选。"
    )
    write_json(out_dir / "g2_multiseed_summary.json", summary, script_path)
    return summary


# =============================================================================
# 4. main
# =============================================================================

def main() -> int:
    ap = base_parser("DMSA-Net 多随机种子重复（★★）")
    ap.add_argument("--seeds", default="43,44",
                    help="要额外跑的种子，逗号分隔（默认 43,44）")
    ap.add_argument("--data-root", required=False, default=None,
                    help=f"Combined Dataset 路径（默认 {FIXED_DATA_ROOT}）")
    ap.add_argument("--mode", choices=["final", "cv"], default="final",
                    help="final=全量 126 epoch；cv=五折 200 epoch")
    ap.add_argument("--keep-code", action="store_true",
                    help="把改写后的副本一起写入 out-dir 便于留档")
    ap.add_argument("--template-script", default=None,
                    help="作为改写模板的 final_train 脚本路径（默认自动在基线目录里找）")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    default=True,
                    help="不覆盖上一次产物（默认：重跑即覆盖，输出路径保持不变）")
    ap.add_argument("--overwrite-dry-run", action="store_true",
                    help="只打印“将被删除什么”，不删除、不训练（覆盖预演）")
    ap.add_argument("--progress-interval", type=float, default=5.0,
                    help="进度行最小输出间隔（秒，默认 5）")
    a = ap.parse_args()

    script_path = Path(__file__).resolve()
    out_dir = resolve_out_dir(a.out_dir, "g2")
    out_dir.mkdir(parents=True, exist_ok=True)

    bdir = find_baseline_dir(a.baseline_dir)
    print("=" * 78)
    print("G2 · 多随机种子重复")
    print("=" * 78)
    print(f"baseline dir : {bdir}")
    print(f"out dir      : {out_dir}")
    print(f"mode         : {a.mode}")
    print(f"seeds        : {a.seeds}")
    print(f"覆盖策略     : {'重跑即覆盖（路径不变）' if a.overwrite else 'NO-OVERWRITE（遇到已有产物即报错）'}"
          + ("  ⚠ 本次为覆盖预演，不会删除、不会训练" if a.overwrite_dry_run else ""))

    # ---- 数据根目录 ----
    # 默认取 gpu_lib.FIXED_DATA_ROOT（用户实测确认的云端位置），--data-root 可覆盖。
    # ⚠️ 副本里的 `DATA_ROOT` **总是**改写成本次解析出的位置：基线 `final_train.py`
    # 硬编码的那条路径与实测位置不一致，不改写副本就会去读不存在的目录。
    # 该字段不在 `_check_final_protocol()` 的 6 项断言内，改写不会触发硬门。
    a.data_root = resolve_data_root(a.data_root)
    print(f"data root    : {a.data_root}  ← 将写入副本 DATA_ROOT")
    print(f"包版本      : {PKG_VERSION}")

    # ---- 先做协议校验：任何实验开跑前都必须过 ----
    man = bdir / TRAIN_MANIFEST_NAME
    man_sha = sha256_file(man)
    if man_sha != TRAIN_MANIFEST_SHA256:
        raise RuntimeError(
            f"train manifest SHA 不符！\nExpected: {TRAIN_MANIFEST_SHA256}\n"
            f"Actual:   {man_sha}"
        )
    assert_train_manifest(read_manifest(man))
    print("协议校验    : PASS（manifest SHA + 结构不变量）")

    seeds = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]
    if BASE_SEED in seeds:
        raise ValueError(
            f"--seeds 里不要包含基线种子 {BASE_SEED}；它已经跑过了。"
        )

    if a.dry_run:
        res = cpu_smoke(bdir)
        print("\n-- dry-run：在临时目录里试改写一次源码，验证改写规则 ----------------")
        base_script = bdir / "final_train.py"
        with tempfile.TemporaryDirectory() as td:
            info = make_seed_variant(
                base_script, Path(td) / f"final_train_seed{seeds[0]}.py",
                seed=seeds[0], output_dir="/tmp/dry",
                run_name=f"dry_seed{seeds[0]}",
                data_root=a.data_root,
            )
        print(f"  改写 PASS  seed={info['seed']}")
        print(f"  源文件 SHA 前后一致: "
              f"{info['source_sha256'] == info['source_sha256_after']} ✅")
        write_json(out_dir / "g2_dryrun.json", {"cpu_smoke": res, "rewrite_probe": info}, script_path)
        return 0 if res.get("ok") else 1

    # ---- 正式执行 ----
    if a.mode == "cv":
        raise SystemExit(
            "cv 模式成本为 5 折 × N 种子，通常远超预算且五折本身已在 seed=42 上完成。\n"
            "如确实需要，请显式改用 train_5fold.py 并自行管理输出目录，"
            "本脚本不代跑 cv 模式以免误用。"
        )

    # ---- 模板脚本解析 ----
    # 云机器上没有 E: 盘，所以不能依赖硬编码的本地路径。
    # 按优先级依次尝试，全部不存在就明确报错（而不是静默用错脚本）。
    candidates: List[Path] = []
    if getattr(a, "template_script", None):
        candidates.append(Path(a.template_script).expanduser())
    candidates += [
        bdir / "final_train_code_used.py",   # 正式跑过的那份快照（优先）
        bdir / "final_train.py",             # 基线源码
        Path(  # 本地开发时的便利回退；云端不存在会被跳过
            r"E:/Code/NeuroEvo/NeuroEvo_AD/dmsa_final_train(1)/"
            r"dmsa_final_train_seed42_epoch126/final_train_code_used.py"
        ),
    ]
    base_script = next((p for p in candidates if p.is_file()), None)
    if base_script is None:
        raise FileNotFoundError(
            "找不到模板脚本。请把 final_train.py（或 final_train_code_used.py）"
            "放到基线目录，或用 --template-script 显式指定。\n已尝试：\n  "
            + "\n  ".join(str(p) for p in candidates)
        )
    print(f"模板脚本     : {base_script}")
    print(f"模板 SHA-256 : {sha256_file(base_script)[:16]}…")

    work = out_dir / "_variants"
    work.mkdir(parents=True, exist_ok=True)

    # ⚠️ 必须把冻结 manifest 放到副本旁边！
    # final_train.py 用 `script_dir = Path(__file__).resolve().parent` 找 manifest，
    # 即 `<副本所在目录>/pcv_v2_folds_seed42.csv`。副本在 _variants/ 下，
    # 而 manifest 在基线目录里 —— 不放过来就会 FileNotFoundError。
    # 逐字节复制，保证 SHA-256 校验仍然通过。
    manifest_src = bdir / TRAIN_MANIFEST_NAME
    manifest_dst = work / TRAIN_MANIFEST_NAME
    shutil.copy2(manifest_src, manifest_dst)
    if sha256_file(manifest_dst) != TRAIN_MANIFEST_SHA256:
        raise RuntimeError("复制到 _variants/ 的 manifest SHA-256 不符，拒绝继续。")
    print(f"manifest     : 已放到 {manifest_dst}")

    per_seed: List[dict] = []
    for seed in seeds:
        run_name = f"dmsa_final_train_seed{seed}_epoch{FINAL_EPOCHS}"
        run_dir = Path(a.out_dir or out_dir) / run_name
        variant = work / f"final_train_seed{seed}.py"

        print(f"\n---- seed {seed} ----")
        info = make_seed_variant(
            base_script, variant, seed=seed,
            output_dir=str(Path(a.out_dir or out_dir)),
            run_name=run_name,
            data_root=a.data_root,
        )
        print(f"  改写 PASS  variant_sha={info['variant_sha256'][:16]}…")

        # ---- 重跑即覆盖：先清掉上一次的 run_dir 与本次 seed 的旧日志 ----
        # 为什么必须先清：基线 final_train.py 有一道守卫——`RESUME_FROM` 为空
        # 且 run_dir 非空时直接 FileExistsError（防止覆盖正式结果）。
        # 所以"重跑覆盖"必须由本脚本先把目录清干净，路径本身保持不变。
        # 删除前过 safe_rmtree 的包含性校验：只删 out_dir 之下的产物，
        # 数据集目录 / out_dir 本身 / 符号链接 / 盘符根 一律拒绝。
        log_path = out_dir / f"log_seed{seed}.txt"
        progress_path = out_dir / f"progress_seed{seed}.log"

        if run_dir.exists():
            if not a.overwrite:
                raise RuntimeError(
                    f"已存在产物目录：{run_dir}\n"
                    "当前是 --no-overwrite 模式。\n"
                    "  · 想保留它 → 换一个 --out-dir\n"
                    "  · 想重跑覆盖 → 去掉 --no-overwrite（默认就是覆盖）"
                )
            # ⚠️ 变量名不能叫 info：上面 make_seed_variant 的返回值已经是 info，
            #    这里若重名会把它覆盖掉，导致后面 rec["variant_sha256"] KeyError。
            rm = safe_rmtree(run_dir, must_be_under=out_dir,
                             dry=a.overwrite_dry_run)
            verb = "将删除" if a.overwrite_dry_run else "已清理"
            print(f"  {verb}      : {rm['path']}  "
                  f"（{rm.get('files', 0)} 文件 / {human_mb(rm.get('bytes', 0))}）")
        else:
            print(f"  产物目录    : {run_dir}（不存在，本次新建）")

        if a.overwrite and not a.overwrite_dry_run:
            for stale in (log_path, progress_path):
                safe_unlink(stale, must_be_under=out_dir)

        if a.overwrite_dry_run:
            print("  [覆盖预演] 跳过训练")
            continue

        rc = run_variant(variant, dry=False, log_path=log_path,
                         progress_path=progress_path, seed=seed,
                         progress_interval=a.progress_interval)

        hist = collect_history(run_dir)
        rec = {
            "seed": seed,
            "run_dir": str(run_dir),
            "exit_code": rc,
            "variant_sha256": info["variant_sha256"],
            "n_epochs_logged": len(hist) if hist else 0,
            "final_train_loss": (
                float(hist[-1]["train_loss"]) if hist else None
            ),
            "first_train_loss": (
                float(hist[0]["train_loss"]) if hist else None
            ),
        }
        per_seed.append(rec)
        print(f"  → epochs={rec['n_epochs_logged']}  "
              f"final_train_loss={rec['final_train_loss']}")

    if a.overwrite_dry_run:
        print("\n" + "=" * 78)
        print("覆盖预演结束：上面就是“将被删除”的清单，**未删除、未训练**。")
        print("确认无误后，去掉 --overwrite-dry-run 即可真正重跑覆盖。")
        print("=" * 78)
        return 0

    if a.keep_code:
        for f in work.glob("final_train_seed*.py"):
            shutil.copy2(f, out_dir / f.name)

    summarize(per_seed, script_path, out_dir)
    write_csv(
        out_dir / "g2_multiseed_table.csv", per_seed,
        ["seed", "exit_code", "n_epochs_logged", "first_train_loss",
         "final_train_loss", "variant_sha256", "run_dir"],
    )

    # ⚠️ 退出码必须反映真实结果：以前这里无条件 return 0，导致
    # 「所有种子都失败」也会打印「G2 完成」。产物照写（便于排查），但退出码非 0。
    failed = [r for r in per_seed
              if r["exit_code"] != 0 or r["n_epochs_logged"] == 0]
    if failed:
        print("\n" + "=" * 78)
        print(f"❌ G2 未成功：{len(failed)}/{len(per_seed)} 个种子没有产出可用结果")
        for r in failed:
            _lp = out_dir / f"log_seed{r['seed']}.txt"
            print(f"  seed {r['seed']}: exit_code={r['exit_code']}  "
                  f"epochs={r['n_epochs_logged']}  日志={_lp}")
        print(f"  {out_dir / 'g2_multiseed_summary.json'}")
        print("产物已写出（便于排查），但**本次结果不可用于验收**：accept.py 会判失败。")
        print("=" * 78)
        return 1

    print("\n" + "=" * 78)
    print(f"G2 完成：{len(per_seed)} 个种子全部产出结果。产物：")
    print(f"  {out_dir / 'g2_multiseed_summary.json'}")
    print(f"  {out_dir / 'g2_multiseed_table.csv'}")
    print("提醒：sealed test 未被本实验触碰。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
