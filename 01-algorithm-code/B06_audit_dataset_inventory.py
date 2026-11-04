#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 0.0 输入盘点（v2.7）
========================

v2.1–v2.3 反复出现"某任务标着 0 成本、实际缺输入"的问题。
本脚本把路线里所有"零成本/零重训"任务的**前置输入**逐项检查并落盘，
缺失项直接标记 `blocked_missing_inputs`，不允许用空泛的"0 GPU 小时"掩盖文件缺失。

v2.7：数据已到货且落在**嵌套目录**（`Alzheimer's MRI Dataset/Alzheimer's MRI Dataset/
Combined Dataset/{train,test}`），因此不再写死相对路径，改为 `--data` 显式指定或按
`DATA_CANDIDATES` 自动探测；并新增阶段 0.0c（数据集到货验证）一项输入检查。

用法：
    python a0_inventory.py --root .. --out _a0_out
    python a0_inventory.py --root .. --data "../Alzheimer's MRI Dataset/Alzheimer's MRI Dataset/Combined Dataset"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Dict, List

#: N6（v2.6）：阶段 0 是流水线第一步，必须能在"numpy 还没装"时直接跑。
#: 因此这里只依赖**纯标准库**的 lib_provenance，不 import lib_dedup。
from B16_lib_provenance_audit import write_artifact

BASELINE_FILES = [
    "DMSA_net/smoke_test.py",
    "DMSA_net/train_5fold.py",
    "DMSA_net/final_train.py",
    "DMSA_net/final_test.py",
    "DMSA_net/data_prepare_check.py",
    "DMSA_net/pcv_v2_folds_seed42.csv",
    "DMSA_net/pcv_test_manifest_SEALED.csv",
]

# 路线中的"零成本"任务 -> 依赖哪些输入
ZERO_COST_TASKS = [
    dict(task="阶段 0.0c 数据集到货验证（v2.7 新增）",
         inputs=["原始图片目录 train", "原始图片目录 test"],
         blocked_action="无法确认到货数据与冻结 manifest 一致；T14 会失败；"
                        "此时禁止进入 A1/A2"),
    dict(task="阶段 0.1 A1 源图映射", inputs=["原始图片目录 train", "原始图片目录 test"],
         blocked_action="无法建立 source_id；阶段 1 全部降级，只能做已有 manifest 的审计"),
    dict(task="阶段 0.2 A2 train<->test 审计", inputs=["原始图片目录 train", "原始图片目录 test"],
         blocked_action="R6 无法判定；sealed test 的独立性只能写'未验证'"),
    dict(task="阶段 0.3 基线冻结", inputs=BASELINE_FILES,
         blocked_action="无法证明基线未被改动，fingerprint 校验失去意义"),
    dict(task="阶段 0.4 五折产物盘点",
         inputs=["五折 metrics.csv", "best_model_state.pth x5", "final_model_state.pth"],
         blocked_action="阶段 1.4/1.5 的预测侧分析只能改为重训；§5.5 固定 epoch 对照只能读折内日志"),
    dict(task="阶段 0.5 协议溯源（v1_val_fold / 论文 0.9990）",
         inputs=["旧 split 文件", "旧 checkpoint", "旧训练日志"],
         blocked_action="只能写'基于当前材料暂不可复现'，不得升级为'不可复现'"),
    dict(task="阶段 3.4 / 5.4 A6-s 静态权重映射", inputs=["torch", "torchvision"],
         blocked_action="贡献 1 的判定推迟；论文需注明未经本项验证"),
    dict(task="阶段 1.9 subject-level 可得性判定",
         inputs=["发布者 patient/subject ID 或元数据"],
         blocked_action="只能报告原图级独立，并显式写 limitation"),
]

#: 已知的数据集落位（2026-09-23 到货）。`--data` 未给时按此顺序探测。
DATA_CANDIDATES = [
    "Alzheimer's MRI Dataset/Alzheimer's MRI Dataset/Combined Dataset",
    "Alzheimer's MRI Dataset/Combined Dataset",
    "Combined Dataset",
]



def _sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _find_images(root: pathlib.Path, data: pathlib.Path | None = None):
    """返回 (总量字典, 逐类明细)。数据落位在 v2.7 之后是**嵌套目录**，不能写死相对路径。"""
    EXTS = {".jpg", ".jpeg", ".png"}
    cands: List[pathlib.Path] = []
    if data is not None:
        cands = [data / "train", data / "test"]
    else:
        cands = [root / n for n in ("Combined Dataset/train", "Combined Dataset/test")]
        cands += [root / p / "train" for p in DATA_CANDIDATES]
        cands += [root / p / "test" for p in DATA_CANDIDATES]
    out, detail = {}, {}
    seen = set()
    for d in cands:
        if not d.is_dir() or str(d) in seen:
            continue
        seen.add(str(d))
        per = {}
        for cls in sorted(p for p in d.iterdir() if p.is_dir()):
            per[cls.name] = sum(1 for f in cls.iterdir()
                                if f.is_file() and f.suffix.lower() in EXTS)
        out[str(d)] = sum(per.values())
        detail[str(d)] = per
    return out, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..", help="项目根目录（含 DMSA_net/ 与数据目录）")
    ap.add_argument("--data", default="",
                    help="含 train/ 与 test/ 的目录；留空则按已知落位自动探测")
    ap.add_argument("--out", default="_a0_out")
    a = ap.parse_args()

    root = pathlib.Path(a.root).resolve()
    data = pathlib.Path(a.data).resolve() if a.data else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for rel in BASELINE_FILES:
        p = root / rel
        files.append(dict(path=rel, exists=p.exists(),
                          sha256=_sha(p) if p.exists() else None))

    imgs, img_detail = _find_images(root, data)
    art = {}
    for pat in ("**/metrics.csv", "**/best_model_state*.pth",
                "**/final_model_state*.pth", "**/best_metrics.json"):
        hits = [str(p) for p in root.glob(pat)][:20]
        art[pat] = hits

    try:
        import torch  # noqa
        import torchvision  # noqa
        torch_ok = f"{torch.__version__} / torchvision {torchvision.__version__}"
    except Exception as e:
        torch_ok = f"UNAVAILABLE: {type(e).__name__}"

    report = dict(
        root=str(root),
        baseline_files=files,
        image_dirs=imgs,
        image_detail=img_detail,
        artifacts=art,
        torch=torch_ok,
        tasks=[],
    )

    print("=" * 70)
    print("阶段 0.0 输入盘点")
    print("=" * 70)
    for f in files:
        print(f"  [{'OK ' if f['exists'] else 'MISS'}] {f['path']}"
              + (f"  {f['sha256'][:16]}..." if f["sha256"] else ""))
    if imgs:
        for d, n in imgs.items():
            print(f"  [OK ] {d}  = {n} 张  {img_detail.get(d, {})}")
    else:
        print("  图片目录：未找到（用 --data 指定，或检查 --root）")
    print(f"  torch    ：{torch_ok}")


    n_blocked = 0
    for t in ZERO_COST_TASKS:
        missing = []
        for need in t["inputs"]:
            if need.startswith("原始图片目录"):
                if not imgs:
                    missing.append(need)
            elif need in ("torch", "torchvision"):
                if torch_ok.startswith("UNAVAILABLE"):
                    missing.append(need)
            elif "/" in need or need.endswith(".csv") or need.endswith(".py"):
                if not any(f["path"] == need and f["exists"] for f in files):
                    if not any(root.glob("**/" + need.split("/")[-1])):
                        missing.append(need)
            else:
                if not any(root.glob("**/*" + need.replace(" ", "*") + "*")):
                    missing.append(need)
        blocked = bool(missing)
        n_blocked += blocked
        t2 = dict(t, missing=missing, blocked=blocked)
        report["tasks"].append(t2)
        print(f"\n  [{'BLOCKED' if blocked else 'READY  '}] {t['task']}")
        if blocked:
            print(f"      缺失：{missing}")
            print(f"      后果：{t['blocked_action']}")

    report["n_blocked_tasks"] = n_blocked
    #: N6：走统一盖加入口，产物自带 code_fingerprint（原来这里没有）
    write_artifact(out / "inventory.json", report)
    (out / "inventory.md").write_text(
        "\n".join([
            "# 阶段 0.0 输入盘点\n",
            f"- 根目录：`{root}`",
            f"- 被阻塞的零成本任务数：**{n_blocked}**",
            "",
            "| 任务 | 状态 | 缺失输入 | 后果 |",
            "|---|---|---|---|",
        ] + [f"| {t['task']} | {'BLOCKED' if t['blocked'] else 'READY'} | "
             f"{', '.join(t['missing']) or '-'} | {t['blocked_action'] if t['blocked'] else '-'} |"
             for t in report["tasks"]]), encoding="utf-8")
    print(f"\n输出：{out/'inventory.json'}、{out/'inventory.md'}")
    print(f"被阻塞的零成本任务数：{n_blocked}")


if __name__ == "__main__":
    main()
