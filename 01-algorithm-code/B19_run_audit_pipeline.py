#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键顺序执行（v2.7）
====================

    python run_all.py --root ..            # 全流程
    python run_all.py --root .. --only 0,1 # 只跑阶段 0、1

硬约束：**前置门（T1–T12 + T14）未通过，禁止进入真实数据扫描**。
阶段 0 输入盘点若报告 BLOCKED，对应任务必须走其 blocked_action，不得跳过。
**收口门 T13 永远最后跑**（任何一行 .py 改动都会让既有产物失效）。
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
PY = sys.executable

STEPS = [
    (0, "输入盘点", [PY, str(HERE / "a0_inventory.py"), "--root", "@root"]),
    (0.5, "数据集到货验证（**只读**：结构/帧几何/SHA 对账/只读纪律）",
     [PY, str(HERE / "a0b_dataset_verify.py"), "--data", "@data"]),
    (0.6, "检测器判别力实测（**真实数据**：null 池 vs manifest 自述正例）",
     [PY, str(HERE / "a0c_detector_power.py"), "--data", "@data",
      "--manifest", "@manifest", "--out", "_a0_out"]),
    (1, "前置门：T1-T12 + T14/T15 验收（单元+脚本+可复现性+到货只读+阈值强制）",
     [PY, str(HERE / "test_acceptance.py"), "--only",
      "T1,T2,T3,T4,T5,T6,T8,T9,T10,T11,T12,T14,T15"]),
    (2, "A1-cal 标定（标定集选阈值 / 留出集报告）", [PY, str(HERE / "a1cal_calibrate.py")]),
    (3, "A1 源图映射（train + test，真实数据）",
     [PY, str(HERE / "a1_source_map.py"),
      "--train", "@train", "--test", "@test", "--config", "@config"]),
    (4, "A2 train<->test 近重复审计（真实数据）",
     [PY, str(HERE / "a2_cross_audit.py"),
      "--train", "@train", "--test", "@test", "--config", "@config"]),
    (5, "阶段 1.2-1.3 结构泄漏审计 + ESS/DEFF",
     [PY, str(HERE / "audit_leakage.py"), "--root", "@root"]),
    (6, "A6-s 预训练权重静态映射（四列 + 代码事实）",
     [PY, str(HERE / "a6s_pretrain_probe.py"), "--model-module", "@model-module"]),
    (7, "阶段 3.1-3.2/4.1-4.2 架构探针",
     [PY, str(HERE / "arch_probe.py"), "--smoke", "@smoke"]),
    (8, "阶段 1.4-1.6/2.x 源图级统计（demo 自检）",
     [PY, str(HERE / "stage2_stats.py"), "--demo"]),
    (9, "sealed test 访问日志状态（硬门）", [PY, str(HERE / "log_test_access.py"),
                                    "--log", "_test_access_log.jsonl", "--status"]),
    #: T13 必须**最后**跑：它校验"产物是否由当前这一版代码产生"，
    #: 而任何一行 .py 改动都会让既有产物立刻失效。
    #: 放在最后 = 先产出、再验证新鲜度，避免"用旧产物证明新代码正确"。
    (10, "T13 产物溯源覆盖率与新适度（**收口门**）",
     [PY, str(HERE / "test_acceptance.py"), "--only", "T13"]),
]

#: `@xxx` 占位符 → 真实值。**用 `@` 前缀而不是裸 `--flag`**，
#: 因为裸 flag 充当占位符时无法同时表达"把值追加到 flag 之后"，
#: 会导致 `--config` 被替换成路径、flag 本身丢失（v2.6 版既有隐患）。
_PLACEHOLDERS = {
    "@root": lambda a: str(pathlib.Path(a.root).resolve()),
    "@data": lambda a: str(_resolve_data(a)),
    "@train": lambda a: str(_resolve_data(a) / "train"),
    "@test": lambda a: str(_resolve_data(a) / "test"),
    "@config": lambda a: str(pathlib.Path(a.config).resolve()),
    "@manifest": lambda a: str((HERE / ".." / "DMSA_net" / "pcv_v2_folds_seed42.csv").resolve()),
    "@model-module": lambda a: str((HERE / ".." / "DMSA_net" / "smoke_test.py").resolve()),
    "@smoke": lambda a: str((HERE / ".." / "DMSA_net" / "smoke_test.py").resolve()),
}

#: 到货数据的已知落位（嵌套目录），顺序即优先级
_DATA_CANDIDATES = [
    "Alzheimer's MRI Dataset/Alzheimer's MRI Dataset/Combined Dataset",
    "Alzheimer's MRI Dataset/Combined Dataset",
    "Combined Dataset",
]


def _resolve_data(a) -> pathlib.Path:
    if getattr(a, "data", ""):
        return pathlib.Path(a.data).resolve()
    root = pathlib.Path(a.root).resolve()
    for rel in _DATA_CANDIDATES:
        p = root / rel
        if (p / "train").is_dir() and (p / "test").is_dir():
            return p.resolve()
    return (root / _DATA_CANDIDATES[0]).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--data", default="",
                    help="含 train/ 与 test/ 的目录；留空则按已知落位自动探测")
    ap.add_argument("--config", default=str(HERE / "dedup_config.json"))
    ap.add_argument("--model-module", default="../DMSA_net/smoke_test.py")
    ap.add_argument("--only", default="", help="只跑指定步骤序号，如 `0,0.5,1`")
    a = ap.parse_args()

    want = {float(x) for x in a.only.split(",") if x.strip()} if a.only else None
    print("=" * 70)
    print("DMSA-Net v2.7 执行流水线（0 GPU 阶段）")
    print("=" * 70)
    print(f"数据目录：{_resolve_data(a)}")

    for idx, name, cmd in STEPS:
        if want is not None and idx not in want:
            continue
        #: 把命令里所有占位符就地替换成真实路径（不止最后一个）
        full = [_PLACEHOLDERS[tok](a) if tok in _PLACEHOLDERS else tok for tok in cmd]
        print(f"\n>>> [{idx}] {name}\n    $ {' '.join(full)}\n")
        r = subprocess.run(full, cwd=str(HERE))
        if r.returncode != 0:
            print(f"\n!!! 步骤 {idx}（{name}）退出码 {r.returncode}")
            if idx == 1:
                print("前置门未通过 —— 按纪律禁止进入真实数据扫描。")
            elif r.returncode == 2:
                print("该步骤报告 BLOCKED（缺输入/缺依赖）—— 见其输出与 "
                      "a0_inventory 的 blocked_missing_inputs，不要跳过。")
            else:
                print("按附录手册的'常见报错'一节定位；不要靠改阈值救结果。")
            return r.returncode
    print("\n全部步骤完成。产物在 _a0_out / _a1cal_out / _a1v2_out / _a2_out / "
          "_audit_out / _a6s_out / _arch_out / _s2_demo / _test_access_log.jsonl。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
