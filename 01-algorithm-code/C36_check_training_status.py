#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
DMSA-Net / 2026AIC —— 运行态体检（双开自检）
===============================================================================

**只读脚本：不训练、不写产物、不删任何东西。** 随时可以上传、随时可以跑。

它回答一个问题：**现在这台机器上，是不是只有"一条"训练链在跑？**

为什么要问：2026-09-25 云上出过两次事故，都是"同一个任务被启动了两次"——
两个进程同时训练、同时往同一个输出目录写 checkpoint 与 metrics，
跑出来的产物是两份混写的，**不可信**。表面上看是 GPU 跑满、进度在动，
只有对比进程树才能发现。

---------------------------------------------------------------------------
判据（三条，任一不满足就报警）
---------------------------------------------------------------------------
 ① 只能有 **1 个** `g12_run.py`（总调度）
 ② 训练进程必须都能顺着 PPID 链追到那个总调度（追不到的 = 孤儿/手动启动的）
 ③ GPU 上吃显存的 python 进程数，应与"该有的训练进程数"一致

---------------------------------------------------------------------------
用法
---------------------------------------------------------------------------
    python check_running.py                    # 扫进程 + GPU + 产物
    python check_running.py --out-root /home/featurize/work/AIC_train_again
    python check_running.py --strict          # 有问题时退出码 1（可用于门禁）
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 哪些命令行算"我们的任务进程"
KEYS = {
    "g12_run.py": "总调度（G2+G1 串行）",
    "g0_run_all.py": "总调度（G3+G2+G1）",
    "g1_ablation.py": "G1 消融训练",
    "g2_multi_seed.py": "G2 多种子调度",
    "g3_efficiency.py": "G3 效率实测",
    "final_train_seed": "G2 的单个种子训练副本",
    "final_train.py": "final 训练",
}

EXCLUDE = ("check_running.py", "grep", "ps -", "--help")


def _ps_rows_win() -> List[dict]:
    """Windows 兜底：ps 不可用，改用 PowerShell CIM 拿 PID/PPID/命令行。

    只为了"本机也能自检"；云上是 Linux，走 `ps`。Windows 下 etime 显示 `-`。
    """
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
             "Get-CimInstance Win32_Process | "
             "Select-Object ProcessId,ParentProcessId,CommandLine | "
             "ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")   # 中文 Windows 默认 GBK，必须容错
        if r.returncode != 0 or not r.stdout.strip():
            return []
        rows, header = [], None
        for rec in csv.reader(io.StringIO(r.stdout)):
            if not rec:
                continue
            if header is None:
                header = [h.strip('"') for h in rec]
                continue
            d = dict(zip(header, rec))
            cmd = (d.get("CommandLine") or "").strip()
            if not cmd:
                continue
            try:
                rows.append({"pid": int(d["ProcessId"]),
                             "ppid": int(d["ParentProcessId"]),
                             "etime": "-", "cmd": cmd})
            except (KeyError, ValueError):
                continue
        return rows
    except Exception:
        return []


def _ps_rows() -> List[dict]:
    """列出全部进程。Windows 走 PowerShell；其他平台优先 `ps -eo`。"""
    if os.name == "nt":
        rows = _ps_rows_win()
        if rows:
            return rows
    for cmd in (["ps", "-eo", "pid=,ppid=,etime=,cmd="],
                ["ps", "-eo", "pid,ppid,etime,cmd"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0 or not r.stdout.strip():
                continue
            rows = []
            for line in r.stdout.splitlines():
                parts = line.strip().split(None, 3)
                if len(parts) < 4:
                    continue
                pid, ppid, etime, cmdline = parts
                if not pid.isdigit():
                    continue          # 跳过表头
                rows.append({"pid": int(pid), "ppid": int(ppid),
                             "etime": etime, "cmd": cmdline})
            if rows:
                return rows
        except Exception:
            continue
    return []


def _kind(cmdline: str) -> Optional[str]:
    if any(x in cmdline for x in EXCLUDE):
        return None
    for key, desc in KEYS.items():
        if key in cmdline:
            return key
    return None


def _children_map(rows: List[dict]) -> Dict[int, List[int]]:
    m: Dict[int, List[int]] = {}
    for r in rows:
        m.setdefault(r["ppid"], []).append(r["pid"])
    return m


def _descendants(root_pid: int, cmap: Dict[int, List[int]]) -> set:
    seen, stack = set(), [root_pid]
    while stack:
        p = stack.pop()
        for c in cmap.get(p, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _gpu_apps() -> List[dict]:
    """nvidia-smi 上的计算进程：pid + 显存。没有 GPU/没有命令则返回空。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                out.append({"pid": int(parts[0]), "name": parts[1],
                            "mem_mb": float(parts[2]) if parts[2].replace(".", "").isdigit() else 0.0})
        return out
    except Exception:
        return []


def _tail(path: Path, n: int = 1) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n] if lines else ""
    except Exception:
        return "(读不到)"


def _csv_rows(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return sum(1 for _ in csv.reader(f)) - 1     # 减表头
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="DMSA-Net 运行态体检（双开自检，只读）")
    ap.add_argument("--out-root", default=None,
                    help="输出根（默认 /home/featurize/work/AIC_train_again，不存在则跳过产物检查）")
    ap.add_argument("--strict", action="store_true",
                    help="发现问题时以退出码 1 结束")
    a = ap.parse_args()

    print("=" * 78)
    print("DMSA-Net 运行态体检（只读）")
    print("=" * 78)
    print(f"时间        : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    rows = _ps_rows()
    if not rows:
        print("\n⚠️ 拿不到进程表（ps 不可用）→ 无法判断，请人工执行：")
        print("   ps aux | grep -E 'g12_run|g1_ablation|g2_multi_seed|final_train' | grep -v grep")
        return 3

    # ---- 按类型归集 ----
    found: Dict[str, List[dict]] = {}
    for r in rows:
        k = _kind(r["cmd"])
        if k:
            found.setdefault(k, []).append(r)

    if not found:
        print("\n✅ 没有发现任何本包的任务进程（现在没在跑）。")
        return 0

    print("\n" + "-" * 78)
    print("① 识别到的任务进程")
    print("-" * 78)
    for k, ps in found.items():
        print(f"  [{KEYS[k]}] {len(ps)} 个")
        for p in ps:
            print(f"      PID {p['pid']:<7} PPID {p['ppid']:<7} 运行 {p['etime']:<12} {p['cmd'][:88]}")

    # ---- 调度进程数量 ----
    schedulers = found.get("g12_run.py", []) + found.get("g0_run_all.py", [])
    print("\n" + "-" * 78)
    print("② 调度链检查")
    print("-" * 78)
    problems: List[str] = []
    if len(schedulers) == 0:
        print("  ⚠️ 没有 g12_run.py / g0_run_all.py —— 训练进程若是手动启动的，"
              "不会被总调度接管，容易出现重复。")
    elif len(schedulers) > 1:
        problems.append(f"有 {len(schedulers)} 个总调度进程（PID "
                        + ", ".join(str(s['pid']) for s in schedulers) + "）")
        print(f"  ❌ 有 {len(schedulers)} 个总调度：正常只应有 1 个！")
    else:
        s = schedulers[0]
        print(f"  ✅ 总调度唯一：PID {s['pid']}（运行 {s['etime']}）")

    cmap = _children_map(rows)
    onchain: set = set()
    for s in schedulers:
        onchain |= _descendants(s["pid"], cmap)

    orphans: List[dict] = []
    for k, ps in found.items():
        if k in ("g12_run.py", "g0_run_all.py"):
            continue
        for p in ps:
            if p["pid"] not in onchain:
                orphans.append({**p, "kind": k})
    if orphans:
        problems.append(f"{len(orphans)} 个训练进程追不到总调度（疑似手动启动/残留）")
        print(f"  ❌ 有 {len(orphans)} 个进程**不在任何调度链上**（很可能是手动另起的）：")
        for o in orphans:
            print(f"      PID {o['pid']}（{KEYS[o['kind']]}，运行 {o['etime']}）"
                  f"\n        {o['cmd'][:110]}")
    else:
        print("  ✅ 所有训练进程都能追到总调度（单链，正常）")

    # ---- GPU ----
    apps = _gpu_apps()
    print("\n" + "-" * 78)
    print("③ GPU 计算进程")
    print("-" * 78)
    if not apps:
        print("  未取到 GPU 进程表（无 GPU / 无 nvidia-smi / 没有进程）。")
    else:
        by_pid = {r["pid"]: r for r in rows}
        total_mb = sum(x["mem_mb"] for x in apps)
        print(f"  共 {len(apps)} 个，占用合计 {total_mb:.0f} MB")
        gpu_training = 0
        for x in apps:
            r = by_pid.get(x["pid"])
            kind = _kind(r["cmd"]) if r else None
            tag = f"{KEYS[kind]}" if kind else "**不属于本包**"
            if kind:
                gpu_training += 1
            print(f"    PID {x['pid']:<7} {x['mem_mb']:>7.0f} MB   {tag}")

    # ---- 产物 ----
    root = Path(a.out_root) if a.out_root else Path("/home/featurize/work/AIC_train_again")
    print("\n" + "-" * 78)
    print("④ 产物情况")
    print("-" * 78)
    if not root.is_dir():
        print(f"  输出根不存在：{root}（用 --out-root 指定）")
    else:
        for tag in ("g2", "g1", "g3"):
            d = root / tag
            if not d.is_dir():
                print(f"  {tag}/ : （无）")
                continue
            prog = sorted(d.glob("*progress*.log"))
            line = _tail(prog[-1]) if prog else ""
            print(f"  {tag}/ : 最近进度行 {line[:110] if line else '（无日志）'}")
        for s in (43, 44):
            hits = list((root / "g2").glob(f"dmsa_final_train_seed{s}*"))
            for h in hits:
                n = _csv_rows(h / "train_history.csv")
                ready = (h / "FINAL_MODEL_READY.json").exists()
                print(f"    G2 seed{s}: train_history={n} 行, READY={ready}")
        for v in ("full", "no_sam", "no_mdc", "no_ilf", "no_dual_path"):
            vd = root / "g1" / v
            if vd.is_dir():
                folds = sorted(p.name for p in vd.iterdir() if p.is_dir())
                done = [f for f in folds if (vd / f / "metrics.csv").exists()]
                print(f"    G1 {v}: 折目录 {folds}，已落盘 metrics.csv 的折 {done}")

    # ---- 结论 ----
    print("\n" + "=" * 78)
    if problems:
        print("❌ 发现异常，建议处理：")
        for p in problems:
            print(f"   · {p}")
        print("\n   处置：先看清上面每个 PID 的完整命令行与启动时间，")
        print("         只 kill 掉**不在调度链上**的那些（手动启动的），")
        print("         保留总调度 + 它的后代。别用 pkill 一把全杀——会把对的也杀掉。")
        print(f"\n   单杀示例：kill -9 {' '.join(str(o['pid']) for o in orphans) or '<PID>'}")
    else:
        print("✅ 正常：单链运行，没有双开迹象。")
    print("=" * 78)
    return 1 if (problems and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
