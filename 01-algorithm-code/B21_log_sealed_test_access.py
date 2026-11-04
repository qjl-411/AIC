#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sealed test 访问日志（v2.4）—— GPT P1-3 / T7 要求
==================================================

为什么必须有这个文件：
    sealed test 一旦被用于任何调参或模型选择，它就不再是独立测试集。
    基线脚本（`final_test.py` 等）**不允许改动**，因此访问记录必须由
    本独立工具承担。每次访问 sealed test 前调用它写一条 append-only 记录。

纪律（写进路线方案 §6.3）：
    · `purpose="audit"`       只做审计/统计，不参与任何选择 → sealed 仍完好
    · `purpose="tuning"`      **一旦出现，sealed 立即作废**，只能称 holdout
    · `purpose="formal_eval"` 正式评价，**只允许一次**
    · `purpose="sanity"`      仅检查文件可读性，不看标签

用法：
    # 记录一次审计性访问
    python log_test_access.py --log _test_access_log.jsonl --purpose audit \
        --protocol-id source_group_v1 --n 1279 --operator qiu
    # 记录（并因此作废）一次调参性访问
    python log_test_access.py --log _test_access_log.jsonl --purpose tuning --formal
    # 查看状态
    python log_test_access.py --log _test_access_log.jsonl --status
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from B15_lib_deduplication import TestAccessLog

PURPOSES = ("audit", "tuning", "formal_eval", "sanity")


def main():
    ap = argparse.ArgumentParser(description="sealed test 访问日志")
    ap.add_argument("--log", default="_test_access_log.jsonl")
    ap.add_argument("--purpose", default="audit", choices=PURPOSES)
    ap.add_argument("--formal", action="store_true",
                    help="计为一次正式评价（只允许一次）")
    ap.add_argument("--protocol-id", default="unknown")
    ap.add_argument("--operator", default="")
    ap.add_argument("--n", type=int, default=0, help="本次访问的样本数")
    ap.add_argument("--note", default="")
    ap.add_argument("--status", action="store_true", help="只打印当前状态，不写记录")
    ap.add_argument("--require-intact", action="store_true",
                    help="若 sealed 已破则返回非零退出码（供流水线做硬门）")
    a = ap.parse_args()

    tal = TestAccessLog(pathlib.Path(a.log))

    if a.status:
        s = tal.summary()
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return 0 if (s["sealed_intact"] or not a.require_intact) else 3

    if a.require_intact and not tal.is_sealed_intact():
        print("[REFUSE] sealed 已被破坏（曾出现 tuning/formal 访问）。")
        print("        该 test 集不得再被称作 independent test。")
        return 3

    if a.purpose == "formal_eval":
        prev = tal.summary()
        if prev["n_formal"] >= 1:
            print(f"[REFUSE] 已存在 {prev['n_formal']} 次正式评价。"
                  f"纪律：同一个 final run 不得重复正式评价。")
            return 4

    rec = tal.log(a.purpose, formal=(a.formal or a.purpose == "formal_eval"),
                  protocol_id=a.protocol_id, operator=a.operator,
                  n_items=a.n, note=a.note)
    print("已记录：")
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    s = tal.summary()
    print(f"\n累计访问 {s['n_accesses']} 次；sealed_intact = {s['sealed_intact']}")
    if not s["sealed_intact"]:
        print("⚠️ sealed 已破 —— 后续只能称 holdout，不得声称 independent test。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
