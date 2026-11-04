#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =============================================================================
# g1_eta.sh —— 只读：算云端 G1 的**真实**剩余时间
#
# 为什么不能直接看日志里的 `剩余≈`：
#   `g1_ablation.py`（≤ v2026-09-24.3）的跨 run 进度累加是错的（见
#   `.workbuddy/memory/2026-09-25.md` §15.2），`step x/614400` 系统性少记，
#   因此它外推出的 `剩余≈` 会**偏高约 10 小时**（实测 45.9% vs 真实 79.3%）。
#
# 本脚本改用 **epoch 口径**自算（不依赖那个错的 step 计数）：
#   done_epochs = (run_i - 1) * epochs_per_run + current_epoch
#   total_epochs = run_total * epochs_per_run
#   速率 = done_epochs / 日志里的 `用时`（该字段是全局口径、可信）
#
# 用法：  bash g1_eta.sh [日志路径]
# 只读，不碰任何产物；训练中随时可跑。
# =============================================================================
set -u

LOG="${1:-/home/featurize/work/AIC_train_again/g1/g1_progress.log}"

if [ ! -f "$LOG" ]; then
    echo "❌ 找不到日志：$LOG"
    echo "   用 find 定位一下："
    echo "   find /home/featurize/work -maxdepth 4 -name g1_progress.log 2>/dev/null"
    exit 2
fi

LATEST="$(grep -F '[进度]' "$LOG" | grep -F 'val_acc=' | tail -1)"
if [ -z "$LATEST" ]; then
    echo "⚠️  日志里还没有带 val_acc 的进度行（可能刚启动，或每 epoch 收尾行尚未落盘）"
    echo "    最近 3 行原始内容："
    grep -F '[进度]' "$LOG" | tail -3
    exit 1
fi

echo "$LATEST" | awk '{
    # awk 折叠连续空白后的字段位（已实测核对）：
    #   $4=variant $6=epoch e/N  $14=用时 HH:MM:SS  $17=val_acc  $19=run i/N
    split($19, r, "/"); split($6, e, "/"); split($14, t, ":");
    if (r[2] == "" || e[2] == "" || t[3] == "") {
        print "⚠️  字段解析异常，请先手动 tail -1 看格式："; print "    " $0; exit 3;
    }
    done  = (r[1] - 1) * e[2] + e[1];
    total = r[2] * e[2];
    el    = t[1] * 3600 + t[2] * 60 + t[3];      # 全局已用秒
    rate  = done / el;                            # epoch/s
    rem   = (total - done) / rate;                # 剩余秒
    printf "\n  ── 读数 ────────────────────────────────────────\n";
    printf "   变体 / run      : %s   (run %s)\n", $4, $19;
    printf "   本 run 进度     : epoch %s\n", $6;
    printf "   全局已用时      : %s\n", $14;
    printf "   val_acc         : %s\n", substr($17, 9);   # 去掉 "val_acc=" 前缀
    printf "\n  ── 真实进度（日志里的百分比偏低，勿信）──────\n";
    printf "   %d / %d epoch = %.2f%%\n", done, total, 100 * done / total;
    printf "   速率            : %.4f epoch/s  (%.1f s/epoch)\n", rate, 1 / rate;
    printf "\n  ── 剩余 ────────────────────────────────────────\n";
    printf "   ≈ %.2f 小时 = %.0f 分钟\n", rem / 3600, rem / 60;
    printf "   ⇒ 用你**现在**的本地时间加上这段时间，就是收工时刻。\n";
    printf "     （日志时间戳是 UTC，比北京时间慢 8 小时，别被吓到。）\n\n";
}'
