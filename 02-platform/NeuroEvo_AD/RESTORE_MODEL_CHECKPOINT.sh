#!/usr/bin/env sh
# 还原 DMSA-Net 生产权重。
#
# Gitee 单文件上限 50MB，而 final_model_epoch126.pth 有 217.5MB，
# 因此把它切成 5 块存放（final_model_epoch126.pth.part01 ... part05）。
# 本脚本按顺序拼回原文件，并校验 SHA-256。
set -eu
cd "$(dirname "$0")"

PARTS_DIR="model_artifacts"
TARGET="$PARTS_DIR/final_model_epoch126.pth"
EXPECT_SHA="ba96575888e35764e3ff196674e52757d9a5f9c90742a38aa8f1d1f9a8457658"
EXPECT_SIZE=228051470

if [ -f "$TARGET" ]; then
  echo "[NeuroEvo] $TARGET 已存在，跳过。"
  exit 0
fi

echo "[NeuroEvo] 合并分块 ..."
cat "$PARTS_DIR"/final_model_epoch126.pth.part* > "$TARGET"

SIZE=$(wc -c < "$TARGET" | tr -d ' ')
if [ "$SIZE" != "$EXPECT_SIZE" ]; then
  echo "[NeuroEvo] 大小不符: 得到 $SIZE, 期望 $EXPECT_SIZE" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  GOT=$(sha256sum "$TARGET" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  GOT=$(shasum -a 256 "$TARGET" | awk '{print $1}')
else
  GOT=""
  echo "[NeuroEvo] 未找到 sha256sum/shasum，跳过哈希校验。"
fi

if [ -n "$GOT" ] && [ "$GOT" != "$EXPECT_SHA" ]; then
  echo "[NeuroEvo] SHA-256 不符: 得到 $GOT" >&2
  echo "[NeuroEvo] 期望 $EXPECT_SHA" >&2
  exit 1
fi

echo "[NeuroEvo] 完成: $TARGET ($SIZE 字节)"
[ -n "$GOT" ] && echo "[NeuroEvo] SHA-256 校验通过"
