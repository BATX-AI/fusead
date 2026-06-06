#!/usr/bin/env bash
# =============================================================================
# 【目标机器：有公网、无需代理】直连 figshare 下载 DyAD 三个数据集
# 数据: https://doi.org/10.6084/m9.figshare.23659323
# 用法: bash download_direct.sh        (默认下到 ./data)
# 说明: 目标机能直连 AWS S3 时, figshare 的 302→S3 跳转会满速完成, 无需任何中继/代理。
# =============================================================================
set -uo pipefail
DATA_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data}"
mkdir -p "$DATA_DIR"; cd "$DATA_DIR"

# figshare file_id : 名称 : 期望字节数(用于校验)
FILES=(
  "41520222:battery_brand1.tar.gz:1191991306"
  "41519265:battery_brand2.tar.gz:236920600"
  "41519169:battery_brand3.tar.gz:55469605"
)
for rec in "${FILES[@]}"; do
  IFS=':' read -r fid name size <<<"$rec"
  if [ -f "$name" ] && [ "$(stat -c%s "$name")" = "$size" ] && gzip -t "$name" 2>/dev/null; then
    echo "[skip] $name 已存在且完整"; continue
  fi
  echo "[get ] $name ($((size/1024/1024)) MB) ..."
  curl -fL --retry 5 --retry-delay 3 -C - \
       "https://ndownloader.figshare.com/files/$fid" -o "$name"
  got=$(stat -c%s "$name" 2>/dev/null || echo 0)
  if [ "$got" = "$size" ] && gzip -t "$name" 2>/dev/null; then
    echo "[ ok ] $name 校验通过 ($got bytes)"
  else
    echo "[FAIL] $name 大小=$got 期望=$size — 重跑本脚本续传"; exit 1
  fi
done

echo "=== 解压 ==="
for f in battery_brand1 battery_brand2 battery_brand3; do
  [ -f "$f.tar.gz" ] && { echo "解压 $f"; tar xzf "$f.tar.gz"; }
done

echo "=== 校验目录结构 ==="
# brand1/2: train/ + test/ ; brand3: data/
for b in 1 2 3; do
  n=$(find "battery_brand$b" -name '*.pkl' 2>/dev/null | wc -l)
  echo "  battery_brand$b: $n 个 pkl"
done
echo "期望: brand1≈476739, brand2≈194245, brand3≈29598"
echo "完成。下一步见 HANDOFF_PLAN.md 第 2 步(生成五折划分缓存)。"
