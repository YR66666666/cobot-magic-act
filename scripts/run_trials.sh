#!/bin/bash
# 连续跑 N 轮：每轮 move_home + 推理 + 自动停止，然后留 PAUSE 秒给操作员放回方块
# 用法: bash run_trials.sh [轮数] [间隔秒] [权重目录] [权重文件名]
N=${1:-10}
PAUSE=${2:-20}
CKPT_DIR=${3:-/home/agilex/train_cube_v3}
CKPT=${4:-policy_epoch_2100_seed_0.ckpt}
OUT=/tmp/aloha_logs/trials.log
: > "$OUT"

echo "批次开始：$N 轮，每轮间隔 ${PAUSE}s，权重 $CKPT_DIR/$CKPT" | tee -a "$OUT"
for i in $(seq 1 "$N"); do
    echo "===== ROUND $i START $(date +%H:%M:%S) =====" >> "$OUT"
    bash /home/agilex/cobot_magic/run_policy.sh "$CKPT_DIR" "$CKPT" >> "$OUT" 2>&1 < /dev/null
    echo "===== ROUND $i END $(date +%H:%M:%S) =====" >> "$OUT"
    if [ "$i" -lt "$N" ]; then
        echo "----- 第 $i 轮结束，请把方块放回抓取位，${PAUSE} 秒后开始第 $((i+1)) 轮 -----" >> "$OUT"
        sleep "$PAUSE"
    fi
done
echo "===== ALL DONE $(date +%H:%M:%S) =====" >> "$OUT"
