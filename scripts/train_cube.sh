#!/bin/bash
# ============================================================================
# 一条命令训练 ACT（本机右臂抓方块任务）
#
#   用法：
#     bash ~/cobot_magic/train_cube.sh [任务名] [训练目录] [epochs] [batch]
#
#   例：
#     bash ~/cobot_magic/train_cube.sh                          # 默认 grasp_cube → ~/train_grasp_cube
#     bash ~/cobot_magic/train_cube.sh grasp_cube_rand          # 用随机位置数据集训练
#     bash ~/cobot_magic/train_cube.sh grasp_cube /home/agilex/train_cube_v4 3000 8
#
#   它做了什么：
#     1) 数一遍 ~/data/<任务名>/episode_*.hdf5 的条数（train.py 的 --num_episodes 必须等于真实条数）
#     2) 先跑一遍质检（有不合格条目就拒绝开训，可用 SKIP_CHECK=1 跳过）
#     3) 后台跑训练，日志同时写到 <训练目录>/train_<时间>.log
#     4) 结束时打印推荐用哪个权重、下一步推理命令
#
#   长时间训练建议（关掉终端也不掉）：
#     setsid nohup bash ~/cobot_magic/train_cube.sh > /tmp/aloha_logs/train_cube.log 2>&1 < /dev/null &
#     然后 tail -f <训练目录>/train_*.log 看进度
#
#   继续训练（从已有权重起步）：PRETRAIN=~/train_cube_v3/policy_epoch_2100_seed_0.ckpt
# ============================================================================
set -o pipefail

TASK=${1:-grasp_cube}
TRAIN_DIR=${2:-/home/agilex/train_${TASK}}
EPOCHS=${3:-3000}
BATCH=${4:-8}

DATA_ROOT=/home/agilex/data
DATA_DIR="$DATA_ROOT/$TASK"
ACT_ROOT=/home/agilex/cobot_magic/aloha-devel
PY=/home/agilex/miniconda3/envs/aloha/bin/python
LOG_DIR=/tmp/aloha_logs
mkdir -p "$LOG_DIR"

red() { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }

# ---- 1. 数据检查 ----------------------------------------------------------
if [ ! -d "$DATA_DIR" ]; then
  red "❌ 数据目录不存在: $DATA_DIR"
  exit 1
fi
N=$(ls "$DATA_DIR"/episode_*.hdf5 2>/dev/null | wc -l)
NBAD=$(ls "$DATA_DIR"/*.bad_* 2>/dev/null | wc -l)
echo "==> 数据集: $DATA_DIR"
echo "    合格条数: $N   （不合格/已改名: $NBAD）"
if [ "$N" -lt 5 ]; then
  red "❌ 条数太少（$N < 5），先多采一些"
  exit 1
fi
if [ -d "$TRAIN_DIR" ] && ls "$TRAIN_DIR"/policy_epoch_*.ckpt >/dev/null 2>&1; then
  if [ "${FORCE:-0}" != "1" ]; then
    red "❌ 训练目录已存在且有权重: $TRAIN_DIR"
    echo "   换一个目录，或 FORCE=1 覆盖（旧权重会被同名文件覆盖）"
    exit 1
  fi
fi

if [ "${SKIP_CHECK:-0}" != "1" ]; then
  echo "==> 质检（约 $((N / 3)) 秒，SKIP_CHECK=1 可跳过）"
  if ! "$PY" /home/agilex/cobot_magic/check_act_data.py "$DATA_DIR" --strict > "$LOG_DIR/qc_train.log" 2>&1; then
    red "❌ 有不合格条目，拒绝开训："
    grep -E "❌" "$LOG_DIR/qc_train.log" | head -20
    echo "  （详情见 $LOG_DIR/qc_train.log；确认无碍可用 SKIP_CHECK=1 强行训练）"
    exit 1
  fi
  green "  ✅ 全部通过"
fi

mkdir -p "$TRAIN_DIR"
STAMP=$(date +%Y%m%d_%H%M)
LOG="$TRAIN_DIR/train_$STAMP.log"

# 预估：本机 RTX 4060 / 51 条 400 帧 / batch 8 实测约 2.5 秒每 epoch
EST=$(awk -v e="$EPOCHS" 'BEGIN{printf "%.0f", e*2.5/60}')
echo
echo "==> 开始训练"
echo "    任务名      : $TASK"
echo "    episode 数  : $N   （按 8:2 随机分成训练/验证）"
echo "    ckpt 目录   : $TRAIN_DIR"
echo "    epochs/batch: $EPOCHS / $BATCH"
echo "    预计耗时    : 约 $EST 分钟（每 100 epoch 存一个权重，每个约 380 MB）"
echo "    日志        : $LOG"
echo

cd "$ACT_ROOT" || exit 1
export PYTHONPATH="$ACT_ROOT:/opt/ros/noetic/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

"$PY" act/train.py \
  --dataset_dir "$DATA_ROOT" \
  --task_name "$TASK" \
  --ckpt_dir "$TRAIN_DIR" \
  --num_episodes "$N" \
  --num_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  ${PRETRAIN:+--pretrain_ckpt "$PRETRAIN"} \
  2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo
if [ "$rc" -ne 0 ]; then
  red "❌ 训练异常退出 rc=$rc，日志: $LOG"
  exit "$rc"
fi

green "==> 训练结束"
ls -1 "$TRAIN_DIR"/policy_epoch_*.ckpt 2>/dev/null | tail -3
LAST_CKPT=$(ls -1 "$TRAIN_DIR"/policy_epoch_*.ckpt 2>/dev/null | sort -t_ -k3 -n | tail -1)
LAST_CKPT_NAME=$(basename "${LAST_CKPT:-policy_best.ckpt}")
echo
echo "下一步："
echo "  1) 离线比对权重（在录制好的数据上算右臂跟踪误差，越小越好；一次一个）："
echo "     $PY ~/cobot_magic/eval_ckpt.py --ckpt $TRAIN_DIR/${LAST_CKPT_NAME} --stats_dir $TRAIN_DIR"
echo "  2) 实机跑："
echo "     bash ~/cobot_magic/run_policy.sh $TRAIN_DIR ${LAST_CKPT_NAME}"
echo "  提示：权重每个约 380 MB，3000 epoch 会存 30 个（约 11 GB），可只保留需要的几个。"
