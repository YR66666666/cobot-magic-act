#!/bin/bash
# ============================================================================
# 空格键触发采集：摆好姿态后按【空格】才开始录制，全程由你控制节奏
#
#   用法（在你自己的终端里跑）：
#       bash ~/cobot_magic/record_here.sh [任务名] [起始编号] [每条帧数]
#   例：
#       bash ~/cobot_magic/record_here.sh grasp_cube 1 400
#
#   每条流程：按空格 -> 录制 13.3s(400帧) -> 自动质检 -> 通过则编号+1，不通过则自动改名重录
#   退出：按 q 或 Ctrl+C
# ============================================================================
set -o pipefail
source /opt/ros/noetic/setup.bash 2>/dev/null

TASK=${1:-grasp_cube}
START=${2:-0}
STEPS=${3:-400}

DATA_DIR=/home/agilex/data
COLLECT=/home/agilex/cobot_magic/collect_data/collect_data.py
CHECK=/home/agilex/cobot_magic/check_act_data.py
PYQC=/home/agilex/miniconda3/envs/aloha/bin/python
# 相机话题映射（本机实测）：/camera_r=前方头视 → cam_high, /camera_l=右腕 → cam_right_wrist
# 注意：当前只起 2 路相机（grasp_2cams.launch），/camera_f 不存在。
#       collect_data.py 的 get_frame() 要求 img_left/img_right/img_front 三个队列都非空才放行，
#       而 cam_left_wrist 不在 camera_names 里（不存盘），所以 img_left 直接指向 /camera_l，
#       只用于时间同步。若哪天重新接左腕相机，改回 /camera_f 即可。
CAM_NAMES="cam_high,cam_right_wrist"
IMG_ARGS="--camera_names $CAM_NAMES --img_front_topic /camera_r/color/image_raw --img_left_topic /camera_l/color/image_raw --img_right_topic /camera_l/color/image_raw"

SECS=$(awk -v n="$STEPS" 'BEGIN{printf "%.1f", n/30}')

mkdir -p "$DATA_DIR/$TASK"
idx=$START

printf '\n'
printf '============================================================\n'
printf '  空格键触发采集  任务=%s  每条 %s 帧 (约 %s 秒)\n' "$TASK" "$STEPS" "$SECS"
printf '  按【空格】开始录制 · 按【q】退出\n'
printf '  录制过程中按【x】= 立刻取消本回合（不保存，编号不变，可重录）\n'
printf '============================================================\n'

while true; do
  f="$DATA_DIR/$TASK/episode_$idx.hdf5"

  # 已存在且合格 -> 自动跳过，避免误覆盖好数据
  if [ -f "$f" ] && "$PYQC" "$CHECK" "$f" --strict >/dev/null 2>&1; then
    printf '  (episode_%s 已存在且合格，自动跳过)\n' "$idx"
    idx=$((idx + 1))
    continue
  fi

  printf '\n---------------- episode_%s ----------------\n' "$idx"
  printf '  摆好球和机械臂后，按【空格】开始录制\n'
  printf '  > '

  key=""
  while true; do
    read -rsn1 key
    if [ "$key" = "q" ] || [ "$key" = "Q" ]; then
      printf '\n已退出。当前下一个编号：episode_%s\n\n' "$idx"
      exit 0
    fi
    # 空格 / 回车 都可以
    if [ "$key" = " " ] || [ "$key" = "" ]; then
      break
    fi
    printf '\n  请按【空格】开始，或按【q】退出\n  > '
  done

  printf '\n  >>> %s 开录，请开始动作（%s 秒）；做错了就按【x】取消\n\n' "$(date +%H:%M:%S)" "$SECS"
  python3 "$COLLECT" --dataset_dir "$DATA_DIR" --task_name "$TASK" \
    --episode_idx "$idx" --max_timesteps "$STEPS" --frame_rate 30 $IMG_ARGS
  rc=$?

  if [ $rc -eq 9 ]; then
    printf '\n  ⛔ 本回合已取消，未保存（还是 episode_%s）\n  按空格重新录\n' "$idx"
    continue
  fi

  if [ $rc -ne 0 ]; then
    printf '\n  ❌ 采集失败(rc=%s)，可能有话题断流。按空格重试\n' "$rc"
    continue
  fi

  printf '\n  >>> 录制结束，质检中…\n'
  qc=$("$PYQC" "$CHECK" "$f" 2>&1)
  echo "$qc" | grep -E '❌|=>|帧数'

  if echo "$qc" | grep -q '❌'; then
    bad="$f.bad_$(date +%H%M%S)"
    mv "$f" "$bad"
    printf '\n  ❌ 这条不合格，已改名 %s\n  按空格重录 episode_%s\n' "$(basename "$bad")" "$idx"
    continue
  fi

  sz=$(du -h "$f" | awk '{print $1}')
  printf '\n  ✅ episode_%s 通过质检 (%s)\n  下一条：episode_%s\n' "$idx" "$sz" "$((idx + 1))"
  idx=$((idx + 1))
done
