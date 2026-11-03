#!/bin/bash
# ============================================================================
# 定时自动采集（双手都占着也能用）
#   桌面通知读秒（静默，默认不发声） → 自动开录 → 自动停 → 自动质检 → 不合格自动重录
#
#   用法: bash collect_timed.sh [条数] [任务名] [起始编号] [每条帧数] [准备秒数] [最多重试次数]
#   默认: 3 条  grasp_test  从 0 开始  500 帧(≈16.7s)  准备 30 秒  每条最多录 3 次
#
#   质检不过的条目会被重命名为 episode_N.hdf5.bad_HHMMSS 并自动重录
# ============================================================================
set -o pipefail
source /opt/ros/noetic/setup.bash 2>/dev/null

N=${1:-3}
TASK=${2:-grasp_test}
START=${3:-0}
STEPS=${4:-500}
PREP=${5:-30}
MAXTRY=${6:-3}

DATA_DIR=/home/agilex/data
COLLECT=/home/agilex/cobot_magic/collect_data/collect_data.py
CHECK=/home/agilex/cobot_magic/check_act_data.py
PYQC=/home/agilex/miniconda3/envs/aloha/bin/python
# 相机话题映射：/camera_r=前方头视 → cam_high, /camera_l=右腕 → cam_right_wrist
# 当前只起 2 路相机，img_left 仅为时间同步存在（cam_left_wrist 不存盘），所以指向 /camera_l。
CAM_NAMES="cam_high,cam_right_wrist"
IMG_ARGS="--camera_names $CAM_NAMES --img_front_topic /camera_r/color/image_raw --img_left_topic /camera_l/color/image_raw --img_right_topic /camera_l/color/image_raw"

export DISPLAY=${DISPLAY:-:0}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}

# 声音默认全部关闭：只发桌面通知（静默）。需要语音/提示音时用 VOICE=1 / BEEP=1 启动
VOICE=${VOICE:-0}
BEEP=${BEEP:-0}
say()   { notify-send -t 2500 "ACT 采集" "$1" 2>/dev/null; echo "[$(date +%H:%M:%S)] $1"; }
voice() { [ "$VOICE" = "1" ] || return 0; spd-say -l zh "$1" >/dev/null 2>&1 || true; }
beep()  { [ "$BEEP" = "1" ] || return 0; paplay /usr/share/sounds/freedesktop/stereo/bell.oga >/dev/null 2>&1 || true; }

mkdir -p "$DATA_DIR/$TASK"
say "准备：${PREP} 秒后开始第 1 条（共 $N 条，每条最多录 $MAXTRY 次）"
voice "准备，${PREP} 秒后开始"

left=$PREP
while [ $left -gt 0 ]; do
  if [ $left -le 10 ] || [ $((left % 10)) -eq 0 ]; then
    say "倒计时 ${left} 秒"
    if [ $left -le 5 ]; then voice "$left"; fi
  fi
  sleep 1
  left=$((left - 1))
done

OK_CNT=0
FAIL_CNT=0

for idx in $(seq "$START" $((START + N - 1))); do
  k=$((idx - START + 1))
  attempt=1
  done_ok=0

  while [ $attempt -le $MAXTRY ]; do
    if [ $attempt -gt 1 ]; then
      say "🔁 重录第 $k/$N 条（第 $attempt 次尝试）—— 手摆球 + 机械臂回 home"
      voice "重录，摆好球和机械臂"
      sleep 12
    fi

    say "⏳ 3 秒后开录第 $k/$N 条 —— 手离开桌面！"
    voice "第 $k 条，三秒后开始，手离开桌面"
    sleep 3
    say "🔴 开录 第 $k/$N 条 —— 现在开始动作！"
    beep
    voice "开始动作"
    echo "------ collect episode_$idx attempt $attempt ------"
    python3 "$COLLECT" --dataset_dir "$DATA_DIR" --task_name "$TASK" \
      --episode_idx "$idx" --max_timesteps "$STEPS" --frame_rate 30 $IMG_ARGS
    rc=$?
    beep

    if [ $rc -ne 0 ]; then
      say "❌ 第 $k 条采集失败(rc=$rc) —— 可能有话题断流"
      attempt=$((attempt + 1)); continue
    fi

    f="$DATA_DIR/$TASK/episode_$idx.hdf5"
    qc=$("$PYQC" "$CHECK" "$f" 2>&1)
    echo "$qc"

    if echo "$qc" | grep -q "❌"; then
      why=$(echo "$qc" | grep -m2 "❌" | sed 's/  */ /g' | tr '\n' ';')
      say "❌ 第 $k 条质检不过：${why:0:80} —— 自动重录"
      voice "这条无效，自动重录"
      mv "$f" "$f.bad_$(date +%H%M%S)"
      attempt=$((attempt + 1))
    else
      sz=$(du -h "$f" 2>/dev/null | awk '{print $1}')
      say "✅ 第 $k/$N 条通过质检 (${sz})"
      voice "第 $k 条完成"
      done_ok=1
      break
    fi
  done

  if [ $done_ok -eq 1 ]; then
    OK_CNT=$((OK_CNT + 1))
  else
    FAIL_CNT=$((FAIL_CNT + 1))
    say "⚠️ 第 $k 条已试 $MAXTRY 次仍不合格，跳过（稍后手动补录 episode_$idx）"
  fi

  if [ "$k" -lt "$N" ]; then
    say "⏸ 停 15 秒：用手把球摆回起点 + 机械臂回 home（已停止录制）"
    sleep 15
  fi
done

say "全部结束：成功 $OK_CNT 条，失败 $FAIL_CNT 条"
voice "采集结束，成功 $OK_CNT 条"
echo "==================== 结果 ===================="
ls -lh "$DATA_DIR/$TASK"/ 2>/dev/null
