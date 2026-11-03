#!/bin/bash
# ============================================================================
# AgileX Mobile ALOHA + ACT 抓取任务 上电/自检/采集 工具
#   用法: bash ~/cobot_magic/bringup.sh <子命令>
#
#   1) can      配置 CAN（需 sudo）：can_left / can_right(1M) + can0(500k 底盘)
#   2) arms     启动双机械臂节点（可指定 mode，默认 0=采集模式）
#   3) cams     启动 2 路 Dabai DC1 相机（camera_r=头视 / camera_l=右腕，关深度）
#   4) check    全链路自检：话题 + 频率 + 主从臂数据对比（含 ACT 可用性判定）
#   5) collect  批量采集 N 条示范（右臂抓取用）
#   6) armstop  只停机械臂、保留相机（mode=0 采集 → mode=1 推理 切换时用）
#   7) stop     关闭本工具启动的所有节点
#   8) status   查看 4 臂使能状态 / 主从跟随误差（不动硬件）
#
#   注意：主从遥操模式下主臂固件会持续把从臂拉回使能，软件无法让机械臂
#         长时间失能；不用机械臂时请直接关掉机械臂电源。
#
#   典型顺序（注意：不需要 sudo 跑整个脚本，内部只在配置 CAN 时提权一次）：
#     bash ~/cobot_magic/bringup.sh can
#     bash ~/cobot_magic/bringup.sh arms 0     # 终端 A（采集：发布主臂/从臂状态）
#     bash ~/cobot_magic/bringup.sh cams       # 终端 B
#     bash ~/cobot_magic/bringup.sh check      # 终端 C（边check边手拖右主臂）
#     bash ~/cobot_magic/bringup.sh collect 3
#
#   推理时改用 arms 1（订阅 /master/joint_* 驱动从臂）
# ============================================================================
set -o pipefail

# 路径从脚本位置推导，避免 `sudo bash ...` 时 $HOME 变成 /root 导致 ~ 失效
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COBOT_DIR="$SCRIPT_DIR"                    # /home/agilex/cobot_magic
HOME_DIR="$(dirname "$COBOT_DIR")"         # /home/agilex
PIPER_WS="$COBOT_DIR/Piper_ros_private-ros-noetic"
CAM_WS="$COBOT_DIR/camera_ws"
COLLECT_DIR="$COBOT_DIR/collect_data"
DATA_DIR="$HOME_DIR/data"
CAM_NAMES="cam_high,cam_right_wrist"
# 只起 2 路相机时 /camera_f 不存在；img_left 仅用于时间同步（cam_left_wrist 不存盘），指向 /camera_l
IMG_ARGS="--camera_names $CAM_NAMES --img_front_topic /camera_r/color/image_raw --img_left_topic /camera_l/color/image_raw --img_right_topic /camera_l/color/image_raw"
LOG_DIR=/tmp/aloha_logs
mkdir -p "$LOG_DIR"

cyan()  { echo -e "\033[36m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
red()   { echo -e "\033[31m$*\033[0m"; }

need_ros() {
  if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/noetic/setup.bash
  fi
}

cmd_can() {
  need_ros
  cyan "==> 配置 CAN（需要输入密码）"
  sudo bash "$PIPER_WS/can_config.sh" || { red "can_config.sh 失败"; exit 1; }
  echo
  cyan "==> 复核结果"
  ip -br link | grep -E "can0|can_left|can_right" || red "未找到 CAN 接口"
  for i in can_left can_right can0; do
    s=$(ip -details link show "$i" 2>/dev/null | grep -oP 'bitrate \K\d+')
    st=$(ip link show "$i" 2>/dev/null | grep -q UP && echo UP || echo DOWN)
    if [ -n "$s" ]; then
      echo "  $i : $st, bitrate=$s"
    else
      red "  $i : 不存在（检查 USB 是否插好）"
    fi
  done
  green "预期：can_left=1000000, can_right=1000000, can0=500000，且都是 UP"
}

cmd_arms() {
  need_ros
  local mode=${1:-0}
  source "$PIPER_WS/devel/setup.bash"
  cyan "==> 启动双机械臂（mode=$mode）"
  echo "    mode=0【采集】发布 /puppet/joint_*（从臂实测）+ /master/joint_*（随动传入的示教值），不控制"
  echo "    mode=1【推理】订阅 /master/joint_* 驱动从臂，自动使能；不发布 master"
  echo "    提示：mode=0 不会发使能指令；机械臂使能由主从固件决定（通电即遥操）"
  roslaunch piper start_ms_piper.launch mode:="$mode"
}

cmd_cams() {
  need_ros
  source "$CAM_WS/devel/setup.bash"
  cyan "==> 启动相机（cam_high=/camera_r + cam_right_wrist=/camera_l，关深度）"
  roslaunch astra_camera grasp_2cams.launch
}

cmd_check() {
  need_ros
  cyan "==> 1. 话题检查"
  local topics
  topics=$(timeout 10 rostopic list 2>/dev/null)
  if [ -z "$topics" ]; then
    red "无法连接 ROS master，请先启动 roslaunch（arms / cams）"; return 1
  fi
  for t in /camera_l/color/image_raw /camera_r/color/image_raw \
           /puppet/joint_left /puppet/joint_right /master/joint_left /master/joint_right; do
    if echo "$topics" | grep -qx "$t"; then green "  ✅ $t"; else red "  ❌ $t （缺失）"; fi
  done

  cyan "==> 2. 频率检查（各 5 秒，期望 30 / 200 / 200 Hz）"
  for t in /camera_r/color/image_raw /puppet/joint_right /master/joint_right; do
    hz=$(timeout 6 rostopic hz "$t" 2>/dev/null | grep -m1 "average rate" | awk '{print $3}')
    [ -z "$hz" ] && hz="无数据"
    echo "  $t : $hz Hz"
  done

  cyan "==> 3. 主从臂数据对比（接下来 5 秒请用手拖动【右主臂（leader）】，从臂会跟着动）"
  python3 "$COBOT_DIR/diag_joints.py" 5 /master/joint_right /puppet/joint_right
  echo
  cyan "==> 4. 结论"
  echo "  正常：master 随你手拖动明显变化，puppet 跟随，两者有 0.02~0.3 rad 的跟随误差"
  echo "        → 可以采数据"
  echo "  异常：master 全程不变 / 与 puppet 完全相等 / 无数据"
  echo "        → 先别采（会训出恒等映射的废策略）；检查固件随动主从配置与 mode 是否用错"
}

cmd_collect() {
  need_ros
  local n=${1:-3}
  local task=${2:-grasp_test}
  local start=${3:-0}
  cyan "==> 采集 $n 条示范，任务名=$task，起始编号=$start"
  echo "    规则：每条开始前把机械臂摆回同一个 home 位；动作慢、连续、不停顿；只保留成功示范。"
  echo
  cd "$COLLECT_DIR" || exit 1
  for i in $(seq "$start" $((start + n - 1))); do
    echo "------------------------------------------------------------------"
    read -r -p "第 $i 条：摆好 home 位后按回车开始（输入 s 跳过，q 退出）: " ans
    [ "$ans" = "q" ] && break
    [ "$ans" = "s" ] && continue
    python3 collect_data.py --dataset_dir "$DATA_DIR" --task_name "$task" \
      --episode_idx "$i" --max_timesteps 500 --frame_rate 30 $IMG_ARGS
    echo "  已保存 episode_$i.hdf5"
    read -r -p "  回放检查这一条？(y/N): " rp
    if [ "$rp" = "y" ]; then
      python3 visualize_episodes.py --dataset_dir "$DATA_DIR/$task" --episode_idx "$i"
    fi
  done
  echo
  green "采集结束。数据在 $DATA_DIR/$task/"
  ls -lh "$DATA_DIR/$task"/ 2>/dev/null | tail -5
}

cmd_armstop() {
  cyan "==> 只停止机械臂节点（保留相机，用于 mode=0 → mode=1 切换）"
  pkill -f "roslaunch piper" 2>/dev/null
  pkill -f piper_start_ms_node 2>/dev/null
  sleep 2
  pgrep -a -f piper_start_ms_node || green "  机械臂节点已停止（相机仍在运行）"
}

cmd_stop() {
  cyan "==> 关闭 roslaunch / 相机 / 机械臂节点"
  pkill -f "roslaunch astra_camera" 2>/dev/null
  pkill -f "roslaunch piper" 2>/dev/null
  pkill -f astra_camera_node 2>/dev/null
  pkill -f piper_start_ms_node 2>/dev/null
  sleep 2
  green "已停止。残留检查："
  pgrep -a -f "astra_camera_node|piper_start_ms_node" || echo "  无残留"
}

# 4 臂状态查询（只读，不动硬件）
cmd_status() { cyan "==> 4 臂使能状态"; bash "$COBOT_DIR/armctl.sh" status; }

case "${1:-}" in
  can)     cmd_can ;;
  armstop) cmd_armstop ;;
  arms)    shift; cmd_arms "$@" ;;
  cams)    cmd_cams ;;
  check)   cmd_check ;;
  collect) shift; cmd_collect "$@" ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  *)
    sed -n '3,/^# =====/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    ;;
esac
