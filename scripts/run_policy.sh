#!/bin/bash
# ============================================================================
# 上机推理：用训练好的 ACT 策略驱动从臂（需要机械臂处于 mode=1）
#
#   用法（机械臂已是 mode=1 时）：
#       bash ~/cobot_magic/run_policy.sh [ckpt目录] [权重文件名]
#   默认:  ~/train_cube_v2  policy_best.ckpt   （夹方块任务）
#
#   如果报错说"还在 mode=0"，先停掉采集用的 roslaunch，再执行：
#       bash ~/cobot_magic/bringup.sh arms 1
# ============================================================================
set -o pipefail
source /opt/ros/noetic/setup.bash 2>/dev/null

ACT_DIR=/home/agilex/cobot_magic/aloha-devel/act
CKPT_DIR=${1:-/home/agilex/train_cube_v2}
DATA_DIR=${DATA_DIR:-/home/agilex/data/grasp_cube}
CKPT_NAME=${2:-policy_best.ckpt}
PY=/home/agilex/miniconda3/envs/aloha/bin/python
MOVE_HOME=/home/agilex/cobot_magic/move_home.py

export PYTHONPATH=/home/agilex/cobot_magic/aloha-devel:/opt/ros/noetic/lib/python3/dist-packages
export MPLBACKEND=Agg

echo "==> 1/3 检查权重"
if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
  echo "  ❌ 找不到 $CKPT_DIR/$CKPT_NAME"
  ls "$CKPT_DIR" 2>/dev/null | head
  exit 1
fi
ls -lh "$CKPT_DIR/$CKPT_NAME" | awk '{print "  ✅ "$NF" ("$5")"}'

echo "==> 2/3 检查 ROS 与机械臂"
timeout 3 rosparam list >/dev/null 2>&1 || { echo "  ❌ ROS master 未启动"; exit 1; }
# 策略只用 cam_high(/camera_r) + cam_right_wrist(/camera_l)；/camera_f 已不再启动
for t in /puppet/joint_left /puppet/joint_right /camera_r/color/image_raw /camera_l/color/image_raw; do
  if ! timeout 3 rostopic info "$t" >/dev/null 2>&1; then
    echo "  ❌ 话题不存在: $t （机械臂或相机节点没起？）"; exit 1
  fi
done
echo "  ✅ 必需话题齐全"

PUB=$(timeout 3 rostopic info /master/joint_right 2>/dev/null | sed -n '/Publishers:/,/Subscribers:/p' | grep -c "^ \* ")
if [ "$PUB" -gt 0 ]; then
  echo "  ❌ /master/joint_right 有发布者 → 机械臂还在 mode=0【采集模式】"
  echo "     请先停掉采集的 roslaunch，然后执行:  bash ~/cobot_magic/bringup.sh arms 1"
  exit 1
fi
echo "  ✅ 机械臂处于 mode=1【推理模式】，从臂已使能"

echo "==> 3/4 移动到训练起点姿态（每次测试都从同一姿态出发，结果才可复现）"
echo "     速度约 0.2 rad/s，很慢；Ctrl+C 可随时中止"
echo
echo "  ⚠️  按回车后会发生两件事："
echo "     ① 双臂缓慢移动到训练起点姿态"
echo "     ② 紧接着策略开始动作（30 Hz）"
echo "     右手放在急停上，确认工作区内无人、无遮挡。"
echo
read -r _

"$PY" "$MOVE_HOME" "$DATA_DIR" || { echo "  ❌ 移动到起点失败，已中止"; exit 1; }

echo
echo "==> 4/4 启动策略 —— Ctrl+C 停止"
# 注意：inference.py 的 get_frame() 要求 img_left/img_right/img_front 三个图像队列
# 都非空才放行（仅用于时间同步），而策略实际只用 cam_high(/camera_r) 与
# cam_right_wrist(/camera_l)。所以 img_left 也指向 /camera_l，否则缺相机时永远 syn fail。
"$PY" "$ACT_DIR/inference.py" \
  --ckpt_dir "$CKPT_DIR" --ckpt_name "$CKPT_NAME" \
  --publish_rate 30 \
  --task_name grasp_cube \
  --img_front_topic /camera_r/color/image_raw \
  --img_left_topic  /camera_l/color/image_raw \
  --img_right_topic /camera_l/color/image_raw \
  --stop_after_release_sec "${STOP_AFTER_RELEASE:-2.0}" 2>&1 | tee -a /tmp/aloha_logs/policy_run.log
