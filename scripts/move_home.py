#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把从臂缓慢移动到「训练数据的起始姿态（中位数）」—— 在 mode=1 下工作。

为什么需要它：
    ACT 的推理结果强烈依赖起始状态（关节角 + 画面）。每次测试都从同一个
    训练起点出发，成功率才可比、可复现。

用法（机械臂处于 mode=1 时）：
    source /opt/ros/noetic/setup.bash
    python3 ~/cobot_magic/move_home.py [数据集目录] [速度 rad/步]

安全：
    - 每步 0.01 rad、20 Hz，约 0.2 rad/s，全程最慢也是"慢慢挪"
    - 屏幕实时打印剩余误差，Ctrl+C 立即停止（停在当前位置）
"""
import sys, os, glob, time
import numpy as np
import rospy
from sensor_msgs.msg import JointState

DATASET_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/data/grasp_cube')
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.004
RATE_HZ = 100          # 必须够快：主臂曾以 200Hz 抢总线，低速指令会被覆盖
TOL = 0.04             # 2.3°，够严；J3 受关节限位只能到 0（比训练起点差 0.027），TOL 太小会白等 20s

import h5py
paths = sorted(glob.glob(os.path.join(DATASET_DIR, 'episode_*.hdf5')),
               key=lambda p: int(p.split('_')[-1].split('.')[0]))
if not paths:
    print('❌ %s 下没有数据，无法确定起始姿态' % DATASET_DIR)
    sys.exit(1)
S = np.array([h5py.File(p, 'r')['/observations/qpos'][0] for p in paths])
TARGET = np.median(S, axis=0)
print('起始姿态取自 %d 条数据的 qpos[0] 中位数' % len(paths))

cur = {'L': None, 'R': None}
rospy.init_node('act_move_home', anonymous=True)
rospy.Subscriber('/puppet/joint_left', JointState, lambda m: cur.__setitem__('L', np.array(m.position[:7], dtype=float)))
rospy.Subscriber('/puppet/joint_right', JointState, lambda m: cur.__setitem__('R', np.array(m.position[:7], dtype=float)))
pub = {'L': rospy.Publisher('/master/joint_left', JointState, queue_size=10),
       'R': rospy.Publisher('/master/joint_right', JointState, queue_size=10)}

rate = rospy.Rate(RATE_HZ)
t0 = rospy.Time.now()
while (cur['L'] is None or cur['R'] is None) and (rospy.Time.now() - t0).to_sec() < 10:
    rate.sleep()
if cur['L'] is None or cur['R'] is None:
    print('❌ 收不到 /puppet/joint_left|right —— 机械臂节点是否已启动？')
    sys.exit(1)
if pub['L'].get_num_connections() == 0:
    print('⚠️  /master/joint_left 没有订阅者 —— 机械臂可能不在 mode=1，指令不会被执行')

cmd = {'L': cur['L'].copy(), 'R': cur['R'].copy()}
tgt = {'L': TARGET[:7].copy(), 'R': TARGET[7:].copy()}

for k in ('L', 'R'):
    print('  %s臂 起始 %s' % ('左' if k == 'L' else '右',
                            ' '.join('%6.2f' % v for v in cur[k])))
    print('  %s臂 目标 %s' % ('左' if k == 'L' else '右',
                            ' '.join('%6.2f' % v for v in tgt[k])))

msg = JointState()
msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

last_print = 0.0
t_start = time.time()
TIMEOUT = 20.0          # 兜底：左主臂仍上电时会跟它抢，可能永远到不了目标
while not rospy.is_shutdown():
    if time.time() - t_start > TIMEOUT:
        print('\n⚠️  移动超时 20s，按当前姿态继续（右臂是重点，左臂仍由它的主臂控制）')
        break
    done = True
    for k in ('L', 'R'):
        d = tgt[k] - cmd[k]
        if np.abs(d).max() > TOL:
            done = False
            # 大误差用较大步长，接近目标时自动收敛到 STEP
            step = np.clip(d, -STEP, STEP)
            cmd[k] = cmd[k] + step
        msg.header.stamp = rospy.Time.now()
        msg.position = list(cmd[k])
        pub[k].publish(msg)
    if done:
        break
    now = rospy.Time.now().to_sec()
    if now - last_print > 0.5:
        last_print = now
        eL = np.abs(cur['L'] - tgt['L']).max() if cur['L'] is not None else 9
        eR = np.abs(cur['R'] - tgt['R']).max() if cur['R'] is not None else 9
        print('  移动中… 左剩 %.3f  右剩 %.3f rad' % (eL, eR))
    rate.sleep()

# 保持 1 秒，让从臂稳定到位
for _ in range(RATE_HZ):
    for k in ('L', 'R'):
        msg.header.stamp = rospy.Time.now()
        msg.position = list(cmd[k])
        pub[k].publish(msg)
    rate.sleep()

eL = np.abs(cur['L'] - tgt['L']).max() if cur['L'] is not None else 9
eR = np.abs(cur['R'] - tgt['R']).max() if cur['R'] is not None else 9
print('\n✅ 已到位：左臂残差 %.3f rad，右臂残差 %.3f rad' % (eL, eR))
print('   现在可以启动 run_policy.sh（它会从这个姿态直接开始）')
