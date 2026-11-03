#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
起始姿态对齐器：把左右从臂拖到"训练数据的中位起始姿态"，保证推理时
的初始状态在训练分布内。

用法（机械臂处于 mode=0 随动模式时，用手拖主臂）：
    source /opt/ros/noetic/setup.bash
    python3 ~/cobot_magic/home_pose.py [数据集目录]

判定：右臂 6 关节 + 左臂 6 关节误差都 < 0.06 rad 时显示 ✅ 可以开始推理
"""
import sys, os, glob
import numpy as np
import rospy
from sensor_msgs.msg import JointState

DATASET_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/data/grasp_test')
TOL = 0.06


def load_target():
    import h5py
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, 'episode_*.hdf5')),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    if not paths:
        return None, 0
    S = np.array([h5py.File(p, 'r')['/observations/qpos'][0] for p in paths])
    return np.median(S, axis=0), len(paths)


target, n_eps = load_target()
if target is None:
    print('❌ %s 下没有 episode_*.hdf5' % DATASET_DIR)
    sys.exit(1)

cur = {'L': None, 'R': None}


def cb(side, msg):
    cur[side] = np.array(msg.position[:7], dtype=float)


rospy.init_node('act_home_pose', anonymous=True)
rospy.Subscriber('/puppet/joint_left', JointState, lambda m: cb('L', m), queue_size=10)
rospy.Subscriber('/puppet/joint_right', JointState, lambda m: cb('R', m), queue_size=10)

print('目标姿态取自 %d 条数据的起始中位数' % n_eps)
print('等待 /puppet/joint_left|right …（机械臂节点是否已启动？）')

rate = rospy.Rate(5)
NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', '夹爪']


def fmt(side):
    c = cur[side]
    t = target[:7] if side == 'L' else target[7:]
    d = np.abs(c - t)
    lines = []
    for i, nm in enumerate(NAMES):
        mark = '✅' if d[i] < TOL else ('⚠️' if d[i] < 0.15 else '❌')
        lines.append('    %s %-4s 当前 %7.3f  目标 %7.3f  差 %5.3f' % (mark, nm, c[i], t[i], d[i]))
    return lines, d.max()


import select
import time

stable_since = None
stdin_ok = True     # stdin 关闭（非交互运行）时不再检测回车
ACCEPT_HOLD = 1.5   # 连续保持到位多少秒后自动结束

while not rospy.is_shutdown():
    rate.sleep()
    if cur['L'] is None or cur['R'] is None:
        continue
    lL, mL = fmt('L')
    lR, mR = fmt('R')
    os.system('clear')
    print('=' * 60)
    print(' 起始姿态对齐   容差 %.02f rad   右臂最大偏差 %.3f  左臂最大偏差 %.3f' % (TOL, mR, mL))
    print('=' * 60)
    print('  左臂（索引 0-6）')
    print('\n'.join(lL))
    print('  右臂（索引 7-13）')
    print('\n'.join(lR))
    print('-' * 60)
    if mR < TOL and mL < TOL:
        if stable_since is None:
            stable_since = time.time()
        held = time.time() - stable_since
        if held >= ACCEPT_HOLD:
            print('\n  ✅ 左右臂均已到位并保持 %.1f 秒 —— 对齐完成！' % held)
            sys.stdout.flush()
            rospy.signal_shutdown('aligned')
            break
        print('  ✅ 已经进入容差内，保持不动 %.1f / %.1f 秒即自动完成…'
              % (held, ACCEPT_HOLD))
    else:
        stable_since = None
        print('  ⚠️  请用手拖主臂（随动模式），把差缩小到 %.02f 以内' % TOL)
    print('  （也可以直接按【回车】强制接受当前姿态）')
    sys.stdout.flush()

    if stdin_ok:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            line = sys.stdin.readline()
            if line == '':
                stdin_ok = False      # 非交互环境，忽略
            else:
                print('\n  ✅ 已手动确认当前姿态\n')
                rospy.signal_shutdown('accepted')
                break
