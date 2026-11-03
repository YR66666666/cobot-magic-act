#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主从臂链路诊断：判断 /master 与 /puppet 是否构成有效的 (动作, 状态) 对。

ACT 训练里：
    observations/qpos <- /puppet/…   （从臂实际状态）
    action            <- /master/…   （示教值）

判定必须同时满足：
    1) 手拖主臂时 /master 有【足够大的运动幅度】（> 0.1 rad 量级）
    2) /puppet 同步跟随（幅度相当）
    3) 两者存在合理跟随误差（不是同一路反馈）

用法：
    python3 diag_joints.py [秒数] [master话题] [puppet话题] [--wait-motion]
        --wait-motion : 最多等 120 秒，检测到主臂明显运动后再采样 6 秒
"""
import sys
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState

args = [a for a in sys.argv[1:] if not a.startswith('--')]
flags = [a for a in sys.argv[1:] if a.startswith('--')]
DUR = float(args[0]) if len(args) > 0 else 8.0
MASTER = args[1] if len(args) > 1 else '/master/joint_right'
PUPPET = args[2] if len(args) > 2 else '/puppet/joint_right'
WAIT = '--wait-motion' in flags
MAX_WAIT = 120.0
MOTION_TH = 0.10          # 判定"主臂动了"的关节幅度阈值 (rad)
MOTION_KEEP = 6.0         # 检测到运动后再采多久

rec = {MASTER: [], PUPPET: []}


def mk(topic):
    def cb(msg):
        rec[topic].append(np.array(msg.position[:7], dtype=float))
    return cb


def rng_of(topic):
    a = np.array(rec[topic])
    return (a.max(0) - a.min(0)) if len(a) else np.zeros(7)


rospy.init_node('act_diag_joints', anonymous=True)
rospy.Subscriber(MASTER, JointState, mk(MASTER))
rospy.Subscriber(PUPPET, JointState, mk(PUPPET))

print('采样中… master=%s  puppet=%s' % (MASTER, PUPPET))
if WAIT:
    print('>> 请现在用手拖动【右主臂】，最多等 120 秒，检测到动作后自动收尾 <<')
    sys.stdout.flush()

t0 = time.time()
t_motion = None
rate = rospy.Rate(20)
while not rospy.is_shutdown():
    rate.sleep()
    elapsed = time.time() - t0
    if WAIT:
        if t_motion is None:
            if len(rec[MASTER]) > 20 and rng_of(MASTER).max() > MOTION_TH:
                t_motion = time.time()
                print('>> 检测到主臂运动 (%.1f 秒)，继续采样 %.0f 秒…' % (elapsed, MOTION_KEEP))
                sys.stdout.flush()
            elif elapsed > MAX_WAIT:
                print('>> 等待 %.0f 秒仍未检测到主臂运动，结束 <<' % MAX_WAIT)
                break
        elif time.time() - t_motion > MOTION_KEEP:
            break
    else:
        if elapsed > DUR:
            break

for t in (MASTER, PUPPET):
    n = len(rec[t])
    print('\n  %-26s 收到 %d 帧' % (t, n))
    if not n:
        print('      ❌ 完全无数据 —— 节点没起来或话题名不对')
        continue
    r = rng_of(t)
    print('      各关节变化幅度 = [%s] rad' % ' '.join('%.3f' % v for v in r))
    print('      最大幅度 = %.3f rad, 当前值 = [%s]'
          % (r.max(), ' '.join('%.3f' % v for v in np.array(rec[t])[-1])))

verdict = 'UNKNOWN'
if rec[MASTER] and rec[PUPPET]:
    r_m, r_p = rng_of(MASTER).max(), rng_of(PUPPET).max()
    n = min(len(rec[MASTER]), len(rec[PUPPET]), 400)
    d = np.abs(np.array(rec[MASTER][-n:]) - np.array(rec[PUPPET][-n:]))
    print('\n  跟随误差 |master-puppet|：均值 %.4f, 最大 %.4f rad' % (d.mean(), d.max()))
    print('\n  ===== 判定 =====')
    if r_m < MOTION_TH:
        verdict = 'NO_MOTION'
        print('  ❌ 主臂几乎没有运动 (%.3f rad < %.2f)' % (r_m, MOTION_TH))
        print('     可能原因：① 你没拖动主臂 / 拖的是从臂')
        print('               ② 主臂未被固件随动主从识别（检查 0xFA/0xFC 配置）')
        print('               ③ 主臂实际未上电或 CAN 未接')
        print('     → 现在不能采数据（action 恒定 = 学不到东西）')
    elif r_p < MOTION_TH * 0.5:
        verdict = 'NO_FOLLOW'
        print('  ⚠️  主臂动了 (%.3f) 但从臂几乎没跟随 (%.3f)' % (r_m, r_p))
        print('     → 固件随动主从未生效，先确认从臂是否跟着手一起动')
    elif d.max() < 1e-4:
        verdict = 'IDENTICAL'
        print('  ⚠️  master 与 puppet 完全相等 → 同一路反馈，不能用于采集')
    else:
        verdict = 'OK'
        print('  ✅ 主臂运动 %.3f rad，从臂跟随 %.3f rad，误差 %.4f rad' % (r_m, r_p, d.max()))
        print('     → 主从链路正常，可以开始采集')
    print('\n  VERDICT=%s' % verdict)
