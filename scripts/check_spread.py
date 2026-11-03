#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8 -*-
"""检查一批示范里"夹住瞬间的右臂姿态"有多分散 —— 用来确认方块位置真的变了。

为什么看这个：
    ACT 只能学到示教里出现过的情况。如果所有示范都在同一个点夹方块，
    那么"夹住瞬间的右臂 6 轴"几乎不变（std 很小），策略就只会去那一个点抓。
    采随机位置数据时，这个 std 应该明显变大。

用法：
    python3 check_spread.py [数据目录]        # 默认 ~/data/grasp_cube

判读：
    各轴 std  ~0.06 rad  → 位置几乎没变（旧固定位置数据集的水平）
    各轴 std  >0.15 rad  → 覆盖了明显的位置变化
"""
import glob
import os
import sys

import h5py
import numpy as np

DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/data/grasp_cube')
FILES = sorted(glob.glob(os.path.join(DIR, 'episode_*.hdf5')))
if not FILES:
    print('没找到 %s/episode_*.hdf5' % DIR)
    sys.exit(1)

GRASP_OPEN = 0.040      # 夹爪"张开"的判据（示教里张开会超过这个值）
GRASP_CLOSED = 0.032    # 判定"已夹住"（降到这个值以下）
NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

poses, times, skipped = [], [], []
for p in FILES:
    with h5py.File(p, 'r') as f:
        q = f['observations/qpos'][()]
    g = q[:, 13]
    t_open = next((t for t in range(len(g)) if g[t] > GRASP_OPEN), None)
    t_g = next((t for t in range(t_open, len(g)) if g[t] < GRASP_CLOSED), None) \
        if t_open is not None else None
    if t_g is None:
        skipped.append(os.path.basename(p))
        continue
    poses.append(q[t_g, 7:13])
    times.append(t_g)

P = np.array(poses)
if len(P) < 2:
    print('可用示范太少（%d 条），无法统计' % len(P))
    sys.exit(1)

med = np.median(P, axis=0)
print('数据目录: %s' % DIR)
print('参与统计: %d 条（跳过 %d 条：没检测到"张开→夹住"）' % (len(P), len(skipped)))
if skipped:
    print('  跳过: %s' % ', '.join(skipped[:6]) + (' ...' if len(skipped) > 6 else ''))
print()
print('夹住瞬间的右臂姿态（rad）：')
print('  %-4s %9s %9s %9s %9s' % ('', '中位数', '标准差', '最小', '最大'))
for i, n in enumerate(NAMES):
    print('  %-4s %9.3f %9.4f %9.3f %9.3f'
          % (n, med[i], P[:, i].std(), P[:, i].min(), P[:, i].max()))

dev = np.abs(P - med).max(axis=1)
print()
print('每条离中位数的最大单轴偏差：中位 %.3f / 90%% %.3f / 最大 %.3f rad'
      % (np.median(dev), np.percentile(dev, 90), dev.max()))
print('夹住时刻：中位 %.1f 帧（%.2f s），范围 %d~%d 帧'
      % (np.median(times), np.median(times) / 30.0, min(times), max(times)))

s = float(np.mean(P.std(axis=0)))
print()
if s < 0.08:
    print('结论: 平均 std %.3f rad → 【位置几乎没变】，和固定位置数据集一个水平。' % s)
    print('      如果这就是新采的"随机位置"数据，说明方块实际上没挪动，需要重采。')
elif s < 0.15:
    print('结论: 平均 std %.3f rad → 有一定位置变化，但范围偏小。' % s)
    print('      建议检查 10cm 方格是不是没铺开，或采集时漏掉了边角格点。')
else:
    print('结论: 平均 std %.3f rad → 【位置变化明显】，数据覆盖到了不同抓取点。' % s)

# 完全重复的抓取点（差 < 0.01 rad）会被列出来，便于发现"忘了换位置"
dups = []
for i in range(len(P)):
    for j in range(i + 1, len(P)):
        if np.abs(P[i] - P[j]).max() < 0.01:
            dups.append((os.path.basename(FILES[i]), os.path.basename(FILES[j])))
if dups:
    print('\n注意: 有 %d 对示范的抓取姿态几乎完全相同（差 <0.01 rad），例如：' % len(dups))
    for a, b in dups[:5]:
        print('  %s ≈ %s' % (a, b))
