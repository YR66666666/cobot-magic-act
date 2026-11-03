#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT 采集数据体检（索引约定：0-6 = 左臂, 7-13 = 右臂，与 collect_data.py 的
qpos = concat(puppet_arm_left, puppet_arm_right) 一致）
"""
import sys, glob, os
import numpy as np
import h5py

CAMS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
L, R = slice(0, 7), slice(7, 14)
DEFAULT_DIR = os.path.expanduser('~/data/grasp_cube')
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
STRICT = '--strict' in sys.argv[1:]
if _args and _args[0].endswith('.hdf5'):
    paths = [os.path.expanduser(_args[0])]
else:
    task_dir = os.path.expanduser(_args[0]) if _args else DEFAULT_DIR
    paths = sorted(glob.glob(os.path.join(task_dir, 'episode_*.hdf5')))
if not paths:
    print('没找到 episode_*.hdf5'); sys.exit(1)

summary = []
for p in paths:
    print('=' * 78)
    print(os.path.basename(p))
    with h5py.File(p, 'r') as f:
        n = f['/action'].shape[0]
        act, qpos = f['/action'][()], f['/observations/qpos'][()]
        base = f['/base_action'][()] if 'base_action' in f else np.zeros((n, 2))
        print('  帧数 %d (%.1f 秒 @30Hz)' % (n, n / 30.0))

        def rng(x):
            return x.max(0) - x.min(0)

        ra, rq = rng(act), rng(qpos)
        # 逐关节 min/max（右臂）
        print('  右臂各关节 (索引7-13)  qpos[min -> max]  幅度')
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', '夹爪']
        for i, nm in enumerate(names):
            k = 7 + i
            print('    %-4s  %7.3f -> %7.3f   %.3f' % (nm, qpos[:, k].min(), qpos[:, k].max(), rq[k]))

        grip = qpos[:, 13]
        # 夹爪"有效动作"：相邻帧变化超过 0.5mm 的帧占比
        dg = np.abs(np.diff(grip))
        grip_active = (dg > 5e-4).mean()

        # 残帧检测：单帧野值（如 J6=-292、夹爪=1895）会污染 dataset_stats 的 std，
        # 使该维归一化被压扁，必须拦下来。
        outlier_val = float(np.abs(qpos).max())
        outlier_jump = float(max(np.abs(np.diff(qpos, axis=0)).max(),
                                 np.abs(np.diff(act, axis=0)).max()))
        bad_frames = np.argwhere(np.abs(qpos) > 10)[:, 0].tolist()
        if not bad_frames:
            bad_frames = [int(i) + 1 for i in
                          np.argwhere(np.abs(np.diff(qpos, axis=0)) > 0.5)[:, 0]]

        checks = [
            ('无残帧/野值(<10rad, <0.5rad每帧)', outlier_val < 10 and outlier_jump < 0.5,
             'max %.1f, 跳变 %.1f%s' % (outlier_val, outlier_jump,
                                      (', 坏帧 %s' % bad_frames[:5]) if bad_frames else '')),
            ('右臂运动足够(>0.15rad)', max(ra[R][:6].max(), rq[R][:6].max()) > 0.15,
             'max %.3f' % max(ra[R][:6].max(), rq[R][:6].max())),
            ('右夹爪开合(>10mm)', (grip.max() - grip.min()) > 0.010,
             '范围 %.3f m, 变化帧占比 %.0f%%' % (grip.max() - grip.min(), grip_active * 100)),
            ('左臂静止(<0.05rad)', max(ra[L].max(), rq[L].max()) < 0.05,
             'max %.3f' % max(ra[L].max(), rq[L].max())),
            ('主从不相等', np.abs(act - qpos).max() > 1e-3,
             'max %.3f' % np.abs(act - qpos).max()),
            ('base 全零', np.abs(base).max() < 1e-6, 'max %.4f' % np.abs(base).max()),
            ('无长时间静止(>2s)', int((np.abs(np.diff(qpos[:, R][:6], axis=0)).max(1) < 1e-5).sum()) < 60,
             '静帧 %d 帧' % int((np.abs(np.diff(qpos[:, R][:6], axis=0)).max(1) < 1e-5).sum())),
        ]

        idxs = np.linspace(0, n - 1, 6).astype(int)
        present = [c for c in CAMS if ('/observations/images/' + c) in f]
        if not present:
            print('  ❌ 文件里没有任何图像')
            ok_all = False
        for c in present:
            ds = f[f'/observations/images/{c}']
            ms = np.array([[ds[i].mean(), ds[i].std()] for i in idxs])
            checks.append(('图像 %s' % c, ms[:, 0].mean() > 5 and ms[:, 1].mean() > 5,
                           'mean %.1f std %.1f' % (ms[:, 0].mean(), ms[:, 1].mean())))

    ok_all = True
    for name, ok, info in checks:
        print('  %s %-22s %s' % ('✅' if ok else '❌', name, info))
        ok_all &= ok
    summary.append((os.path.basename(p), ok_all))
    print('  => %s' % ('这条可用' if ok_all else '这条需要重采'))
    print()

print('=' * 78)
for n, ok in summary:
    print('  %s  %s' % ('✅' if ok else '❌', n))
_ok = all(o for _, o in summary)
print('\n结论：%s' % ('全部可用' if _ok else '有需要重采的条目（见上）'))
if STRICT:
    sys.exit(0 if _ok else 1)
