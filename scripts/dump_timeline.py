#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8 -*-
"""把一个 episode 的 示教/预测 夹爪曲线打成时间线，看抓取与释放时刻。"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/agilex/cobot_magic')
from eval_ckpt import build_policy, eval_episode  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else '/home/agilex/train_cube_v3/policy_epoch_2100_seed_0.ckpt'
EPS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '10,25').split(',')]

with open('/home/agilex/train_cube_v3/dataset_stats.pkl', 'rb') as f:
    stats = pickle.load(f)
policy = build_policy(32).cuda()
policy.deserialize(torch.load(CKPT, map_location='cuda'))
policy.eval()


def phases(g):
    """返回 接近张开/抓取/释放 三个时刻（帧号，30fps）"""
    open1 = None
    for t in range(len(g)):
        if g[t] > 0.035:
            open1 = t
            break
    grasp = rel = None
    if open1 is not None:
        for t in range(open1, len(g)):
            if g[t] < 0.028:
                grasp = t
                break
    if grasp is not None:
        for t in range(grasp, len(g)):
            if g[t] > 0.032:
                rel = t
                break
    return open1, grasp, rel


for ep in EPS:
    p = f'/home/agilex/data/grasp_cube/episode_{ep}.hdf5'
    gt, pred = eval_episode(policy, stats, p, 32)
    g_gt, g_pr = gt[:, 13], pred[:, 13]
    o1, gr, rl = phases(g_gt)
    o1p, grp, rlp = phases(g_pr)
    f = lambda v: '-' if v is None else f'{v}({v/30:.1f}s)'
    print(f'\n===== episode_{ep} =====')
    print(f'示教：张开@ {f(o1)}  夹住@ {f(gr)}  释放@ {f(rl)}   夹住值 {g_gt[gr] if gr else 0:.4f}  释放峰 {g_gt[rl:].max() if rl else 0:.4f}')
    print(f'预测：张开@ {f(o1p)}  夹住@ {f(grp)}  释放@ {f(rlp)}  夹住值 {g_pr[grp] if grp else 0:.4f}  释放峰 {g_pr[rlp:].max() if rlp else 0:.4f}')
    print(f'{"t":>4} {"示教夹爪":>8} {"预测夹爪":>8} | {"示教右臂J1..J6":>0}')
    for t in range(0, len(g_gt), 10):
        arm = ' '.join(f'{v:+.2f}' for v in gt[t, 7:13])
        print(f'{t:>4} {g_gt[t]:8.4f} {g_pr[t]:8.4f} | {arm}')
