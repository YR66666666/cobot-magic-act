#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8 -*-
"""离线验证一个 ACT 检查点：在【录制好的观测】上逐帧推理，对比预测动作与示教动作。

用途：上机之前先判断
  1) 右臂跟踪误差有多大（rad）
  2) 夹爪曲线有没有"闭合 → 搬运 → 释放"的形状（决定要不要靠兜底释放）
  3) 释放时刻比示教早/晚多少

用法：
  python eval_ckpt.py --ckpt ~/train_cube_v3/policy_epoch_2200_seed_0.ckpt \
      --stats_dir ~/train_cube_v3 --episodes 0,10,25,40,50
"""
import argparse
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from einops import rearrange

ACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cobot_magic', 'aloha-devel', 'act')
ACT_DIR = os.environ.get('ACT_DIR', ACT_DIR)
# act/ 目录里是 policy.py；它的上一级 aloha-devel/ 里是仓库自带的精简版 robomimic。
# 两个都要在 sys.path 里，否则会报 ModuleNotFoundError: No module named 'robomimic'
sys.path.insert(0, os.path.dirname(ACT_DIR))
sys.path.insert(0, ACT_DIR)
from policy import ACTPolicy  # noqa: E402

CAMERAS = ['cam_high', 'cam_right_wrist']


def build_policy(chunk_size=32):
    cfg = {
        'lr': 4e-5, 'lr_backbone': 4e-5, 'backbone': 'resnet18', 'masks': False,
        'weight_decay': 1e-4, 'dilation': False, 'position_embedding': 'sine',
        'loss_function': 'l1', 'chunk_size': chunk_size, 'camera_names': CAMERAS,
        'use_depth_image': False, 'use_robot_base': False, 'kl_weight': 10,
        'hidden_dim': 512, 'dim_feedforward': 3200, 'enc_layers': 4, 'dec_layers': 7,
        'nheads': 8, 'dropout': 0.1, 'pre_norm': False,
    }
    return ACTPolicy(cfg)


def eval_episode(policy, stats, path, chunk):
    with h5py.File(path, 'r') as f:
        qpos = f['observations/qpos'][:]            # (T,14)
        gt = f['action'][:]                         # (T,14)
        imgs = {c: f['observations/images/' + c] for c in CAMERAS}   # 惰性读，省内存
        T = qpos.shape[0]
        pred = np.zeros_like(gt)
        for t in range(T):
            batch = np.stack([rearrange(imgs[c][t], 'h w ch -> ch h w') for c in CAMERAS], axis=0)
            image = torch.from_numpy(batch / 255.0).float().cuda().unsqueeze(0)
            q = torch.from_numpy(((qpos[t] - stats['qpos_mean']) / stats['qpos_std'])).float().cuda().unsqueeze(0)
            with torch.inference_mode():
                a = policy(image, None, q)          # (1,chunk,14) 归一化空间
            a = a.cpu().numpy()[0] * stats['qpos_std'] + stats['qpos_mean']
            pred[t] = a[0]
    return gt, pred


def grip_events(g, thr=0.018):
    """返回 (闭合时刻, 闭合后峰值, 峰值时刻)。夹爪是索引 13。"""
    close_t = None
    for t in range(len(g)):
        if g[t] < thr:
            close_t = t
            break
    if close_t is None:
        return None, None, None
    seg = g[close_t:]
    pk = seg.max()
    return close_t, pk, close_t + int(seg.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--stats_dir', required=True, help='含 dataset_stats.pkl 的目录')
    ap.add_argument('--data', default='/home/agilex/data/grasp_cube')
    ap.add_argument('--episodes', default='0,10,25,40,50')
    ap.add_argument('--chunk_size', type=int, default=32)
    args = ap.parse_args()

    with open(os.path.join(os.path.expanduser(args.stats_dir), 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    policy = build_policy(args.chunk_size).cuda()
    sd = torch.load(os.path.expanduser(args.ckpt), map_location='cuda')
    ok = policy.deserialize(sd)
    policy.eval()
    print(f'ckpt: {args.ckpt}  加载: {ok}')
    print(f'episodes: {args.episodes}\n')

    eps = [int(x) for x in args.episodes.split(',')]
    all_err = []
    print(f"{'ep':>3} {'右臂MAE':>9} {'右夹爪MAE':>9} {'左臂MAE':>9} | "
          f"{'示教闭合@':>9} {'示教释放峰':>10} {'示教峰@':>8} | {'预测闭合@':>9} {'预测释放峰':>10} {'预测峰@':>8}")
    for ep in eps:
        p = os.path.join(os.path.expanduser(args.data), f'episode_{ep}.hdf5')
        if not os.path.exists(p):
            print(f'{ep:>3}  文件不存在')
            continue
        gt, pred = eval_episode(policy, stats, p, args.chunk_size)
        err = np.abs(pred - gt)
        r_arm, r_grip, l_arm = err[:, 7:13].mean(), err[:, 13].mean(), err[:, 0:6].mean()
        all_err.append((ep, r_arm, r_grip, l_arm))
        gc, gpk, gpt = grip_events(gt[:, 13])
        pc, ppk, ppt = grip_events(pred[:, 13])
        f_ = lambda v: '-' if v is None else (f'{v:.4f}' if isinstance(v, float) else f'{v}')
        print(f'{ep:>3} {r_arm:9.4f} {r_grip:9.4f} {l_arm:9.4f} | {f_(gc):>9} {f_(gpk):>10} {f_(gpt):>8} | '
              f'{f_(pc):>9} {f_(ppk):>10} {f_(ppt):>8}')

    if all_err:
        a = np.array([x[1:] for x in all_err])
        print(f'\n平均  右臂MAE {a[:,0].mean():.4f} rad   右夹爪MAE {a[:,1].mean():.4f}   左臂MAE {a[:,2].mean():.4f} rad')
        print('（上机经验：右臂 MAE < 0.10 一般能完成抓取搬运；夹爪 < 0.01 说明两态判断很干净）')


if __name__ == '__main__':
    main()
