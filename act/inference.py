#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math
import threading


import sys
sys.path.append("./")

task_config = {'camera_names': ['cam_high', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

# ============================================================================
# 夹爪两态化（关键修复）
#   已用腕部相机实测确认物理含义：
#     grippers_angle = 0     → 爪尖相碰（全闭，靠力矩限幅夹住物体）
#     grippers_angle = 68000 → 最大张开
#   为什么必须两态：
#     示教数据里"运球段"的夹爪值(≈0.041)是【夹爪被球撑住的测量值】，不是目标值——
#     操作员当时是把夹爪一路夹到 0，靠力矩夹住球。模型学到的这个值如果直接下发，
#     等于"刚好贴住球、零夹紧力"→ 球一碰就滑；盒子处同理，开不到球径之外 → 不松手。
#   为什么不能用固定阈值：
#     实测本机曲线 第一峰 0.071(靠近球)、夹住球后掉到 ~0、第二个峰只有 0.054(盒子处)。
#     两个峰差 25%，任何"绝对电平"或"全局峰值比例"都会把释放判丢。
#   判定规则（按每次开合循环各自计峰）：
#     张开态：value 跌破【本次打开以来的峰值】的 85% → 闭合（下行 15%）
#     闭合态：value 从谷值回升到 (谷值 + 60% × (闭合时的峰值-谷值)) → 张开
# ============================================================================
grip_state_open = True          # 默认张开：远离球时张开无害，且避免"闭着爪去撞球"
grip_peak = -9.0                # 本次张开以来的峰值
grip_low = 9.0                  # 本次张开以来的最小值（判断有没有足够起伏）
grip_base = 0.0                 # 触发闭合那一刻的峰值
grip_trough = 9.0               # 闭合以来的谷值
grip_hist = deque()

# 兜底释放状态：ACT 有时会收敛到"回合结束"的姿态（夹住不动），此时模型输出
# 不再给出释放峰，光靠阈值永远等不到松手。这里用实测数据兜底。
dwell_anchor = None             # 闭合瞬间的右臂关节姿态
dwell_prev = None               # 上一帧实测姿态（判断是否静止）
dwell_moved = 0.0               # 闭合以来右臂最大位移（确认是真的搬运过）
dwell_still_since = None        # 开始静止的时刻
dwell_hold_until = 0.0          # 强制释放后保持张开到这个时刻
place_since = None              # 进入"放置位"范围的起始时刻（到位释放用）


def reset_gripper_state():
    global grip_state_open, grip_peak, grip_low, grip_base, grip_trough
    global dwell_anchor, dwell_prev, dwell_moved, dwell_still_since, dwell_hold_until, place_since
    grip_state_open = True
    grip_peak, grip_low, grip_base, grip_trough = -9.0, 9.0, 0.0, 9.0
    grip_hist.clear()
    dwell_anchor, dwell_prev, dwell_moved, dwell_still_since, dwell_hold_until = None, None, 0.0, None, 0.0
    place_since = None


def dwell_release_check(meas, args, now):
    """兜底释放：已闭合 + 实测量到夹着东西 + 搬运后静止够久 → 强制张开。
    meas: 实测右臂 [j0..j5, 夹爪]；返回下发值，None 表示不干预。"""
    global grip_state_open, grip_peak, grip_low
    global dwell_anchor, dwell_prev, dwell_moved, dwell_still_since, dwell_hold_until, place_since
    q6 = np.array(meas[:6], dtype=float)
    gmeas = float(meas[6])

    if now < dwell_hold_until:                     # 强制释放后的保持期
        grip_state_open = True
        grip_peak, grip_low = -9.0, 9.0            # 重新开始计峰谷，避免立刻又被判闭合
        dwell_anchor, dwell_moved, dwell_still_since = None, 0.0, None
        dwell_prev = q6
        return args.grip_open

    if grip_state_open:                            # 张开态：累积总位移，重置静止计时
        if dwell_anchor is None:
            dwell_anchor = q6.copy()
        dwell_moved = max(dwell_moved, float(np.abs(q6 - dwell_anchor).max()))
        dwell_prev = q6.copy()
        dwell_still_since = None
        return None

    if dwell_anchor is None:
        dwell_anchor = q6.copy()

    # 【到位释放】夹着东西 + 右臂已到达示教里的放置位 → 强制张开。
    # 为什么需要：模型偶尔会在盒子上方"抱着不放"（输出停在夹住值、不给释放峰），
    # 而"静止兜底"要求完全静止，从臂的残余微动会让它永远等不到。
    # 49 条示教的释放姿态高度一致（各轴标准差 0.03~0.06 rad），所以按位置判定可靠。
    if gmeas > args.grip_hold_meas:
        _dev = float(np.abs(q6 - np.array(args.place_pose, dtype=float)).max())
        if _dev < args.place_radius:
            if place_since is None:
                place_since = now
            elif now - place_since >= args.place_hold_sec:
                print('  🔓 到位释放：放置位偏差 %.2f rad，持续 %.1fs，实测夹爪 %.4f → 强制张开' % (
                    _dev, now - place_since, gmeas))
                dwell_hold_until = now + args.grip_hold_keep
                grip_state_open = True
                grip_peak, grip_low = -9.0, 9.0
                dwell_anchor, dwell_moved, dwell_still_since = None, 0.0, None
                place_since = None
                return args.grip_open
        else:
            place_since = None
    else:
        place_since = None

    moved_now = float(np.abs(q6 - dwell_prev).max()) if dwell_prev is not None else 0.0
    dwell_prev = q6.copy()
    dwell_moved = max(dwell_moved, float(np.abs(q6 - dwell_anchor).max()))
    if moved_now > args.grip_hold_move_eps:
        dwell_still_since = None
    elif dwell_still_since is None:
        dwell_still_since = now

    if (dwell_still_since is not None and now - dwell_still_since >= args.grip_hold_still_sec
            and dwell_moved >= args.grip_hold_moved and gmeas > args.grip_hold_meas):
        print('  🔓 兜底释放：静止 %.1fs，搬运 %.2f rad，实测夹爪 %.4f → 强制张开' % (
            now - dwell_still_since, dwell_moved, gmeas))
        dwell_hold_until = now + args.grip_hold_keep
        grip_state_open = True
        grip_peak, grip_low = -9.0, 9.0
        dwell_anchor, dwell_moved, dwell_still_since = None, 0.0, None
        return args.grip_open
    return None



def decide_gripper(v, args):
    """v: 模型输出的夹爪值(rad)；返回要下发的两态值"""
    global grip_state_open, grip_peak, grip_low, grip_base, grip_trough
    v = float(v)
    grip_hist.append(v)
    if grip_state_open:
        grip_peak = max(grip_peak, v)
        grip_low = min(grip_low, v)
        if grip_peak - grip_low > args.grip_min_span and v < args.grip_close_frac * grip_peak:
            grip_state_open = False
            grip_base = grip_peak
            grip_trough = v
    else:
        grip_trough = min(grip_trough, v)
        if v > grip_trough + args.grip_reopen_frac * (grip_base - grip_trough):
            grip_state_open = True
            grip_peak, grip_low = v, v
    return args.grip_open if grip_state_open else args.grip_close


def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    result = [pre_action]
    post_action = post_process(actions[0])
    # print("pre_action:", pre_action[7:])
    # print("actions_interpolation1:", post_action[:, 7:])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j])/steps[j]) for j in range(pre_action.shape[0])])
        inter = np.linspace(result[-1], post_action[i], step+2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size+1:
        result.append(result[-1])
    result = np.array(result)[1:args.chunk_size+1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_model_config(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)
   
    # 如果是ACT策略
    # fixed parameters
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'kl_weight': args.kl_weight,        # kl散度权重
                         'hidden_dim': args.hidden_dim,      # 隐藏层维度
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'dropout': args.dropout,
                         'pre_norm': args.pre_norm
                         }
    elif args.policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': 1,     # 查询
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base
                         }

    elif args.policy_class == 'Diffusion':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'observation_horizon': args.observation_horizon,
                         'action_horizon': args.action_horizon,
                         'num_inference_timesteps': args.num_inference_timesteps,
                         'ema_power': args.ema_power
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_step,
        'state_dim': args.state_dim,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names': task_config['camera_names'],
    }
    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, config, ros_operator, policy, stats, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            # 相机 30Hz、推理 ~45ms，按整帧(33ms)睡会白等掉一次机会；
            # 短睡重试可把有效控制频率从 ~16Hz 提到 ~22Hz。
            rospy.sleep(0.002)
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        obs = collections.OrderedDict()
        image_dict = dict()
        _img_by_name = {'cam_high': img_front, 'cam_left_wrist': img_left,
                        'cam_right_wrist': img_right}
        for _cn in config['camera_names']:
            image_dict[_cn] = _img_by_name[_cn]


        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            _depth_by_name = {'cam_high': img_front_depth, 'cam_left_wrist': img_left_depth,
                              'cam_right_wrist': img_right_depth}
            for _cn in config['camera_names']:
                image_depth_dict[_cn] = _depth_by_name[_cn]
            obs['images_depth'] = image_depth_dict

        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        # qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])
        start_time = time.time()
        all_actions = policy(curr_image, curr_depth_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']
        # print("obs['qpos']:", obs['qpos'][7:])
        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # 1 创建模型数据  继承nn.Module
    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)
    
    # 2 加载模型权重
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value
    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    # 3 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    # 4 加载统计值
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

    max_publish_step = config['episode_len']
    chunk_size = config['policy_config']['chunk_size']

    # 【已修改】原代码会先把双臂插值推到"零位"(left0/right0，夹爪 3.5578) 并等待回车，
    # 再推到 left1/right1 —— 这会破坏推理前对齐好的起始姿态，使策略从中等
    # 训练分布外的状态起步，导致动作完全失控。这里改为直接从当前姿态开始推理。
    action = None
    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            max_t = 0
            reset_gripper_state()
            rate = rospy.Rate(args.publish_rate)
            if config['temporal_agg']:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config['state_dim']])
            wall_t0 = time.time()
            _seen_close, _release_at = False, None
            while t < max_publish_step and not rospy.is_shutdown():
                # start_time = time.time()
                # query policy
                if config['policy_class'] == "ACT":
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, config, ros_operator,
                                                                  policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
                action = post_process(raw_action[0])
                # 夹爪两态化（自适应阈值）：见文件头 decide_gripper 说明
                if args.binarize_gripper:
                    raw_g = float(action[13])
                    action[13] = decide_gripper(raw_g, args)
                    if len(ros_operator.puppet_arm_right_deque):
                        _meas = ros_operator.puppet_arm_right_deque[-1].position[:7]
                        _forced = dwell_release_check(_meas, args, time.time())
                        if _forced is not None:
                            action[13] = _forced
                    if t % 15 == 0:
                        _lag, _mg = 0.0, float('nan')
                        if len(ros_operator.puppet_arm_right_deque):
                            _mr = np.array(ros_operator.puppet_arm_right_deque[-1].position[:6], dtype=float)
                            _mg = float(ros_operator.puppet_arm_right_deque[-1].position[6])
                            _lag = float(np.abs(_mr - np.array(action[7:13], dtype=float)).max())
                        print('  t=%5.2fs 墙钟%5.1fs 夹爪 模型=%.4f → 下发=%.4f 实测=%.4f  (峰值 %.4f)  %s | 右臂 %s | 跟随偏差 %.2f' % (
                            t / args.publish_rate, time.time() - wall_t0, raw_g, action[13], _mg, max(grip_hist),
                            '张开' if action[13] > 0.03 else '闭合',
                            ' '.join('%+.2f' % x for x in action[7:13]), _lag))
                # 自动停止：完成一次"夹住 → 松开"就退出，避免策略接着跑第二遍（会把方块又抓起来）
                if args.stop_after_release_sec > 0:
                    if action[13] <= 0.03:
                        _seen_close = True
                    elif _seen_close and _release_at is None:
                        _release_at = time.time()
                        print('  🏁 检测到释放，%.1f 秒后自动停止（要再跑请重新启动）' % args.stop_after_release_sec)
                    if _release_at is not None and time.time() - _release_at >= args.stop_after_release_sec:
                        print('  🏁 已自动停止')
                        rospy.signal_shutdown('task done')
                        return
                left_action = action[:7]  # 取7维度
                right_action = action[7:14]
                ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                if args.use_robot_base:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)
                t += 1
                # end_time = time.time()
                # print("publish: ", t)
                # print("time:", end_time - start_time)
                # print("left_action:", left_action)
                # print("right_action:", right_action)
                rate.sleep()


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    # 夹爪两态化（见文件头 decide_gripper 说明）
    parser.add_argument('--binarize_gripper', action='store', type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                        default=True, required=False, help='把右夹爪改成 全开/全闭 两态')
    parser.add_argument('--grip_open', action='store', type=float, default=0.0685, required=False,
                        help='张开时下发的值')
    parser.add_argument('--grip_close', action='store', type=float, default=0.0, required=False,
                        help='闭合时下发的值（0=全闭，靠力矩夹住物体）')
    parser.add_argument('--grip_close_frac', action='store', type=float, default=0.85, required=False,
                        help='跌破本次峰值该比例 → 闭合（默认下行15%）')
    parser.add_argument('--grip_reopen_frac', action='store', type=float, default=0.35, required=False,
                        help='从谷值回升该比例(×峰谷差) → 张开。方块任务的释放峰只比夹住值高 ~0.02，'
                             '0.60 时 51 条示教里只有 37 条能触发，0.35 时 50/50 全部触发')
    parser.add_argument('--grip_min_span', action='store', type=float, default=0.012, required=False,
                        help='峰谷差小于此值时不判定（避免起步噪声抖动）')
    parser.add_argument('--grip_hold_meas', action='store', type=float, default=0.018, required=False,
                        help='兜底释放：实测夹爪大于此值才算"夹着东西"')
    parser.add_argument('--grip_hold_moved', action='store', type=float, default=0.30, required=False,
                        help='兜底释放：闭合后右臂至少移动这么多 rad（确认搬运过，排除原地夹持）')
    parser.add_argument('--grip_hold_still_sec', action='store', type=float, default=2.0, required=False,
                        help='兜底释放：右臂静止超过该秒数才强制松手（避免半路停顿误放）')
    parser.add_argument('--grip_hold_move_eps', action='store', type=float, default=0.004, required=False,
                        help='兜底释放：单帧关节变化大于此值视为在动')
    parser.add_argument('--grip_hold_keep', action='store', type=float, default=1.0, required=False,
                        help='兜底释放后强制保持张开的秒数')
    parser.add_argument('--stop_after_release_sec', action='store', type=float, default=2.0, required=False,
                        help='完成一次"夹住→松开"后多少秒自动退出（0=不自动停，策略会继续跑第二遍）')
    parser.add_argument('--place_pose', action='store', type=float, default=[0.885, 2.030, -1.700, 0.264, 1.222, -0.698],
                        required=False, help='放置位（示教释放瞬间右臂 6 轴中位数），到位释放用')
    parser.add_argument('--place_radius', action='store', type=float, default=0.25, required=False,
                        help='进入放置位的判定半径（每轴最大偏差，49 条示教最大 0.21 rad）')
    parser.add_argument('--place_hold_sec', action='store', type=float, default=1.0, required=False,
                        help='在放置位持续多久才强制松手（示教里进入该范围到真正释放的中位时间是 1.2s）')

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == '__main__':
    main()
# python act/inference.py --ckpt_dir ~/train0314/