# ALOHA 右臂夹方块 · ACT 实机版

> 一句话：把「人手拖着机械臂示范 51 次」变成「机械臂自己看画面、把方块夹进盒子」。
> 本仓库是**在真机上跑通、带调试后处理**的版本。未修改的原厂代码在另一个仓库：
> **[act-factory](https://github.com/YR66666666/act-factory)**

---

## 目录

1. [30 秒看懂这套代码](#1-30-秒看懂这套代码)
2. [快速开始（4 条命令）](#2-快速开始4-条命令)
3. [代码地图：每个关键文件干什么](#3-代码地图每个关键文件干什么) ★
4. [一次推理到底发生了什么](#4-一次推理到底发生了什么) ★
5. [三个关键机制（为什么原厂逻辑不够用）](#5-三个关键机制为什么原厂逻辑不够用) ★
6. [数据格式与维度约定](#6-数据格式与维度约定)
7. [常用命令速查](#7-常用命令速查)
8. [与原厂的差异](#8-与原厂的差异)
9. [不包含的东西 / 环境要求 / 文档](#9-不包含的东西--环境要求--文档)

---

## 1. 30 秒看懂这套代码

```
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │  ① 采集       │    │  ② 训练       │    │  ③ 推理       │
 │ collect_data  │ →  │ train.py     │ →  │ inference.py │
 │              │    │              │    │              │
 │ 人手拖主臂    │    │ 51 条示范    │    │ 30 Hz 闭环    │
 │ 录 图像+关节  │    │ 训成 380MB模型│    │ 看画面 → 动臂 │
 └──────────────┘    └──────────────┘    └──────────────┘
     数据 36 GB          ckpt + pkl          ROS 发布指令
```

| 项目 | 内容 |
|------|------|
| 输入 | 2 路图像（头视 480×640 + 右腕 480×640）+ 14 维关节角 |
| 输出 | 未来 32 帧 × 14 维目标关节角（约 1 秒的动作块） |
| 控制频率 | 30 Hz（单次前向约 40 ms） |
| 维度 | 左臂 7（夹爪索引 6）+ 右臂 7（夹爪索引 13） |
| 执行者 | **只有右臂**，左臂全程不动（示教里就没动） |
| 夹爪 | 只下发两态：全开 `0.0685` / 全闭 `0.0` |

---

## 2. 快速开始（4 条命令）

```bash
conda activate aloha

# ① 上电（相机要先起，它兼任 ROS master）
bash scripts/bringup.sh can          # 需 sudo 密码
bash scripts/bringup.sh cams         # 终端 A，全程别关
bash scripts/bringup.sh arms 0       # 终端 B，采集模式（主臂上电）
bash scripts/bringup.sh check        # 终端 C，自检

# ② 采集（空格开始 / x 取消 / q 退出）
bash scripts/record_here.sh grasp_cube 0 400

# ③ 训练（约 2 小时，后台跑）
setsid nohup bash scripts/train_cube.sh grasp_cube ~/train_cube_v4 3000 8 \
    > /tmp/aloha_logs/train.log 2>&1 < /dev/null &

# ④ 推理（主臂必须断电！）
bash scripts/bringup.sh armstop
bash scripts/bringup.sh arms 1       # 推理模式
bash scripts/run_policy.sh ~/train_cube_v4 policy_epoch_3000_seed_0.ckpt
```

> 路径默认 `/home/agilex/cobot_magic/...`，换机器需改（见 `docs/07` 第五节）。

---

## 3. 代码地图：每个关键文件干什么

### 3.1 算法核心（`act/`）——只需要看懂这 4 个

| 文件 | 干什么 | 关键函数 / 关键参数 | 改过？ |
|------|--------|---------------------|--------|
| **`act/inference.py`** | **推理主程序**：连 ROS、读相机和关节、跑模型、发指令。整套真机闭环都在这里 | `model_inference()` 主循环<br>`inference_process()` 每帧推理<br>`decide_gripper()` 夹爪两态化<br>`dwell_release_check()` 到位/静止释放<br>`RosOperator` ROS 收发<br>`get_frame()` 时间同步取帧 | ✅ 大改 (+213 行) |
| **`act/train.py`** | **训练主程序**：读 HDF5、算归一化、训 3000 epoch、每 100 epoch 存权重 | `train()` 入口/配置<br>`train_process()` 训练循环<br>`forward_pass()` 单次前反向<br>`TASK_CONFIGS` 数据集路径规则 | ✅ 小改 (1 行) |
| **`act/policy.py`** | **策略封装**：把模型包装成 `train()` / `serialize()` / `deserialize()` 三个方法，训练和推理共用 | `ACTPolicy`（本项目用的）<br>`DiffusionPolicy` / `CNNMLPPolicy`（没用上） | ❌ 未改 |
| **`act/utils.py`** | **数据管道**：读 HDF5、算 qpos/action 的均值方差、切训练/验证集、`EpisodicDataset`（一个样本=一条 episode 的一段 chunk） | `load_data()`<br>`get_norm_stats()`<br>`EpisodicDataset.__getitem__()`<br>`compute_dict_mean()` / `set_seed()` | ❌ 未改 |

### 3.2 模型本体（`act/detr/`）

| 文件 | 干什么 |
|------|--------|
| `detr/models/detr_vae.py` | **网络结构**：`build()` 组装 ACT（ResNet 编码图像 + Transformer 编解码 + CVAE 分支），`Diffusion` 是另一条没用的分支 |
| `detr/models/backbone.py` | 图像主干：ResNet18 + 空间 softmax，把 480×640 压成一组特征点 |
| `detr/models/transformer.py` | Transformer 编码器/解码器（4 层编码 + 7 层解码） |
| `detr/models/position_encoding.py` | 位置编码 |
| `detr/main.py` | 参数定义 + `build_ACT_model_and_optimizer()` |

> **不需要动 `detr/`**。想改网络容量改 `--hidden_dim` / `--enc_layers` 就够。

### 3.3 采集程序（`collect_data.py`）

| 关键部分 | 干什么 |
|----------|--------|
| `KeyPoller` | **新增**：录制中按 `x` 立即取消本回合，按 `q` 退出（非 TTY 自动禁用） |
| `RecordingCancelled` | **新增**：取消时抛这个异常，进程以退出码 9 结束，外层脚本判定"作废" |
| `save_data()` | 把 `action / qpos / qvel / effort / base_action / images` 写进 HDF5 |
| `RosOperator.get_frame()` | 时间同步：三路图像取最小时间戳，四个关节消息都要 ≥ 它，否则 `syn fail` |
| `process()` | 主循环：30 Hz 采样、等空格、计时、写盘 |

### 3.4 工具脚本（`scripts/`）

| 脚本 | 什么时候用 | 关键参数 |
|------|-----------|---------|
| `bringup.sh` | 每天开工 | `can` / `cams` / `arms 0\|1` / `check` / `armstop` / `stop` / `status` |
| `record_here.sh` | 采数据（推荐） | `任务名 起始编号 帧数`，如 `grasp_cube 0 400` |
| `collect_timed.sh` | 双手都占着时采 | 同上（倒计时自动开录） |
| `train_cube.sh` | 训练 | `任务名 训练目录 epochs batch` |
| `run_policy.sh` | 上机推理 | `权重目录 权重文件名`；`STOP_AFTER_RELEASE=0` 可循环 |
| `run_trials.sh` | 批量测成功率 | `轮数 间隔秒 权重目录 权重名` |
| `check_act_data.py` | 采集后质检（8 项） | `数据目录` 或单条 hdf5；`--strict` |
| `check_spread.py` | 验证随机位置数据是否真采开 | `数据目录` |
| `eval_ckpt.py` | 不用机械臂，离线比权重 | `--ckpt --stats_dir --episodes` |
| `dump_timeline.py` | 看某条的夹爪时间线 | `ckpt路径 0,10` |
| `analyze_trials.py` | 批量日志判成败 | 日志路径（默认 `/tmp/aloha_logs/trials.log`） |
| `move_home.py` | 回训练起点（`run_policy.sh` 自动调用） | 数据目录 |
| `diag_joints.py` | 主从跟随诊断 | 秒数 话题A 话题B |
| `armctl.py` / `armctl.sh` | 查/控 4 臂使能 | `status` |

---

## 4. 一次推理到底发生了什么

`act/inference.py` 里 `model_inference()` 的循环，每 1/30 秒执行一遍：

```python
for t in count():                                    # ① 起始：移到训练起点（move_home 已完成）
    result = ros_operator.get_frame()                # ② 取一帧对齐的数据
    #    └ 2 路图像 + 左/右从臂 qpos(14维)；任一路没数据就 sleep(0.002) 重试
    obs = 组装成 OrderedDict(qpos, images...)         # ③
    qpos = (qpos - stats['qpos_mean']) / stats['qpos_std']   # ④ 归一化（stats 来自 dataset_stats.pkl）

    all_actions = policy(qpos, images)               # ⑤ 模型：输出 32×14 动作块
    all_actions = all_actions * stats['action_std'] + stats['action_mean']   # ⑥ 反归一化

    raw_action = 时间集成(all_actions, 历史)           # ⑦ exp(-0.01·i) 加权平均 → 平滑
    raw_g = raw_action[13]                            # ⑧ 右夹爪原始值
    g = decide_gripper(raw_g)                         # ⑨ 两态化 → 0.0685 或 0.0
    g, reason = dwell_release_check(...)              # ⑩ 到位释放 / 静止兜底（可能强制张开）

    ros_operator.puppet_arm_publish(left, right)      # ⑪ 发到 /master/joint_* → 从臂执行
    if 完成释放: sleep(stop_after_release_sec); break  # ⑫ 自动停止（0 = 一直循环）
```

对应关系：

| 步骤 | 代码位置 | 一句话 |
|------|---------|--------|
| ② | `RosOperator.get_frame()` | 4 个关节消息 + 3 路图像都到齐才放行 |
| ④⑥ | `stats` ← `<ckpt_dir>/dataset_stats.pkl` | **stats 必须和 ckpt 同一次训练产生**，否则动作整体偏移 |
| ⑤ | `policy(qpos, images)` | ACT 前向，`chunk_size=32` |
| ⑦ | `temporal_agg` 分支 | 每帧都重新预测，再和历史预测加权融合 |
| ⑨ | `decide_gripper()` | 连续值 → 两态（见第 5 节） |
| ⑩ | `dwell_release_check()` | 兜底松手（见第 5 节） |

**日志**（每 15 帧一行，排查全靠它）：

```
t= 5.00s 墙钟  5.0s 夹爪 模型=0.0364 → 下发=0.0000 实测=0.0281 (峰值 0.0540) 闭合 | 右臂 +0.64 +1.63 -1.05 +0.17 +0.99 -0.96 | 跟随偏差 0.01
```

| 字段 | 含义 | 判读 |
|------|------|------|
| `模型` | 模型原始输出 | 看它有没有给"释放峰" |
| `下发` | 两态化后的值 | `0.0685`=张开，`0`=闭合 |
| `实测` | 从臂真实开度 | **0.023~0.028 = 夹着方块**；0.060+ = 空手张开；0.000 = 空手闭合 |
| `跟随偏差` | 实测与目标最大差 | <0.1 正常；>0.3 要停 |

---

## 5. 三个关键机制（为什么原厂逻辑不够用）

### 5.1 夹爪两态化 —— `decide_gripper()`

**问题**：示教里"搬运段"夹爪显示 `≈0.041`，那是**方块把爪子撑住的测量值**，不是指令值。
操作员当时是把爪子一路夹到 0、靠力矩夹住方块。如果照抄 0.041 下发，等于"刚好贴住方块、零夹紧力"，一碰就滑；
到盒子处也张不到方块宽度之外，**松不了手**。

**做法**：模型给连续值，程序只发两态。用**每次开合循环各自的峰谷**判断，不用固定阈值
（本机实测靠近方块时峰值 0.071、到盒子释放峰只有 0.054，差 25%，任何绝对电平都会判丢）。

| 状态 | 触发 | 下发 |
|------|------|------|
| 张开 → 闭合 | 从峰值回落 15%（`--grip_close_frac 0.85`） | `0.0` |
| 闭合 → 张开 | 从谷值回升峰谷差的 35%（`--grip_reopen_frac 0.35`） | `0.0685` |

### 5.2 到位释放 —— `dwell_release_check()` 第一分支

**问题**：模型偶尔在盒子上方"抱着方块不松手"（输出一直停在夹住值）。实机见过：抱着方块飞回原位、半路掉落、在盒子上方停 20 秒。

**做法**：夹着东西 + 右臂进入盒子区域 + 持续 1 秒 → **强制张开**。

```python
--place_pose  0.885 2.030 -1.700 0.264 1.222 -0.698   # 49 条示教"释放瞬间"中位数
--place_radius 0.25                                    # 每轴偏差阈值（49/49 条落在 0.25 内）
--place_hold_sec 1.0                                   # 持续 1 秒才松手
```

日志：`🔓 到位释放：放置位偏差 0.12 rad，持续 1.0s，实测夹爪 0.0266 → 强制张开`

> 盒子换位置要改 `--place_pose`。实测 193 秒长跑：4 次到位释放、0 次卡死。

### 5.3 静止兜底 —— `dwell_release_check()` 第二分支

**问题**：策略收敛到某个姿态后彻底不动（模型不再给释放峰）。

**做法**：夹着东西 + 闭合后右臂移动过 ≥0.3 rad + **完全静止 ≥2 秒** → 强制张开。

> 注意：从臂在目标位置常有 0.1 rad/s 残余速度，这条经常等不到——这正是 5.2 存在的原因（5.2 只看位置，不看静止）。

### 5.4 自动停止 —— `--stop_after_release_sec`

完成一次"夹住→松开"后 N 秒（默认 2）退出；设 `0` = 一直循环。
不停止的话，方块已在盒子里，策略会跑回去"抓空气"。

---

## 6. 数据格式与维度约定

### HDF5（每条约 700 MB，400 帧）

```
/action                            (400, 14) float32   ← 主臂角度（人手动作）= 训练目标
/observations/qpos                 (400, 14) float32   ← 从臂实测（模型输入）
/observations/images/cam_high      (400, 480, 640, 3) uint8  ← /camera_r 头视
/observations/images/cam_right_wrist (400, 480, 640, 3) uint8 ← /camera_l 右腕
/observations/qvel | /effort | /base_action              ← 训练不用
```

### 14 维

| 索引 | 0–5 | **6** | 7–12 | **13** |
|------|-----|-------|------|--------|
| 含义 | 左臂 J1–J6 | 左夹爪 | 右臂 J1–J6 | 右夹爪 |

### 夹爪实测含义

| 值 | 状态 |
|----|------|
| `0.000` | 空手闭合（没夹到） |
| `0.023 ~ 0.028` | **夹着方块**（方块 23 mm） |
| `0.060 ~ 0.0685` | 空手张开 |

### 相机映射（只有 2 路，没有 `/camera_f`）

| 物理 | 话题 | ACT 名字 |
|------|------|---------|
| 头部 | `/camera_r/color/image_raw` | `cam_high` |
| 右腕 | `/camera_l/color/image_raw` | `cam_right_wrist` |

> `get_frame()` 要求 front/left/right 三个图像队列都非空才放行（时间同步用）。
> 所以采集/推理脚本里 `--img_left_topic` 也指向 `/camera_l`——**只参与同步，不入库**。

---

## 7. 常用命令速查

```bash
# 自检
bash scripts/bringup.sh status
bash scripts/bringup.sh check

# 采集 / 质检
bash scripts/record_here.sh grasp_cube_rand 0 400
python3 scripts/check_act_data.py ~/data/grasp_cube_rand --strict
python3 scripts/check_spread.py ~/data/grasp_cube_rand

# 训练 / 看进度
bash scripts/train_cube.sh grasp_cube_rand ~/train_rand 3000 8
tail -f ~/train_rand/train_*.log

# 离线比权重（不用机械臂）
python3 scripts/eval_ckpt.py --ckpt ~/train_cube_v3/policy_epoch_2100_seed_0.ckpt \
        --stats_dir ~/train_cube_v3 --episodes 0,10,25,40,50

# 上机
bash scripts/run_policy.sh ~/train_cube_v3 policy_epoch_2100_seed_0.ckpt
STOP_AFTER_RELEASE=0 bash scripts/run_policy.sh ~/train_cube_v3 policy_epoch_2100_seed_0.ckpt

# 批量 + 统计
bash scripts/run_trials.sh 10 20 ~/train_cube_v3 policy_epoch_2100_seed_0.ckpt
python3 scripts/analyze_trials.py
```

---

## 8. 与原厂的差异

| 文件 | 变化 |
|------|------|
| `act/inference.py` | **+213 / -18**：夹爪两态化、到位释放、静止兜底、日志、自动停止、取帧提频 |
| `act/train.py` | +1 / -1：固定相机 `['cam_high','cam_right_wrist']` |
| `collect_data.py` | +82 / -7：`KeyPoller`（录制中按 `x` 取消） |

其余（`policy.py` / `utils.py` / `detr/**` / `robomimic/**`）**未改**。
逐项说明见 **[DIFF.md](DIFF.md)**，完整 diff：

```bash
diff -u <原厂仓库>/act/inference.py act/inference.py
```

---

## 9. 不包含的东西 / 环境要求 / 文档

**不包含**（体积超过 GitHub 限制）：

| 内容 | 大小 |
|------|------|
| 训练数据 `~/data/grasp_cube`（51 条） | 36 GB |
| 训练权重 `~/train_cube_v3` | 8.2 GB（单个 ckpt 380 MB） |
| ROS 工作区 `camera_ws` / `Piper_ros_private-ros-noetic` | 126 MB（换机器要重新 `catkin_make`） |

**环境**：

| 组件 | 版本 |
|------|------|
| 系统 | Ubuntu 20.04 + ROS Noetic |
| Python | **3.8**（conda env `aloha`）—— 升 3.9 会让 `cv_bridge` 不可用 |
| PyTorch | 1.10.0 + cu113 |
| GPU | RTX 4060 8 GB（训练 batch 8，推理 ~30 Hz） |

**详细文档**（`docs/`）：

| 文档 | 内容 |
|------|------|
| `01_系统组成与数据流.md` | 硬件、话题表、mode 0/1、维度、夹爪标定 |
| `02_上电与自检.md` | 开工顺序、CAN、相机、自检输出解读 |
| `03_数据采集.md` | 示范规范、10 cm 随机位置采法、8 项质检 |
| `04_训练.md` | 参数、两个语义陷阱、真实收敛曲线、选权重 |
| `05_推理与实机运行.md` | 推理循环、日志解读、已知限制、调参速查 |
| `06_排错手册.md` | 按症状索引的 18 个问题 |
| `07_文件与数据清单.md` | 文件清单、库存、清理、迁移 |
| `cube_grid_10cm.pdf` | 采随机位置数据用的 1:1 方格纸 |

---

## 许可

`robomimic/` 与 ACT 原始实现版权归原作者；本仓库为内部使用。
