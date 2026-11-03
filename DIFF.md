# 原厂 vs 当前：改动清单

对比对象：

- `factory/` ← 从出厂包 `cobot_magic.zip`（2024-04-17）解压的原始代码
- `current/` ← 本机 2026-10-30 ~ 11-03 实际在跑的版本

```
factory/act/inference.py  765 行  →  current/act/inference.py  960 行
```

---

## 一、总体对照

| 文件 | 新增行 | 删除行 | 一句话 |
|------|--------|--------|--------|
| `act/inference.py` | +213 | -18 | 夹爪两态化 + 到位释放 + 静止兜底 + 日志 + 循环控制 |
| `act/train.py` | +1 | -1 | 固定相机为 `['cam_high','cam_right_wrist']` |
| `collect_data.py` | +82 | -7 | 录制中可按 `x`/`q` 立即取消本回合 |

其余文件（`policy.py`、`utils.py`、`aloha-devel/detr/**`、`robomimic/**`）**完全未改**。

> 查看完整差异：
> ```bash
> diff -u factory/act/inference.py current/act/inference.py
> diff -u factory/collect_data.py current/collect_data.py
> ```

---

## 二、`act/train.py`（+1 / -1）

```diff
     TASK_CONFIGS = {
         args.task_name: {
             'dataset_dir': os.path.join(DATA_DIR, args.task_name),
+            'camera_names': ['cam_high', 'cam_right_wrist'],
             'num_episodes': args.num_episodes
         }
     }
```

- 原厂没写 `camera_names`，会因 `KeyError` 或默认三相机而找不到 `cam_left_wrist` 图像
- 本项目只有 2 路相机，固定为头视 + 右腕

---

## 三、`act/inference.py`（+213 / -18）★ 主要改动

### 1. 新增函数

| 函数 | 作用 |
|------|------|
| `decide_gripper(v, args)` | 把模型输出的**连续夹爪值**转成**两态**（全开 0.0685 / 全闭 0.0） |
| `dwell_release_check(meas, args, now)` | 两条兜底释放：**静止兜底** + **到位释放** |
| `reset_gripper_state()` | 每回合开始时重置夹爪状态机 |

### 2. 新增命令行参数（15 个）

**夹爪两态化**

| 参数 | 默认 | 含义 |
|------|------|------|
| `--binarize_gripper` | True | 是否启用两态化 |
| `--grip_open` | 0.0685 | 张开时下发的值 |
| `--grip_close` | 0.0 | 闭合时下发的值 |
| `--grip_close_frac` | 0.85 | 峰值回落 15% 判定"开始闭合" |
| `--grip_reopen_frac` | 0.35 | 谷值回升 35% 判定"开始张开" |
| `--grip_min_span` | 0.012 | 峰谷差至少这么大才算一次有效开合 |

**静止兜底释放**

| 参数 | 默认 | 含义 |
|------|------|------|
| `--grip_hold_meas` | 0.018 | 实测夹爪 > 18 mm 才算"夹着东西" |
| `--grip_hold_moved` | 0.3 | 闭合后右臂至少移动 0.3 rad |
| `--grip_hold_move_eps` | 0.004 | 每帧位移小于此值算"静止" |
| `--grip_hold_still_sec` | 2.0 | 静止持续 2 秒触发 |
| `--grip_hold_keep` | 1.0 | 触发后强制张开保持 1 秒 |

**到位释放（11-03 新增）**

| 参数 | 默认 | 含义 |
|------|------|------|
| `--place_pose` | `0.885 2.030 -1.700 0.264 1.222 -0.698` | 盒子处右臂 6 轴姿态（49 条示教释放姿态中位数） |
| `--place_radius` | 0.25 | 每轴偏差 < 0.25 rad 视为"到盒子了" |
| `--place_hold_sec` | 1.0 | 在此区域内持续 1 秒 → 强制张开 |

**循环控制**

| 参数 | 默认 | 含义 |
|------|------|------|
| `--stop_after_release_sec` | 2.0 | 完成一次"夹住→松开"后 N 秒自动退出；`0` = 一直循环 |

### 3. 其它行为修改

| 位置 | 改动 | 原因 |
|------|------|------|
| 主循环取帧 | `rospy.sleep(0.033)` → `rospy.sleep(0.002)` 重试 | 相机 30 Hz、推理 40 ms，整帧睡会白等，有效频率从 ~16 Hz 提到 ~22 Hz |
| 日志 | 每 15 帧打印：模型值 / 下发值 / 实测值 / 峰值 / 状态 / 右臂 6 轴 / 跟随偏差 | 排查"夹没夹到、在哪松手"必需 |
| 回合结束 | 检测到释放后按 `--stop_after_release_sec` 自动停止 | 不停止会拿盒子里的方块当目标跑第二遍 |

### 4. 为什么原厂逻辑不够用（实测结论）

| 问题 | 原厂行为 | 后果 |
|------|----------|------|
| 连续值夹爪 | 模型输出 0.041 直接下发 | 示教里 0.041 是**被方块撑住的测量值**，照抄等于"贴住方块、零夹紧力"，一碰就滑；到盒子也张不到方块宽度外，松不了手 |
| 无释放兜底 | 只信模型的释放峰 | 模型偶尔停在夹住值，抱着方块飞回原位、半路掉落、或在盒子上方停 20 秒 |
| 无自动停止 | 跑满 `max_publish_step` | 会重复执行任务 |

三道保险的优先级：**模型给释放峰 → 到位释放（位置判据）→ 静止兜底（时间判据）**。
实测 193 秒长跑：4 次到位释放、0 次静止兜底、0 次卡死。

---

## 四、`collect_data.py`（+82 / -7）

```python
class RecordingCancelled(Exception): ...   # 用于中断录制
class KeyPoller:                           # 非阻塞读键盘
    """录制过程中按 x / q 立即取消本回合（不保存）。
       非 TTY（被管道/重定向）时自动禁用，不影响原有用法。"""
```

- 原厂：`input("please input s:")` 阻塞，一旦开录只能录满
- 现在：`record_here.sh` 里按**空格**开始、按 **x** 立即取消（数据不落盘、编号不变）、按 **q** 退出
- 取消时抛 `RecordingCancelled`，进程以退出码 **9** 结束，外层脚本据此判定"本回合作废"

---

## 五、没有改动但要知道的点

| 文件 | 说明 |
|------|------|
| `policy.py` / `detr/**` | 模型结构（ACT + Diffusion）与 2024 原厂一致 |
| `utils.py` | 数据加载 / 归一化 / `EpisodicDataset` 一致（一个样本 = 一条 episode 的一段 chunk） |
| `train.sh` | 原作者示例脚本，路径写的是别人机器（`/media/lin/T7/...`），**不要直接跑**；用 `current/scripts/train_cube.sh` |
| `factory/act/inference.py` | 原厂推理脚本没有夹爪后处理，**直接跑会夹住不松手**，仅作对照 |

---

## 六、时间线

| 日期 | 改动 |
|------|------|
| 2024-04-17 | 出厂包（`factory/`） |
| 2026-10-30 | `train.py` 固定 2 相机；`collect_data.py` 加 `KeyPoller` |
| 2026-10-31 | `inference.py` 夹爪两态化 + 静止兜底 + 自动停止 + 日志 |
| 2026-11-03 | `inference.py` 到位释放判据；取帧短睡提频 |
| 2026-11-03 | 新增 `scripts/`（训练、检查、评估、批量测试）与 `docs/`（7 篇教程） |
