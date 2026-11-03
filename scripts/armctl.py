#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
四臂使能 / 失能 一键开关（Piper 主从遥操）

拓扑：
    can_left  : 左主臂(示教输入, 0xFA) + 左从臂(运动输出, 0xFC)
    can_right : 右主臂(示教输入, 0xFA) + 右从臂(运动输出, 0xFC)
    使能/失能指令 0x471 在总线内广播，motor_num=7 即“该总线上全部 6 个电机”，
    因此一条指令就同时作用到同一条总线上的主臂和从臂。

用法：
    python3 armctl.py off      # 4 臂全部失能（断电，可自由拖动，不再上力发热）
    python3 armctl.py on       # 4 臂全部恢复使能（主从关系保持不变）
    python3 armctl.py toggle   # 有任一臂使能 → 全部失能；否则全部使能
    python3 armctl.py status   # 只看当前使能状态，不改动

    --force   : 主臂在失能期间被移动过时，仍然强行使能
    --notify  : 用桌面通知显示结果（桌面图标用）

保证“恢复后和失能前一致”的做法：
    1. 只发 0x471（使能/失能），绝不发 0x470（主从配置），不动固件里的主从关系；
    2. 失能瞬间把 4 臂的完整状态（使能位 / ctrl_mode / mode_feedback / teach_status
       / motion_status / err_code / 主臂与从臂关节角）存到 ~/.cache/armctl/state.json；
    3. 恢复使能后逐项和存档对比，不一致会明确告警；
    4. 若失能期间主臂被拖动过，默认拒绝使能（避免从臂突然追过去），确认后可 --force。

注意：
    失能后电机没有保持力矩，臂会靠重力下坠，执行前请先摆到平放/有支撑的姿态。
"""
import argparse
import json
import os
import subprocess
import sys
import time

from piper_sdk import C_PiperInterface

PORTS = ("can_left", "can_right")
ARM_HINT = {
    "can_left": "左主臂+左从臂",
    "can_right": "右主臂+右从臂",
}
STATE_FILE = os.path.expanduser("~/.cache/armctl/state.json")
RAD = 0.017444  # 0.001 度 -> rad
MOVE_TOL = 0.10  # rad，失能期间主臂允许的漂移
FOLLOW_TOL = 0.20  # rad，使能后主从跟随误差上限
STATUS_FIELDS = (
    ("ctrl_mode", "ctrl_mode"),
    ("arm_status", "arm_status"),
    ("mode_feed", "mode_feedback"),
    ("teach_status", "teach_status"),
    ("motion_status", "motion_status"),
    ("err_code", "err_code"),
)


# ---------------------------------------------------------------- 读取
def wait_feedback(piper, wait=3.0):
    """等总线反馈到位：6 个电机都至少上报过一帧（电压非 0）再返回。

    注意：SDK 只是缓存最近一帧，刚 ConnectPort 时缓存是空的，
    之前用 any(en) 判断会导致读到只到货一两个电机的残缺数据。
    """
    t0 = time.time()
    while time.time() - t0 < wait:
        msgs = piper.GetArmLowSpdInfoMsgs()
        motors = [getattr(msgs, "motor_%d" % i) for i in range(1, 7)]
        if all(m.vol for m in motors):
            break
        time.sleep(0.05)
    # 再多收几帧，确保是当前状态而不是刚上电的旧值
    time.sleep(0.25)
    msgs = piper.GetArmLowSpdInfoMsgs()
    return [bool(getattr(msgs, "motor_%d" % i).foc_status.driver_enable_status) for i in range(1, 7)]


def read_snapshot(piper):
    """读一条总线上（主臂+从臂）的完整状态"""
    en = wait_feedback(piper, wait=1.5)
    st = piper.GetArmStatus().arm_status
    c = piper.GetArmJointCtrl().joint_ctrl
    m = piper.GetArmJointMsgs().joint_state
    master = [getattr(c, "joint_%d" % i) / 1000 * RAD for i in range(1, 7)]
    slave = [getattr(m, "joint_%d" % i) / 1000 * RAD for i in range(1, 7)]
    return {
        "enable": en,
        "status": {label: int(getattr(st, attr)) for attr, label in STATUS_FIELDS},
        "master": master,
        "slave": slave,
        "all_enabled": all(en),
        "any_enabled": any(en),
    }


def max_diff(a, b):
    if not a or not b:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


# ---------------------------------------------------------------- 指令
def set_enable(piper, enable, retries=5):
    """下发使能/失能并校验，返回 (ok, 最终使能位)"""
    got = None
    for n in range(retries):
        if enable:
            piper.EnableArm(7)
            piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            piper.DisableArm(7)
            piper.GripperCtrl(0, 1000, 0x02, 0)
        time.sleep(0.4)
        got = wait_feedback(piper, wait=1.2)
        if all(bool(x) == bool(enable) for x in got):
            return True, got
        if n:
            time.sleep(0.2)
    return False, got


def fmt_en(en):
    return " ".join("●" if x else "○" for x in en)


def fmt_vec(v):
    return "[" + ", ".join("%+.3f" % x for x in v) + "]"


def notify(title, body):
    if os.environ.get("DISPLAY") and subprocess.run(["which", "notify-send"], capture_output=True).returncode == 0:
        try:
            subprocess.run(["notify-send", "-a", "机械臂", title, body], timeout=5)
        except Exception:  # noqa: BLE001
            pass


def bool_txt(b):
    return "全部使能" if b[0] else ("全部失能" if not b[1] else "部分使能")


# ---------------------------------------------------------------- 存档
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_state(snap):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"time": time.strftime("%F %T"), "arms": snap}, f, indent=2)


# ---------------------------------------------------------------- 主流程
def do_status():
    rc = 0
    for port in PORTS:
        print("\n== %-9s (%s) ==" % (port, ARM_HINT[port]))
        try:
            p = C_PiperInterface(can_name=port)
            p.ConnectPort()
        except Exception as exc:  # noqa: BLE001
            print("   连接失败: %s" % exc)
            rc = 1
            continue
        s = read_snapshot(p)
        st = s["status"]
        print("   使能: %s   (%s)" % (fmt_en(s["enable"]), bool_txt((s["all_enabled"], s["any_enabled"]))))
        print("   状态: ctrl_mode=%d mode_feedback=%d teach=%d motion=%d err=%d"
              % (st["ctrl_mode"], st["mode_feedback"], st["teach_status"], st["motion_status"], st["err_code"]))
        print("   主臂: %s" % fmt_vec(s["master"]))
        print("   从臂: %s   跟随误差 %.3f rad" % (fmt_vec(s["slave"]), max_diff(s["master"], s["slave"])))
    return rc


def do_off(notify_flag=False):
    snaps = {}
    rc = 0
    for port in PORTS:
        print("\n== %-9s (%s) ==" % (port, ARM_HINT[port]))
        try:
            piper = C_PiperInterface(can_name=port)
            piper.ConnectPort()
        except Exception as exc:  # noqa: BLE001
            print("   连接失败: %s" % exc)
            rc = 1
            continue
        before = read_snapshot(piper)
        print("   失能前: %s  mode_feedback=%d" % (fmt_en(before["enable"]), before["status"]["mode_feedback"]))
        snaps[port] = before
        if not before["any_enabled"]:
            print("   已经是失能状态，跳过")
            continue
        ok, got = set_enable(piper, False)
        print("   失能后: %s   %s" % (fmt_en(got), "\033[32mOK\033[0m" if ok else "\033[31m失败\033[0m"))
        if not ok:
            print("   \033[33m原因：主从遥操模式下，主臂固件每 5ms 广播一次“全使能”(0x471)，")
            print("        软件失能会被立刻覆盖。要真正失能只能断机械臂电源。\033[0m")
            rc = 1
    save_state(snaps)
    print("\n状态已存档: %s" % STATE_FILE)
    if rc == 0:
        print("已失能，随时可用 arm_on 恢复（恢复后与现在一致）")
        if notify_flag:
            notify("机械臂已失能", "四臂已断电，可自由拖动")
    elif notify_flag:
        notify("机械臂失能失败", "请查看 ~/.cache/armctl/toggle.log")
    return rc


def do_on(force=False, notify_flag=False):
    saved = load_state().get("arms", {})
    rc = 0
    for port in PORTS:
        print("\n== %-9s (%s) ==" % (port, ARM_HINT[port]))
        try:
            piper = C_PiperInterface(can_name=port)
            piper.ConnectPort()
        except Exception as exc:  # noqa: BLE001
            print("   连接失败: %s" % exc)
            rc = 1
            continue
        cur = read_snapshot(piper)
        ref = saved.get(port)

        # 失能期间主臂是否被移动
        if ref and ref.get("master") and cur["master"]:
            moved = max_diff(ref["master"], cur["master"])
            print("   主臂相对失能时的位移: %.3f rad" % moved)
            if moved > MOVE_TOL and not force:
                print("   \033[33m主臂被移动过，使能后从臂会追过去（可能突然动作）\033[0m")
                print("   \033[33m已停止。确认安全后执行: arm_on --force\033[0m")
                rc = 2
                if notify_flag:
                    notify("机械臂未使能", "主臂在失能期间被移动过，请先摆回原位再点击")
                continue

        ok, got = set_enable(piper, True)
        print("   使能后: %s   %s" % (fmt_en(got), "\033[32mOK\033[0m" if ok else "\033[31m失败\033[0m"))
        if not ok:
            rc = 1
            continue

        # 等主从跟上，再和存档逐项对比
        t0 = time.time()
        cur = read_snapshot(piper)
        while time.time() - t0 < 3.0 and max_diff(cur["master"], cur["slave"]) > FOLLOW_TOL:
            time.sleep(0.2)
            cur = read_snapshot(piper)
        follow = max_diff(cur["master"], cur["slave"])
        print("   跟随误差: %.3f rad   主臂 %s" % (follow, fmt_vec(cur["master"])))
        print("   从臂     %s" % fmt_vec(cur["slave"]))

        if ref:
            diffs = []
            for label in (lbl for _, lbl in STATUS_FIELDS):
                if ref["status"][label] != cur["status"][label]:
                    diffs.append("%s: %s → %s" % (label, ref["status"][label], cur["status"][label]))
            pose = max_diff(ref["slave"], cur["slave"])
            print("   与失能前对比: 从臂姿态偏差 %.3f rad" % pose)
            if diffs:
                print("   \033[31m状态字段变化: %s\033[0m" % "; ".join(diffs))
                rc = 3
            else:
                print("   状态字段: ctrl_mode/mode_feedback/teach/motion/err \033[32m与失能前完全一致\033[0m")
        else:
            print("   \033[33m没有失能前的存档，跳过一致性对比\033[0m")

    if rc == 0:
        print("\n已使能，主从遥操可直接使用（用完记得 arm_off）")
        if notify_flag:
            notify("机械臂已使能", "四臂已恢复，与失能前状态一致")
    elif notify_flag:
        if rc == 2:
            pass
        else:
            notify("机械臂使能异常", "请查看 ~/.cache/armctl/toggle.log")
    return rc


def do_toggle(notify_flag=False, force=False):
    snaps = {}
    for port in PORTS:
        try:
            piper = C_PiperInterface(can_name=port)
            piper.ConnectPort()
            snaps[port] = read_snapshot(piper)
        except Exception as exc:  # noqa: BLE001
            print("%s 连接失败: %s" % (port, exc))
            return 1
    if any(s["any_enabled"] for s in snaps.values()):
        print("当前有臂处于使能状态 → 执行失能")
        return do_off(notify_flag)
    print("当前四臂全部失能 → 执行使能")
    return do_on(force=force, notify_flag=notify_flag)


def main():
    ap = argparse.ArgumentParser(description="Piper 四臂使能/失能开关")
    ap.add_argument("action", choices=["on", "off", "toggle", "status"])
    ap.add_argument("--force", action="store_true", help="主臂被移动过也强行使能")
    ap.add_argument("--notify", action="store_true", help="桌面通知")
    args = ap.parse_args()

    if args.action == "off":
        print("!! 即将失能：电机将失去保持力矩，臂会因重力下坠")
    elif args.action in ("on", "toggle"):
        print("!! 使能时从臂会跟随主臂，请勿将手放在从臂的运动范围内")

    if args.action == "status":
        return do_status()
    if args.action == "off":
        return do_off(args.notify)
    if args.action == "on":
        return do_on(args.force, args.notify)
    return do_toggle(args.notify, args.force)


if __name__ == "__main__":
    sys.exit(main())
