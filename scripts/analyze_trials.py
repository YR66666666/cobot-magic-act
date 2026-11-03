#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8 -*-
"""解析 /tmp/aloha_logs/trials.log，统计每一轮：是否夹到方块 / 是否释放 / 释放位置 / 用时。"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else '/tmp/aloha_logs/trials.log'

LINE = re.compile(
    r't=\s*([\d.]+)s 墙钟\s*([\d.]+)s 夹爪 模型=([\d.]+) → 下发=([\d.]+) 实测=([\d.]+).*?右臂 '
    r'([+-][\d.]+) ([+-][\d.]+) ([+-][\d.]+) ([+-][\d.]+) ([+-][\d.]+) ([+-][\d.]+)')

rounds = []
cur = None
for raw in open(LOG, errors='ignore'):
    for seg in raw.replace('\r', '\n').split('\n'):
        s = seg.strip()
        m = re.match(r'===== ROUND (\d+) START', s)
        if m:
            cur = {'idx': int(m.group(1)), 'lines': [], 'release': False, 'fallback': False,
                   'place': False, 'stop': False, 'err': None}
            rounds.append(cur)
            continue
        if cur is None:
            continue
        if '兜底释放' in s:
            cur['fallback'] = True
        if '到位释放' in s:
            cur['place'] = True
        if '检测到释放' in s:
            cur['release'] = True
        if '已自动停止' in s:
            cur['stop'] = True
        if 'Traceback' in s or 'Error' in s:
            cur['err'] = s[:80]
        mm = LINE.search(s)
        if mm:
            cur['lines'].append({
                't': float(mm.group(1)), 'wall': float(mm.group(2)), 'model': float(mm.group(3)),
                'cmd': float(mm.group(4)), 'meas': float(mm.group(5)),
                'arm': [float(mm.group(i)) for i in range(6, 12)]})

print(f'{"轮":>3} {"结果":>6} {"夹住时刻":>9} {"夹住实测":>9} {"释放时刻":>9} {"释放位置J1/J2/J3":>18} {"方式":>6}')
n_ok = n_grasp = n_release = 0
for r in rounds:
    # 夹住 = 已下发闭合 + 实测夹爪在"夹着方块"的区间（0.018~0.045）。
    # 上限 0.045 是为了排除"刚闭合、爪子还在收拢"的过渡帧（此时实测仍是 0.06 左右）。
    held = [l for l in r['lines'] if l['cmd'] <= 0.03 and 0.018 <= l['meas'] < 0.045]
    grasp_t = held[0]['t'] if held else None
    grasp_v = max((l['meas'] for l in held), default=0.0)
    rel = None
    for l in r['lines']:
        if grasp_t is not None and l['t'] > grasp_t and l['cmd'] > 0.03:
            rel = l
            break
    ok = bool(held) and r['release'] and rel is not None
    n_ok += ok
    n_grasp += bool(held)
    n_release += bool(rel)
    pos = f"{rel['arm'][0]:+.2f} {rel['arm'][1]:+.2f} {rel['arm'][2]:+.2f}" if rel else '-'
    how = '兜底' if r['fallback'] else ('到位' if r['place'] else ('模型' if rel else '-'))
    rel_t = f"{rel['t']:.1f}s" if rel else '-'
    grasp_t_s = f"{grasp_t:.1f}s" if grasp_t is not None else '-'
    grasp_v_s = f"{grasp_v:.4f}" if held else '-'
    print(f"{r['idx']:>3} {'✅' if ok else '❌':>5} {grasp_t_s:>9} {grasp_v_s:>9} "
          f"{rel_t:>9} {pos:>18} {how:>6}")

print(f'\n总计 {len(rounds)} 轮：夹到方块 {n_grasp}，完成释放 {n_release}，完整成功 {n_ok}')
if rounds and rounds[0]['err']:
    print('首轮报错:', rounds[0]['err'])
