#!/usr/bin/env python3
# -- coding: UTF-8 --
"""生成 1:1 的 10cm×10cm 方格纸（A4），用于采随机位置数据时摆放方块。

打印要求：A4、纵向、"实际大小 / 100%"（不要"适应页面"）。
打印后用尺子量一下底部的校验尺，正好 10 cm 才算对。

重新生成：
    python3 ~/cobot_magic/docs/make_cube_grid.py
输出：
    ~/cobot_magic/docs/cube_grid_10cm.pdf
    ~/cobot_magic/docs/cube_grid_10cm.png（预览用）
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 中文字体（本机有 Noto Sans CJK；没有会显示成方块）
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC',
                                          'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
A4_W, A4_H = 21.0, 29.7           # cm
SIDE = 10.0                       # 方格边长 cm
N = 5                             # 每边格数（5×5 = 25 个格点，间距 2cm）
CELL = SIDE / N

fig = plt.figure(figsize=(A4_W / 2.54, A4_H / 2.54))   # 英寸
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, A4_W)
ax.set_ylim(0, A4_H)
ax.axis('off')

# ---------------------------------------------------------------- 标题
ax.text(A4_W / 2, 28.2, '方块随机位置采集 · 10 cm 方格（1:1）',
        ha='center', va='center', fontsize=15, fontweight='bold')

# ---------------------------------------------------------------- 方格
x0 = (A4_W - SIDE) / 2
y0 = 14.0

ax.add_patch(Rectangle((x0, y0), SIDE, SIDE, fill=False, lw=1.6, ec='black'))
for k in range(1, N):
    ax.plot([x0 + k * CELL, x0 + k * CELL], [y0, y0 + SIDE], lw=0.5, color='0.55')
    ax.plot([x0, x0 + SIDE], [y0 + k * CELL, y0 + k * CELL], lw=0.5, color='0.55')

rows = ['A', 'B', 'C', 'D', 'E']
for r in range(N):
    for c in range(N):
        cx = x0 + (c + 0.5) * CELL
        cy = y0 + (N - 1 - r + 0.5) * CELL
        center = (r == N // 2 and c == N // 2)
        if center:
            ax.plot(cx, cy, 'o', ms=11, mfc='#d62728', mec='#d62728', zorder=5)
        else:
            ax.plot(cx, cy, 'o', ms=5.5, mfc='white', mec='black', mew=1.1, zorder=5)

# 上边列号 / 左边行号
for c in range(N):
    ax.text(x0 + (c + 0.5) * CELL, y0 + SIDE + 0.25, str(c + 1),
            ha='center', va='bottom', fontsize=9)
for r in range(N):
    ax.text(x0 - 0.35, y0 + (N - 1 - r + 0.5) * CELL, rows[r],
            ha='right', va='center', fontsize=9)

# 尺寸标注
ax.annotate('', xy=(x0, y0 + SIDE + 0.95), xytext=(x0 + SIDE, y0 + SIDE + 0.95),
            arrowprops=dict(arrowstyle='<->', lw=0.8))
ax.text(x0 + SIDE / 2, y0 + SIDE + 1.05, '10.0 cm', ha='center', va='bottom', fontsize=9)
ax.annotate('', xy=(x0 - 1.3, y0), xytext=(x0 - 1.3, y0 + SIDE),
            arrowprops=dict(arrowstyle='<->', lw=0.8))
ax.text(x0 - 1.85, y0 + SIDE / 2, '10.0 cm', ha='center', va='center',
        rotation=90, fontsize=9)

ax.text(x0 + SIDE / 2, y0 - 0.55, 'C3 = 原固定抓取点（方格中心）',
        ha='center', va='top', fontsize=9.5, color='#d62728')

# ---------------------------------------------------------------- 使用说明
ax.text(2.0, 12.0, '使用说明', fontsize=10, fontweight='bold')
lines = [
    '打印：A4 纵向，缩放选「实际大小 / 100%」，不要「适应页面」',
    '校验：下方标尺用尺子量，必须是 10.0 cm',
    '固定：方格中心 C3 对准原来的固定抓取点，四角用胶带贴牢',
    '摆放：方块始终摆正（面与网格线平行），只改变位置，不改变朝向',
    '数量：25 个格点每个至少 1 条，共 40 条；打乱顺序采，别按 A1→E5',
    '记录：采之前先写下 40 个点的顺序，采完在表上勾掉',
]
for i, s in enumerate(lines):
    ax.text(2.0, 11.1 - i * 0.7, '· ' + s, fontsize=8.5, va='center')

# ---------------------------------------------------------------- 校验尺
rx, ry = 2.0, 5.6
ax.plot([rx, rx + 10.0], [ry, ry], lw=2.4, color='black', solid_capstyle='butt')
for k in range(11):
    ax.plot([rx + k, rx + k], [ry, ry + (0.9 if k % 5 == 0 else 0.5)], lw=1.0, color='black')
ax.text(rx + 5.0, ry - 0.55, '校验尺：两端之间应为 10.0 cm', ha='center', va='top', fontsize=8.5)

# ---------------------------------------------------------------- 顺序示例
ax.text(2.0, 3.6, '采集顺序示例（自己打乱后填）：', fontsize=8.5, fontweight='bold')
ax.text(2.0, 3.0, 'B2  D4  A3  C5  E1  C2  A5  D1  B4  E3  …（40 个，其中 15 个格点重复）',
        fontsize=8, color='0.35')

pdf = os.path.join(HERE, 'cube_grid_10cm.pdf')
png = os.path.join(HERE, 'cube_grid_10cm.png')
fig.savefig(pdf)
fig.savefig(png, dpi=110)
print('已生成:', pdf)
print('预览图:', png)
