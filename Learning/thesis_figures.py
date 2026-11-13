'''
論文の図（Docs/thesis/fig/）。値は各タスクの test の判定ファイルから写した（出典はコメント）。
- fig_gaps.png：3 つのタスクの「腕全体 − 手先」の差（位置誤差・衝突 AUPRC）と H5 balanced、数値・画像、95% CI
- fig_h5_bias.png：H5 の枝ごとの正解率（偏り）と、2 つ目のタスクのずれの補正の前後
使い方: python Learning/thesis_figures.py
'''
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'Docs/thesis/fig'; OUT.mkdir(parents=True, exist_ok=True)
NUM, IMG = '#2a78d6', '#eb6834'   # dataviz の参照パレット（categorical slot 1・2）
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
TASKS = ['Task 1\npush + cylinder', 'Task 2\nshelf placement', 'Task 3\nforearm push']

# (点推定, 下限, 上限)。数値, 画像
# 位置誤差の差 = 手先 − 腕全体（mm、正 = 腕全体のほうが良い）
#   Task 1: results_main_v1v2_ckpt/test_judgement.md の H1 pos_err 低下（Task A）
#   Task 2: results_t2/test_judgement.md の H1 pos_err 低下（Task A 行）
#   Task 3: results_t3/t3_h1_test.md（Task B 行、同じ episode）
POS = {'num': [(-3.775, -4.059, -3.482), (-1.739, -4.668, 1.274), (10.5, 3.3, 18.6)],
       'img': [(-4.090, -4.458, -3.720), (-2.738, -5.589, 0.336), (7.9, 1.6, 15.7)]}
# 衝突（接触）AUPRC の差 = 腕全体 − 手先（各 test_judgement.md の H2）
COLL = {'num': [(0.028, 0.009, 0.047), (0.034, 0.005, 0.071), (0.446, 0.399, 0.497)],
        'img': [(0.050, 0.025, 0.074), (-0.009, -0.023, 0.006), (0.426, 0.333, 0.506)]}
# H5 balanced（腕全体モデル）。手先は構造上 0.5
H5 = {'num': [(0.842, 0.785, 0.893), (0.900, 0.750, 0.986), (1.0, 1.0, 1.0)],
      'img': [(0.719, 0.638, 0.799), (0.623, 0.541, 0.707), (1.0, 1.0, 1.0)]}


def panel(ax, data, title, ylabel, ref):
    x = np.arange(3); w = 0.32
    for k, (f, col, lab) in enumerate([('num', NUM, 'Numeric'), ('img', IMG, 'Image')]):
        v = np.array(data[f]); xs = x + (k - 0.5) * w
        ax.errorbar(xs, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='o', color=col, ms=6, lw=2, capsize=0, label=lab, zorder=3)
    ax.axhline(ref, color=INK2, lw=1, ls='--', zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(TASKS); ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_ylabel(ylabel); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
panel(axs[0], POS, 'a  Final object position', 'EEF − Full error (mm)\n> 0: full-body better', 0)
panel(axs[1], COLL, 'b  Collision / contact', 'Full − EEF AUPRC\n> 0: full-body better', 0)
panel(axs[2], H5, 'c  Matched pairs (H5)', 'Full-body balanced accuracy\n(EEF = 0.5 by construction)', 0.5)
axs[2].set_ylim(0.45, 1.05)
axs[0].legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(OUT / 'fig_gaps.png', dpi=200); plt.close(fig)

# H5 の枝ごと（test）。Task 1: progress_en §4 / test_judgement、Task 2: results_t2/test_judgement.md、Task 3 は両方 1.0
BR = {'Task 1': {'num': (0.728, 0.956), 'img': (0.500, 0.938)},   # wrist_down / wrist_up 側が当たった組
      'Task 2': {'num': (0.800, 1.000), 'img': (0.262, 0.985)}}   # 肩 A / B 側が止まった組
fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
ax = axs[0]; x = np.arange(4); w = 0.38
lab = ['T1 num', 'T1 img', 'T2 num', 'T2 img']
s1 = [BR['Task 1']['num'][0], BR['Task 1']['img'][0], BR['Task 2']['num'][0], BR['Task 2']['img'][0]]
s2 = [BR['Task 1']['num'][1], BR['Task 1']['img'][1], BR['Task 2']['num'][1], BR['Task 2']['img'][1]]
ax.bar(x - w / 2, s1, w - 0.02, color='#1baf7a', label='branch 0 side collides (T1 wrist-down, T2 shoulder A)')
ax.bar(x + w / 2, s2, w - 0.02, color='#6250d6', label='branch 1 side collides (T1 wrist-up, T2 shoulder B)')
ax.axhline(0.5, color=INK2, lw=1, ls='--'); ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 1.08)
ax.set_title('a  H5 accuracy by branch (test)', loc='left', fontsize=10); ax.set_ylabel('Accuracy'); ax.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.08))
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax = axs[1]
# Task 2 のずれの補正（results_t2/h5_bias.md、探索的）
ax.bar([0, 1], [0.900, 0.623], 0.36, color=[NUM, IMG], alpha=0.45, label='threshold 0')
ax.bar([0.38, 1.38], [0.938, 0.792], 0.36, color=[NUM, IMG], label='threshold re-centred on val')
ax.axhline(0.5, color=INK2, lw=1, ls='--'); ax.set_xticks([0.19, 1.19]); ax.set_xticklabels(['Full-Num', 'Full-Img']); ax.set_ylim(0, 1.08)
ax.set_title('b  Task 2: bias correction (exploratory)', loc='left', fontsize=10); ax.set_ylabel('Balanced accuracy (test)')
ax.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.08)); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / 'fig_h5_bias.png', dpi=200); plt.close(fig)
print('written', OUT)
