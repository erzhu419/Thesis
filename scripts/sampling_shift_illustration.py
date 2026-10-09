"""Fixed-seed illustration of finite sampling and empirical resampling, not bus data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/figures'
FONT = '/mnt/c/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.axisbelow': True, 'legend.frameon': False, 'mathtext.fontset': 'stix',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
SEED, ALPHA, M, Q = 20261002, 1.5, 10000, .01
rng = np.random.default_rng(SEED)
threshold = Q ** (-1 / ALPHA)
true_cdf = lambda x: 1 - np.asarray(x) ** (-ALPHA)
def ks_distance(sample):
    values = np.sort(sample)
    f = true_cdf(values)
    n = len(values)
    return float(max(np.max(np.arange(1, n + 1) / n - f),
                     np.max(f - np.arange(n) / n)))
def ecdf(sample, grid):
    return np.searchsorted(np.sort(sample), grid, side='right') / len(sample)
records, data = [], {}
for n in (50, 500):
    for group in ('A', 'B'):
        original = 1 + rng.pareto(ALPHA, n)
        repeated = rng.choice(original, M, replace=True)
        data[n, group] = original, repeated
        records.append({'n': n, 'group': group, 'resample_n': M,
            'original_ks': ks_distance(original), 'resampled_ks': ks_distance(repeated),
            'original_tail_count': int((original > threshold).sum()),
            'original_tail_probability': float((original > threshold).mean()),
            'resampled_tail_probability': float((repeated > threshold).mean()),
            'original_max': float(original.max()), 'resampled_max': float(repeated.max())})
fig = plt.figure(figsize=(6.5, 4.25))
gs = fig.add_gridspec(2, 2, height_ratios=(.95, 2.5), hspace=.37, wspace=.25)
flow = fig.add_subplot(gs[0, :]); flow.set_axis_off(); flow.set_xlim(0, 1); flow.set_ylim(0, 1)
labels = ['真实运行规律\nPareto(尺度1，形状1.5)', '有限观测\nn=50或500，各两组',
          '估计环境\n各组样本的经验分布', '训练样本\n各再采10000次']
for i, label in enumerate(labels):
    x = .013 + i * .248
    flow.add_patch(FancyBboxPatch((x, .15), .222, .62, boxstyle='round,pad=.008',
        facecolor='#F0F4F7', edgecolor='#8A9BA8', linewidth=.8))
    flow.text(x + .111, .46, label, ha='center', va='center', fontsize=7.8, linespacing=1.5)
    if i < 3:
        flow.annotate('', xy=(x + .240, .46), xytext=(x + .227, .46),
                      arrowprops={'arrowstyle': '->', 'color': '#626F79', 'lw': 1})
flow.set_title('(a) 同一分布下的有限采样与二次采样', loc='left', pad=0)
xgrid = np.geomspace(1, 200, 1600)
for col, n in enumerate((50, 500)):
    ax = fig.add_subplot(gs[1, col])
    ax.axhline(0, color='#787F87', lw=.8)
    for group, color in zip(('A', 'B'), ('#286A98', '#C65A46')):
        original, repeated = data[n, group]
        ax.plot(xgrid, ecdf(original, xgrid) - true_cdf(xgrid), color=color,
                lw=1.5, label=f'样本{group}（n={n}）')
        ax.plot(xgrid, ecdf(repeated, xgrid) - true_cdf(xgrid), color=color,
                lw=1.15, ls='--', label=f'{group}再采样（m=10000）')
    ax.set_xscale('log'); ax.set_xlim(1, 200); ax.set_ylim(-.18, .18)
    ax.set_xlabel('运行量 x（对数坐标）'); ax.set_title(f'({"bc"[col]}) 首次样本量 n={n}', loc='left')
    ax.grid(True, color='#DCE1E5', linewidth=.45); ax.legend(fontsize=7, loc='lower right')
    if col == 0:
        ax.set_ylabel('经验CDF − 真实CDF')
fig.subplots_adjust(left=.11, right=.985, bottom=.13, top=.955)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'fig5_sampling_chain.{ext}', dpi=360)
summary = {'seed': SEED, 'distribution': {'name': 'Pareto', 'scale': 1, 'shape': ALPHA},
    'true_tail_probability': Q, 'tail_threshold': threshold, 'records': records,
    'figure_inches': [6.5, 4.25], 'figure_png_pixels': [2340, 1530],
    'interpretation': 'Same generating distribution; finite estimation error and resampling noise only.'}
(OUT / 'source_data' / 'sampling_shift_illustration.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
