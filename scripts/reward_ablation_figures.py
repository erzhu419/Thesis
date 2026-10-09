"""Compare the completed reward treatments using common evaluation metrics."""

import csv
import json
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import build_figures as b


ARMS = ['ridge', 'exp60', 'exp60_hold', 'exp1_hold']
LABELS = ['计划对齐', '指数60秒', '指数60秒＋动作项', '指数1秒＋动作项']
SEEDS = [17001, 17002, 17003]
SCENES = ['normal', 'fast_switch']
COLORS = [b.BLUE, b.GREEN, b.PURPLE, b.RED]


def render():
    root = b.ROOT / 'experiments/legacy_bus/runs/reward_v1'
    rows = []
    for arm, seed in product(ARMS, SEEDS):
        folder = root / f'{arm}_seed{seed}' / 'results'
        completion = json.loads((folder / 'completed.json').read_text())
        assert completion['training_complete'] and completion['evaluation_episodes'] == 6
        for scene, ev in product(SCENES, [15001, 15002, 15003]):
            row = json.loads((folder / f'{scene}_seed{ev}.json').read_text())
            assert (row['method'], row['training_seed'], row['evaluation_seed'], row['scene']) == (arm, seed, ev, scene)
            assert row['horizon_seconds'] == 24000
            count = sum(w['decisions'] for w in row['windows'])
            rows.append(dict(method=arm, training_seed=seed, scene=scene, evaluation_seed=ev,
                             common_return_K=row['reward']/1000,
                             cost_seconds=row['generalized_seconds_per_passenger'],
                             imbalance_seconds=sum(w['imbalance_sum'] for w in row['windows'])/count,
                             hold_seconds=sum(w['hold_sum'] for w in row['windows'])/count,
                             completion_percent=100*row['arrived_passengers']/row['generated_passengers']))
    metrics = ['common_return_K', 'cost_seconds', 'imbalance_seconds', 'hold_seconds', 'completion_percent']
    seeds = []
    for arm, scene, seed in product(ARMS, SCENES, SEEDS):
        group = [r for r in rows if (r['method'], r['scene'], r['training_seed']) == (arm, scene, seed)]
        seeds.append(dict(method=arm, scene=scene, training_seed=seed,
                          **{k:float(np.mean([r[k] for r in group])) for k in metrics}))
    means = []
    for arm, scene in product(ARMS, SCENES):
        group = [r for r in seeds if (r['method'], r['scene']) == (arm, scene)]
        means.append(dict(method=arm, scene=scene,
                          **{k:float(np.mean([r[k] for r in group])) for k in metrics}))
    report = dict(evaluations=len(rows), training_seeds=3, evaluation_seeds_per_policy=3,
                  primary_metric='cost_seconds', common_return='unchanged plan-aligned environment reward',
                  aggregation='average evaluation seeds within each training seed, then average training seeds',
                  training_seed_means=seeds, means=means)
    (b.DATA / 'reward_ablation_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (b.DATA / 'reward_ablation_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig = plt.figure(figsize=(5.75, 6.0))
    gs = fig.add_gridspec(2, 2, left=.16, right=.97, top=.95, bottom=.24,
                         height_ratios=[1.4, 1], hspace=.72, wspace=.5)
    ax = fig.add_subplot(gs[0, :])
    for arm, color in zip(ARMS, COLORS):
        for scene, marker in zip(SCENES, ['o', 's']):
            group = [r for r in seeds if (r['method'], r['scene']) == (arm, scene)]
            mean = next(r for r in means if (r['method'], r['scene']) == (arm, scene))
            ax.scatter([r['cost_seconds']/60 for r in group], [r['imbalance_seconds'] for r in group],
                       color=color, alpha=.32, marker=marker, s=19, linewidths=0)
            ax.scatter(mean['cost_seconds']/60, mean['imbalance_seconds'],
                       color=color, edgecolor='white', linewidth=.7, marker=marker, s=60, zorder=4)
    ax.set(title='(a) 奖励设计改变规律性与乘客时间的权衡',
           xlabel='人均广义时间 / 分钟', ylabel='前后时距不平衡 / 秒',
           xlim=(18, 38), ylim=(35, 180), xticks=[20, 25, 30, 35])
    ax.grid()
    ax.text(.03, .93, '左下方对应两项指标较低', transform=ax.transAxes, va='top', fontsize=8, color=b.GRAY)
    for col, (key, title, xlabel) in enumerate([
            ('common_return_K', '(b) 同一控制目标下的回报', '计划对齐回报 / 千'),
            ('hold_seconds', '(c) 停站动作的变化', '平均附加停站 / 秒')]):
        ax = fig.add_subplot(gs[1, col])
        for i, (arm, color) in enumerate(zip(ARMS, COLORS)):
            for j, (scene, marker) in enumerate(zip(SCENES, ['o', 's'])):
                group = [r[key] for r in seeds if (r['method'], r['scene']) == (arm, scene)]
                y = i+(j-.5)*.28
                ax.plot([min(group), max(group)], [y, y], color=color, lw=.8)
                ax.scatter(group, [y]*3, facecolor='white', edgecolor=color, s=15, marker=marker, zorder=3)
                ax.scatter(np.mean(group), y, color=color, s=25, marker=marker, zorder=4)
        ax.set(yticks=range(4), yticklabels=['计划对齐', '指数60秒', '60秒＋动作项', '1秒＋动作项'] if col == 0 else [],
               ylim=(3.55, -.55), xlabel=xlabel)
        ax.set_title(title, loc='left', fontsize=9)
        ax.xaxis.set_major_locator(b.matplotlib.ticker.MaxNLocator(3))
        ax.grid(axis='x')
    handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in zip(COLORS, LABELS)]
    handles += [Line2D([], [], color=b.GRAY, marker=m, lw=0, label=l)
                for m, l in [('o', '正常运行'), ('s', '快速切换')]]
    fig.legend(handles=handles, ncol=2, loc='lower center', bbox_to_anchor=(.53, .015), fontsize=8)
    b.save(fig, 'fig3_reward_comparison', '计划对齐奖励保持较高控制回报，指数尺度与动作项改变规律性和乘客时间的权衡',
           [str(root)], '72次完整评价；每组3个训练种子，各评价3次；先按训练种子平均；大实心点为均值，小点为训练重复；跨组统一计算计划对齐回报，不比较异尺度训练奖励')
    path = b.OUT / 'figure_sources.json'
    existing = {r['figure']:r for r in json.loads(path.read_text())}
    existing.update({r['figure']:r for r in b.MANIFEST})
    path.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(means, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    render()
