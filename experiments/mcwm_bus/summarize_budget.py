"""Chinese budget and physical-mismatch figures from completed SUMO evaluations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

from data import HERE
from run_budget import policy_tag

NAMES = {'calibrated': '已标定', 'speed25': '速度偏差25%', 'od25': 'OD强度偏差25%',
         'signals25': '额外路口密度25%', 'dwell25': '停站时间偏差25%',
         'joint25': '联合偏差25%', 'joint50': '联合偏差50%', 'mechanistic': '无经验运行时间标定'}
COLORS = {'h2o': '#447EAC', 'mcwm': '#248878'}
LABELS = {'h2o': 'H2O+', 'mcwm': '世界模型辅助'}


def specifications(p):
    result = []
    for count in p['offline_block_counts']:
        result.append(dict(tag=f'blocks{count}_offline', blocks=count, method='offline',
                           source_events=0, scenario='calibrated'))
        for budget in p['source_event_budgets']:
            for method in COLORS:
                result.append(dict(tag=policy_tag(count, method, budget, 'calibrated'), blocks=count,
                                   method=method, source_events=budget, scenario='calibrated'))
        if count == p['mismatch_block_count']:
            for scene in p['mismatch_scenarios'][1:]:
                for method in COLORS:
                    budget = max(p['source_event_budgets'])
                    result.append(dict(tag=policy_tag(count, method, budget, scene['name']), blocks=count,
                                       method=method, source_events=budget, scenario=scene['name']))
    return result


def save(fig, out, name):
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{name}.{suffix}', dpi=240)
    plt.close(fig)


def main(out):
    p = json.loads((out / 'protocol.json').read_text())
    specs = specifications(p)
    if {spec['tag'] for spec in specs} != set(p['methods']):
        raise ValueError('Summary does not cover the frozen evaluation methods')
    rows = []
    for spec in specs:
        metadata = json.loads((out / f"{spec['tag']}.json").read_text()) if spec['source_events'] else {}
        for demand in p['evaluation_demand_scales']:
            for block in p['evaluation_blocks']:
                result = json.loads((out / 'evaluation' / f"{spec['tag']}_demand{demand:g}_seed{block['sumo_seed']}.json").read_text())
                if result['horizon_seconds'] != p['evaluation_horizon_seconds']:
                    raise ValueError('An evaluation did not reach the full horizon')
                pax = result['passengers']
                rows.append({**spec, 'demand': demand, **block,
                             'actual_source_events': metadata.get('actual_source_events', 0),
                             'cost': result['generalized_seconds_per_departed_passenger'],
                             'completion_percent': 100*pax['completion_rate']})
    with (out / 'paired_budget_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    font = '/mnt/c/Windows/Fonts/msyh.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'font.size': 8, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    get = lambda c, m, e, s, d, metric: float(np.mean([r[metric] for r in rows if
        (r['blocks'], r['method'], r['source_events'], r['scenario'], r['demand']) == (c, m, e, s, d)]))
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.1), layout='constrained')
    for i, demand in enumerate(p['evaluation_demand_scales']):
        for j, count in enumerate(p['offline_block_counts']):
            ax = axes[i, j]
            zero = get(count, 'offline', 0, 'calibrated', demand, 'cost')
            for method in COLORS:
                values = [zero] + [get(count, method, budget, 'calibrated', demand, 'cost')
                                   for budget in p['source_event_budgets']]
                ax.plot(np.array([0]+p['source_event_budgets'])/1000, values, '-o',
                        ms=4, lw=1.3, color=COLORS[method], label=LABELS[method])
            ax.set_title(f'({chr(97+2*i+j)}) {count*24}个离线回合，需求倍率{demand:g}', loc='left', fontsize=9)
            ax.set_xlabel('源仿真决策预算 / 千次')
            ax.set_ylabel('人均广义乘客时间 / 秒')
            ax.set_xticks(np.array([0]+p['source_event_budgets'])/1000, ['0', '4.096', '16.384'])
            ax.grid(alpha=.18)
    axes[0, 0].legend(frameon=False)
    save(fig, out, '公交模型_离线数据与交互预算')

    count, budget = p['mismatch_block_count'], max(p['source_event_budgets'])
    scenes = [scene['name'] for scene in p['mismatch_scenarios']]
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 7.2), layout='constrained')
    contrasts = []
    for j, demand in enumerate(p['evaluation_demand_scales']):
        for method, color in COLORS.items():
            values = np.array([get(count, method, budget, scene, demand, 'cost') for scene in scenes])
            y = np.arange(len(scenes)) + (-.10 if method == 'h2o' else .10)
            axes[0, j].scatter(values, y, color=color, s=22, label=LABELS[method])
            axes[1, j].scatter(100*(values/values[0]-1), y, color=color, s=22)
        for i in range(2):
            ax = axes[i, j]
            ax.set_yticks(range(len(scenes)), [NAMES[s] for s in scenes] if j == 0 else [])
            ax.set_ylim(len(scenes)-.5, -.5)
            ax.set_title(f'({chr(97+2*i+j)}) 需求倍率{demand:g}', loc='left', fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='x', alpha=.2)
        axes[0, j].set_xlabel('人均广义乘客时间 / 秒')
        axes[1, j].set_xlabel('相对本方法已标定参照的时间增加 / %')
        axes[1, j].axvline(0, color='#888888', lw=.6)
        for scene in scenes:
            h2o, mcwm = (get(count, m, budget, scene, demand, 'cost') for m in COLORS)
            comp = (get(count, 'mcwm', budget, scene, demand, 'completion_percent') -
                    get(count, 'h2o', budget, scene, demand, 'completion_percent'))
            contrasts.append(dict(scenario=scene, demand=demand, h2o_cost=h2o, mcwm_cost=mcwm,
                                  mcwm_reduction_percent=100*(1-mcwm/h2o), completion_change_pp=comp))
    axes[0, 0].legend(frameon=False, loc='best')
    save(fig, out, '公交模型_转移机制失配')
    summary = {'completed_evaluations': len(rows), 'training_seed': p['training_seed'],
               'mismatch_contrasts': contrasts, 'rows': rows}
    (out / 'budget_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    text = ['# 公交数据预算与转移机制失配结果', '',
            '| 源仿真条件 | 需求倍率 | H2O+人均时间（秒） | 世界模型人均时间（秒） | 时间下降（%） | 完成率变化（百分点） |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in contrasts:
        text.append(f"| {NAMES[row['scenario']]} | {row['demand']:g} | {row['h2o_cost']:.2f} | {row['mcwm_cost']:.2f} | {row['mcwm_reduction_percent']:.3f} | {row['completion_change_pp']:.3f} |")
    text += ['', '## 解释范围', '',
             '两档数据均从随机参数重新训练10,000步，源模型额外拟合与可信度更新耗时另计。预算图为各档离线数据下的实际学习结果，不与既有50,000步初始化混合。',
             '每个条件采用一个训练种子、每档需求一个共同环境块，无置信区间或显著性判断。信号路口以额外遭遇及延误建模，未修改SUMO路网；OD偏差只改变强度。无经验运行时间标定组仍使用既有路网、OD和上下客参数。',
             '预算图横轴为名义上限，逐场景CSV保留未拆分同时决策批次后的实际交互数。所有评价均结算全部24,000秒服务时域。']
    (out / '预算与失配补充结果.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'completed_evaluations': len(rows), 'mismatch_contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'runs/budget_v2')
    main(parser.parse_args().out)
