"""Join frozen decompositions with existing development truth; no fitting."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
VARIANTS=('zero_memory','memory_base','level_only','action_only','full')
VARIANT_COLUMNS=['delta_min','delta_mean','min_head_low','min_head_high','choice',
                 'regret300','direction_correct','fixed_low_head_delta','selection_gap']
HEAD_COLUMNS=['zero_delta','memory_base_delta','level','action_delta','full_delta']
STATE_COLUMNS=['scenario','slot','sequence','benefit300','variants','heads',
               'base_drift_min','direct_memory_min','min_nonadditivity','coverage']


def display_numbers(value):
    if isinstance(value,float):return float(format(value,'.7g'))
    if isinstance(value,list):return [display_numbers(v) for v in value]
    if isinstance(value,dict):return {k:display_numbers(v) for k,v in value.items()}
    return value


def read(path):
    return json.loads(Path(path).read_text())


def sig(value):
    return (value>0)-(value<0)


def aggregate(rows):
    result={}
    for i,name in enumerate(VARIANTS):
        values=[r[4][i] for r in rows]
        result[name]=dict(n=len(values),correct=sum(v[6] for v in values),
            high=sum(v[4]==1 for v in values),regret=sum(v[5] for v in values),
            head_switches=sum(v[2]!=v[3] for v in values))
    return result


def analyze(root):
    p=read(HERE/'protocol.json')
    cases={};states=[];checks={}
    for scenario in p['scenarios']:
        parent=read(root/scenario/'summary.json')
        cover=read(root/scenario/'coverage.json')
        chunks=[json.loads(line) for line in (root/scenario/'heads.jsonl').read_text().splitlines()]
        checks[scenario+'_parent']=parent['integrity']['all_pass']
        checks[scenario+'_six_states']=len(chunks)==6 and len({v['slot'] for v in chunks})==6
        selected={v['slot']:v for v in cover['selected']}
        rows=[]
        for chunk in chunks:
            original=read(chunk['source'])
            difference=original['comparison']['high_minus_low']['fixed300']
            if difference['status']!='complete':raise RuntimeError('Existing300s truth is missing')
            benefit=-difference['cost']
            d=chunk['decomposition']
            metrics=[]
            for name in VARIANTS:
                v=d['variants'][name];choice=v['choice']
                regret=max(-benefit,0) if choice else max(benefit,0)
                metrics.append([v['delta_min'],v['delta_mean'],*v['min_heads'],choice,regret,
                    sig(v['delta_min'])==sig(benefit),v['fixed_low_head_delta'],v['selection_gap']])
            # Raw float32 matrices remain resident; compact per-head terms are display-only.
            component=d['components'];variants=d['variants']
            heads=[[float(format(x,'.7g')) for x in values] for values in zip(
                variants['zero_memory']['delta_heads'],variants['memory_base']['delta_heads'],
                [v[0] for v in component['level_heads']],component['action_delta_heads'],
                variants['full']['delta_heads'])]
            supports=selected[chunk['slot']]['supports']
            compact_support={key:value[:10] for key,value in supports.items()}
            row=[scenario,chunk['slot'],chunk['sequence'],benefit,metrics,heads,
                d['between_learner_base_drift']['min_advantage_difference'],
                d['direct_memory_same_weights']['full_minus_base_min'],
                d['direct_memory_same_weights']['min_nonadditivity'],compact_support]
            states.append(row);rows.append(row)
        cases[scenario]=dict(checks=parent['integrity']['checks'],overall=aggregate(rows),
            by_window={window:aggregate([r for r in rows if r[1].endswith('/'+window)])
                       for window in ('early','tail')},
            coverage=dict(metadata=cover['metadata'],all_actions=cover['all_actions'],
                by_line={line:cover['by_line'][line] for line in p['focus_lines']},
                by_line_window={key:value for key,value in cover['by_line_window'].items()
                    if key.split('/')[0] in p['focus_lines']},
                selected_v34_training_windows={key:value[0] for key,value in cover['v34_window_training'].items()
                    if key.split('/')[0] in p['focus_lines']},residual_design=cover['residual_design']))
    return display_numbers(dict(status='complete' if all(checks.values()) else 'integrity_failed',
        integrity=dict(all_pass=all(checks.values()),checks=checks),
        columns=dict(states=STATE_COLUMNS,variants=list(VARIANTS),variant=VARIANT_COLUMNS,
            heads=HEAD_COLUMNS,coverage=cover['columns']['selected_support'][:10],
            actions=cover['columns']['action'],head_indices='zero-based',
            display_precision='7 significant digits; exact scores, matrices and checks stay resident'),
        budget=p['compute_budget'],states=states,cases=cases,
        limitations=['Same-weight masking is a dependency diagnostic, not retraining or policy effect.',
            'Memory base and zero comparator have different learned weights.',
            'Old12 states and overlapping300s truth remain development data; native Q uses a different long-term backup.',
            'Action range/neighborhoods and coordinate context ranges describe coverage, not sufficient joint support; structural redundant memory lowers design rank.']))


def report(s):
    lines=['# v35 冻结 critic 分解','',f"完整性：{s['status']}。所有权重与原更新计数保持，新增仿真、优化和训练标签均为0。",'',
        '|场景/状态|真收益|zero ΔQ|base ΔQ|level-only ΔQ|action-only ΔQ|full ΔQ|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in s['states']:
        lines.append('|'+r[0]+'/'+r[1]+'|'+format(r[3],'.7g')+'|'+
            '|'.join(format(v[0],'.7g') for v in r[4])+'|')
    lines+=['','完整逐头矩阵、两动作min-head与覆盖见服务器heads.jsonl/coverage.json；小汇总包含逐头差分及分线路、时窗的旧事实动作覆盖。',
            '',*s['limitations']]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE/'runs')
    parser.add_argument('--out',type=Path,default=HERE/'decomposition_summary.json')
    args=parser.parse_args()
    try:
        s=analyze(args.root)
        compact=json.dumps(s,separators=(',',':'),allow_nan=False)
        if len(compact.encode())+1>20000:raise RuntimeError('Small summary exceeds20000 bytes')
        with args.out.open('x') as f:f.write(compact+'\n')
        with (args.out.parent/'RESULTS.md').open('x') as f:f.write(report(s))
    except Exception as error:
        with (args.out.parent/'decomposition_summary_failure.json').open('x') as f:
            json.dump(dict(type=type(error).__name__,message=str(error)),f)
        raise
