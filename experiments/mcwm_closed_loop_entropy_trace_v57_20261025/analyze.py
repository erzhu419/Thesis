"""Bounded summary of level, exposure and factual-state target differences."""
from __future__ import annotations

import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
STAT=('n','mean','std','min','max')


def stats(value):
    return [value[key] for key in STAT]


def compact_group(value):
    fit=value['exposure_fit']
    pair=value['adjacent_same_line_factual_roots']
    return [value['rows'],stats(value['shift_across_factual_roots']),value['block_sampling_sd_rms'],
        value['shift_after_global_centering_rms'],value['shift_after_line_centering_rms'],
        stats(value['trace_minus_root']['beta0']),stats(value['trace_minus_root']['beta1']),
        stats(value['future_mass']),[fit[key] for key in ('intercept','mass_slope','residual_std','variance_explained')],
        [pair['pairs'],stats(pair['beta0_difference']),stats(pair['beta1_difference']),
            stats(pair['difference_change']),pair['sign_changes'],pair['beta0_zero_differences'],pair['beta1_zero_differences']],
        [stats(value['targets_across_factual_roots'][key]) for key in ('native','beta0','beta1')]]


def compact_arm(value):
    metrics=value['metrics']
    def small_group(group):
        return [group['rows'],stats(group['shift']),stats(group['remaining_real_span_seconds']),stats(group['future_mass'])]
    return dict(lambda_zero_gate=value['lambda_zero_gate'],beta0_exact=value['beta0_original_recurrence_exact'],
        rounding_max_abs=value['rounding64_max_abs'],terminal_dependency=value['terminal_dependency'],
        installed_training_lambda=value['installed_training_lambda'],probe_lambda=1.,
        supported_roots=metrics['supported_roots'],excluded_no_successor=metrics['excluded_no_successor'],
        coverage=[[line,item['chain_rows'],item['supported_roots']] for line,item in metrics['coverage'].items()],
        groups={name:compact_group(group) for name,group in metrics['groups'].items()},
        duration_groups={name:small_group(group) for name,group in metrics['duration_groups'].items()},
        span_groups={name:small_group(group) for name,group in metrics['remaining_span_groups'].items()},
        mass_groups={name:small_group(group) for name,group in metrics['future_mass_groups'].items()})


def combine(cases):
    if [c['scenario'] for c in cases]!=['stable','local_speed_drop']:
        raise RuntimeError('Both original scenarios are required')
    if any(c['status']!='complete' or not c['integrity']['all_pass'] for c in cases):
        raise RuntimeError('Frozen fidelity gate failed')
    return dict(status='complete',integrity_all_pass=True,
        scenarios={case['scenario']:dict(integrity=case['integrity'],
            arms={arm:compact_arm(value) for arm,value in case['arms'].items()},
            cross_arm_beta_difference_in_differences=case['cross_arm_beta_difference_in_differences'],wall=case['wall']) for case in cases},
        stat_columns=list(STAT),group_columns=['rows','shift_stats','block_sampling_sd_rms',
            'global_centered_rms','line_centered_rms','beta0_minus_root','beta1_minus_root','future_mass',
            'exposure_fit','adjacent_same_line_factual_roots','native_beta0_beta1_targets'],
        fit_columns=['intercept','mass_slope','residual_std','variance_explained'],
        adjacent_columns=['pairs','beta0_difference','beta1_difference','difference_change','sign_changes','beta0_zeros','beta1_zeros'],
        small_group_columns=['rows','shift_stats','remaining_real_span_seconds','future_mass'],
        coverage_columns=['line','chain_rows','supported_roots'],
        preserved_contract=dict(initializer_updates=50000,feedback_slots=285,optimizer_updates=456,
            horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        new_budget=dict(new_optimizer_updates=0,new_native_branches=0,new_training_labels=0,
            truth_admitted=0,checkpoints_downloaded=0),
        decision=dict(advance_to_training=False,advance_to_control=False,
            scope='Frozen objective sensitivity only. Negative means and factual-root variation do not establish action-response improvement.'),
        limitations=['Different factual roots are not same-state counterfactual action pairs.',
            'Known entropy subtraction introduces no new outcomes; remaining-span/exposure effects are reported separately.',
            'Beta1 changes the lambda>0 soft fixed point; no unbiased or pure passenger-cost-Q claim.'])


def report(value):
    lines=['# v57：未来trace去熵冻结预检','',
        'λ=0根M8与β=0原递推保真检查通过。表中单位为训练目标值，不是乘客成本。','',
        '|场景/模型/线路|根数|β0平均校正|β1平均校正|移位标准差|质量拟合R²|拟合后残差std|',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'缺失' if x is None else format(x,'.7g')
    for scene,item in value['scenarios'].items():
        for arm,model in item['arms'].items():
            for line,g in model['groups'].items():
                cells=[g[0],g[5][1],g[6][1],g[1][2],g[8][3],g[8][2]]
                lines.append('|'+scene+'/'+arm+'/'+line+'|'+'|'.join(fmt(x) for x in cells)+'|')
    lines.extend(['','相邻同线事实根的目标差变化不等于动作收益改善。最后一行无真实后继，单列且不计入270根变化统计。',
        '均值翻负、非恒定移位或事实根差分翻号均不作为训练通过条件。本轮不训练、不新增原生分支。'])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    cases=[json.loads((HERE/'runs'/scene/'summary.json').read_text()) for scene in ('stable','local_speed_drop')]
    result=combine(cases)
    raw=json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode())>json.loads((HERE/'protocol.json').read_text())['summary_max_bytes']:
        raise RuntimeError('Reduce resident summary before collection; byte limit exceeded')
    out=HERE/'analysis'
    out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(raw)
    (out/'RESULTS.md').write_text(report(result))
    print(json.dumps(dict(status='complete',summary_bytes=len(raw.encode()))))
