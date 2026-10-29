"""Keep exact head contrasts small; never equate endpoint paths with causes."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compact_critic(item):
    full = item['variants']['full']
    return dict(q_heads=full['q_heads'], delta_heads=full['delta_heads'],
        delta_min=full['delta_min'], min_heads=full['min_head'],
        base_delta_heads=item['component_delta_heads']['base'],
        interaction_delta_heads=item['component_delta_heads']['action_interaction'],
        shared_level_heads=item['components']['shared_level'],
        variants_delta_min={name: value['delta_min'] for name, value in item['variants'].items()},
        same_weight_direct_effect=item['same_weight_direct_effect'], numeric=item['numeric'])


def compact_increment(item):
    return dict(min_advantage=item['min_advantage'],
        head_common_level_shift=item['parts']['full']['head_common_level_shift'],
        part_head_contrast_increments={name: value['head_action_contrast_increment']
            for name, value in item['parts'].items()},
        reference_head_parts=item['reference_head_parts'],
        reference_head_component_sum_error=item['reference_head_component_sum_error'])


def compact_case(value):
    rows = []
    for issue in value['issues']:
        rows.append(dict(slot=issue['slot'], issue=issue['issue'], sequence=issue['sequence'],
            benefit300=issue['benefit300'],
            critics={name: {arm: compact_critic(result) for arm, result in arms.items()}
                for name, arms in issue['decomposition'].items()},
            increments={name: compact_increment(result)
                for name, result in issue['trace_minus_one_step'].items()}))
    return dict(status=value['status'], scenario=value['scenario'], integrity=value['integrity'],
        issue_prediction_replay_all_pass=all(item['full_old_prediction_exact'] for item in value['issue_proofs']),
        issues=rows, frozen_credit=value['frozen_credit'],
        actual_training_target_accounting=value['actual_training_target_accounting'],
        budget=value['budget'], wall_seconds=value['wall_seconds'])


def compact_credit(value):
    """Keep signed/absolute components and ranges without repeated field names."""
    atoms=list(value['arms']['trace']['groups']['all']['components'])
    selected_groups=('all','102X','705X','7X')
    def group(item):
        return [item['rows'], item['reward']['mean'], item['bootstrap_minus_q']['mean'],
            item['total']['mean'], item['total']['mean_abs'], item['total']['min'],
            item['total']['max'], item['positive_total'], item['negative_total'],
            [[item['components'][name]['mean'],item['components'][name]['mean_abs']]
                for name in atoms]]
    arms={}
    for arm, item in value['arms'].items():
        arms[arm]={key: data for key,data in item.items() if key != 'groups'}
        arms[arm]['groups']={name:group(item['groups'][name]) for name in selected_groups}
    return dict(probe_lambda=value['probe_lambda'],installed_lambda=value['installed_lambda'],
        atoms=atoms, component_columns=['mean','mean_abs'],
        group_columns=['rows','reward_mean','BQ_mean','total_mean','total_mean_abs',
            'total_min','total_max','positive_rows','negative_rows','components'],
        seeds=value['seeds'],eta=value['eta'],chain_rows=value['chain_rows'],arms=arms,
        trace_minus_single={name:group(value['trace_minus_single'][name]) for name in selected_groups},
        groups_scope='All271 real network rows included in all; only102X/705X/7X are expanded here. All13 line groups stay resident.',
        scope=value['scope'])


NUMERIC_COLUMNS=('native_float32_forward_max_abs_error',
    'float64_component_sum_vs_native_max_abs_error',
    'shared_level_native_action_contrast_max_roundoff','native_float32_roundoff_bound')
VARIANTS=('full','no_level','no_interaction','base')
PARTS=('base','shared_level','action_interaction','full')


def pack_issue(issue, scenario):
    critics={}
    for name, arms in issue['critics'].items():
        critics[name]={}
        for arm, item in arms.items():
            critics[name][arm]=[item['q_heads'],item['delta_heads'],item['base_delta_heads'],
                item['interaction_delta_heads'],item['shared_level_heads'],
                [item['variants_delta_min'][v] for v in VARIANTS],item['min_heads'],
                [item['numeric'][key] for key in NUMERIC_COLUMNS]]
    increments={}
    for name,item in issue['increments'].items():
        minimum=item['min_advantage']
        increments[name]=[[minimum[key] for key in ('one_step','trace','increment',
            'one_step_low_action_reference_head','fixed_reference_head_contrast_increment',
            'min_head_selection_gap_increment','decomposition_error')],
            item['head_common_level_shift'],
            [item['part_head_contrast_increments'][key] for key in PARTS],
            [item['reference_head_parts'][key] for key in PARTS],
            item['reference_head_component_sum_error']]
    return dict(scenario=scenario,slot=issue['slot'],issue=issue['issue'],sequence=issue['sequence'],
        benefit300=issue['benefit300'],critics=critics,increments=increments)


def combine(cases):
    if [case['scenario'] for case in cases] != ['stable', 'local_speed_drop']:
        raise RuntimeError('Both predeclared scenarios are required')
    if any(case['status'] != 'complete' or not case['integrity']['all_pass'] or
           not case['issue_prediction_replay_all_pass'] for case in cases):
        raise RuntimeError('Frozen-model/source integrity did not pass')
    for case in cases:
        if [row['slot'] for row in case['issues']] != ['102X/early', '102X/late', '705X/early', '705X/late']:
            raise RuntimeError('All four existing issue slots must be retained')
    rows = [pack_issue(issue,case['scenario']) for case in cases for issue in case['issues']]
    stats = {}
    for case in cases:
        scene_rows = case['issues']
        stats[case['scenario']] = dict(states=len(scene_rows),
            trace_positive_predictions=sum(row['critics']['critic']['trace']['delta_min'] > 0 for row in scene_rows),
            true_positive_benefits=sum(row['benefit300']['cost'] > 0 for row in scene_rows),
            head_switch_states={arm: sum(len(set(row['critics']['critic'][arm]['min_heads'])) > 1
                for row in scene_rows) for arm in ('one_step','trace')},
            direct_level_effect_max_abs=max(abs(row['critics']['critic'][arm]['variants_delta_min']['full']-
                row['critics']['critic'][arm]['variants_delta_min']['no_level'])
                for row in scene_rows for arm in ('one_step','trace')))
    return dict(status='complete', integrity_all_pass=True, states=rows,
        scene_summary=stats,
        critic_columns=['q_heads','delta_heads','base_delta_heads','interaction_delta_heads',
            'shared_level_heads','variants_delta_min','min_heads','numeric'],
        variant_order=list(VARIANTS),critic_numeric_columns=list(NUMERIC_COLUMNS),
        increment_columns=['min_advantage','head_common_level_shift','part_head_contrast_increments',
            'reference_head_parts','reference_head_component_sum_error'],
        min_advantage_columns=['one_step','trace','increment','reference_head',
            'fixed_reference_head_contrast_increment','min_head_selection_gap_increment','decomposition_error'],
        part_order=list(PARTS),
        frozen_credit={case['scenario']:compact_credit(case['frozen_credit']) for case in cases},
        actual_training_target_accounting={case['scenario']:case['actual_training_target_accounting'] for case in cases},
        preserved_contract=dict(initializer_updates=50000,feedback_slots=285,optimizer_updates=456,
            horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        new_budget=dict(new_training_labels=0,new_optimizer_updates=0,new_native_branches=0,
            new_native_seconds=0,truth_admitted=0,checkpoints_downloaded=0),
        scope=['Exact v55 frozen function attribution and same-weight deletion.',
            'Actual v54 logged training-target accounting, without per-issue optimizer attribution.',
            'Frozen CRN v53 chain target sensitivity; this chain is not the v55 held-out branch chain.'],
        limitations=['Endpoint decompositions do not identify historical reward-vs-bootstrap optimizer causes.',
            'No300s target is fabricated for either v55 action; truth is annotation only.',
            'This diagnostic does not deploy or train an actor.'])


def report(value):
    lines=['# v56 冻结动作差分归因','',
        '两场景、8个旧issue仅重评分。正B表示7.5秒降低300秒乘客成本；Q差是7.5−0。', '',
        '|场景/状态|真B|单步ΔminQ|traceΔminQ|增量：base / 交互 / min-head选择|',
        '|---|---:|---:|---:|---|']
    for row in value['states']:
        online=row['critics']['critic']
        change=row['increments']['critic']
        parts=change[3]
        values=[parts[0],parts[2],change[0][5]]
        lines.append('|'+row['scenario']+'/'+row['slot']+'|'+format(row['benefit300']['cost'],'.7g')+'|'+
            format(online['one_step'][5][0],'.7g')+'|'+format(online['trace'][5][0],'.7g')+'|'+
            ' / '.join(format(x,'.7g') for x in values)+'|')
    lines.extend(['','base/交互增量使用单步0动作的min-head作固定参照，min-head变化另列。共享level代数上无直接动作差；原生float32舍入单列。',
        '训练target日志和冻结v53链敏感性是两组独立证据，不是8个留出状态的历史因果份额。'])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    cases=[json.loads((HERE/'runs'/scene/'summary.json').read_text()) for scene in ('stable','local_speed_drop')]
    value=combine(cases)
    raw=json.dumps(value,separators=(',', ':'),allow_nan=False)
    bound=json.loads((HERE/'protocol.json').read_text())['summary_max_bytes']
    if len(raw.encode())+1 > bound:
        raise RuntimeError('Reduce resident summary before collection: exceeds byte budget')
    out=HERE/'analysis'
    out.mkdir(exist_ok=False)
    with (out/'summary.json').open('x') as handle:
        handle.write(raw+'\n')
    with (out/'RESULTS.md').open('x') as handle:
        handle.write(report(value))
    print(json.dumps(dict(status='complete', states=len(value['states']),summary_bytes=len(raw.encode())+1)))
