"""Small support matrix and physically aligned soft-Q/cost bridges."""
from __future__ import annotations

import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
COMPONENTS=('passenger_cost300','actual_endpoint_excess_cost','reward_scale_rounding',
    'entropy','terminal_nonentropy_bootstrap','mixed_continuation_gap','minus_TD_residual_sum','native_assembly_rounding')


def compact_case(value):
    current=[]
    for row in value['current']:
        current.append(dict(slot=row['slot'],issue=row['issue'],holds=row['holds'],benefit300=row['benefit300'],
            scaled_benefit300=row['scaled_benefit300'],root_Q_contrast=row['root_Q_contrast'],
            unidentified_non300_remainder=row['unidentified_non300_remainder'],
            paired=row['same_root_seed_continuation'],full_event_chain_exists=row['full_event_chain_exists'],
            first_TD={arm:dict(status=a['status'],raw_physical_transition_available=a.get('raw_physical_transition_available'),
                missing_fields=a.get('missing_fields')) for arm,a in row['first_transition_support'].items()},
            future_reward_entropy_tail_split_supported=False))
    old=[]
    for row in value['historical']:
        support=row['support']
        models={}
        for model,result in row['models'].items():
            pair=result['pair'];bridge=pair['fixed300_Q_bridge']
            models[model]=[bridge['high_minus_low_root_Q'],
                *[bridge['high_minus_low_components'][name] for name in COMPONENTS],
                bridge['identity_error'],pair['target_lag_contrast'],pair['mixed_vs_recorded_target_Q_contrast']]
        old.append(dict(slot=row['slot'],issue=row['issue'],benefit300=row['benefit300'],holds=row['holds'],
            paired=support['paired_root_supported'],reaches300=support['both_chains_reach_300'],
            spans=[support[arm]['span_seconds'] for arm in ('low','high')],
            exact300_states=[support[arm]['actual_300_state'] for arm in ('low','high')],
            events=[support[arm]['rows'] for arm in ('low','high')],
            zero_dt=[support[arm]['zero_duration_rows'] for arm in ('low','high')],models=models))
    short=[dict(line=row['line'],paired=row['support']['paired_root_supported'],
        spans=[row['support'][arm].get('span_seconds') for arm in ('low','high')],
        events=[row['support'][arm]['rows'] for arm in ('low','high')],
        reaches300=row['support']['both_chains_reach_300']) for row in value['short_pairs']]
    inv={version:[dict(role=item['role'],status=item['status'],count=item['count'],bytes=item['bytes'])
        for item in cases[value['scenario']]['artifacts']] for version,cases in value['inventory'].items()}
    return dict(status=value['status'],scenario=value['scenario'],integrity=value['integrity'],
        contract=value['contract'],current=current,historical=old,short_pairs=short,
        factual_chain=value['factual_chain'],inventory=inv,budget=value['budget'],wall=value['wall'])


def combine(cases):
    if [c['scenario'] for c in cases]!=['stable','local_speed_drop'] or any(
            c['status']!='complete' or not c['integrity']['all_pass'] for c in cases):
        raise RuntimeError('Both source/integrity audits must complete')
    return dict(status='complete',integrity_all_pass=True,
        contract=cases[0]['contract'],model_bridge_columns=['root_Q_high_minus_low',*COMPONENTS,
            'identity_error','target_lag_contrast','mixed_vs_recorded_target_Q_contrast'],
        scenarios={c['scenario']:{key:value for key,value in c.items() if key not in ('contract','scenario')} for c in cases},
        cohorts=dict(current='v54 frozen M8 predictions on v55mu14 0/7.5s; no full successor chains.',
            historical='v37r1 cached M1 mixed-Q on v38 original50000 controller 0/1.875s; legal separate cohort.',
            short='v43mu24 paired45-event chains; no300s support.',factual='v53mu14 single271-event chain; no same-state action alternative.'),
        decision=dict(latest_same_state_reward_entropy_tail_identified=False,
            advance_to_training=False,advance_to_native=False,
            interpretation='Reward units can be aligned; long softQ and finite300 cost are distinct targets. Preserve unsupported contributions as unknown.'),
        new_budget=dict(new_optimizer_updates=0,new_native_branches=0,new_training_labels=0,
            truth_admitted=0,checkpoints_loaded=0,checkpoints_downloaded=0),
        preserved_contract=dict(initializer_updates=50000,feedback_slots=285,optimizer_updates=456,
            horizon_seconds=24000,F_unchanged=True,trust_unchanged=True))


def report(value):
    lines=['# v58：评价口径与现有链支持','',
        '原生奖励为−(2ΔW+ΔR)，训练缩放1e−4、gamma1。300秒真实乘客成本与长期soft混合Q需分开评价。','',
        '|场景/现有v55状态|300秒真实收益|缩放收益|trace ΔQ|未识别的非300项合计|',
        '|---|---:|---:|---:|---:|']
    for scene,data in value['scenarios'].items():
        for row in data['current']:
            numbers=[row['benefit300'],row['scaled_benefit300'],row['root_Q_contrast']['trace'],row['unidentified_non300_remainder']['trace']]
            lines.append('|'+scene+'/'+row['slot']+'|'+'|'.join(format(x,'.7g') for x in numbers)+'|')
    lines.extend(['','v55无完整后继链，未识别合计不能进一步归因给奖励、熵或尾项。下表只使用独立的旧v38/v37r1 shared_level缓存。','',
        '|旧队列/状态|真收益|ΔQ|成本项|熵|末端非熵B|mixed gap|−ΣTD|实际端点补差|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for scene,data in value['scenarios'].items():
        for row in data['historical']:
            a=row['models']['shared_level']
            selected=[row['benefit300'],a[0],a[1],a[4],a[5],a[6],a[7],a[2]]
            lines.append('|'+scene+'/'+row['slot']+'|'+'|'.join(format(x,'.7g') for x in selected)+'|')
    lines.extend(['','ΔQ=1e−4收益300−1e−4Δ(J_T−J300)+Δ熵+Δ末端非熵bootstrap+Δmixed gap−ΔΣTD+舍入。T为真实事件端点，未生成s300。',
        '旧v38数据不可作为v55后继；分解是固定缓存的恒等式，不是历史训练因果归因，也不是长期soft Q真值认证。'])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    value=combine([json.loads((HERE/'runs'/scene/'summary.json').read_text()) for scene in ('stable','local_speed_drop')])
    raw=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode())>json.loads((HERE/'protocol.json').read_text())['summary_max_bytes']:
        raise RuntimeError('Resident output must be reduced before collection')
    out=HERE/'analysis';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(raw);(out/'RESULTS.md').write_text(report(value))
    print(json.dumps(dict(status='complete',summary_bytes=len(raw.encode()))))
