"""Separate an available log diagnostic from an unavailable post24 snapshot."""
import json
import math
from pathlib import Path

COLUMNS=['slot','arm','links','baseline_L_old','baseline_L_geom','baseline_overlap_mean',
         'oracle_L_old','oracle_L_geom','oracle_overlap_mean']
AVERAGE_COLUMNS=['L_old','L_geom','L_arith','overlap_mean','atom_overlap_mean',
                 'continuous_overlap_mean','full_old_log_weight','full_fixed_state_geom_log_weight']


def compact(value):
    if isinstance(value,float):return float(f'{value:.10g}')
    if isinstance(value,list):return [compact(v) for v in value]
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
    return value


def brief_journal(j):
    def row(r):
        return [r['packet'],r['processed_at'],r['policy_updates_before'],r['p_hold']]
    return dict(packets=j['packets'],updates=j['updates'],
        milestone_columns=['packet','processed_at','actor_updates_before','sampled_batch_p_hold'],
        milestones=[row(r) for r in j['milestones']],
        sampled_first_threshold={key:None if r is None else row(r)
                                  for key,r in j['sampled_minibatch_first_threshold'].items()},
        batch_window_columns=['window','packets','actor_updates_before_range','clock_range','p_mean','p_min','p_max'],
        batch_windows=[[key,r['packets'],r['policy_update_before_range'],r['processed_at_range'],
            r['p_hold_mean'],r['p_hold_min'],r['p_hold_max']] for key,r in j['sampled_minibatch_windows'].items()],
        same_state_crossings=dict(counts=j['fixed_state_gate_threshold']['counts'],
            first=j['fixed_state_gate_threshold']['first_crossing'],
            first24_count=len(j['fixed_state_gate_threshold']['first24_crossings']),
            first24_intervals=sorted({tuple(r['policy_update_interval'])
                for r in j['fixed_state_gate_threshold']['first24_crossings']}),
            by_line=j['fixed_state_gate_threshold']['by_line'],
            crossing_packets=j['fixed_state_gate_threshold']['all_crossing_packets']),
        per_state_probabilities_saved=False,intermediate_weights_saved=False)


def averages(rows,variant,scope):
    keys=('L_old','L_geom','L_arith','overlap_mean','atom_overlap_mean','continuous_overlap_mean',
          'full_old_log_weight','full_fixed_state_geom_log_weight')
    return {key:math.fsum(row['comparisons'][variant][scope][key] for row in rows)/len(rows)
            for key in keys}


def analyze(runs,out,inventory,exits,sync):
    cases={};clocks=[]
    for scenario in ('stable','local_speed_drop'):
        root=runs/scenario
        if not (root/'summary.json').exists():
            failure=json.loads((root/'failure.json').read_text()) if (root/'failure.json').exists() else None
            cases[scenario]=dict(status='failed',failure=failure);continue
        source=json.loads((root/'summary.json').read_text());clocks.append(source['wall']);models={}
        for model,rows in source['models'].items():
            table=[]
            for row in rows:
                a=row['comparisons']['baseline']['first_reward']
                b=row['comparisons']['oracle_gate_alignment']['first_reward']
                table.append([row['slot'],row['arm'],a['links'],a['L_old'],a['L_geom'],a['overlap_mean'],
                              b['L_old'],b['L_geom'],b['overlap_mean']])
            models[model]=dict(journal=brief_journal(source['journals'][model]),
                first_reward_rows=table,
                first_reward_average={variant:list(averages(rows,variant,'first_reward').values())
                    for variant in ('baseline','oracle_gate_alignment')},
                full44_average={variant:{key:val for key,val in averages(rows,variant,'full44').items()
                    if key in ('L_old','L_geom','overlap_mean')}
                    for variant in ('baseline','oracle_gate_alignment')},
                actual_post24_comparison='unavailable')
        cases[scenario]=dict(status=source['status'],integrity=source['integrity'],counts=source['counts'],
            errors=source['errors'],models=models,
            wall_seconds=source['wall']['finished']-source['wall']['started'])
    complete=len(clocks)==2 and all(code==0 for code in exits.values()) and all(
        c.get('integrity',{}).get('all_pass') for c in cases.values())
    overlap=min(c['finished'] for c in clocks)-max(c['started'] for c in clocks) if len(clocks)==2 else 0.
    value=dict(status='complete' if complete else 'partial',priority='high',
        integrity=dict(all_pass=complete),exitcodes=exits,cases=cases,
        snapshot_inventory=dict(initializer=inventory['initializer'],
            columns=['version','saved_files','saved50024_candidates'],
            versions=[[name,len(record['files']),record['saved50024_name_candidates']]
                for name,record in inventory['versions'].items()],
            resident_details='runs/snapshot_inventory.json',actual_post24_snapshot_available=False),
        actual_scenario_overlap_seconds=max(0.,overlap),schema=dict(columns=COLUMNS,
            first_reward_average_columns=AVERAGE_COLUMNS,
            metric_timing='first step in packet, before its actor optimizer step; varying minibatch',
            fixed_state_evidence='same-packet before/after proposals only certify crossing p=0.5',
            oracle='future target gate copied into hypothetical behavior, not a saved or issue-visible policy',
            precision='10 significant digits in small summary; resident records retain full precision'),
        sync_plan_reference='sync_plan.json',
        sync_contract=dict(initializer=sync['initializer_updates'],horizon=sync['target_horizon_seconds'],
            total_slots=285,ordinary_slots=15,response_slots=270,warmup_updates=24,response_updates=432,
            issue_window=[9000,9300],release_clock=9630,behavior_snapshot_after24=True,
            F_unchanged=True,trust_unchanged=True,mixed_backup_unchanged=True,
            current_actor_ratio_recomputed_each_update=True),decision=dict(
            proposed_causal_sync_defined=True,actual_post24_snapshot_available=False,
            actual_post24_to_post456_credit_pass=None,collection_recommended=False,
            collection_launched=False,training_launched=False,
            next_gate='Intermediate post24 weights are absent. Existing batch logs and oracle gate alignment cannot certify the prospective behavior. No collection or training starts; resolve snapshot/target timing evidence within the stated budget.'),
        compute_budget=dict(new_native_prefixes=0,new_native_branches=0,new_native_seconds=0,
            new_optimizer_updates=0,new_training_labels=0,checkpoints_loaded=0,checkpoints_downloaded=0,
            diagnostic_feedback_admitted=0),
        limitations='Old time-series metrics use changing samples; old final actors are not issue-visible post24 policies. Oracle expectations condition on old states under a hypothetical behavior. New trajectory timing and current-actor drift across432 future updates remain unobserved.')
    encoded=json.dumps(compact(value),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>24000:raise RuntimeError('Small-result summary exceeds24000bytes')
    out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(encoded)
    return value
