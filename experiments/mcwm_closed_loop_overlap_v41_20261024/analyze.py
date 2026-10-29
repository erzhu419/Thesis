"""Compact old-state conditional evidence, with no automatic collection."""
import json
import math
from pathlib import Path

COLUMNS=('slot','arm','links','L_old','L_geom','L_arith','old_minus_geom',
         'input_log_c','gate','mean','std','overlap_min','overlap_mean',
         'atom_overlap_mean','continuous_overlap_mean')


def compact(value):
    if isinstance(value,float):return float(f'{value:.10g}')
    if isinstance(value,list):return [compact(v) for v in value]
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
    return value


def average(rows,scope):
    count=len(rows)
    keys=('links','L_old','L_geom','L_arith','old_minus_geom','jensen_gap','input_log_c',
          'discount_log_weight','overlap_mean','atom_overlap_mean','continuous_overlap_mean')
    result={key:math.fsum(r[scope][key] for r in rows)/count for key in keys}
    result['shapley']={key:math.fsum(r[scope]['shapley'][key] for r in rows)/count
                       for key in ('gate','mean','std')}
    return result


def analyze(runs,out,parent,exits):
    cases={};clocks=[]
    for scenario in ('stable','local_speed_drop'):
        root=runs/scenario
        if not (root/'summary.json').exists():
            failure=json.loads((root/'failure.json').read_text()) if (root/'failure.json').exists() else None
            cases[scenario]=dict(status='failed',failure=failure);continue
        source=json.loads((root/'summary.json').read_text());models={}
        clocks.append(source['wall'])
        for model,rows in source['models'].items():
            table=[]
            for row in rows:
                f=row['first_reward']
                table.append([row['slot'],row['arm'],f['links'],f['L_old'],f['L_geom'],f['L_arith'],
                    f['old_minus_geom'],f['input_log_c'],*[f['shapley'][key] for key in ('gate','mean','std')],
                    f['overlap_min'],f['overlap_mean'],f['atom_overlap_mean'],f['continuous_overlap_mean']])
            models[model]=dict(first_reward_average=average(rows,'first_reward'),
                fixed44_average=average(rows,'full44'),first_reward_rows=table,
                zero_dt=sum(r['dt0'] for r in rows),missing=0,unsupported=0)
        cases[scenario]=dict(status=source['status'],integrity=source['integrity'],counts=source['counts'],
            errors=source['errors'],models=models,wall_seconds=source['wall']['finished']-source['wall']['started'])
    complete=len(clocks)==2 and parent['all_pass'] and all(code==0 for code in exits.values()) and all(
        case.get('integrity',{}).get('all_pass') for case in cases.values())
    overlap=min(c['finished'] for c in clocks)-max(c['started'] for c in clocks) if len(clocks)==2 else 0.
    value=dict(status='complete' if complete else 'partial',priority='high',
        integrity=dict(all_pass=complete),exitcodes=exits,parent_frozen=parent,
        actual_scenario_overlap_seconds=max(0.,overlap),cases=cases,
        schema=dict(columns=COLUMNS,natural_log=True,table_scope='successor links before first existing reward witness',
            attribution='input baseline + three-factor Shapley = L_old; no causal SGD attribution',
            discount='k*log0.9 is separate, including zero-duration events',
            L_geom='sum E_mu logc at the SAME old states, not a rollout estimate',
            L_arith='sum log conditional overlap; not a new trajectory success probability',
            summary_precision='10 significant digits; full event precision remains resident'),
        preserved=dict(initializer=50000,feedback_slots=285,planned_updates=456,target_horizon=24000,
            issue_window=[9000,9300],release_clock=9630,ordinary_slots=15,response_slots=270,
            warmup_updates=24,response_updates=432,F_unchanged=True,trust_unchanged=True,
            final_v40_release_plan_equal=True),
        decision=dict(collection_launched=False,training_ready=False,training_launched=False,
            next_gate='Judge L_old versus conditional L_geom and L_arith, and parameter attribution. Any sampling recommendation must still require a new-prefix native feasibility precheck; no rollout or training is authorized by this diagnostic.'),
        compute_budget=dict(new_native_prefixes=0,new_native_branches=0,new_native_seconds=0,
            new_training_labels=0,new_optimizer_updates=0,diagnostic_feedback_admitted=0,checkpoints_downloaded=0),
        limitations='All results condition on old policy-selected states. Shapley partitions frozen parameter changes, not training causes. New-state occupancy, reward timing,45-event reach and control improvement remain unobserved.')
    encoded=json.dumps(compact(value),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>24000:raise RuntimeError('Resident summary exceeds small-result limit')
    out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(encoded)
    return value
