"""Read resident factual traces and pre-action roots only; no native runtime."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, os.environ['THESIS_H2O_ROOT'])
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_INDEX as EI
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_reference_collection_v84_20261027.action_pair import choose_pair
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support
from experiments.mcwm_closed_loop_issue_conditions_v91_20261028.conditions import CorridorHistory
import numpy as np
import torch

V90 = HERE.parent/'mcwm_closed_loop_stratified_collection_v90_20261028'
SCENES = ('stable', 'local_speed_drop')
WINDOWS = ((12000, 12600), (12600, 15000), (15000, 18000))
LINES = ('7X','102X','705X')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def event_readout(row):
    event = row['event']
    value = {name: float(event[index]) for name,index in EI.items()}
    value['event_load_proxy'] = value['onboard_passengers']/value['bus_capacity']
    value['issue_clock'] = row['started_at']
    value['event_age_seconds'] = row['started_at']-value['sim_time_seconds']
    return value


def condition_readout(history, now):
    full = history.snapshot(now)
    keys = ('status','reason','value','historical_only','issue_clock','baseline_counts',
            'baseline_medians_seconds','latest_completed_at','latest_observed_at','latest_age_seconds')
    value = {key:full[key] for key in keys}
    value['baseline_last_observed_at'] = max((x for x in
        full['baseline_latest_observed_at'].values() if x is not None),default=None)
    value['latest_sources'] = [{key:row[key] for key in ('line_id','trip_id',
        'completed_at','observed_at','nonholding_ready_seconds','baseline_nonholding_seconds',
        'normalized_slowdown')} for row in full['latest_sources']]
    return value


def replay_action(row, transform):
    state = np.asarray(row['s'],np.float32)
    proposal = np.asarray(row['proposed_a'],np.float32)
    pair = choose_pair(transform,state,proposal,simulation_step=1.,action_step=1.)
    support = action_support(pair['reference_hold_seconds'],pair['candidate_hold_seconds'],1.,1.)
    current = int(np.clip(round(float(state[2])*15),0,15))
    slot = state[5:].reshape(16,14)[current]
    forward,backward = slot[11]>.5,slot[12]>.5
    reserve = float(np.clip(.5*((slot[4]-1)*slot[6]-(slot[3]-1)*slot[5])*720,0,60))
    floor = bool(.15<=state[0]<.75 and forward and not backward and slot[9]>0 and slot[8]<=.25)
    exact = bool(np.array_equal(np.asarray(pair['reference_action'],np.float32),np.asarray(row['a'],np.float32)))
    return dict(reference_exact=exact,math_support=support['supported'],reason=support['reason'],
        reference_hold=pair['reference_hold_seconds'],candidate_hold=pair['candidate_hold_seconds'],
        endpoint_holds=[float(.5*(transform.numpy(state,np.asarray([u],np.float32))[0]+1)*60)
                        for u in (-1.,1.)],reserve_seconds=reserve,completion_floor=floor)


def analyze_scene(scene):
    root = V90/'runs'/scene
    declarations = [read(path) for path in sorted((root/'pairs').glob('*/*/root.json'))]
    contract = read(declarations[0]['reference']['source_contract_path'])['contract']
    transform = ExecutionAction(contract)
    # Only an unchanged reference arm may extend the factual prefix beyond12600.
    first = min((row for row in declarations if not (scene=='stable' and
        row['root']['line_id']=='7X' and row['load_bin']=='heavier')),
        key=lambda row:(row['root']['started_at'],row['root']['sequence']))
    branch = root/'pairs'/first['load_bin']/first['root']['line_id']/'low/trace.jsonl'
    parent = root/'prefix_trace.jsonl'
    parent_rows = {row['sequence']:row for row in rows(parent)}
    cutoff = first['root']['sequence']
    history = CorridorHistory(EI)
    selected = {row['root']['sequence']:row for row in declarations}
    roots, candidates, future_slots = [], [], {}
    counts = {f'{start}-{end}/{line}':Counter() for start,end in WINDOWS for line in LINES}
    replay = dict(all_actions_exact=True,reference_overlap_exact=True,
                  prefix_rows=len(parent_rows),overlap_rows=0,full_rows=0,
                  all_event_clocks_visible=True)
    ages = []

    def consume(row):
        replay['full_rows'] += 1
        event = event_readout(row)
        replay['all_event_clocks_visible'] &= event['sim_time_seconds']<=row['started_at']
        history.observe(row)
        if row['sequence'] in selected:
            saved = selected[row['sequence']]
            roots.append(dict(line_id=row['line_id'],stratum=saved['load_bin'],
                root={key:row[key] for key in ('sequence','started_at','trip_id','station_id')},
                local_service_features=saved['root']['service']['features'],
                event=event,
                condition=condition_readout(history,row['started_at']),
                reference_actor_source=saved['reference']['source_actor_path']))
        now = row['started_at']
        if row['line_id'] not in LINES or not 12000<=now<18000:
            return
        action = replay_action(row,transform)
        replay['all_actions_exact'] &= action['reference_exact']
        start,end = next((a,b) for a,b in WINDOWS if a<=now<b)
        counter = counts[f'{start}-{end}/'+row['line_id']]
        counter['issues'] += 1
        counter['supported' if action['math_support'] else 'unsupported'] += 1
        proxy_group = 'lighter' if event['event_load_proxy'] < (.13 if row['line_id']=='7X' else .5) else 'heavier'
        counter[('supported_' if action['math_support'] else 'unsupported_')+proxy_group] += 1
        key = row['line_id']+'/'+proxy_group
        if now<15000 and action['math_support'] and key not in future_slots:
            condition = condition_readout(history,now)
            if condition['value'] is not None:
                future_slots[key] = dict(root={name:row[name] for name in
                    ('sequence','started_at','trip_id','station_id')},
                    event_load_proxy=event['event_load_proxy'],event_age=event['event_age_seconds'],
                    condition=condition,action=action,
                    old_v90_same_root=next((saved['load_bin'] for saved in declarations
                        if saved['root']['sequence']==row['sequence']),None))
        if not action['math_support']:
            counter['reason/'+action['reason']] += 1
            counter['completion_floor' if action['completion_floor'] else 'without_completion_floor'] += 1
        if row['line_id']=='705X':
            # Retain only a few pre-action examples, no truth or native IDs list.
            tag=(start,end,action['math_support'],proxy_group)
            n=sum(tuple(item['tag'])==tag for item in candidates)
            if n<1:
                candidates.append(dict(tag=list(tag),sequence=row['sequence'],issue_clock=now,
                    station_id=row['station_id'],trip_id=row['trip_id'],
                    event_load_proxy=event['event_load_proxy'],event_age=event['event_age_seconds'],
                    action=action))
        ages.append(event['event_age_seconds'])

    for sequence in sorted(parent_rows):
        if sequence<cutoff:
            consume(parent_rows[sequence])
    for row in rows(branch):
        if row['sequence'] in parent_rows:
            replay['overlap_rows'] += 1
            replay['reference_overlap_exact'] &= row==parent_rows[row['sequence']]
        consume(row)
    replay['condition_historical_only'] = all(row['condition']['historical_only'] for row in roots)
    return dict(scenario=scene,sources=dict(parent=str(parent),reference_branch=str(branch)),
        replay=replay,counts={key:dict(value) for key,value in counts.items()},
        root_conditions=roots,examples_705x=candidates,future_slots=future_slots,
        event_age_range=[min(ages),max(ages)],history_summary=history.summary(),
        available_native_issue_counts='Selected roots only; remaining per-issue native contexts were not logged',
        native_at_stop_and_action_step='Not recoverable for unselected issues; math support is not a new physical preflight')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=dict(scope='Existing factual traces and before_suffixes roots only; no labels read',
        scenarios=[analyze_scene(scene) for scene in SCENES],
        budget=dict(new_native_seconds=0,new_branches=0,new_labels=0,new_fits=0,
                    neural_updates=0,v87_truth_used=0),priority='high')
    result['all_pass']=all(all(row['replay'][key] for key in ('all_actions_exact',
        'reference_overlap_exact','all_event_clocks_visible','condition_historical_only'))
        for row in result['scenarios'])
    raw=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode())>48000:
        raise ValueError('Reduce readout; only a small JSON may be pulled')
    with args.out.open('x') as handle:handle.write(raw)
    print(json.dumps(dict(status='complete',all_pass=result['all_pass'],bytes=len(raw.encode()))),flush=True)
    if not result['all_pass']:raise SystemExit('Factual trace or pre-issue condition check failed')


if __name__=='__main__':main()
