"""Resident six-day component readout; no training, branch query or raw download."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path
from statistics import mean, stdev

PERIODS = {'full_day':(0,24000), 'early':(9000,12000), 'post_budget':(18300,24000)}
FAMILIES = ('critic','value','actor','alpha')
CONTRASTS = ('decoupled_minus_coupled','coupled_minus_fixed_no_fresh')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def prefix_equal(left, right, stop):
    def rows(path):
        with path.open() as handle:
            for text in handle:
                if text.strip():
                    row = json.loads(text)
                    if row['started_at'] < stop:
                        yield row
    if not left.is_file() or not right.is_file() or stop is None:
        return dict(all_pass=False, rows=0, differing_rows=None)
    count = different = 0
    for a,b in zip_longest(rows(left),rows(right)):
        count += 1
        different += a != b
    return dict(all_pass=count > 0 and different == 0, rows=count, differing_rows=different)


def metric_windows(directory, summary):
    path = directory/'metrics.jsonl'
    points = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    by_clock = {row['time_seconds']:row for row in points}
    all_points = [row['time_seconds'] for row in points] == list(range(0,24001,300))
    windows = {}
    for name,(start,end) in PERIODS.items():
        if start not in by_clock or end not in by_clock:
            windows[name] = None
            continue
        first,last = by_clock[start],by_clock[end]
        w = last['accrued_wait_seconds']-first['accrued_wait_seconds']
        r = last['accrued_in_vehicle_seconds']-first['accrued_in_vehicle_seconds']
        windows[name] = dict(wait=w,ride=r,cost=2*w+r,departed=15079,cost_per_passenger=(2*w+r)/15079)
    cost = summary['cost']
    final = by_clock.get(24000,{})
    ledger = all(final.get(key) == cost.get(key) for key in ('accrued_wait_seconds','accrued_in_vehicle_seconds'))
    ledger &= cost.get('departed_passengers') == 15079 and cost.get('generalized_passenger_seconds') == (
        2*cost.get('accrued_wait_seconds',0)+cost.get('accrued_in_vehicle_seconds',0))
    return windows, dict(exact_metric_clocks=all_points,complete_windows=all(v is not None for v in windows.values()),final_ledger=ledger)


def load_day(directory, cached, spec):
    summary = read(directory/'summary.json')
    if summary is None:
        failure = read(directory/'failure.json')
        return dict(seed=spec['seed'],scenario=spec['scenario'],status='failed' if failure else 'missing',
            checks=dict(complete=False),periods={},budget=(failure or {}).get('consumed_budget'),
            native_seconds=(failure or {}).get('native_time'),failure=failure)
    config,preflight = read(directory/'protocol.json') or {},read(directory/'preflight.json') or {}
    old,old_config = read(cached/'summary.json'),read(cached/'protocol.json')
    old = old or {}; old_config = old_config or {}
    budget = summary['budget']
    stop = summary.get('clock_audit',{}).get('first_feedback_applied_at')
    prefix = prefix_equal(directory/'prefix_trace.jsonl',cached/'prefix_trace.jsonl',stop)
    periods,checks = metric_windows(directory,summary)
    post = summary.get('post_budget_counts',{})
    checks.update(complete=summary['status'] == 'complete' and summary['integrity']['all_pass']
        and bool(summary['integrity']['checks']) and all(summary['integrity']['checks'].values()),
        exact_cell=all(summary['cell'].get(key) == spec[key] for key in ('seed','scenario','arm','external_seed')),
        first8=preflight.get('technical_gate_pass') is True,
        reused_v75_initializer=summary.get('initializer') == old.get('initializer')
            and summary.get('initializer',{}).get('initial_updates') == 50000,
        policy_config=bool(config.get('policy_config')) and config.get('policy_config') == old_config.get('policy_config'),
        common_F=bool(summary.get('runtime_contract')) and summary.get('runtime_contract') == old.get('runtime_contract'),
        pre_update_prefix=prefix['all_pass'] and stop == old.get('clock_audit',{}).get('first_feedback_applied_at'),
        coupled_record_path=summary['integrity']['checks'].get('coupled_covariate_outcome_records') is True,
        feedback285=budget.get('target_feedback_slots') == budget.get('training_fresh_outcomes') == 285,
        packets57=budget.get('packets') == 57,
        native24000=budget.get('native_seconds') == 24000,
        updates456=all(budget.get(name+'_updates') == 456 for name in (*FAMILIES,'policy')),
        zero_world_trust_queries=all(budget.get(key) == 0 for key in ('world_updates','trust_updates','counterfactual_queries')),
        frozen_after_budget=isinstance(post.get('before'),dict) and post['before'] == post.get('after')
            and all(post['before'].get(name) == 456 for name in (*FAMILIES,'policy')))
    selection = summary.get('capacity_selection',{})
    replay = summary.get('replay',{})
    return dict(seed=spec['seed'],scenario=spec['scenario'],status=summary['status'],checks=checks,
        periods=periods,budget=budget,native_seconds=budget.get('native_seconds'),prefix=prefix,
        information=dict(covariate_selections=budget.get('target_feedback_slots'),
            complete_outcomes=budget.get('training_fresh_outcomes'),
            coupled_distinct_records=len({seq for receipt in summary['receipts'] for seq in receipt['sequence_ids']}),
            risk_windows=selection.get('risk_windows'),covered_risk_windows=selection.get('selected_covered_risk_windows'),
            replacements=selection.get('replacements'),recent_window=replay.get('critic_fresh_outcome_window')),
        wall_seconds=summary.get('wall_seconds'),extra_cql_wall_seconds=summary.get('cql_audit',{}).get('regularizer_wall_seconds'))


def difference(left,right):
    if left is None or right is None:
        return None
    if left['departed'] != right['departed'] or left['departed'] != 15079:
        raise ValueError('All windows use the same full-day departed-passenger denominator')
    return dict(wait=(left['wait']-right['wait'])/15079,
                ride=(left['ride']-right['ride'])/15079,cost=(left['cost']-right['cost'])/15079)


def group(cells,contrast,period,expected):
    values = [row['comparisons'][contrast][period] for row in cells]
    valid = [value for value in values if value is not None]
    delta = [value['cost'] for value in valid]
    wins,losses = [v for v in delta if v < 0],[v for v in delta if v > 0]
    return dict(expected=expected,available=len(valid),complete=len(valid) == expected,
        mean={key:mean(v[key] for v in valid) if valid else None for key in ('wait','ride','cost')},
        wins=len(wins),losses=len(losses),ties=sum(v == 0 for v in delta),
        win_fraction=len(wins)/len(valid) if valid else None,
        winning_mean_reduction=-mean(wins) if wins else None,losing_mean_increase=mean(losses) if losses else None,
        best=min(delta) if delta else None,worst=max(delta) if delta else None)


def analyze(summaries,reference,protocol):
    refs = {(row['identity'],row['scenario']):row for row in reference['cells']}
    cells,missing = [],[]
    seen = set()
    for day in summaries:
        key = day['seed'],day['scenario']
        if key in seen:
            raise ValueError('Duplicate day cannot replace a missing prescribed cell')
        seen.add(key)
        old = refs.get(key)
        comparisons = {name:{} for name in CONTRASTS}
        for period in PERIODS:
            new = day['periods'].get(period)
            old_period = old['periods'].get(period) if old else None
            arms = old_period['arms'] if old_period else {}
            comparisons[CONTRASTS[0]][period] = difference(arms.get('fresh'),new)
            comparisons[CONTRASTS[1]][period] = difference(new,arms.get('no_fresh'))
        if not all(day['checks'].values()) or any(value is None for c in comparisons.values() for value in c.values()):
            missing.append(dict(seed=key[0],scenario=key[1],status=day['status'],
                failed_checks=[name for name,passed in day['checks'].items() if not passed],failure=day.get('failure')))
        cells.append(dict(**day,comparisons=comparisons))
    expected={(spec['seed'],spec['scenario']) for spec in protocol['native_plan']}
    if seen-expected:
        raise ValueError('Undeclared native day')
    missing.extend(dict(seed=seed,scenario=scene,status='missing') for seed,scene in sorted(expected-seen))
    technical = len(cells) == len(expected) == 6 and not missing
    aggregates={contrast:{period:dict(overall=group(cells,contrast,period,6),
        by_scenario={scene:group([c for c in cells if c['scenario'] == scene],contrast,period,3)
                     for scene in protocol['scenarios']}) for period in PERIODS} for contrast in CONTRASTS}
    seeds=[]
    for seed in protocol['development_seeds']:
        rows=[c for c in cells if c['seed'] == seed]
        record=dict(seed=seed,contrasts={})
        for contrast in CONTRASTS:
            record['contrasts'][contrast]={period:group(rows,contrast,period,2) for period in PERIODS}
        seeds.append(record)
    primary=aggregates[CONTRASTS[0]]['full_day']['overall']
    means=[r['contrasts'][CONTRASTS[0]]['full_day']['mean']['cost'] for r in seeds
           if r['contrasts'][CONTRASTS[0]]['full_day']['complete']]
    native=[r['native_seconds'] for r in summaries]
    known=[n for n in native if n is not None]
    updates={name:sum((r.get('budget') or {}).get(name+'_updates',0) for r in summaries) for name in FAMILIES}
    labels=sum((r.get('budget') or {}).get('training_fresh_outcomes',0) for r in summaries)
    passed=technical and primary['mean']['cost'] < 0 and primary['wins'] > 3
    return dict(status='complete' if technical else 'hold',technical_complete=technical,missing_or_failed=missing,
        cells=cells,aggregates=aggregates,seed_readout=dict(independent_unit='Training initialization; two scenes averaged within seed',
            rows=seeds,complete_seeds=len(means),mean=mean(means) if means else None,
            sample_sd=stdev(means) if len(means)>1 else None,
            wins=sum(v<0 for v in means),losses=sum(v>0 for v in means)),
        budget=dict(new_native_days_complete=sum(r['status']=='complete' for r in summaries),
            known_new_native_seconds=sum(known),native_spend_complete=len(known)==6,
            native_cap=144000,new_complete_outcome_labels=labels,updates_per_optimizer=updates,
            new_initialization_updates=0,reused_v75_initializers=3,
            reused_decoupled_days=6,reused_fixed_no_fresh_days=6),
        decision=dict(component_mean_majority_pass=passed,no_automatic_extension=True,
            complete_negative_result=technical and not passed),
        interpretation=dict(primary='Cached v75 decoupled minus new coupled; negative favors separate replay',
            secondary='New coupled minus cached latest5 no_fresh is a fixed reference, not the coupled method own matched control',
            scope='Three already-exposed initializer/environment units; same label/update count but different unique-record unions. Not a record-count-matched causal module isolation or new independent confirmation.',
            windows='Early9000-12000; post-update18300-24000; all use full-day15079 denominator and unfinished-trip accrued costs'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reference','protocol','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    p=read(args.protocol); reference=read(args.reference)
    repo=args.protocol.resolve().parents[2]
    cached_root=repo/p['cached_reference']['root']
    records=[]
    for spec in p['native_plan']:
        directory=args.root/Path(spec['relative_out']).relative_to('runs')
        cached=cached_root/p['cached_reference']['relative'].format(**spec)
        records.append(load_day(directory,cached,spec))
    result=analyze(records,reference,p)
    encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>p['summary_max_bytes']:
        raise ValueError('Compact component readout exceeds64KB; raw artifacts stay resident')
    with args.out.open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=result['status'],component_pass=result['decision']['component_mean_majority_pass'],bytes=len(encoded.encode()))),flush=True)


if __name__=='__main__':
    main()
