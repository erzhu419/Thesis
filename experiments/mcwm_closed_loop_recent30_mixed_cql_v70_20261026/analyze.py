"""V70 full-day/tail decision, with immutable aligned-seed v68 comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import metrics as old
from experiments.mcwm_closed_loop_recent30_cql_v69_20261026.learning import recency_checks

MAX_BYTES = 48000
PERIODS = old.PERIODS
REFERENCES = dict(v68_fresh=('fresh','v68_fresh'),v68_no_fresh=('fresh','v68_no_fresh'))


def native_checks(summary, config, *, seed, scene, arm, cached):
    """Retain v68 admission, adding the new replay only to current arms."""
    checks = old._run_checks(summary, arm)
    checks.update(old.target_checks(summary, arm, historical=cached))
    checks.update(old.schedule_checks(summary))
    checks.update(old.cql_checks(summary, config.get('policy_config', {})))
    checks.update(correct_cell=summary.get('cell', {}).get('seed') == seed
        and summary.get('cell', {}).get('scenario') == scene
        and summary.get('cell', {}).get('arm') == arm
        and summary.get('cell', {}).get('candidate') == 'executed_cql',
        all_native_checks=bool(summary.get('integrity', {}).get('checks'))
            and all(x is True for x in summary.get('integrity', {}).get('checks', {}).values()))
    if not cached:
        checks.update({'recency/'+key: value for key, value in
            recency_checks(summary.get('replay', {}), arm, 456, seed).items()})
    return checks


def statistics(cells,protocol):
    expected=len(protocol['development_seeds'])*len(protocol['scenarios'])
    return {comparison:{period:dict(overall=old.group(cells,comparison,period,expected),
        by_scenario={scene:old.group([c for c in cells if c['scenario']==scene],comparison,
            period,len(protocol['development_seeds'])) for scene in protocol['scenarios']})
        for period,_ in PERIODS} for comparison in REFERENCES}


def analyze(root=HERE):
    root=Path(root)
    p=old.read(root/'protocol.json')
    main=old.analyze(root)
    cells,checks,missing,evidence,failures,failed_work=[],{},[],[],[],[]
    for seed in p['development_seeds']:
        for scene in p['scenarios']:
            name=f'seed{seed}/{scene}'
            current=root/'runs/executed_cql/development'/f'seed{seed}'/scene
            ref=p['original_cql_reference']
            reference=root.parent/ref['directory']/ref['relative'].format(seed=seed,scenario=scene)
            runs,points,dirs,configs={}, {}, {}, {}
            for arm in ('fresh','no_fresh','v68_fresh','v68_no_fresh'):
                cached=arm.startswith('v68_')
                native_arm=arm.removeprefix('v68_')
                directory=(reference if cached else current)/native_arm
                row=old.read(directory/'summary.json')
                if row is None:
                    failure=old.read(directory/'failure.json')
                    (failures if failure else missing).append(name+'/'+arm)
                    if failure:
                        failed_work.append(dict(cell=name+'/'+arm,**{key:failure.get(key)
                            for key in ('type','native_time','consumed_budget','wall_seconds')}))
                    continue
                runs[arm],dirs[arm]=row,directory
                configs[arm]=old.read(directory/'protocol.json') or {}
                points[arm]=old.read_metrics(directory,row)
                checks.update({name+'/'+arm+'/'+key:value is True for key,value in
                    native_checks(row,configs[arm],seed=seed,scene=scene,
                        arm=native_arm,cached=cached).items()})
                checks[name+'/'+arm+'/required_cost_endpoints']=all(
                    old.window(row,points[arm],*bounds)['status']=='complete' for _,bounds in PERIODS)
                if not cached:
                    checks[name+'/'+arm+'/recipe']=row.get('cell',{}).get('recipe')==p['recipe']
            if len(runs)!=4:
                continue
            fresh,nofresh,cached=runs['fresh'],runs['no_fresh'],runs['v68_no_fresh']
            exact_trace=old.compare_prefix(dirs['no_fresh']/'prefix_trace.jsonl',
                dirs['v68_no_fresh']/'prefix_trace.jsonl',float('inf'))
            exact_trace.pop('stop_exclusive_seconds',None)
            exact_trace['scope']='complete issued trace; no time cutoff'
            exact=dict(ledger=nofresh['cost']==cached['cost'],
                metrics=points['no_fresh']==points['v68_no_fresh'],
                issued_trace=exact_trace['all_pass'] and exact_trace['compared_rows']==
                    nofresh.get('factual_trace_rows')==cached.get('factual_trace_rows'),
                feedback_receipts=nofresh.get('receipts')==cached.get('receipts'),
                initializer=nofresh.get('initializer')==cached.get('initializer'),
                runtime=old._runtime(nofresh)==old._runtime(cached),
                update_schedule=nofresh.get('actor_schedule_audit')==cached.get('actor_schedule_audit'),
                target_path=nofresh.get('target_audit')==cached.get('target_audit'),
                config=configs['no_fresh'].get('policy_config')==
                    configs['v68_no_fresh'].get('policy_config'))
            checks.update({name+'/matched_v68_no_fresh/'+k:v for k,v in exact.items()})
            periods,comparisons={}, {key:{} for key in REFERENCES}
            for period,bounds in PERIODS:
                values={arm:old.window(row,points[arm],*bounds) for arm,row in runs.items()}
                periods[period]=values
                for key,(left,right) in REFERENCES.items():
                    comparisons[key][period]=old.difference(values[left],values[right])
            cells.append(dict(identity=seed,scenario=scene,periods=periods,comparisons=comparisons))
            evidence.append(dict(seed=seed,scenario=scene,no_fresh_exact=exact,
                full_trace=exact_trace,
                recency={arm:runs[arm]['replay'] for arm in ('fresh','no_fresh')},
                targets={arm:runs[arm]['target_audit'] for arm in ('fresh','no_fresh')}))
    stats=statistics(cells,p)
    expected=len(p['development_seeds'])*len(p['scenarios'])
    technical=(main['integrity']['all_pass'] and len(cells)==expected and not missing and not failures
        and bool(checks) and all(checks.values()))
    matched=old.cost_gate(main['aggregates']['matched'])
    absolute=old.cost_gate(main['aggregates']['original_no_fresh'])
    cql=old.cost_gate(stats['v68_no_fresh'])
    tail=main['aggregates']['matched']['post_budget']
    tail_groups=[tail['overall'],*tail['by_scenario'].values()]
    tail_ok=all(x['complete_cells']==x['expected_cells'] and x['complete_cells']>0
        and x['mean_cost_difference'] is not None and x['mean_cost_difference']<=0 for x in tail_groups)
    old_deltas=[c['periods']['full_day']['v68_fresh']['generalized_seconds_per_departed_passenger']-
        c['periods']['full_day']['v68_no_fresh']['generalized_seconds_per_departed_passenger'] for c in cells
        if all(c['periods']['full_day'][a]['status']=='complete' for a in ('v68_fresh','v68_no_fresh'))]
    new_worst=stats['v68_no_fresh']['full_day']['overall']['worst_cost_difference']
    old_worst=max(old_deltas) if len(old_deltas)==expected else None
    worst_ok=new_worst is not None and old_worst is not None and new_worst<=old_worst
    passed=technical and matched and absolute and cql and tail_ok and worst_ok
    failed=[k for k,v in checks.items() if not v]
    main['integrity']['checked_conditions']+=len(checks)
    main['integrity']['failed_checks'].extend(failed)
    main['integrity']['all_pass']=bool(technical)
    status=('complete' if technical else 'integrity_failed' if failures or
        main['status']=='integrity_failed' or any(x is not True for x in checks.values()) else 'pending')
    main.update(status=status,
        complete=bool(technical),recipe=p['recipe'],missing=sorted(set(main['missing']+missing)),
        failures=sorted(set(main['failures']+failures)),
        failed_run_work=list({row['cell']:row for row in main['failed_run_work']+failed_work}.values()),
        recency_summary_file='recency.json',
        acceptance_protocol=dict(cost_gate=p['cost_gate'],early_window='descriptive_only',
            historical_v68_verdict_unchanged=True,
            scope='New v70 gate frozen before new results; not a reclassification of v68.'),
        optimization_assessment=dict(conditions=dict(matched_tail_nonharm=tail_ok,worst_cell_not_worse=worst_ok),
            previous_v68_worst=old_worst,new_worst=new_worst),
        decision=dict(advance_to_confirmation=bool(passed),scientific_gate_pass=bool(passed),
            matched_cost_gate_pass=bool(technical and matched),original_reference_gate_pass=bool(technical and absolute),
            cached_cql_reference_gate_pass=bool(technical and cql),post_budget_gate_pass=bool(technical and tail_ok),
            worst_cell_gate_pass=bool(technical and worst_ok),ordinary_dripping_goal_achieved=False,
            reason='development_pass_requires_root_review' if passed else
                'incomplete_or_integrity_failed' if not technical else 'candidate_closed_cost_gate_failed'),
        interpretation='Fixed exposed6201/6203 development only. Early window descriptive, full-day and tail costs decide. '
            'Fresh restores the original mixed bootstrap; recent30 and CQL stay fixed. '
            'No_fresh must reproduce immutable v68 complete native results. No automatic confirmation.')
    extra=dict(recipe=p['recipe'],stage='development',status=main['status'],aggregates=stats,
        cells=[dict(identity=c['identity'],scenario=c['scenario'],periods={period:dict(
            comparisons={name:old.compact_comparison(c['comparisons'][name][period]) for name in REFERENCES},
            v68_arms={arm:old.cost_view(c['periods'][period][arm]) for arm in ('v68_fresh','v68_no_fresh')})
            for period,_ in PERIODS}) for c in cells],native_evidence=evidence,missing=missing,
        integrity=dict(all_pass=bool(technical),failed_checks=failed),
        prior_results_modified=False,confirmation_started=False)
    return main,extra


def export(root,out):
    main,recency=analyze(root)
    cost,policy=old.split_summary(main)
    out=Path(out)
    payloads={out:old.encoded(cost),out.with_name('policy_drift.json'):old.encoded(policy),
        out.with_name('recency.json'):old.encoded(recency)}
    for path,raw in payloads.items():
        if len(raw)>MAX_BYTES:
            raise ValueError(f'{path.name} exceeds48KB: {len(raw)}')
    out.parent.mkdir(parents=True,exist_ok=True)
    for path,raw in payloads.items():
        with path.open('xb') as handle:
            handle.write(raw)
    return dict(status=cost['status'],decision=cost['decision'],files={p.name:len(raw) for p,raw in payloads.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(HERE,args.out),allow_nan=False))
