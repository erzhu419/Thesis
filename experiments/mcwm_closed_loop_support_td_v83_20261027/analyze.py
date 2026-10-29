"""Fixed eight-day singleton-TD gradient trial; keep v75/v81 references visible."""
from __future__ import annotations

import argparse
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.analyze import native_checks
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.analyze import pool_view
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.analyze import compare_update_logs
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import metrics as old

ARMS = ('fresh', 'no_fresh')
PERIODS = old.PERIODS
COMPARISONS = dict(matched=('fresh','no_fresh'),
    v75_no_fresh=('fresh','v75_no_fresh'), v75_fresh=('fresh','v75_fresh'),
    v81_fresh=('fresh','v81_fresh'),
    no_fresh_v75_shift=('no_fresh','v75_no_fresh'),
    no_fresh_v81_shift=('no_fresh','v81_no_fresh'))
MAX_BYTES = 48000


def run_dir(root, seed, scene, arm):
    return Path(root)/'runs/executed_cql/development'/f'seed{seed}'/scene/arm


def reference_dir(root, protocol, seed, scene, arm,version='v75'):
    phase='validation' if version=='v75' else 'development'
    return (Path(root).parent/protocol[version+'_reference']/
        f'runs/executed_cql/{phase}/seed{seed}'/scene/arm)


def native_arm(label):
    return label.removeprefix('v75_').removeprefix('v81_')


def compare_jsonl(current,cached):
    """Read complete native evidence on the server; return small counts."""
    if not current.is_file() or not cached.is_file():
        return dict(all_pass=False,compared_rows=0,first_difference='missing_log')
    count,first=0,None
    with current.open() as left,cached.open() as right:
        for index,(a,b) in enumerate(zip_longest(left,right)):
            if a is None or b is None or json.loads(a)!=json.loads(b):
                first=index
                break
            count+=1
    return dict(all_pass=first is None and count>0,compared_rows=count,first_difference=first)


def compare_checkpoints(current,cached):
    """Detect drift between logged updates without transferring either snapshot."""
    if not current.is_file() or not cached.is_file():
        return dict(all_pass=False,checks=dict(final_snapshot_present=False))
    import torch
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal
    left=torch.load(current,map_location='cpu',weights_only=False)
    right=torch.load(cached,map_location='cpu',weights_only=False)
    checks={key:key in left and key in right and _equal(left[key],right[key])
        for key in ('state','global_rng','outcome_rng','config','counts','frozen_at')}
    return dict(all_pass=all(checks.values()),checks=checks,
        scope='Final modules,Adam,counters,RNG and configuration; snapshots remain resident')


def no_fresh_regression(current,cached,summary,reference):
    trace=old.compare_prefix(current/'prefix_trace.jsonl',cached/'prefix_trace.jsonl',math.inf)
    trace.pop('stop_exclusive_seconds',None)
    trace['scope']='complete issued trace; no cutoff'
    updates=compare_update_logs(current/'updates.jsonl',cached/'updates.jsonl')
    contexts={name:compare_jsonl(current/name,cached/name)
        for name in ('context_issues.jsonl','context_selected.jsonl')}
    checkpoint=compare_checkpoints(current/'final_learner.pt',cached/'final_learner.pt')
    checks=dict(full_trace=trace['all_pass'] and trace['compared_rows']==
            summary.get('factual_trace_rows')==reference.get('factual_trace_rows'),
        cost=summary.get('cost')==reference.get('cost'),
        metrics=old.read_metrics(current,summary)==old.read_metrics(cached,reference),
        logged_updates=updates['all_pass'],
        contexts=all(row['all_pass'] for row in contexts.values()),
        final_model_Adam_RNG=checkpoint['all_pass'],
        receipts=summary.get('receipts')==reference.get('receipts'),
        post_budget_counts=summary.get('post_budget_counts')==reference.get('post_budget_counts'))
    return dict(all_pass=all(checks.values()),checks=checks,trace=trace,updates=updates,
        contexts=contexts,final_snapshot=checkpoint,
        scope='Complete no_fresh reproduction against immutablev81; only CQL profiler wall is ignored')


def seed_readout(cells,protocol):
    rows=[]
    for seed in protocol['development_seeds']:
        selected=[row for row in cells if row['identity']==seed]
        periods={}
        for period,_ in PERIODS:
            values=[row['comparisons']['matched'][period]['candidate_minus_baseline']
                ['generalized_seconds_per_departed_passenger'] for row in selected
                if row['comparisons']['matched'][period]['status']=='complete']
            complete=len(values)==len(protocol['scenarios'])
            periods[period]=dict(status='complete' if complete else 'missing',
                paired_scenario_mean=sum(values)/len(values) if complete else None)
        rows.append(dict(seed=seed,periods=periods))
    values=[row['periods']['full_day']['paired_scenario_mean'] for row in rows
        if row['periods']['full_day']['status']=='complete']
    return dict(rows=rows,expected_seeds=len(protocol['development_seeds']),complete_seeds=len(values),
        mean_paired_difference=sum(values)/len(values) if len(values)==len(rows) else None,
        improved_seeds=sum(value<0 for value in values),degraded_seeds=sum(value>0 for value in values),
        interpretation='Four exposed scenario cells belong to two training seeds; win fractions are not future probabilities.')


def context_checks(value):
    value = value or {}
    integrity, first = value.get('integrity', {}), value.get('first_packet', {})
    return dict(context_complete=integrity.get('all_pass') is True,
        context_audits=bool(integrity.get('checks'))
            and all(flag is True for flag in integrity['checks'].values()),
        first8_reused=first.get('actual_updates') == 8
            and first.get('reused_main_day') is True and first.get('no_extra_steps') is True,
        first8_audits=bool(first.get('checks'))
            and all(flag is True for flag in first['checks'].values()))


def context_view(value):
    value = value or {}
    integrity, first = value.get('integrity', {}), value.get('first_packet', {})
    return dict(all_pass=integrity.get('all_pass'),
        checked_conditions=len(integrity.get('checks', {})),
        failed_checks=[key for key, flag in integrity.get('checks', {}).items() if flag is not True],
        first_packet={key:first.get(key) for key in ('actual_updates','reused_main_day','no_extra_steps')},
        issues_counts=value.get('issues', {}).get('counts'),
        selected_counts=value.get('selected', {}).get('counts'),
        native_observation=value.get('native_observation'))


def statistics(cells, protocol):
    expected = len(protocol['development_seeds'])*len(protocol['scenarios'])
    return {name:{period:dict(overall=old.group(cells, name, period, expected),
        by_scenario={scene:old.group([row for row in cells if row['scenario'] == scene],
            name, period, len(protocol['development_seeds'])) for scene in protocol['scenarios']})
        for period, _ in PERIODS} for name in COMPARISONS}


def compact_statistics(value):
    """Retain exact costs and cell counts without redundant derived fields."""
    if isinstance(value,dict):
        return {key:compact_statistics(item) for key,item in value.items()
            if key not in ('percent_change','missing_cells')}
    return value


def cost_assessment(cells, aggregates, expected):
    """Keep tail harm separate from the frozen full-day advancement conditions."""
    full_old = aggregates['v75_fresh']['full_day']['overall']
    full_v81 = aggregates['v81_fresh']['full_day']['overall']
    worst = aggregates['matched']['full_day']['overall']['worst_cost_difference']
    old_deltas = [row['periods']['full_day']['v75_fresh']['generalized_seconds_per_departed_passenger']
        - row['periods']['full_day']['v75_no_fresh']['generalized_seconds_per_departed_passenger']
        for row in cells if all(row['periods']['full_day'][arm]['status'] == 'complete'
            for arm in ('v75_fresh','v75_no_fresh'))]
    old_worst = max(old_deltas) if len(old_deltas) == expected else None
    conditions = dict(matched_full_day=old.cost_gate(aggregates['matched']),
        v75_absolute_full_day=old.cost_gate(aggregates['v75_no_fresh']),
        fresh_mean_not_worse_than_v75=full_old['complete_cells'] == expected
            and full_old['mean_cost_difference'] is not None and full_old['mean_cost_difference'] <= 0,
        fresh_mean_not_worse_than_v81=full_v81['complete_cells'] == expected
            and full_v81['mean_cost_difference'] is not None and full_v81['mean_cost_difference'] <= 0,
        worst_cell_not_worse=worst is not None and old_worst is not None and worst <= old_worst)
    stable = [row for row in cells if row['scenario'] == 'stable']
    tails = {}
    for name in ('matched','v75_no_fresh','v75_fresh','v81_fresh'):
        group = aggregates[name]['post_budget']['by_scenario']['stable']
        complete = group['complete_cells'] == group['expected_cells'] and group['complete_cells'] > 0
        tails[name] = dict(complete=complete,
            every_cell_non_degrading=bool(complete and all(
                row['comparisons'][name]['post_budget']['status'] == 'complete'
                and row['comparisons'][name]['post_budget']['candidate_minus_baseline']
                    ['generalized_seconds_per_departed_passenger'] <= 0 for row in stable)),
            mean_cost_difference=group['mean_cost_difference'],
            worst_cost_difference=group['worst_cost_difference'])
    return dict(conditions=conditions, matched_worst=worst, v75_matched_worst=old_worst,
        stable_tail_non_degradation=tails,
        tail_is_new_hard_gate=False, old_gate_conclusions_unchanged=True)


def analyze(root=HERE):
    root = Path(root)
    protocol = old.read(root/'protocol.json')
    checks, missing, failures, cells, evidence, pools, training = {}, [], [], [], [], [], []
    for seed in protocol['development_seeds']:
        for scene in protocol['scenarios']:
            name = f'seed{seed}/{scene}'
            directories = {arm:run_dir(root, seed, scene, arm) for arm in ARMS}
            directories.update({f'{version}_{arm}':reference_dir(root, protocol, seed, scene, arm,version)
                for version in ('v75','v81') for arm in ARMS})
            runs, points, configs, preflights = {}, {}, {}, {}
            for label, directory in directories.items():
                summary = old.read(directory/'summary.json')
                if summary is None:
                    failed = old.read(directory/'failure.json')
                    if failed:
                        failures.append(dict(cell=name+'/'+label, **{key:failed.get(key) for key in
                            ('type','message','native_time','consumed_budget','wall_seconds')}))
                    else:
                        missing.append(name+'/'+label)
                    continue
                arm = native_arm(label)
                current = label in ARMS
                config = old.read(directory/'protocol.json') or {}
                runs[label], configs[label] = summary, config
                points[label] = old.read_metrics(directory, summary)
                audit = native_checks(summary, config, seed=seed, scene=scene, arm=arm, current=True)
                audit.update(external_seed=summary.get('cell', {}).get('external_seed') == seed+protocol['external_seed_offset'],
                    departed_cohort=summary.get('cost', {}).get('departed_passengers') == protocol['full_day_departed_passengers'],
                    complete_cost_endpoints=all(old.window(summary, points[label], *bounds)['status'] == 'complete'
                        for _, bounds in PERIODS))
                final = {point['time_seconds']:point for point in points[label]}.get(24000, {})
                audit['final_metrics_match_ledger'] = all(final.get(key) == summary.get('cost', {}).get(key)
                    for key in ('accrued_wait_seconds','accrued_in_vehicle_seconds'))
                if current:
                    preflights[arm] = old.read(directory/'preflight.json') or {}
                    context = old.read(directory/'context_summary.json') or {}
                    audit.update(context_checks(context))
                    gate=summary.get('td_gradient_gate',{})
                    audit.update(td_gate_complete=gate.get('all_pass') is True,
                        td_gate_audits=bool(gate.get('checks')) and all(x is True for x in gate['checks'].values()))
                    audit['first_packet_preflight'] = preflights[arm].get('technical_gate_pass') is True
                    audit['recipe'] = summary.get('cell', {}).get('recipe') == protocol['recipe']
                    pools.append(dict(seed=seed, scenario=scene, arm=arm,
                        **pool_view(summary), context=context_view(context),td_gradient_gate=gate))
                    schedule = summary.get('actor_schedule_audit', {})
                    training.append(dict(seed=seed, scenario=scene, arm=arm, budget=summary.get('budget'),
                        schedule={key:schedule.get(key) for key in ('packets','critic_updates','value_updates',
                            'actor_updates','alpha_updates','schedule','optimizer_order_exact')},
                        first_packet=context.get('first_packet'),td_gradient_gate=gate,
                        final_learner_path=summary.get('final_learner_path'),
                        post_budget_counts=summary.get('post_budget_counts'), wall_seconds=summary.get('wall_seconds'),
                        extra_CQL_wall_seconds=summary.get('cql_audit', {}).get('regularizer_wall_seconds')))
                checks.update({name+'/'+label+'/'+key:flag is True for key, flag in audit.items()})
            if len(runs) != 6:
                continue
            first = [run.get('clock_audit', {}).get('first_feedback_applied_at') for run in runs.values()]
            valid_first = all(isinstance(x, (int,float)) and math.isfinite(x) and x > 0 for x in first)
            prefixes = {label:old.compare_prefix(directories['fresh']/'prefix_trace.jsonl',
                directories[label]/'prefix_trace.jsonl', min(first) if valid_first else 0.)
                for label in ('no_fresh','v75_fresh','v75_no_fresh','v81_fresh','v81_no_fresh')}
            parallel = old._overlap(runs['fresh'], runs['no_fresh'])
            pair = dict(common_prefix=valid_first and len(set(first)) == 1
                    and all(value['all_pass'] for value in prefixes.values()),
                same_initializer=all(runs['fresh'].get('initializer') == run.get('initializer') for run in runs.values()),
                same_F_and_runtime=all(old._runtime(runs['fresh']) == old._runtime(run) for run in runs.values()),
                same_policy_config=all(configs['fresh'].get('policy_config') == conf.get('policy_config') for conf in configs.values()),
                actual_parallel=parallel['positive_overlap'])
            for key in ('initializer','first_feedback','execution_contract'):
                values = [preflights[arm].get(key) for arm in ARMS]
                pair['preflight_'+key] = bool(values[0]) and values[0] == values[1]
            regression=no_fresh_regression(directories['no_fresh'],directories['v81_no_fresh'],
                runs['no_fresh'],runs['v81_no_fresh'])
            pair['no_fresh_v81_exact']=regression['all_pass']
            checks.update({name+'/'+key:flag is True for key, flag in pair.items()})
            periods, comparisons = {}, {key:{} for key in COMPARISONS}
            for period, bounds in PERIODS:
                periods[period] = {label:old.window(run, points[label], *bounds) for label, run in runs.items()}
                for comparator, (left,right) in COMPARISONS.items():
                    comparisons[comparator][period] = old.difference(periods[period][left],periods[period][right])
            cells.append(dict(identity=seed, scenario=scene, periods=periods, comparisons=comparisons))
            evidence.append(dict(seed=seed, scenario=scene, checks=pair, prefixes=prefixes,
                actual_parallel=parallel, initializer=runs['fresh'].get('initializer'),
                no_fresh_v81_regression=regression))
    expected = len(protocol['development_seeds'])*len(protocol['scenarios'])
    technical = len(cells) == expected and not missing and not failures and bool(checks) and all(checks.values())
    status = 'complete' if technical else ('integrity_failed' if failures or any(not flag for flag in checks.values()) else 'pending')
    aggregates = statistics(cells, protocol)
    assessment = cost_assessment(cells, aggregates, expected)
    passed = technical and all(assessment['conditions'].values())
    complete_runs = len(training) == expected*2
    complete_pools = len(pools) == expected*2
    budget = dict(new_initialization_updates=0, reused_initializers=protocol['development_seeds'],
        planned_native_days=expected*2, planned_native_seconds=expected*2*24000,
        planned_covariate_slots=expected*2*285, planned_complete_fresh_results=expected*285,
        planned_updates_per_optimizer=expected*2*456, diagnostic_branches=0, extra_first8_steps=0,
        actual_covariate_slots=sum(row['replay']['covariate_records_received'] for row in pools) if complete_pools else None,
        actual_complete_fresh_results=sum(row['replay']['outcome_labels_received'] for row in pools) if complete_pools else None,
        actual_distinct_target_records=sum(row['replay']['distinct_target_records_used'] for row in pools) if complete_pools else None,
        actual_native_seconds=sum(row['budget']['native_seconds'] for row in training) if complete_runs else None,
        actual_updates_per_optimizer={family:sum(row['budget'][family+'_updates'] for row in training)
            if complete_runs else None for family in ('critic','value','actor','alpha')})
    integrity = dict(all_pass=technical, checked_conditions=len(checks), failed_checks=[key for key, flag in checks.items() if not flag])
    main = dict(status=status, complete=technical, expected_condition_pairs=expected, available_condition_pairs=len(cells),
        integrity=integrity, missing=missing, failures=failures, aggregates=compact_statistics(aggregates),
        seed_readout=seed_readout(cells, protocol), optimization_assessment=assessment,
        cells=[dict(identity=row['identity'],scenario=row['scenario'],periods={period:dict(
            arms={arm:old.cost_view(value) for arm,value in row['periods'][period].items()},
            comparisons={key:old.compact_comparison(row['comparisons'][key][period]) for key in COMPARISONS})
            for period,_ in PERIODS}) for row in cells], budget=budget,
        decision=dict(scientific_gate_pass=bool(passed), primary_full_day_gate_pass=bool(technical and assessment['conditions']['matched_full_day']),
            automatic_new_experiments=False, old_gate_reopened=False,
            stable_tail_gate='secondary_report_only',
            reason='fixed_development_cost_gates_pass' if passed else 'technical_incomplete_or_failed' if not technical else 'fixed_development_cost_gates_failed'),
        interpretation='Exposed6207/6208 development test, not independent confirmation. Same v75 initializers and paired native seeds; only singleton-F TD gradients into theta are blocked; v81 information and forward scores remain. Negative cost difference is improvement. No automatic retuning.',
        units='RawW/R passenger-seconds; J=(2W+R)/full-day departed cohort, including unfinished accrued costs.')
    return (main, dict(status=status,integrity=integrity,pairs=evidence,missing=missing,failures=failures),
        dict(status=status,cells=pools,information_contract='285covariates per arm; fresh separately285complete outcomes; issue context is newly observed, v79 diagnostic records excluded.'),
        dict(status=status,budget=budget,cells=training,new_initialization_updates=0,extra_first8_steps=0))


def export(root, out):
    values = analyze(root)
    out = Path(out)
    paths = (out,out.with_name('evidence.json'),out.with_name('pools.json'),out.with_name('training.json'))
    payloads = {path:old.encoded(value) for path,value in zip(paths,values)}
    for path,raw in payloads.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f'{path.name} exceeds48KB: {len(raw)}')
    out.parent.mkdir(parents=True,exist_ok=True)
    for path,raw in payloads.items():
        with path.open('xb') as handle:
            handle.write(raw)
    return dict(status=values[0]['status'],decision=values[0]['decision'],files={path.name:len(raw) for path,raw in payloads.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(export(HERE,args.out),allow_nan=False))
