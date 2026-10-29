"""Fixed twelve-day validation of frozen v73, with initialization paid separately."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.analyze import native_checks
from experiments.mcwm_closed_loop_recent30_mixed_cql_v70_20261026 import metrics as old

ARMS = ('fresh', 'no_fresh')
PERIODS = old.PERIODS
MAX_BYTES = 48000


def run_dir(root, seed, scene, arm):
    return Path(root)/'runs/executed_cql/validation'/f'seed{seed}'/scene/arm


def initializer_checks(value, seed):
    value = value or {}
    return dict(complete=value.get('status') == 'complete' and value.get('all_pass') is True,
        correct_seed=value.get('seed') == seed, offline50000=value.get('offline_updates') == 50000,
        no_online_updates=value.get('online_updates') == 0, original_recipe=value.get('recipe_exact') is True,
        saved_initializer=bool(value.get('initializer_path')))


def statistics(cells, protocol):
    return {period: dict(overall=old.group(cells, 'matched', period,
        len(protocol['development_seeds'])*len(protocol['scenarios'])),
        by_scenario={scene: old.group([row for row in cells if row['scenario'] == scene],
            'matched', period, len(protocol['development_seeds'])) for scene in protocol['scenarios']})
        for period, _ in PERIODS}


def seed_readout(cells, protocol):
    """Each training seed contributes one scenario-averaged paired observation."""
    rows = []
    for seed in protocol['development_seeds']:
        selected = [row for row in cells if row['identity'] == seed]
        periods = {}
        for period, _ in PERIODS:
            values = [row['comparisons']['matched'][period] for row in selected]
            complete = len(selected) == len(protocol['scenarios']) and all(
                value['status'] == 'complete' for value in values)
            periods[period] = dict(status='complete' if complete else 'missing',
                paired_scenario_mean=(sum(value['candidate_minus_baseline']
                    ['generalized_seconds_per_departed_passenger'] for value in values)/len(values)
                    if complete else None))
        rows.append(dict(seed=seed, periods=periods))
    full = [row['periods']['full_day']['paired_scenario_mean'] for row in rows
            if row['periods']['full_day']['status'] == 'complete']
    return dict(independent_unit='training seed; stable and shock averaged within seed', rows=rows,
        expected_seeds=len(protocol['development_seeds']), complete_seeds=len(full),
        mean_paired_difference=sum(full)/len(full) if full else None,
        improved_seeds=sum(x < 0 for x in full), degraded_seeds=sum(x > 0 for x in full),
        interpretation='Six condition cells are not six independent training seeds; observed win fractions are not future probabilities.')


def initial_budget(gates):
    valid = len(gates) == 3 and all(all(initializer_checks(value, seed).values())
        for seed, value in gates.items())
    return dict(planned_initializers=3, planned_offline_updates=150000,
        completed_initializers=sum(all(initializer_checks(value, seed).values()) for seed, value in gates.items()),
        actual_offline_updates=sum(value['offline_updates'] for value in gates.values()) if valid else None,
        actual_online_updates=sum(value['online_updates'] for value in gates.values()) if valid else None,
        complete=valid)


def pool_view(summary):
    replay = summary.get('replay', {})
    fields = ('arm', 'covariate_records_received', 'outcome_labels_received', 'covariate_unique_records',
        'outcome_unique_records', 'distinct_target_records_used', 'selected_target_record_union',
        'outcome_state_record_overlap', 'covariate_rows_only_sa', 'complete_outcome_rows',
        'fresh_current_s_a_use', 'fresh_outcome_use', 'critic_fresh_outcome_window',
        'independent_outcome_sampler_seed', 'independent_outcome_sampler_calls',
        'maximum_fresh_outcome_age_in_admitted_labels')
    selection = summary.get('capacity_selection', {})
    return dict(replay={key: replay.get(key) for key in fields}, coverage={key: selection.get(key) for key in
        ('packets', 'selected_records', 'eligible_records', 'eligible_risk_records', 'risk_windows',
         'original_covered_risk_windows', 'selected_covered_risk_windows', 'original_selected_risk_records',
         'selected_risk_records', 'replacements', 'max_replacements_per_packet', 'outcome_fields_used_for_ranking')})


def analyze(root=HERE):
    root = Path(root)
    protocol = old.read(root/'protocol.json')
    reference = root.parent/'mcwm_closed_loop_decoupled_feedback_v73_20261026/runs/executed_cql/development/seed6201/stable/fresh'
    frozen, frozen_config = old.read(reference/'summary.json'), old.read(reference/'protocol.json')
    checks, missing, failures, cells, evidence, pools, training = {}, [], [], [], [], [], []
    gates = {seed: old.read(root/f'inputs/seed{seed}/initializer_gate.json')
             for seed in protocol['development_seeds']}
    for seed, gate in gates.items():
        if gate is None:
            missing.append(f'initializer/seed{seed}')
        checks.update({f'initializer/{seed}/{key}': flag for key, flag in initializer_checks(gate, seed).items()})
    checks['frozen_v73_reference_available'] = frozen is not None and frozen_config is not None
    for seed in protocol['development_seeds']:
        for scene in protocol['scenarios']:
            name = f'seed{seed}/{scene}'
            runs, points, configs, dirs, preflights = {}, {}, {}, {}, {}
            for arm in ARMS:
                directory = run_dir(root, seed, scene, arm)
                summary = old.read(directory/'summary.json')
                if summary is None:
                    failure = old.read(directory/'failure.json')
                    (failures if failure else missing).append(dict(cell=name+'/'+arm,
                        failure=failure) if failure else name+'/'+arm)
                    continue
                config = old.read(directory/'protocol.json') or {}
                preflight = old.read(directory/'preflight.json') or {}
                configs[arm], preflights[arm], runs[arm], dirs[arm] = config, preflight, summary, directory
                points[arm] = old.read_metrics(directory, summary)
                audit = native_checks(summary, config, seed=seed, scene=scene, arm=arm, current=True)
                gate = gates[seed] or {}
                audit.update(initializer_path=summary.get('initializer', {}).get('path') == gate.get('initializer_path'),
                    first_packet_preflight=preflight.get('technical_gate_pass') is True,
                    native_recipe=summary.get('cell', {}).get('recipe') == protocol['recipe'],
                    external_seed=summary.get('cell', {}).get('external_seed') == seed+protocol['external_seed_offset'],
                    departed_cohort=summary.get('cost', {}).get('departed_passengers') == protocol['full_day_departed_passengers'],
                    complete_cost_endpoints=all(old.window(summary, points[arm], *bounds)['status'] == 'complete'
                        for _, bounds in PERIODS),
                    frozen_v73_F_and_runtime=frozen is not None and old._runtime(summary) == old._runtime(frozen),
                    frozen_v73_policy_config=frozen_config is not None
                        and config.get('policy_config') == frozen_config.get('policy_config'))
                final = {row['time_seconds']: row for row in points[arm]}.get(24000, {})
                audit['final_metrics_match_ledger'] = all(final.get(field) == summary.get('cost', {}).get(field)
                    for field in ('accrued_wait_seconds', 'accrued_in_vehicle_seconds'))
                checks.update({name+'/'+arm+'/'+key: flag is True for key, flag in audit.items()})
                pools.append(dict(seed=seed, scenario=scene, arm=arm, **pool_view(summary)))
                schedule = summary.get('actor_schedule_audit', {})
                training.append(dict(seed=seed, scenario=scene, arm=arm, budget=summary.get('budget'),
                    schedule={key: schedule.get(key) for key in ('packets', 'critic_updates', 'value_updates',
                        'actor_updates', 'alpha_updates', 'schedule', 'optimizer_order_exact')},
                    post_budget_counts=summary.get('post_budget_counts'), wall_seconds=summary.get('wall_seconds'),
                    extra_CQL_wall_seconds=summary.get('cql_audit', {}).get('regularizer_wall_seconds')))
            if len(runs) != 2:
                continue
            first = [runs[arm].get('clock_audit', {}).get('first_feedback_applied_at') for arm in ARMS]
            finite_first = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in first)
            prefix = old.compare_prefix(dirs['fresh']/'prefix_trace.jsonl', dirs['no_fresh']/'prefix_trace.jsonl',
                min(first) if finite_first else 0.)
            overlap = old._overlap(runs['fresh'], runs['no_fresh'])
            paired = dict(common_prefix=finite_first and len(set(first)) == 1 and prefix['all_pass'],
                same_initializer=runs['fresh'].get('initializer') == runs['no_fresh'].get('initializer'),
                same_policy_config=configs['fresh'].get('policy_config') == configs['no_fresh'].get('policy_config'),
                same_F_runtime=old._runtime(runs['fresh']) == old._runtime(runs['no_fresh']),
                actual_parallel=overlap['positive_overlap'])
            for key in ('initializer', 'first_feedback', 'execution_contract'):
                values = [preflights[arm].get(key) for arm in ARMS]
                paired['preflight_'+key] = bool(values[0]) and values[0] == values[1]
            checks.update({name+'/'+key: flag is True for key, flag in paired.items()})
            periods, comparisons = {}, {'matched': {}}
            for period, bounds in PERIODS:
                periods[period] = {arm: old.window(runs[arm], points[arm], *bounds) for arm in ARMS}
                comparisons['matched'][period] = old.difference(periods[period]['fresh'], periods[period]['no_fresh'])
            cells.append(dict(identity=seed, scenario=scene, periods=periods, comparisons=comparisons))
            evidence.append(dict(seed=seed, scenario=scene, checks=paired, pre_update_prefix=prefix,
                actual_parallel=overlap, initializer=runs['fresh'].get('initializer')))
    expected = len(protocol['development_seeds'])*len(protocol['scenarios'])
    technical = len(cells) == expected and not missing and not failures and bool(checks) and all(checks.values())
    status = 'complete' if technical else 'integrity_failed' if failures or any(not v for v in checks.values()) else 'pending'
    aggregates = statistics(cells, protocol)
    passed = technical and old.cost_gate(aggregates)
    all_runs = len(training) == 12
    pools_complete = len(pools) == 12
    budget = dict(initialization=initial_budget(gates), response=dict(planned_native_days=12,
        planned_native_seconds=288000, planned_covariate_slots=3420, planned_complete_fresh_results=1710,
        planned_updates_per_optimizer=5472, native_branches=0, cached_cost_baseline_days=0,
        actual_covariate_slots=sum(row['replay']['covariate_records_received'] for row in pools) if pools_complete else None,
        actual_complete_fresh_results=sum(row['replay']['outcome_labels_received'] for row in pools) if pools_complete else None,
        actual_distinct_target_records=sum(row['replay']['distinct_target_records_used'] for row in pools) if pools_complete else None,
        actual_native_seconds=sum(row['budget']['native_seconds'] for row in training) if all_runs else None,
        actual_updates_per_optimizer={key: sum(row['budget'][key+'_updates'] for row in training)
            if all_runs else None for key in ('critic', 'value', 'actor', 'alpha')}))
    main = dict(status=status, complete=technical, expected_condition_pairs=expected, available_condition_pairs=len(cells),
        integrity=dict(all_pass=technical, checked_conditions=len(checks), failed_checks=[key for key, value in checks.items() if not value]),
        missing=missing, failures=failures, recipe=protocol['recipe'], aggregates=aggregates,
        independent_seed_readout=seed_readout(cells, protocol), cells=[dict(identity=row['identity'], scenario=row['scenario'],
            periods={period: dict(arms={arm: old.cost_view(value) for arm, value in row['periods'][period].items()},
                matched=old.compact_comparison(row['comparisons']['matched'][period])) for period, _ in PERIODS}) for row in cells],
        budget=budget, decision=dict(scientific_gate_pass=bool(passed), primary_full_day_gate_pass=bool(passed),
            automatic_new_experiments=False, old_v73_gate_reopened=False, stable_tail_gate='secondary_report_only_in_this_new_study',
            reason='fixed_full_day_validation_pass' if passed else 'technical_incomplete_or_failed' if not technical else 'fixed_full_day_validation_failed'),
        interpretation='New training seeds6206–6208; fixedv73 algorithm. Each seed averages stable and shock. Full-day primary, original stable18300–24000 tail fully reported as secondary. Oldv73 failure remains unchanged.',
        units='Raw W/R passenger-seconds; J=(2W+R)/full-day departed cohort, including unfinished-trip accrued costs.')
    evidence_out = dict(status=status, integrity=main['integrity'], initializers=gates,
        pairs=evidence, missing=missing, failures=failures, fixed_v73_reference=str(reference))
    pools_out = dict(status=status, information_contract='Each fresh day285covariates plus285complete factual outcomes; actual overlap/union reported, cap342. no_fresh uses285s/a and zero new complete outcomes.', cells=pools)
    training_out = dict(status=status, budget=budget, cells=training,
        separate_initialization_budget=True, initializers=gates, additional_diagnostic_branches=0)
    return main, evidence_out, pools_out, training_out


def export(root, out):
    results = analyze(root)
    out = Path(out)
    paths = (out, out.with_name('evidence.json'), out.with_name('pools.json'), out.with_name('training.json'))
    payloads = {path: old.encoded(value) for path, value in zip(paths, results)}
    for path, raw in payloads.items():
        if len(raw) > MAX_BYTES:
            raise ValueError(f'{path.name} exceeds48KB: {len(raw)}')
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in payloads.items():
        with path.open('xb') as handle:
            handle.write(raw)
    return dict(status=results[0]['status'], decision=results[0]['decision'], files={p.name: len(raw) for p, raw in payloads.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(HERE, args.out), allow_nan=False))
