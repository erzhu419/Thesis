"""Read four restored entries and eight frozen continuations; no model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.analyze import (
    finite, near, integrity, overlap, fork_checks, read)

SEEDS = (6208, 6207)
ENTRIES = ('fresh', 'no_fresh')
BRANCHES = {'low': 'fresh', 'high': 'no_fresh'}
PERIODS = ('remaining', 'tail', 'post20100')
DENOMINATOR = 15079
COUNTS = dict(critic=408, value=408, actor=408, alpha=408, policy=408, world=0, trust=0)
COST_FIELDS = ('wait', 'ride', 'cost', 'cost_per_passenger')
MAX_BYTES = 48000


def actor_checks(row, seed, policy):
    value = row.get('actor_snapshot', {})
    return dict(snapshot_seed=value.get('seed') == seed,
        snapshot_arm=value.get('source_arm') == policy,
        snapshot_packet51=value.get('packets') == 51,
        snapshot_counts408=value.get('counts') == COUNTS,
        snapshot_covariates255=value.get('covariates') == 255,
        snapshot_outcomes255_or0=value.get('complete_outcomes') == (255 if policy == 'fresh' else 0),
        snapshot_deadline16200=value.get('publication_deadline') == 16200,
        snapshot_data_deadline16200=value.get('data_deadline') == 16200,
        snapshot_labels_available_by16200=finite(value.get('max_available_at')) and value['max_available_at'] <= 16200,
        snapshot_after_deadline=finite(value.get('published_clock')) and value['published_clock'] >= 16200,
        snapshot_actual_processing_clock=value.get('processed_at') == value.get('published_clock'),
        snapshot_no_future_training=value.get('future_training_used') is False)


def prefix_checks(row, seed, entry):
    budget, cell = row.get('budget', {}), row.get('cell', {})
    checks = dict(complete=row.get('status') == 'complete', native_integrity=integrity(row),
        correct_cell=cell.get('seed') == seed and cell.get('scenario') == 'stable'
            and cell.get('entry_arm') == entry,
        restored_covariates255=budget.get('restoration_feedback_slots') == 255,
        restored_outcomes=budget.get('restoration_complete_outcomes') == (255 if entry == 'fresh' else 0),
        restored_packet51=budget.get('restoration_packets') == 51,
        cutpoint=finite(row.get('prefix_clock')) and 16200 <= row['prefix_clock'] < 24000,
        native_prefix_budget=budget.get('native_prefix_seconds') == row.get('prefix_clock'),
        full_prefix_exact=type(row.get('trace_rows')) is int and row['trace_rows'] > 0
            and row.get('matched_prefix_rows') == row['trace_rows'],
        actual_cut_snapshot=row.get('cut_snapshot', {}).get('time_seconds') == row.get('prefix_clock'),
        own_snapshot_before_cut=finite(row.get('actor_snapshot', {}).get('published_clock'))
            and finite(row.get('prefix_clock')) and row['actor_snapshot']['published_clock'] <= row['prefix_clock'],
        initial50000=row.get('initializer', {}).get('initial_updates') == 50000,
        same_external_seed=row.get('external_seed') == seed+100000)
    checks.update({key+'408': budget.get('restoration_'+key+'_updates') == 408
                   for key in ('critic', 'value', 'actor', 'alpha')})
    checks.update(actor_checks(row, seed, entry))
    return checks


def endpoint(row, period):
    points = row.get('endpoints', {})
    if period == 'remaining':
        return row.get('cut_snapshot', {}), points.get('24000', {}), row.get('switch_time')
    start = 18300 if period == 'tail' else 20100
    return points.get(str(start), {}), points.get('24000', {}), start


def window_checks(row, period):
    lo, hi, start = endpoint(row, period)
    cost = row.get('cost_windows', {}).get(period, {})
    ready = all(finite(cost.get(k)) for k in COST_FIELDS)
    points = all(all(finite(point.get(k)) for k in ('time_seconds', 'accrued_wait_seconds',
        'accrued_in_vehicle_seconds', 'accrued_generalized_passenger_seconds',
        'waiting_passengers', 'in_vehicle_passengers')) for point in (lo, hi))
    return dict(finite=ready and points,
        true_native_endpoints=lo.get('time_seconds') == start and hi.get('time_seconds') == 24000,
        fixed_cohort=cost.get('denominator') == DENOMINATOR,
        weighted_cost=ready and near(cost['cost'], 2*cost['wait']+cost['ride']),
        accrued_wait=ready and points and near(cost['wait'], hi['accrued_wait_seconds']-lo['accrued_wait_seconds']),
        accrued_ride=ready and points and near(cost['ride'], hi['accrued_in_vehicle_seconds']-lo['accrued_in_vehicle_seconds']),
        accrued_cost=ready and points and near(cost['cost'], hi['accrued_generalized_passenger_seconds']-lo['accrued_generalized_passenger_seconds']),
        normalization=ready and near(cost['cost_per_passenger'], cost['cost']/DENOMINATOR))


def suffix_checks(row, seed, entry, policy, prefix):
    cell = row.get('cell', {})
    post = row.get('post_budget_counts', {})
    checks = dict(complete=row.get('status') == 'complete', native_integrity=integrity(row),
        correct_cell=cell.get('seed') == seed and cell.get('scenario') == 'stable'
            and cell.get('entry_arm') == entry and cell.get('policy_arm') == policy,
        same_entry_clock=row.get('switch_time') == prefix.get('prefix_clock'),
        same_entry_ledger=row.get('cut_snapshot') == prefix.get('cut_snapshot'),
        same_initializer=row.get('initializer') == prefix.get('initializer'),
        inherited_external_seed=row.get('external_seed') == prefix.get('external_seed') == seed+100000,
        frozen_counts=post.get('before') == post.get('after') == COUNTS,
        suffix_zero_training=row.get('zero_suffix_training') is True,
        exact_suffix_duration=near(row.get('budget', {}).get('suffix_native_seconds'), 24000-prefix.get('prefix_clock', 24000)),
        final_waiting=row.get('final_waiting') == row.get('endpoints', {}).get('24000', {}).get('waiting_passengers'),
        original_v75_suffix_not_required=row.get('diagonal_reference', {}).get('requires_original_suffix_exact') is False)
    full = full_day_cost(row)
    final = row.get('endpoints', {}).get('24000', {})
    checks.update(full_day_cohort=full['denominator'] == DENOMINATOR,
        full_day_weighted_cost=near(full['cost'], 2*full['wait']+full['ride']) if finite(full['wait']) and finite(full['ride']) else False,
        full_day_wait_ledger=near(full['wait'], final.get('accrued_wait_seconds')),
        full_day_ride_ledger=near(full['ride'], final.get('accrued_in_vehicle_seconds')),
        full_day_cost_ledger=near(full['cost'], final.get('accrued_generalized_passenger_seconds')),
        full_day_normalization=near(full['cost_per_passenger'], full['cost']/DENOMINATOR) if finite(full['cost']) else False)
    checks.update(actor_checks(row, seed, policy))
    for period in PERIODS:
        checks.update({period+'/'+key: value for key, value in window_checks(row, period).items()})
    return checks


def difference(left, right):
    return {key: left[key]-right[key] for key in COST_FIELDS}


def full_day_cost(row):
    cost = row.get('cost', {})
    return dict(wait=cost.get('accrued_wait_seconds'), ride=cost.get('accrued_in_vehicle_seconds'),
        cost=cost.get('generalized_passenger_seconds'), denominator=cost.get('departed_passengers'),
        cost_per_passenger=cost.get('generalized_seconds_per_departed_passenger'))


def comparisons(cells):
    result = {}
    for entry in ENTRIES:
        left, right = (cells[entry][policy] for policy in ENTRIES)
        actions = [row.get('first_issue', {}).get('executed_a') for row in (left, right)]
        result[entry] = dict(switch_time=left['switch_time'],
            policy_fresh_minus_no_fresh={period: difference(left['cost_windows'][period], right['cost_windows'][period])
                for period in PERIODS},
            waiting_stock_difference={clock: left['endpoints'][clock]['waiting_passengers']-
                right['endpoints'][clock]['waiting_passengers'] for clock in ('18300', '20100', '24000')},
            first_action_executed_equal=actions[0] == actions[1] if all(value is not None for value in actions) else None)
    return result


def decision(seed_rows, technical):
    if not technical:
        return dict(reason='incomplete_or_integrity_failed', baseline_protection_may_be_considered=False,
            new_training_authorized_by_diagnostic=False, automatic_seed_expansion=False)
    by_seed = {row['seed']: row for row in seed_rows}
    bad = [by_seed[6208]['comparisons'][entry]['policy_fresh_minus_no_fresh']['remaining']['cost'] for entry in ENTRIES]
    good = [by_seed[6207]['comparisons'][entry]['policy_fresh_minus_no_fresh']['remaining']['cost'] for entry in ENTRIES]
    full = by_seed[6207]['full_day_frozen_family']
    handoff_difference = difference(full['fresh']['no_fresh'], full['no_fresh']['no_fresh'])
    bad_rescued = all(value > 0 for value in bad)
    good_preserved = handoff_difference['cost'] <= 0
    supported = bad_rescued and good_preserved
    reason = 'may_consider_single_takeover_protection' if supported else (
        'no_consistent_bad_case_rescue' if not bad_rescued else 'positive_case_prefix_gain_not_preserved')
    return dict(reason=reason, baseline_protection_may_be_considered=supported,
        bad_case_nofresh_remaining_better_both_entries=bad_rescued,
        positive_case_prefix_gain_preserved_after_nofresh_handoff=good_preserved,
        positive_case_full_day_difference_with_nofresh_handoff=handoff_difference,
        positive_case_remaining_fresh_actor_not_worse_by_entry=dict(zip(ENTRIES, [value <= 0 for value in good])),
        new_training_authorized_by_diagnostic=False, automatic_seed_expansion=False,
        old_v73_gate_reopened=False, original_v75_continuation_causal_decomposition=False)


def analyze(root=HERE):
    root = Path(root)
    checks, missing, failures, cells_out, evidence, seeds = {}, [], [], [], [], []
    prefix_seconds = suffix_seconds = 0.
    prefix_count = suffix_count = 0
    for seed in SEEDS:
        cells, seed_keys, seed_prefixes = {entry: {} for entry in ENTRIES}, [], []
        for entry in ENTRIES:
            name = f'seed{seed}/{entry}'
            base = root/'runs'/f'seed{seed}'/entry
            prefix = read(base/'prefix_summary.json')
            if prefix is None:
                missing.append(name+'/prefix_summary.json')
                failed = read(base/'failure.json')
                if failed: failures.append(dict(cell=name, failure=failed))
                continue
            prefix_count += 1
            duration = prefix.get('budget', {}).get('native_prefix_seconds')
            if finite(duration): prefix_seconds += duration
            entry_checks = prefix_checks(prefix, seed, entry)
            seed_prefixes.append(prefix)
            branches = {}
            for branch, policy in BRANCHES.items():
                path = base/'suffixes'/branch/'summary.json'
                row = read(path)
                if row is None:
                    missing.append(name+'/suffixes/'+branch+'/summary.json')
                    failed = read(path.with_name('failure.json'))
                    if failed: failures.append(dict(cell=name+'/'+branch, failure=failed))
                    continue
                suffix_count += 1
                duration = row.get('budget', {}).get('suffix_native_seconds')
                if finite(duration): suffix_seconds += duration
                branches[branch] = row
                cells[entry][policy] = row
                entry_checks.update({branch+'/'+key: value for key, value in suffix_checks(row, seed, entry, policy, prefix).items()})
                cells_out.append(dict(seed=seed, entry_arm=entry, policy_arm=policy,
                    switch_time=row.get('switch_time'), cost_windows=row.get('cost_windows'),
                    endpoints=row.get('endpoints'), final_waiting=row.get('final_waiting'),
                    full_day_cost=full_day_cost(row), zero_suffix_training=row.get('zero_suffix_training')))
            live = read(base/'suffixes/live_fork.json')
            if live is None:
                missing.append(name+'/suffixes/live_fork.json')
            elif len(branches) == 2:
                entry_checks.update({'fork/'+key: value for key, value in fork_checks(live, prefix, branches).items()})
                entry_checks['same_complete_cut_ledger'] = branches['low']['cut_snapshot'] == branches['high']['cut_snapshot'] == prefix.get('cut_snapshot')
            evidence.append(dict(seed=seed, entry_arm=entry, prefix_clock=prefix.get('prefix_clock'),
                prefix_trace_rows=prefix.get('trace_rows'), matched_prefix_rows=prefix.get('matched_prefix_rows'),
                prefix_checks=entry_checks, budget=prefix.get('budget'), initializer=prefix.get('initializer'),
                actor_snapshots={policy: row.get('actor_snapshot') for policy, row in cells[entry].items()},
                fork_overlap_seconds=overlap(live.get('children', {})) if live else None,
                first_issues={policy: row.get('first_issue') for policy, row in cells[entry].items()}))
            for key, value in entry_checks.items():
                full = name+'/'+key
                checks[full] = value is True
                seed_keys.append(full)
        if len(seed_prefixes) == 2:
            key = f'seed{seed}/same_initializer_across_entries'
            checks[key] = seed_prefixes[0].get('initializer') == seed_prefixes[1].get('initializer')
            seed_keys.append(key)
        ready = all(len(cells[entry]) == 2 for entry in ENTRIES) and bool(seed_keys) and all(checks[key] for key in seed_keys) \
            and not any(path.startswith(f'seed{seed}/') for path in missing) \
            and not any(row['cell'].startswith(f'seed{seed}/') for row in failures)
        seeds.append(dict(seed=seed, role='bad_case' if seed == 6208 else 'positive_case',
            status='complete' if ready else 'unscored', comparisons=comparisons(cells) if ready else None,
            full_day_frozen_family={entry: {policy: full_day_cost(cells[entry][policy]) for policy in ENTRIES}
                for entry in ENTRIES} if ready else None))
    documented = prefix_seconds+suffix_seconds
    checks['fixed_prefix_count4'] = prefix_count == 4
    checks['fixed_suffix_count8'] = suffix_count == 8
    checks['documented_simulation_bound'] = documented <= 127200
    technical = prefix_count == 4 and suffix_count == 8 and not missing and not failures and bool(checks) and all(checks.values())
    failed_checks = [key for key, value in checks.items() if not value]
    status = 'complete' if technical else 'integrity_failed' if failed_checks or failures else 'pending'
    actual_counts = {family: [row.get('budget', {}).get('restoration_'+family+'_updates') for row in evidence]
                     for family in ('critic', 'value', 'actor', 'alpha')}
    covariates = [row.get('budget', {}).get('restoration_feedback_slots') for row in evidence]
    outcomes = [row.get('budget', {}).get('restoration_complete_outcomes') for row in evidence]
    budget = dict(documented_prefixes=prefix_count, documented_suffixes=suffix_count,
        planned_restoration_covariate_slots=1020, planned_restoration_complete_fresh_outcomes=510,
        planned_restoration_updates_per_family=1632,
        restoration_covariate_slots=sum(covariates) if len(covariates) == 4 and all(finite(x) for x in covariates) else None,
        restoration_complete_fresh_outcomes=sum(outcomes) if len(outcomes) == 4 and all(finite(x) for x in outcomes) else None,
        restoration_updates_per_family={family: sum(values) if len(values) == 4 and all(finite(x) for x in values) else None
                                        for family, values in actual_counts.items()},
        documented_prefix_seconds=prefix_seconds, documented_suffix_seconds=suffix_seconds,
        documented_total_native_seconds=documented,
        actual_total_native_seconds=documented if technical else None,
        maximum_native_seconds=127200, new_initialization_updates=0,
        suffix_training_updates=0 if technical else None, diagnostic_truth_entered_training=False if technical else None,
        failed_work=failures, accounting='Documented summaries only; missing or failed work is retained and not encoded as zero.')
    out = dict(version='v76', status=status, complete=bool(technical),
        metric='J=(2W+R)/15079; positive fresh-policy-minus-no_fresh-policy is worse.',
        window_scope='remaining uses each entry actual native cut->24000; tail18300 and post20100 share physical clocks.',
        cells=cells_out, seeds=seeds, budget=budget,
        integrity=dict(all_pass=bool(technical), checked_conditions=len(checks), failed_checks=failed_checks),
        missing=missing, failures=failures, decision=decision(seeds, technical),
        limitations=['Four entry states are from two exposed stable cases, not independent confirmation.',
            'Actors freeze after408 updates; original v75 continues to456, so frozen diagonals are new interventions.',
            'Same inherited entry/exogenous seed does not imply semantic equality of all later path-dependent random draws.',
            'Different entry cut clocks do not define one common16200 state or an exact original-v75 causal decomposition.'])
    detail = dict(version='v76', status=status, integrity=out['integrity'], entries=evidence,
        budget=budget, source='v75 exact restoration to packet51 only; no source continuation replay required',
        new_algorithm_started=False, old_code_and_results_modified=False)
    return out, detail


def export(root, out):
    results = analyze(root)
    paths = (Path(out), Path(out).with_name('evidence.json'))
    payloads = {path: (json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()
                for path, result in zip(paths, results)}
    for path, raw in payloads.items():
        if len(raw) > MAX_BYTES: raise ValueError(f'{path.name} exceeds48KB')
        if path.exists(): raise FileExistsError(path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for path, raw in payloads.items():
        with path.open('xb') as handle: handle.write(raw)
    return dict(status=results[0]['status'], decision=results[0]['decision'], files={p.name: len(raw) for p, raw in payloads.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.out), ensure_ascii=False))
