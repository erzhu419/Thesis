"""Read the fixed two-root lane-RNG experiment; never fit or run a simulator."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import isclose, isfinite
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROLES = ('reference', 'candidate')
COSTS = ('wait', 'ride', 'cost')
ZERO = ('new_initialization_updates', 'prefix_training_updates', 'suffix_updates',
        'labels_fed_back', 'new_fits', 'new_neural_updates')


def finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


def passed(receipt, excluded=()):
    checks = receipt.get('checks', {}) if receipt else {}
    return bool(receipt and receipt.get('all_pass') is True and checks
                and all(value is True for key, value in checks.items() if key not in excluded))


def full_endpoint(ledger, remaining):
    if not ledger or not remaining:
        return None
    wait = ledger['accrued_wait_seconds'] + remaining['wait']
    ride = ledger['accrued_in_vehicle_seconds'] + remaining['ride']
    return dict(wait=wait, ride=ride, cost=2*wait+ride)


def pair_readout(source, pair, declared, future, protocol):
    remaining = pair.get('remaining', {})
    finite_labels = all(role in remaining and all(finite(remaining[role].get(key)) for key in COSTS)
                        for role in ROLES)
    delta = {key: remaining['candidate'][key]-remaining['reference'][key] for key in COSTS} \
        if finite_labels else None
    rng_checks, rng_readout = {}, {}
    for role in ROLES:
        rng, usage = pair.get('rng', {}).get(role, {}), pair.get('rng_usage', {}).get(role, {})
        changed = rng.get('changed_lane_pools', [])
        rng_checks[role] = bool(passed(rng) and rng.get('seed') == future['seed']
            and rng.get('mode') == future['mode'] and rng.get('lane_count') == protocol['rng']['pools']
            and rng.get('other_native_streams') == 'unchanged'
            and changed == ([] if future['seed'] is None else list(range(protocol['rng']['pools'])))
            and rng.get('burn_per_lane') == (0 if future['seed'] is None else protocol['rng']['burn_per_pool'])
            and usage.get('all_pass') is True and usage.get('native_clock') == 24000
            and finite(usage.get('total_draws')) and usage['total_draws'] > 0)
        rng_readout[role] = dict(mode=rng.get('mode'), seed=rng.get('seed'),
            changed_lane_pools=len(changed), pools_advanced=usage.get('pools_advanced'),
            total_draws=usage.get('total_draws'), checks=deepcopy(rng.get('checks', {})))
    endpoints = {role: full_endpoint(source.get('root_ledger'), remaining.get(role))
                 for role in ROLES} if finite_labels else None
    original = None
    if future['id'] == 'F0':
        expected = {role: full_endpoint(source.get('frozen_original_root_ledger'),
                    source.get('frozen_original_remaining', {}).get(role)) for role in ROLES}
        equal = {role: {key: bool(endpoints and endpoints[role] is not None and expected[role] is not None
                      and endpoints[role][key] == expected[role][key]) for key in COSTS} for role in ROLES}
        original = dict(expected_full_endpoint=expected, observed_full_endpoint=endpoints, checks=equal,
                        all_pass=all(value for arm in equal.values() for value in arm.values()))
    checks = dict(runtime_integrity=passed(pair, ('physical_departure_separation',)),
        declared_future=pair.get('future_id') == future['id'] and pair.get('future_seed') == future['seed'],
        finite_labels=finite_labels,
        mature_full_suffix=finite_labels and all(value.get('observed_at') == 24000
            and value.get('available_at') == 24030 and value.get('dt') == 24000-declared['clock']
            and value['wait'] >= 0 and value['ride'] >= 0 and value['cost'] == 2*value['wait']+value['ride']
            for value in remaining.values()),
        exact_delta=delta is not None and pair.get('delta') == delta,
        lane_rng_contract=all(rng_checks.values()),
        paired_native_state=all(pair.get('checks', {}).get(key) is True for key in (
            'reference_before_equals_parent', 'candidate_before_equals_parent',
            'paired_after_physical', 'paired_after_lane_rng', 'paired_after_other_rng')),
        original_endpoints=future['id'] != 'F0' or original['all_pass'],
        original_D=future['id'] != 'F0' or (delta is not None and isclose(
            delta['cost']/protocol['fixed_passenger_denominator'], declared['old_D'], rel_tol=1e-12, abs_tol=1e-12)))
    denominator = protocol['fixed_passenger_denominator']
    return dict(future_id=future['id'], future_seed=future['seed'], valid=all(checks.values()), checks=checks,
        remaining=deepcopy(remaining), full_endpoint=endpoints, delta=delta,
        delta_per_passenger=None if delta is None else {key: value/denominator for key, value in delta.items()},
        weighted_wait_per_passenger=None if delta is None else 2*delta['wait']/denominator,
        D=None if delta is None else delta['cost']/denominator,
        physical_noop=pair.get('physical_noop'), physical=deepcopy(pair.get('physical')),
        rng=rng_readout, F0_reproduction=original)


def analyze(collectors, protocol, probe):
    expected = {row['external_seed']: row for row in protocol['roots']}
    by_seed = {}
    for source in collectors:
        seed = source['external_seed']
        if seed not in expected or seed in by_seed:
            raise ValueError('Undeclared or repeated root; never substitute another root')
        by_seed[seed] = source
    roots, exceptions = [], []
    for seed, declared in expected.items():
        source = by_seed.get(seed)
        if source is None:
            roots.append(dict(external_seed=seed, case=declared['case'], status='missing',
                              complete=False, pairs=[], within_root=None, error='Missing root summary'))
            exceptions.append(dict(external_seed=seed, type='missing_root'))
            continue
        expected_root = dict(started_at=declared['clock'], sequence=declared['sequence'],
            line_id='7X', trip_id=declared['trip_id'], station_id=declared['station_id'])
        futures = {row['id']: row for row in protocol['futures']}
        pairs = {}
        for pair in source.get('pairs', []):
            fid = pair['future_id']
            if fid not in futures or fid in pairs:
                raise ValueError('Undeclared or repeated future; never replace a failed future')
            pairs[fid] = pair_readout(source, pair, declared, futures[fid], protocol)
        budget = source.get('budget', {})
        root_checks = source.get('root_checks', {})
        prefix = source.get('prefix_reproduction') or {}
        fixed_root = source.get('case') == declared['case'] and source.get('scenario') == declared['scenario'] \
            and source.get('root') == expected_root and bool(root_checks) and all(root_checks.values()) \
            and source.get('root_ledger') is not None \
            and source.get('root_ledger') == source.get('frozen_original_root_ledger') \
            and prefix.get('all_pass') is True \
            and prefix.get('matched_issues') == prefix.get('observed_issues') == declared['sequence']
        checks = dict(declared_root=fixed_root, runtime_complete=source.get('status') == 'complete'
            and passed(source.get('integrity')) and source.get('error') is None,
            three_valid_futures=set(pairs) == set(futures) and all(row['valid'] for row in pairs.values()),
            zero_learning=all(budget.get(key) == 0 for key in ZERO),
            exact_prefix_spend=budget.get('native_prefix_seconds') == declared['clock'],
            known_budget=budget.get('total_native_seconds_is_lower_bound') is False
                and budget.get('unknown_suffix_costs') == 0,
            budget_identity=all(finite(budget.get(key)) for key in (
                'total_native_seconds', 'native_prefix_seconds', 'suffix_native_seconds'))
                and budget['total_native_seconds'] == budget['native_prefix_seconds']+budget['suffix_native_seconds'],
            six_suffixes=budget.get('suffixes') == 6 and budget.get('labels_observed') == 6)
        good = [row for row in pairs.values() if row['valid'] and fixed_root]
        values = [row['D'] for row in good]
        complete = all(checks.values())
        within = dict(valid_futures=[row['future_id'] for row in good], complete=complete,
            minimum=min(values) if values else None, maximum=max(values) if values else None,
            span=max(values)-min(values) if values else None,
            mean=sum(values)/len(values) if values else None,
            positive=sum(value > 0 for value in values), negative=sum(value < 0 for value in values),
            zero=sum(value == 0 for value in values),
            sign_flip_observed=any(value < 0 for value in values) and any(value > 0 for value in values),
            robustness_established=False)
        roots.append(dict(external_seed=seed, case=declared['case'], scenario=source.get('scenario'),
            root=deepcopy(source.get('root')), status=source.get('status'), complete=complete, checks=checks,
            pairs=[pairs[fid] for fid in futures if fid in pairs],
            missing_futures=[fid for fid in futures if fid not in pairs], within_root=within,
            budget=deepcopy(budget), error=deepcopy(source.get('error'))))
        if not complete:
            exceptions.append(dict(external_seed=seed, type='incomplete_or_failed_root',
                error=deepcopy(source.get('error')), failed_checks=[key for key, value in checks.items() if not value]))
    budgets = [source.get('budget', {}) for source in collectors]
    known = sum(value['total_native_seconds'] for value in budgets if finite(value.get('total_native_seconds')))
    unknown_roots = len(expected)-len(by_seed)+sum(not finite(value.get('total_native_seconds')) for value in budgets)
    unknown_suffixes = sum(value.get('unknown_suffix_costs', 0) for value in budgets)
    lower_bound = bool(unknown_roots or unknown_suffixes or any(
        value.get('total_native_seconds_is_lower_bound', True) for value in budgets))
    probe_seconds = probe.get('native_seconds') if probe else None
    probe_ok = bool(probe and probe.get('status') == 'complete' and passed(probe)
                    and probe_seconds == protocol['native_seconds']['capability_probe'])
    formal = dict(known_native_seconds=known, native_seconds=None if lower_bound else known,
        is_lower_bound=lower_bound, unknown_roots=unknown_roots, unknown_suffix_costs=unknown_suffixes,
        cap=protocol['native_seconds']['formal_total'],
        within_cap=None if lower_bound else known <= protocol['native_seconds']['formal_total'])
    checks = dict(two_complete_roots=len(roots) == 2 and all(row['complete'] for row in roots),
        six_valid_pairs=sum(pair['valid'] for row in roots for pair in row['pairs']) == protocol['planned_pairs'],
        capability_probe=probe_ok, formal_budget_complete=not lower_bound,
        formal_budget_exact=not lower_bound and known == protocol['native_seconds']['formal_total'])
    complete = all(checks.values())
    observed_flip = any(row.get('within_root') and row['within_root']['sign_flip_observed'] for row in roots)
    learning_known = len(by_seed) == len(expected) and all(
        all(finite(value.get(key)) for key in ZERO) for value in budgets)
    return dict(status='complete' if complete else 'hold', integrity=dict(all_pass=complete, checks=checks),
        roots=roots, exceptions=exceptions,
        accounting=dict(formal=formal, capability_probe=dict(native_seconds=probe_seconds,
            all_pass=probe_ok, error=None if probe is None else probe.get('error')),
            combined_native_seconds=known+probe_seconds if not lower_bound and finite(probe_seconds) else None,
            planned_pairs=protocol['planned_pairs'], distinct_new_future_pairs=protocol['distinct_new_future_pairs'],
            completed_original_future_replications=sum(pair['future_id'] == 'F0' and pair['valid']
                for row in roots for pair in row['pairs']),
            learning_counts={key:sum(value[key] for value in budgets) if learning_known else None for key in ZERO}),
        decision='TECHNICAL_INCOMPLETE' if not complete else
            'OBSERVED_WITHIN_ROOT_SIGN_FLIP' if observed_flip else 'NO_FLIP_OBSERVED_IN_FIXED_FUTURES',
        interpretation=dict(scope='Two exposed physical roots,64 future lane RNG pools; other streams unchanged',
            F0='Identity adapter, no whole-state load; arm endpoint equivalence is reported per root',
            physical_roots=2, planned_suffixes=12, suffixes_are_not_independent_roots=True,
            no_flip_is_not_robustness=True, V_kept_sealed=True,
            no_automatic_refit_or_additional_futures=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--roots', type=Path, nargs='+')
    parser.add_argument('--probe', type=Path, default=HERE/'capability_ready.json')
    parser.add_argument('--out', type=Path, default=HERE/'summary.json')
    args = parser.parse_args(argv)
    load = lambda path: json.loads(path.read_text())
    protocol = load(args.protocol)
    paths = args.roots or [HERE/'roots'/str(row['external_seed'])/'summary.json' for row in protocol['roots']]
    result = analyze([load(path) for path in paths if path.exists()], protocol,
                     load(args.probe) if args.probe.exists() else None)
    with args.out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], decision=result['decision'], accounting=result['accounting'])))
    return result


if __name__ == '__main__':
    main()
