"""Read the frozen 12-root, six-future label diagnostic; never fit or simulate."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, variance


HERE = Path(__file__).resolve().parent
ROLES = ('reference', 'candidate')
COSTS = ('wait', 'ride', 'cost')
ZERO = ('new_initialization_updates', 'prefix_training_updates', 'suffix_updates',
        'labels_fed_back', 'new_fits', 'new_neural_updates')
NATIVE_PAIR_CHECKS = ('reference_before_equals_parent', 'candidate_before_equals_parent',
                      'paired_after_physical', 'paired_after_lane_rng', 'paired_after_other_rng')


def finite(value):
    return type(value) in (int, float) and isfinite(value)


def passed(receipt, excluded=()):
    checks = receipt.get('checks', {}) if receipt else {}
    return bool(receipt and receipt.get('all_pass') is True and checks
                and all(value is True for key, value in checks.items() if key not in excluded))


def sign(value):
    return (value > 0) - (value < 0)


def covariance(left, right):
    if len(left) < 2:
        return None
    a, b = mean(left), mean(right)
    return sum((x-a)*(y-b) for x, y in zip(left, right))/(len(left)-1)


def pair_readout(pair, declared, future, protocol):
    remaining = pair.get('remaining', {})
    finite_labels = all(role in remaining and all(finite(remaining[role].get(k)) for k in COSTS)
                        for role in ROLES)
    delta = {k: remaining['candidate'][k]-remaining['reference'][k] for k in COSTS} \
        if finite_labels else None
    rng_checks = {}
    for role in ROLES:
        rng, usage = pair.get('rng', {}).get(role, {}), pair.get('rng_usage', {}).get(role, {})
        rng_checks[role] = bool(passed(rng) and rng.get('seed') == future['seed']
            and rng.get('mode') == future['mode'] and rng.get('lane_count') == protocol['rng']['pools']
            and rng.get('changed_lane_pool_count') == protocol['rng']['pools']
            and rng.get('all_declared_lane_pools_changed') is True
            and rng.get('burn_per_lane') == protocol['rng']['burn_per_pool']
            and rng.get('other_native_streams') == 'unchanged'
            and usage.get('all_pass') is True and usage.get('native_clock') == protocol['horizon']
            and finite(usage.get('total_draws')) and usage['total_draws'] > 0)
    actions = pair.get('action', {})
    checks = dict(runtime_integrity=passed(pair, ('physical_departure_separation',)),
        declared_future=pair.get('future_id') == future['id'] and pair.get('future_seed') == future['seed'],
        original_actions=actions.get('reference', {}).get('hold_seconds') == declared['reference_hold_seconds']
            and actions.get('candidate', {}).get('hold_seconds') == declared['candidate_hold_seconds'],
        finite_labels=finite_labels,
        mature_full_suffix=finite_labels and all(remaining[role].get('observed_at') == protocol['horizon']
            and remaining[role].get('available_at') == protocol['label_available_at']
            and remaining[role].get('dt') == protocol['horizon']-declared['root']['started_at']
            and remaining[role]['wait'] >= 0 and remaining[role]['ride'] >= 0
            and remaining[role]['cost'] == 2*remaining[role]['wait']+remaining[role]['ride'] for role in ROLES),
        exact_delta=delta is not None and pair.get('delta') == delta,
        lane_rng_contract=all(rng_checks.values()),
        paired_native_state=all(pair.get('checks', {}).get(k) is True for k in NATIVE_PAIR_CHECKS))
    denominator = protocol['fixed_passenger_denominator']
    return dict(future_id=future['id'], future_seed=future['seed'], valid=all(checks.values()), checks=checks,
        remaining=deepcopy(remaining), delta=delta,
        delta_per_passenger=None if delta is None else {k: value/denominator for k, value in delta.items()},
        weighted_wait_per_passenger=None if delta is None else 2*delta['wait']/denominator,
        D=None if delta is None else delta['cost']/denominator,
        physical_noop=pair.get('physical_noop'), physical=deepcopy(pair.get('physical')),
        error=deepcopy(pair.get('error')))


def root_statistics(pairs, prediction, protocol):
    good = [pair for pair in pairs if pair['valid']]
    values = [pair['D'] for pair in good]
    n = len(values)
    average = mean(values) if n else None
    sample_variance = variance(values) if n >= 2 else None
    sd = sqrt(sample_variance) if sample_variance is not None else None
    se = sd/sqrt(n) if sd is not None else None
    complete = n == protocol['replicates_per_root']
    st = protocol['statistics']
    interval = [average-st['pointwise_t_critical']*se, average+st['pointwise_t_critical']*se] \
        if complete else None
    upper = average+st['bonferroni_t_critical']*se if complete else None
    by_id = {pair['future_id']: pair['D'] for pair in good}
    halves = []
    for ids in (('R1', 'R2', 'R3'), ('R4', 'R5', 'R6')):
        halves.append(mean(by_id[k] for k in ids) if all(k in by_id for k in ids) else None)
    error = prediction-average if finite(prediction) and average is not None else None
    return dict(n=n, complete_six=complete, mean_D=average, sample_variance=sample_variance,
        single_label_sd=sd, mean_SE=se, pointwise_95_interval=interval,
        simultaneous_one_sided_95_upper=upper, bound_below_zero=upper < 0 if upper is not None else None,
        minimum=min(values) if n else None, maximum=max(values) if n else None,
        positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values), zero=sum(x == 0 for x in values),
        observed_sign_flip=any(x < 0 for x in values) and any(x > 0 for x in values),
        mean_delta_per_passenger={k: mean(p['delta_per_passenger'][k] for p in good) for k in COSTS} if n else None,
        mean_weighted_wait_per_passenger=mean(p['weighted_wait_per_passenger'] for p in good) if n else None,
        split_half=dict(first_ids=['R1', 'R2', 'R3'], second_ids=['R4', 'R5', 'R6'],
            first_mean=halves[0], second_mean=halves[1],
            same_sign=sign(halves[0]) == sign(halves[1]) if None not in halves else None),
        frozen_service_history6=dict(prediction=prediction, prediction_minus_mean=error,
            squared_error_to_mean=error*error if error is not None else None,
            mean_squared_error_to_single_labels=mean((prediction-x)**2 for x in values) if error is not None else None,
            same_sign_as_mean=sign(prediction) == sign(average) if error is not None else None))


def panel_statistics(roots, protocol):
    complete = [r for r in roots if r['complete']]
    rows = [r['statistics'] for r in complete]
    means = [r['mean_D'] for r in rows]
    within = mean(r['sample_variance'] for r in rows) if rows else None
    between = variance(means) if len(means) >= 2 else None
    noise = within/protocol['replicates_per_root'] if within is not None else None
    left = [r['split_half']['first_mean'] for r in rows]
    right = [r['split_half']['second_mean'] for r in rows]
    errors = [r['frozen_service_history6']['prediction_minus_mean'] for r in rows
              if r['frozen_service_history6']['prediction_minus_mean'] is not None]
    return dict(complete_root_count=len(rows), complete_root_ids=[r['id'] for r in complete],
        panel_complete=len(rows) == protocol['root_count'],
        mean_root_D=mean(means) if means else None, mean_within_root_variance=within,
        sample_variance_of_root_means=between, noise_in_root_means=noise,
        signed_denoised_between_root_variance=between-noise if between is not None else None,
        split_half_covariance=covariance(left, right),
        split_half_same_sign_count=sum(sign(a) == sign(b) for a, b in zip(left, right)),
        frozen_service_history6=dict(roots_with_prediction=len(errors),
            mean_prediction_minus_mean=mean(errors) if errors else None,
            MAE_to_root_means=mean(abs(e) for e in errors) if errors else None,
            RMSE_to_root_means=sqrt(mean(e*e for e in errors)) if errors else None,
            same_sign_count=sum(r['frozen_service_history6']['same_sign_as_mean'] is True for r in rows)),
        inference_unit='Fixed physical roots; future replications estimate conditional lane-RNG uncertainty, not new roots')


def analyze(collectors, protocol):
    expected = {(s, scene) for s in protocol['external_seeds'] for scene in protocol['scenarios']}
    sources, anomalies = {}, []
    for source in collectors:
        key = (source.get('external_seed'), source.get('scenario'))
        if key not in expected or key in sources:
            anomalies.append(dict(type='undeclared_or_duplicate_collector', collector=list(key)))
        else:
            sources[key] = source
    roots, collector_rows, exceptions = [], [], []
    for seed, scene in ((s, scene) for s in protocol['external_seeds'] for scene in protocol['scenarios']):
        source = sources.get((seed, scene))
        declared_roots = [r for r in protocol['roots'] if r['external_seed'] == seed and r['scenario'] == scene]
        if source is None:
            exceptions.append(dict(type='missing_collector', external_seed=seed, scenario=scene))
            for declared in declared_roots:
                roots.append(dict(id=declared['id'], external_seed=seed, scenario=scene,
                    line_id=declared['line_id'], complete=False, pairs=[], statistics=root_statistics([], None, protocol),
                    missing_futures=[f['id'] for f in declared['futures']], checks=dict(collector_present=False)))
            continue
        by_line, local_anomalies = {}, []
        for root in source.get('roots', []):
            line = root.get('line_id')
            if line not in protocol['lines'] or line in by_line:
                local_anomalies.append(dict(type='undeclared_or_duplicate_root', line_id=line))
            else:
                by_line[line] = root
        collector_roots = []
        for declared in declared_roots:
            root = by_line.get(declared['line_id'], {})
            prefix = root.get('prefix_reproduction') or {}
            root_checks = root.get('checks', {})
            fixed_root = root.get('root') == declared['root'] and bool(root_checks) \
                and all(v is True for v in root_checks.values()) and root.get('root_ledger') is not None \
                and prefix.get('all_pass') is True \
                and prefix.get('matched_issues') == prefix.get('observed_issues') == declared['root']['sequence']
            declared_futures = {f['id']: f for f in declared['futures']}
            pairs, future_anomalies = {}, []
            for pair in root.get('pairs', []):
                fid = pair.get('future_id')
                if fid not in declared_futures or fid in pairs:
                    future_anomalies.append(dict(type='undeclared_or_duplicate_future', future_id=fid))
                    continue
                pairs[fid] = pair_readout(pair, declared, declared_futures[fid], protocol)
            ordered = [pairs[fid] for fid in declared_futures if fid in pairs]
            # A label from the wrong physical root is retained but not included in root statistics.
            for pair in ordered:
                pair['checks']['fixed_physical_root'] = fixed_root
                pair['valid'] = pair['valid'] and fixed_root
            prediction = (root.get('frozen_predictions') or {}).get('service_history6')
            checks = dict(fixed_physical_root=fixed_root, no_extra_or_duplicate_futures=not future_anomalies,
                six_declared_valid_futures=set(pairs) == set(declared_futures) and all(p['valid'] for p in ordered),
                frozen_prediction_present=finite(prediction))
            item = dict(id=declared['id'], external_seed=seed, scenario=scene, line_id=declared['line_id'],
                root=deepcopy(root.get('root')), root_ledger=deepcopy(root.get('root_ledger')),
                complete=all(checks.values()), checks=checks, pairs=ordered,
                missing_futures=[fid for fid in declared_futures if fid not in pairs],
                statistics=root_statistics(ordered, prediction if finite(prediction) else None, protocol),
                frozen_predictions=deepcopy(root.get('frozen_predictions')), anomalies=future_anomalies)
            collector_roots.append(item)
            roots.append(item)
            if not item['complete']:
                exceptions.append(dict(type='incomplete_root', id=declared['id'],
                    failed_checks=[k for k, v in checks.items() if not v], anomalies=future_anomalies))
        budget, prefix = source.get('budget', {}), source.get('prefix_reproduction') or {}
        suffix_expected = sum(2*protocol['replicates_per_root']*(protocol['horizon']-r['root']['started_at'])
                              for r in declared_roots)
        checks = dict(runtime_complete=source.get('status') == 'complete' and passed(source.get('integrity'))
                and source.get('error') is None,
            complete_reference_prefix=prefix.get('all_pass') is True and prefix.get('reference_exhausted') is True
                and finite(prefix.get('source_rows'))
                and prefix.get('source_rows') == prefix.get('matched_issues') == prefix.get('observed_issues'),
            three_complete_roots=len(collector_roots) == 3 and all(r['complete'] for r in collector_roots),
            no_extra_or_duplicate_roots=not local_anomalies,
            zero_learning=all(budget.get(k) == 0 for k in ZERO),
            exact_prefix_spend=budget.get('native_prefix_seconds') == protocol['prefix_stop'],
            exact_suffix_spend=budget.get('suffix_native_seconds') == suffix_expected,
            exact_suffix_count=budget.get('suffixes') == 36 and budget.get('labels_observed') == 36,
            known_budget=budget.get('total_native_seconds_is_lower_bound') is False and budget.get('unknown_suffix_costs') == 0,
            budget_identity=all(finite(budget.get(k)) for k in ('total_native_seconds', 'native_prefix_seconds', 'suffix_native_seconds'))
                and budget['total_native_seconds'] == budget['native_prefix_seconds']+budget['suffix_native_seconds'])
        collector_rows.append(dict(external_seed=seed, scenario=scene, complete=all(checks.values()), checks=checks,
            budget=deepcopy(budget), error=deepcopy(source.get('error')), anomalies=local_anomalies))
        if not all(checks.values()):
            exceptions.append(dict(type='incomplete_collector', external_seed=seed, scenario=scene,
                failed_checks=[k for k, v in checks.items() if not v], error=deepcopy(source.get('error')),
                anomalies=local_anomalies))
    budgets = [s.get('budget', {}) for s in collectors]
    known = sum(b['total_native_seconds'] for b in budgets if finite(b.get('total_native_seconds')))
    missing = len(expected-sources.keys())
    unknown_budget_records = sum(not finite(b.get('total_native_seconds')) for b in budgets)
    unknown_suffix_counts = [b.get('unknown_suffix_costs') for b in budgets]
    unknown_suffixes = sum(x for x in unknown_suffix_counts if finite(x))
    lower_bound = bool(missing or unknown_budget_records or unknown_suffixes
        or any(not finite(x) for x in unknown_suffix_counts) or any(
        b.get('total_native_seconds_is_lower_bound') is not False for b in budgets))
    zero_counts = {k: sum(b[k] for b in budgets) if not missing and all(finite(b.get(k)) for b in budgets) else None for k in ZERO}
    checks = dict(four_complete_collectors=len(collector_rows) == len(expected) and all(c['complete'] for c in collector_rows),
        twelve_complete_roots=len(roots) == protocol['root_count'] and all(r['complete'] for r in roots),
        seventy_two_valid_pairs=sum(p['valid'] for r in roots for p in r['pairs']) == protocol['planned_pairs'],
        no_undeclared_or_duplicate_collectors=not anomalies,
        known_native_budget=not lower_bound,
        exact_native_budget=not lower_bound and known == protocol['native_seconds']['total'])
    complete = all(checks.values())
    candidates = [r['id'] for r in roots if r['complete'] and r['statistics']['bound_below_zero'] is True]
    decision = 'HOLD_TECHNICAL_INCOMPLETE' if not complete else \
        'LABEL_SIGNAL_SUPPORTED_FOR_SEPARATE_FEASIBILITY' if candidates else 'CLOSE_FIXED_LABEL_CANDIDATE'
    return dict(status='complete' if complete else 'hold', decision=decision,
        integrity=dict(all_pass=complete, checks=checks), roots=roots, collectors=collector_rows,
        exceptions=exceptions+anomalies, descriptive=panel_statistics(roots, protocol),
        mean_gate=dict(evaluated=complete, passed=bool(candidates) if complete else None,
            multiplicity=protocol['statistics']['multiplicity'], df=protocol['statistics']['df'],
            t_critical=protocol['statistics']['bonferroni_t_critical'],
            roots_with_bound_below_zero=candidates, no_split_half_or_covariance_gate=True),
        accounting=dict(known_native_seconds=known, native_seconds=None if lower_bound else known,
            is_lower_bound=lower_bound, missing_collectors=missing, unknown_budget_records=unknown_budget_records,
            unknown_suffix_costs=unknown_suffixes if all(finite(x) for x in unknown_suffix_counts) and not missing else None,
            declared_native_seconds=protocol['native_seconds']['total'],
            planned_pairs=protocol['planned_pairs'], planned_suffixes=protocol['planned_suffixes'],
            declared_F0_repeats=0, declared_capability_probe=0, learning_counts=zero_counts),
        interpretation=dict(scope=protocol['scope'], assumptions=protocol['statistics']['assumptions'],
            old_labels_in_means=False, fixed_panel_only=True, no_fit_or_deployment=True,
            V_kept_sealed=True, no_automatic_additional_futures=True,
            label_signal_does_not_establish_feature_learnability=True,
            root_average_is_not_a_multi_intervention_policy_cost=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--collectors', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=HERE/'summary.json')
    args = parser.parse_args(argv)
    load = lambda path: json.loads(path.read_text())
    protocol = load(args.protocol)
    paths = args.collectors or [HERE/'runs'/str(seed)/scene/'summary.json'
        for seed in protocol['external_seeds'] for scene in protocol['scenarios']]
    result = analyze([load(p) for p in paths if p.exists()], protocol)
    with args.out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], decision=result['decision'], mean_gate=result['mean_gate'],
                         accounting=result['accounting'])))
    return result


if __name__ == '__main__':
    main()
