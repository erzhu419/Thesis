"""Fixed 30-day historical system comparison; preserve v75 attribution."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ('wait_per_passenger', 'ride_per_passenger', 'cost_per_passenger')
ARMS = ('fresh', 'no_fresh')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def period_values(row, denominator, window=None):
    """Validate units before returning normalized W, R and 2W+R."""
    if not isinstance(row, dict):
        raise ValueError('missing period')
    if window is not None and (row.get('status') != 'complete'
            or [row.get('start'), row.get('end')] != list(window)):
        raise ValueError('incomplete period or wrong actual boundaries')
    values = [row.get(key) for key in ('wait', 'ride', 'cost')]
    if not all(finite(value) and value >= 0 for value in values):
        raise ValueError('missing, nonfinite or negative passenger cost')
    wait, ride, cost = values
    if row.get('departed') != denominator:
        raise ValueError('wrong whole-day passenger denominator')
    if not math.isclose(cost, 2*wait+ride, rel_tol=0, abs_tol=1e-6):
        raise ValueError('cost is not 2W+R in passenger-seconds')
    if not finite(row.get('cost_per_passenger')) or not math.isclose(
            row['cost_per_passenger'], cost/denominator, rel_tol=0, abs_tol=1e-9):
        raise ValueError('normalized cost differs from raw ledger cost')
    return dict(zip(METRICS, (value/denominator for value in values)))


def mean(values):
    return {key: statistics.fmean(row[key] for row in values) for key in METRICS}


def difference(left, right):
    return {key: left[key]-right[key] for key in METRICS}


def distribution(values):
    return dict(n=len(values), mean=mean(values),
        sample_sd={key: statistics.stdev(row[key] for row in values) for key in METRICS},
        minimum={key: min(row[key] for row in values) for key in METRICS},
        maximum={key: max(row[key] for row in values) for key in METRICS})


def expected_runs(protocol):
    return list(itertools.product(protocol['historical_training_seeds'],
        protocol['external_seeds'], protocol['scenarios']))


def read_cached(protocol, cached):
    denominator = protocol['full_day_departed_passengers']
    environments = list(itertools.product(protocol['external_seeds'], protocol['scenarios']))
    errors, result = [], {}
    if not (cached.get('status') == 'complete' and cached.get('complete') is True
            and cached.get('integrity', {}).get('all_pass') is True):
        errors.append('v75 cached validation is incomplete or invalid')
    for cell in cached.get('cells', []):
        key = (cell.get('identity', -1)+protocol['external_seed_offset'], cell.get('scenario'))
        if key not in environments or key in result:
            errors.append(f'undeclared or duplicate v75 cell: {key}')
            continue
        values = {}
        for period in protocol['periods']:
            values[period] = {}
            for arm in ARMS:
                try:
                    values[period][arm] = period_values(
                        cell.get('periods', {}).get(period, {}).get('arms', {}).get(arm), denominator)
                except ValueError as error:
                    errors.append(f'v75 {key}/{period}/{arm}: {error}')
                    values[period][arm] = None
        result[key] = values
    if set(result) != set(environments):
        errors.append('v75 six-cell panel is incomplete')
    return result, errors


def validate_run(protocol, identity, row):
    actor, external, scenario = identity
    errors, periods = [], {}
    if row is None:
        return None, ['missing summary']
    if (row.get('actor_seed'), row.get('external_seed'), row.get('scenario')) != identity:
        errors.append('summary identity differs from its declared slot')
    if row.get('status') != 'complete' or row.get('integrity', {}).get('all_pass') is not True:
        errors.append('native day incomplete or integrity failed')
    denominator = protocol['full_day_departed_passengers']
    if row.get('passenger_denominator') != denominator:
        errors.append('wrong whole-day passenger denominator')
    budget = row.get('budget', {})
    if budget.get('native_seconds') != protocol['horizon_seconds'] or budget.get('native_seconds_is_lower_bound') is not False:
        errors.append('native day duration is incomplete or unknown')
    if any(budget.get(key) != 0 for key in (
            'optimizer_updates', 'target_training_labels', 'world_updates', 'trust_updates')):
        errors.append('frozen historical policy budget differs')
    for period, window in protocol['periods'].items():
        try:
            periods[period] = period_values(row.get('periods', {}).get(period), denominator, window)
        except ValueError as error:
            errors.append(f'{period}: {error}')
    return (periods if not errors else None), errors


def analyze(protocol, cached_validation, historical_runs):
    """Pure reduction of the declared grid; no fit or outcome-selected subset."""
    actors = protocol['historical_training_seeds']
    environments = list(itertools.product(protocol['external_seeds'], protocol['scenarios']))
    declared = expected_runs(protocol)
    cached, cache_errors = read_cached(protocol, cached_validation)
    valid, run_status, missing, invalid = {}, [], [], []
    known_seconds, unknown_spend, lower_bound = 0., 0, False
    for identity in declared:
        row = historical_runs.get(identity)
        values, errors = validate_run(protocol, identity, row)
        item = dict(actor_seed=identity[0], external_seed=identity[1], scenario=identity[2],
                    valid=not errors, errors=errors,
                    action_audit=None if row is None else copy.deepcopy(row.get('action_audit')))
        run_status.append(item)
        if values is not None:
            valid[identity] = values
        else:
            (missing if row is None else invalid).append(item)
        budget = {} if row is None else row.get('budget', {})
        seconds = budget.get('native_seconds')
        if finite(seconds) and seconds >= 0:
            known_seconds += seconds
            lower_bound |= budget.get('native_seconds_is_lower_bound') is not False
        else:
            unknown_spend += 1
    undeclared = [list(key) for key in historical_runs if key not in declared]
    complete = len(valid) == len(declared) and not cache_errors and not undeclared

    cells = []
    for external, scenario in environments:
        periods = {}
        for period in protocol['periods']:
            values = {actor: valid[(actor, external, scenario)][period] for actor in actors
                      if (actor, external, scenario) in valid}
            baseline = distribution(list(values.values())) if len(values) == len(actors) else None
            dynamic = cached.get((external, scenario), {}).get(period, {})
            differences = {arm: difference(dynamic[arm], baseline['mean'])
                if baseline is not None and dynamic.get(arm) is not None and not cache_errors else None
                for arm in ARMS}
            periods[period] = dict(complete=baseline is not None and not cache_errors,
                historical_actor_values=values, historical_distribution=baseline,
                dynamic_values=dynamic, dynamic_minus_historical_mean=differences)
        cells.append(dict(external_seed=external, scenario=scenario, periods=periods))

    actor_rows = []
    for actor in actors:
        actor_complete = all((actor, *environment) in valid for environment in environments)
        periods = {}
        for period in protocol['periods']:
            old_mean = mean([valid[(actor, *environment)][period] for environment in environments]) if actor_complete else None
            dynamic_means = {arm: mean([cached[environment][period][arm] for environment in environments])
                             if not cache_errors else None for arm in ARMS}
            periods[period] = dict(historical_mean=old_mean, dynamic_mean=dynamic_means,
                dynamic_minus_historical={arm: difference(dynamic_means[arm], old_mean)
                    if actor_complete and not cache_errors else None for arm in ARMS})
        actor_rows.append(dict(actor_seed=actor, complete=actor_complete, environments=len(environments), periods=periods))

    aggregate = None
    if complete:
        aggregate = {}
        for period in protocol['periods']:
            old_actor_means = [row['periods'][period]['historical_mean'] for row in actor_rows]
            old = distribution(old_actor_means)
            dynamic = {arm: mean([cached[environment][period][arm] for environment in environments]) for arm in ARMS}
            by_scenario = {}
            for scenario in protocol['scenarios']:
                environment_subset = [key for key in environments if key[1] == scenario]
                old_scene = mean([mean([valid[(actor, *environment)][period]
                    for environment in environment_subset]) for actor in actors])
                dynamic_scene = {arm: mean([cached[environment][period][arm]
                    for environment in environment_subset]) for arm in ARMS}
                by_scenario[scenario] = dict(historical_mean=old_scene, dynamic_mean=dynamic_scene,
                    dynamic_minus_historical={arm: difference(dynamic_scene[arm], old_scene) for arm in ARMS})
            aggregate[period] = dict(historical_actor_mean_distribution=old, dynamic_mean=dynamic,
                dynamic_minus_historical={arm: difference(dynamic[arm], old['mean']) for arm in ARMS},
                by_scenario=by_scenario)

    preserved = {key: copy.deepcopy(cached_validation.get(key)) for key in (
        'status', 'complete', 'cells', 'aggregates', 'independent_seed_readout', 'decision', 'budget')}
    return dict(status='complete' if complete else 'partial', complete=complete,
        decision=dict(state='COMPLETE_DESCRIPTIVE_COMPARISON' if complete else 'HOLD_INCOMPLETE',
            ranking_available=complete, promotion_gate=False, automatic_new_experiments=False),
        integrity=dict(all_pass=complete, cached_validation_errors=cache_errors,
            undeclared_runs=undeclared, valid_historical_days=len(valid), expected_historical_days=len(declared)),
        missing=missing, invalid=invalid, run_status=run_status,
        cached_v75_attribution=preserved, environment_cells=cells,
        historical_actor_means=actor_rows, aggregates=aggregate,
        budget=dict(new_native_seconds=known_seconds if unknown_spend == 0 and not lower_bound else None,
            known_native_seconds=known_seconds, is_lower_bound=bool(unknown_spend or lower_bound),
            unknown_spend_slots=unknown_spend, declared_native_seconds=protocol['new_native_seconds_cap'],
            reused_v75_native_seconds=cached_validation.get('budget', {}).get('response', {}).get('actual_native_seconds')),
        interpretation=dict(units='Comparison W/R and J are seconds per whole-day passenger; J=2W+R.',
            historical_weighting='Each old actor first averages all six fixed environments; all five actor means have equal weight.',
            historical_cell_weighting='Within each environment all five old actors have equal weight; never select a best actor.',
            statistical_unit='Five historical training actors and three v75 training seeds; thirty old days are not thirty independent training replicates.',
            attribution='Cached fresh-minus-matched-no_fresh is preserved. Comparisons with historical final MC-WM rank complete systems with different training budgets and initializers.',
            execution='Historical raw-state F is retained; canonical F is audited passively. Its difference is not asserted to cause the observed degradation.',
            no_pvalue_or_new_gate=True,
            v103='Its fixed-action mean gate closed; history6 causal failure was not tested.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--runs', type=Path, default=HERE/'runs')
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--out', type=Path, default=HERE/'summary.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validation_path = args.validation or ROOT/protocol['v75_results_path']
    cached = json.loads(validation_path.read_text())
    runs = {}
    for actor, external, scenario in expected_runs(protocol):
        path = args.runs/str(actor)/str(external)/scenario/'summary.json'
        runs[(actor, external, scenario)] = json.loads(path.read_text()) if path.exists() else None
    result = analyze(protocol, cached, runs)
    result['sources'] = dict(protocol=str(args.protocol), v75_validation=str(validation_path), runs=str(args.runs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], decision=result['decision'], output=str(args.out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
