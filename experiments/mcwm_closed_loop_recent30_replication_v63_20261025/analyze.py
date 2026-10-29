"""One fixed recent30 replication: independent three seeds and all five retained."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_control_campaign_v60_20261025 import control_analyze as native

DEVELOPMENT_SEEDS = (6201, 6202)
INDEPENDENT_SEEDS = (6203, 6204, 6205)
SCENARIOS = ('stable', 'local_speed_drop')
DEVELOPMENT_PATH = HERE.parent/'mcwm_closed_loop_ordinary_control_v61_20261025/results/recent30/development.json'


def compact_cell(cell, *, seed, scenario, cohort):
    result = dict(seed=seed, scenario=scenario, cohort=cohort, status='missing_or_failed_run',
        candidate_cost=None, baseline_cost=None, cost_difference=None, percent_change=None,
        wait_seconds_per_passenger_difference=None, ride_seconds_per_passenger_difference=None,
        outcome=None)
    if cell is None:
        return result
    full = cell.get('comparisons', {}).get('no_fresh', {}).get('full_day', {})
    ledgers = cell.get('cost_ledgers', {})
    if (full.get('status') != 'complete' or not native._cost_valid(ledgers.get('fresh', {}))
            or not native._cost_valid(ledgers.get('no_fresh', {}))):
        result['status'] = 'invalid_or_missing_cost'
        return result
    fresh, baseline = ledgers['fresh'], ledgers['no_fresh']
    delta = fresh['generalized_seconds_per_departed_passenger']-baseline['generalized_seconds_per_departed_passenger']
    result.update(status='complete', candidate_cost=fresh, baseline_cost=baseline,
        cost_difference=delta, percent_change=100.*delta/baseline['generalized_seconds_per_departed_passenger'],
        wait_seconds_per_passenger_difference=(fresh['accrued_wait_seconds']/fresh['departed_passengers']
            - baseline['accrued_wait_seconds']/baseline['departed_passengers']),
        ride_seconds_per_passenger_difference=(fresh['accrued_in_vehicle_seconds']/fresh['departed_passengers']
            - baseline['accrued_in_vehicle_seconds']/baseline['departed_passengers']),
        outcome='win' if delta < 0 else 'loss' if delta > 0 else 'tie')
    return result


def stats(rows):
    complete = [row for row in rows if row['status'] == 'complete']
    count = len(complete)
    wins = [row for row in complete if row['cost_difference'] < 0]
    losses = [row for row in complete if row['cost_difference'] > 0]
    ties = count-len(wins)-len(losses)
    fresh = sum(row['candidate_cost']['generalized_seconds_per_departed_passenger'] for row in complete)/count if count else None
    baseline = sum(row['baseline_cost']['generalized_seconds_per_departed_passenger'] for row in complete)/count if count else None
    delta = sum(row['cost_difference'] for row in complete)/count if count else None
    return dict(status='complete' if count == len(rows) else 'partial' if count else 'missing',
        expected_cells=len(rows), complete_cells=count, missing_or_invalid_cells=len(rows)-count,
        candidate_mean_cost=fresh, baseline_mean_cost=baseline, mean_cost_difference=delta,
        percent_change=100.*delta/baseline if count and baseline else None,
        mean_wait_seconds_per_passenger_difference=sum(row['wait_seconds_per_passenger_difference'] for row in complete)/count if count else None,
        mean_ride_seconds_per_passenger_difference=sum(row['ride_seconds_per_passenger_difference'] for row in complete)/count if count else None,
        wins=len(wins), losses=len(losses), ties=ties,
        win_percent_of_complete=100.*len(wins)/count if count else None,
        loss_percent_of_complete=100.*len(losses)/count if count else None,
        tie_percent_of_complete=100.*ties/count if count else None,
        winning_mean_cost_reduction=-sum(row['cost_difference'] for row in wins)/len(wins) if wins else None,
        winning_mean_percent_reduction=-sum(row['percent_change'] for row in wins)/len(wins) if wins else None,
        largest_cost_reduction=max((-row['cost_difference'] for row in wins), default=None),
        losing_mean_cost_increase=sum(row['cost_difference'] for row in losses)/len(losses) if losses else None,
        losing_mean_percent_increase=sum(row['percent_change'] for row in losses)/len(losses) if losses else None,
        largest_cost_increase=max((row['cost_difference'] for row in losses), default=None))


def cohort_result(rows, seeds, *, integrity_pass):
    overall = stats(rows)
    by_scenario = {scenario: stats([row for row in rows if row['scenario'] == scenario])
        for scenario in SCENARIOS}
    by_seed = {str(seed): stats([row for row in rows if row['seed'] == seed]) for seed in seeds}
    complete = overall['complete_cells'] == len(seeds)*2 and len(rows) == len(seeds)*2
    cost_conditions = dict(
        overall_lower=complete and overall['mean_cost_difference'] < 0,
        stable_not_higher=complete and by_scenario['stable']['mean_cost_difference'] <= 0,
        shock_lower=complete and by_scenario['local_speed_drop']['mean_cost_difference'] < 0)
    return dict(seeds=list(seeds), overall=overall, by_scenario=by_scenario, by_seed=by_seed,
        gate=dict(all_pass=bool(integrity_pass and complete and all(cost_conditions.values())),
            integrity_pass=bool(integrity_pass), complete_fixed_cohort=complete,
            conditions=cost_conditions, rule='overall<0 and stable<=0 and shock<0'))


def source_cells(report, seeds):
    expected = {(seed, scenario) for seed in seeds for scenario in SCENARIOS}
    index, duplicates, unexpected = {}, [], []
    for cell in report.get('cells', []):
        key = (cell.get('identity'), cell.get('scenario'))
        if key not in expected:
            unexpected.append(dict(identity=key[0], scenario=key[1]))
        elif key in index:
            duplicates.append(dict(identity=key[0], scenario=key[1]))
        else:
            index[key] = cell
    return index, dict(expected_cells=len(expected), available_cells=len(index),
        duplicate_cells=duplicates, unexpected_cells=unexpected,
        exact_fixed_membership=set(index) == expected and not duplicates and not unexpected)


def audit_summary(report, membership):
    return dict(status=report.get('status', 'missing'), complete=report.get('complete', False),
        integrity=report.get('integrity', dict(all_pass=False)), membership=membership,
        expected_cells=report.get('expected_cells'), available_cells=report.get('available_cells'),
        missing=report.get('missing', []), failures=report.get('failures', []),
        failed_run_work=report.get('failed_run_work', []),
        native_evidence=[dict(seed=cell.get('identity'), scenario=cell.get('scenario'),
            prefix=cell.get('pre_intervention_prefix'), actual_parallel=cell.get('actual_parallel'))
            for cell in report.get('cells', [])])


def failed_summary_work(root, report):
    """Retain consumed work when a completed summary fails a native audit."""
    affected = {'/'.join(key.split('/')[:2]) for key in report.get('integrity', {}).get('failed_checks', [])}
    work = []
    for pair in sorted(affected):
        for arm in ('fresh', 'no_fresh'):
            path = root/'runs/recent30/confirmation'/pair/arm/'summary.json'
            value = native.read(path)
            if value is not None:
                work.append(dict(cell=pair+'/'+arm, source='summary.json', status=value.get('status'),
                    consumed_budget=value.get('budget'), wall_seconds=value.get('wall_seconds')))
    return work


def analyze(root=HERE, *, development_path=None):
    root = Path(root)
    independent = native.analyze(root, 'recent30', 'confirmation')
    reference_path = DEVELOPMENT_PATH if development_path is None else Path(development_path)
    development = native.read(reference_path)
    reference_missing = development is None
    development = development or {}
    protocol = native.read(root/'protocol.json')
    protocol_fixed = (tuple(protocol['confirmation_seeds']) == INDEPENDENT_SEEDS
        and tuple(protocol['scenarios']) == SCENARIOS)
    old_index, old_membership = source_cells(development, DEVELOPMENT_SEEDS)
    new_index, new_membership = source_cells(independent, INDEPENDENT_SEEDS)
    old_audit_pass = (development.get('status') == 'complete' and development.get('complete') is True
        and development.get('candidate') == 'recent30' and development.get('stage') == 'development'
        and development.get('integrity', {}).get('all_pass') is True
        and old_membership['exact_fixed_membership'])
    old_failure_preserved = (development.get('decision', {}).get('scientific_gate_pass') is False
        and development.get('decision', {}).get('advance_to_confirmation') is False)
    independent_audit_pass = (independent['complete'] and independent['integrity']['all_pass']
        and new_membership['exact_fixed_membership'] and protocol_fixed)
    rows = []
    for cohort, seeds, index in (('development', DEVELOPMENT_SEEDS, old_index),
            ('independent', INDEPENDENT_SEEDS, new_index)):
        for seed in seeds:
            for scenario in SCENARIOS:
                rows.append(compact_cell(index.get((seed, scenario)), seed=seed, scenario=scenario, cohort=cohort))
    independent_rows = [row for row in rows if row['cohort'] == 'independent']
    cost_table_complete = all(row['status'] == 'complete' for row in rows)
    all_audits_pass = independent_audit_pass and old_audit_pass and old_failure_preserved and cost_table_complete
    independent3 = cohort_result(independent_rows, INDEPENDENT_SEEDS, integrity_pass=independent_audit_pass)
    retained5 = cohort_result(rows, DEVELOPMENT_SEEDS+INDEPENDENT_SEEDS, integrity_pass=all_audits_pass)
    confirmed = all_audits_pass and independent3['gate']['all_pass'] and retained5['gate']['all_pass']
    integrity = dict(all_pass=bool(all_audits_pass), protocol_fixed=protocol_fixed,
        all_ten_cost_ledgers_complete=cost_table_complete,
        independent_native_audit_pass=bool(independent_audit_pass),
        source_development_audit_pass=bool(old_audit_pass),
        source_development_failed_gate_preserved=bool(old_failure_preserved))
    pending = reference_missing or independent.get('status') == 'pending'
    status = 'complete' if all_audits_pass else 'pending' if pending else 'integrity_failed'
    reason = ('independent_and_retained_cost_gates_pass_root_review_required' if confirmed else
        'incomplete_or_integrity_failed' if not all_audits_pass else
        'independent_cost_gate_failed' if not independent3['gate']['all_pass'] else
        'retained_all_five_cost_gate_failed')
    independent_audit = audit_summary(independent, new_membership)
    independent_audit['failed_summary_work'] = failed_summary_work(root, independent)
    return dict(status=status, candidate='recent30', stage='independent_replication',
        complete=bool(all_audits_pass), integrity=integrity,
        development_reference=dict(path=str(reference_path), missing=reference_missing,
            audit=development.get('integrity'), decision=development.get('decision'),
            membership=old_membership),
        independent_audit=independent_audit,
        independent3=independent3, retained5=retained5, expected_cells=10, available_cells=sum(
            row['status'] == 'complete' for row in rows), cells=rows,
        decision=dict(confirmed_control_improvement=bool(confirmed),
            scientific_gate_pass=bool(confirmed), independent3_gate_pass=independent3['gate']['all_pass'],
            retained5_gate_pass=retained5['gate']['all_pass'], ordinary_dripping_goal_achieved=False,
            original_development_gate_failed=bool(old_failure_preserved),
            further_seed_expansion_allowed=False, root_review_required=True, reason=reason),
        metric='Full-day(2W+R)/all departed passengers, including unfinished trips; '
            'candidate=fresh recent30, baseline=no_fresh. Negative paired difference is beneficial.',
        aggregation='Equal weight per fixed seed/scenario cell. Independent3 and retained5 tested separately. '
            'No cell selection or missing-to-zero substitution. Original development gate remains failed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(HERE)
    payload = json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(payload.encode()) > 48000:
        raise ValueError('Replication summary exceeds48KB; do not duplicate native artifacts')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        output.write(payload+'\n')


if __name__ == '__main__':
    main()
