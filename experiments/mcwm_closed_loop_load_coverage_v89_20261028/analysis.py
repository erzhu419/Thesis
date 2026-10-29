"""Freeze a reachable 7X load split from native issue covariates only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_fresh_response_v86_20261027.analysis import _reference_valid
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support

GUARDS = ('clock_unchanged', 'ledger_unchanged', 'rng_unchanged',
          'issue_clock_exact', 'event_clock_exact', 'before_action')
ZERO_BUDGET = ('suffixes', 'labels', 'new_initialization_updates', 'new_fits',
               'new_neural_updates', 'feedback_admission', 'v87_truth_used')


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _protocol_checks(protocol):
    observation, strata, budget = (protocol[name] for name in ('observation', 'stratification', 'budget'))
    return dict(fixed_source=protocol['source_initializer'] == 6207 and protocol['external_seed'] == 716207
            and protocol['scenario'] == 'stable',
        fixed_window_and_horizon=observation['line'] == '7X' and observation['issue_window'] == [12000, 12600]
            and observation['prefix_end'] == 12600 and observation['original_episode_horizon'] == 24000,
        unchanged_action_domain=protocol['action_support'] == dict(reference_min=3., reference_max_exclusive=3.75,
            candidate=7.5, candidate_tolerance=1e-5, minimum_gap_native_steps=2, at_stop_required=True),
        distinct_gap_rule=strata['variable'] == 'issue_native_load_ratio' and strata['rule'] == 'middle_distinct_gap'
            and strata['lighter'] == 'load<cut' and strata['heavier'] == 'load>=cut',
        other_line_strata_unchanged=strata['other_lines_unchanged'] == {'102X': .5, '705X': .5},
        fixed_budget=budget['max_native_simulation_seconds'] == 12600 and budget['prefixes'] == 1
            and all(budget[key] == 0 for key in ZERO_BUDGET))


def _row_readout(row):
    load_complete = (row['observed'] is True and _finite(row['onboard']) and row['onboard'] >= 0
        and _finite(row['capacity']) and row['capacity'] > 0 and _finite(row['load'])
        and row['load'] == row['onboard']/row['capacity'])
    clocks_complete = all(_finite(row[key]) for key in ('issue_clock', 'event_clock',
        'sim_step_seconds', 'action_step_seconds'))
    clocks_complete = clocks_complete and row['sim_step_seconds'] > 0 and row['action_step_seconds'] > 0
    action_complete = all(_finite(row[key]) for key in ('ref_hold', 'cand_hold'))
    support = action_support(row['ref_hold'], row['cand_hold'], row['sim_step_seconds'],
        row['action_step_seconds']) if action_complete and clocks_complete else dict(supported=False)
    expected_inside = row['at_stop'] is True and support['supported']
    checks = dict(native_load_complete=load_complete, native_clocks_complete=clocks_complete,
        atomic_observation=all(_finite(row[key]) for key in ('obs_start', 'obs_end', 'issue_clock'))
            and row['obs_start'] == row['obs_end'] == row['issue_clock'],
        issue_window=_finite(row['issue_clock']) and 12000 <= row['issue_clock'] < 12600,
        event_not_after_issue=_finite(row['event_clock']) and _finite(row['issue_clock'])
            and row['event_clock'] <= row['issue_clock'],
        observational_guards=all(row['guard_checks'][key] is True for key in GUARDS),
        declared_domain=isinstance(row['in_domain'], bool) and row['in_domain'] == expected_inside,
        domain_reason=(row['domain_reason'] is None if row['in_domain'] is True
            else isinstance(row['domain_reason'], str) and bool(row['domain_reason'])))
    return dict(sequence=row['sequence'], trip_id=row['trip_id'], station_id=row['station_id'],
        issue_clock=row['issue_clock'], event_clock=row['event_clock'], in_domain=row['in_domain'],
        obs_start=row['obs_start'], obs_end=row['obs_end'],
        domain_reason=row['domain_reason'], onboard=row['onboard'], capacity=row['capacity'], load=row['load'],
        ref_hold=row['ref_hold'], cand_hold=row['cand_hold'], valid=all(checks.values()), checks=checks)


def _distribution(rows):
    observed = [row for row in rows if row['checks']['native_load_complete']]
    counts = Counter(row['load'] for row in observed)
    values = sorted(counts)
    return dict(issues=len(rows), observed_loads=len(observed), missing_loads=len(rows)-len(observed),
        min_load=values[0] if values else None, max_load=values[-1] if values else None,
        histogram={str(value): counts[value] for value in values}, distinct_values=values,
        distinct_count=len(values), original_half_capacity_split=dict(
            below_half=sum(row['load'] < .5 for row in observed),
            at_or_above_half=sum(row['load'] >= .5 for row in observed)))


def _earliest(rows):
    if not rows:
        return None
    row = min(rows, key=lambda value: (value['issue_clock'], value['sequence']))
    return {key: row[key] for key in ('sequence', 'trip_id', 'station_id', 'issue_clock', 'event_clock',
        'obs_start', 'obs_end', 'onboard', 'capacity', 'load', 'ref_hold', 'cand_hold')}


def _audit_checks(audit):
    return dict(resident_reference_audit=audit['all_pass'] and all(audit['checks'].values()),
        initializer=audit['source_initializer'] == 6207,
        external_seed=audit['external_seed'] == 716207, scenario=audit['scenario'] == 'stable',
        six_original_reference_roots=audit['reference_roots'] == 6,
        zero_native_audit_seconds=audit['native_seconds'] == 0,
        zero_audit_fits=audit['new_fits'] == 0, zero_audit_neural_updates=audit['new_neural_updates'] == 0,
        zero_audit_suffixes=audit['new_suffixes'] == 0, zero_audit_labels=audit['new_labels'] == 0,
        no_v87_truth=audit['v87_truth_used'] == 0)


def analyze(collector, protocol, audit):
    rows = [_row_readout(row) for row in collector['rows']]
    trace = collector['trace_replay']
    budget = collector['budget']
    audit_checks = _audit_checks(audit)
    protocol_checks = _protocol_checks(protocol)
    domain_rows = {name: [row for row in rows if row['in_domain'] is inside]
        for name, inside in (('inside', True), ('outside', False))}
    distributions = {name: _distribution(values) for name, values in domain_rows.items()}
    checks = dict(fixed_protocol=all(protocol_checks.values()),
        reference_audit=all(audit_checks.values()), collector_complete=collector['status'] == 'complete',
        runtime_integrity=collector['integrity']['all_pass'] and all(collector['integrity']['checks'].values()),
        fixed_initializer=collector['seed'] == 6207, fixed_external_seed=collector['external_seed'] == 716207,
        fixed_scenario=collector['scenario'] == 'stable', same_prior_reference=_reference_valid(collector),
        same_reference_actor_path=collector['reference_snapshot']['source_actor_path'] == audit['source_actor_path'],
        reference_actual_seed=collector['reference_snapshot']['heldout_external_seed'] == 716207,
        full_trace_reproduced=trace['all_pass'] and trace['comparison_exact'] and trace['reference_exhausted']
            and trace['observed_issues'] == trace['matched_issues'],
        same_reference_trace=trace['reference_trace_path'] == audit['reference_trace_path'],
        all_window_7x_rows=len(rows) == trace['window_7x_issues'],
        all_window_7x_sequences=sorted(row['sequence'] for row in rows)
            == sorted(trace['window_7x_sequences']),
        unique_issue_sequences=len({row['sequence'] for row in rows}) == len(rows),
        complete_native_loads=all(row['checks']['native_load_complete'] for row in rows),
        all_issue_observations_valid=all(row['valid'] for row in rows),
        aggregate_observation_unchanged=all(collector['observation_checks'][key] is True for key in GUARDS),
        exact_native_prefix=budget['actual_simulated_seconds'] == 12600 and budget['prefixes'] == 1,
        zero_other_budget=all(budget[key] == 0 for key in ZERO_BUDGET),
        training_not_started=collector['training_started'] is False,
        no_label_selection=collector['labels_used_for_selection'] is False)
    integrity = all(checks.values())
    values = distributions['inside']['distinct_values']
    reachable = len(values) >= 2
    passed = integrity and reachable
    index = (len(values)-1)//2 if passed else None
    boundary = (values[index]+values[index+1])/2 if passed else None
    strata = {name: [row for row in domain_rows['inside']
        if (row['load'] < boundary if name == 'lighter' else row['load'] >= boundary)] if passed else []
        for name in ('lighter', 'heavier')}
    return dict(status='complete' if passed else 'hold', integrity=dict(all_pass=integrity, checks=checks),
        audit_checks=audit_checks, protocol_checks=protocol_checks, distributions=distributions,
        domain_reasons={name: dict(Counter(row['domain_reason'] for row in values
            if row['domain_reason'] is not None)) for name, values in domain_rows.items()},
        stratification=dict(status='frozen' if passed else 'hold', line_id='7X',
            boundary=boundary, rule='lighter: load<boundary; heavier: load>=boundary',
            selected_adjacent_distinct_values=[values[index], values[index+1]] if passed else None,
            distinct_gap_index=index,
            sample_counts={name: len(values) if passed else None for name, values in strata.items()},
            earliest_issue={name: _earliest(values) if passed else None for name, values in strata.items()},
            feature_definition_unchanged=True,
            other_line_boundaries=protocol['stratification']['other_lines_unchanged'],
            meaning='Relative load strata inside the original F action domain; heavier does not mean at least half capacity'),
        coverage_gate=dict(passed=passed, checks=dict(technical_integrity=integrity,
            two_distinct_inside_load_values=reachable),
            decision='Reachable covariate-only 7X strata frozen; separately decide collection and fitting'
                if passed else 'HOLD; no boundary frozen and no automatic new prefix or collection'),
        exceptions=dict(invalid_observations=[dict(sequence=row['sequence'], issue_clock=row['issue_clock'],
            failed_checks=[key for key, value in row['checks'].items() if not value]) for row in rows if not row['valid']]),
        source=dict(scenario=collector['scenario'], external_seed=collector['external_seed'],
            prefix_id=collector['prefix_id'], reference_snapshot=collector['reference_snapshot'],
            trace_replay=trace), budget=budget,
        interpretation=dict(stage='native_load_coverage_only', reward_or_cost_used=False,
            new_fits=budget['new_fits'], new_labels=budget['labels'], new_suffixes=budget['suffixes'],
            future_collection_scope='Later collection still requires the original complete atomic service context and native physical pair gates',
            old_v88_gate_unchanged=True, old_v88_high_not_filled=True,
            no_automatic_collection_or_training=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collector', 'protocol', 'audit', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = analyze(json.loads(args.collector.read_text()), json.loads(args.protocol.read_text()),
        json.loads(args.audit.read_text()))
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 16000:
        raise ValueError('Load coverage summary exceeds16KB; raw observations remain resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()), coverage=value['coverage_gate']['passed'])))


if __name__ == '__main__':
    main()
