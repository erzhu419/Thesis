"""Audit one fixed completion query and the twelve-row covariate design; no fit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
from experiments.mcwm_closed_loop_fresh_response_v86_20261027.analysis import _reference_valid
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import (
    action_support, make_design, service_state,
)

SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
BINS = ('lighter', 'heavier')
CUTS = {'7X': .13, '102X': .5, '705X': .5}
CASES = {(scene, line+'/'+group) for scene in SCENARIOS for line in LINES for group in BINS}
NEW_CASE = ('local_speed_drop', '705X/lighter')
COST_KEYS = ('wait', 'ride', 'cost')
ZERO_BUDGET = ('new_initialization_updates', 'prefix_training_updates', 'suffix_updates',
               'labels_fed_back', 'new_fits', 'new_neural_updates', 'v87_truth_used')


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _condition_readout(condition, now, origin):
    sources = condition['latest_sources']
    value = condition['value']
    past = (bool(sources) and all(row['completed_at'] < now and row['observed_at'] < now for row in sources))
    if origin == 'old_v90_fixed':
        # v91's immutable compact manifest retains one maximum observed clock,
        # not the full per-line clock maps. Its source definition is audited.
        baseline = (all(condition['baseline_counts'][line] >= 3 for line in LINES)
            and _finite(condition['baseline_last_observed_at']) and condition['baseline_last_observed_at'] < 9000)
        baseline_provenance = dict(source_schema='v91_fixed_compact',
            baseline_last_observed_at=condition['baseline_last_observed_at'])
    else:
        baseline = (condition['baseline_before'] == 9000 and all(condition['baseline_counts'][line] >= 3
            and condition['baseline_latest_completed_at'][line] < 9000
            and condition['baseline_latest_observed_at'][line] < 9000 for line in LINES))
        baseline_provenance = dict(source_schema='native_completion_full', baseline_before=condition['baseline_before'],
            baseline_latest_completed_at=condition['baseline_latest_completed_at'],
            baseline_latest_observed_at=condition['baseline_latest_observed_at'])
    latest = condition['latest_completed_at']
    age = condition['latest_age_seconds']
    checks = dict(supported=condition['status'] == 'supported' and _finite(value),
        exact_issue_clock=condition['issue_clock'] == now,
        explicitly_historical=condition['historical_only'] is True,
        strict_past_sources=past, three_line_pre9000_baselines=baseline,
        latest_age=_finite(latest) and _finite(age) and 0 < now-latest == age <= 900,
        latest_sources_exact=bool(sources) and latest == max(row['completed_at'] for row in sources)
            and condition['latest_observed_at'] == max(row['observed_at'] for row in sources),
        tied_latest_only=bool(sources) and all(row['completed_at'] == latest for row in sources),
        mean_normalized_sources=bool(sources) and all(_finite(row['normalized_slowdown']) for row in sources)
            and value == mean(row['normalized_slowdown'] for row in sources))
    return dict(valid=all(checks.values()), checks=checks, status=condition['status'], reason=condition['reason'],
        value=value, historical_only=condition['historical_only'], issue_clock=condition['issue_clock'],
        baseline_counts=condition['baseline_counts'], baseline_medians_seconds=condition['baseline_medians_seconds'],
        baseline_provenance=baseline_provenance,
        latest_completed_at=latest, latest_observed_at=condition['latest_observed_at'], latest_age_seconds=age,
        latest_sources=[{key: row[key] for key in ('line_id', 'trip_id', 'completed_at', 'observed_at',
            'nonholding_ready_seconds', 'baseline_nonholding_seconds', 'normalized_slowdown')} for row in sources])


def _pair_readout(pair, origin):
    root, service = pair['root'], pair['service']
    scenario, line, group = pair['scenario'], pair['line_id'], pair['load_bin']
    slot = line+'/'+group
    q = _condition_readout(pair['condition'], root['started_at'], origin)
    reference, candidate = (pair['remaining'][role] for role in ('reference', 'candidate'))
    delta = {key: candidate[key]-reference[key] for key in COST_KEYS}
    support = action_support(pair['reference_hold_seconds'], pair['candidate_hold_seconds'],
        pair['sim_step_seconds'], pair['action_step_seconds'])
    observed = service_state(pair['context'])
    load = service['features'][0] if service['observed'] else None
    actual_bin = ('lighter' if load < CUTS[line] else 'heavier') if load is not None else None
    physical = pair['physical']
    departures = [physical[role]['departure_clock'] for role in ('reference', 'candidate')]
    gap = abs(departures[1]-departures[0]) if all(value is not None for value in departures) else None
    checks = dict(declared_case=(scenario, slot) in CASES,
        fixed_external_seed=pair['external_seed'] == 716207,
        fixed_native_bin=group == actual_bin,
        atomic_service=service == observed and service['observed'] and service['atomic_at_issue']
            and service['issue_clock'] == root['started_at'],
        runtime_integrity=pair['all_pass'] and all(pair['checks'].values()),
        native_action_support=support['supported'],
        q_historical_and_supported=q['valid'],
        actual_pair_parallel=pair['overlap_seconds'] > 0,
        physical_clocks_complete=all(physical[role][field] is not None for role in ('reference', 'candidate')
            for field in ('departure_clock', 'next_arrival_clock')),
        physical_separation=gap is not None and gap >= pair['sim_step_seconds'],
        real_remaining_duration=reference['dt'] == candidate['dt'] == 24000-root['started_at'] > 0,
        real_terminal_maturity=all(row['observed_at'] == 24000 and row['available_at'] == 24030
            for row in (reference, candidate)),
        outcome_cost_identity=all(row['cost'] == 2*row['wait']+row['ride'] for row in (reference, candidate)),
        paired_delta_identity=pair['delta'] == delta and delta['cost'] == 2*delta['wait']+delta['ride'],
        declared_issue_window=12000 <= root['started_at'] < (12900 if origin == 'new_completion' else 12600))
    valid = all(checks.values())
    return dict(scenario=scenario, external_seed=pair['external_seed'], line_id=line, load_bin=group, slot=slot,
        root=root, physical_root_id=[scenario, pair['external_seed'], root['sequence']], origin=origin,
        source_pair_path=pair.get('source_pair_path'), outcome_source_pair_path=pair.get('outcome_source_pair_path'),
        valid=valid, checks=checks, service_features=service['features'],
        reference_hold_seconds=pair['reference_hold_seconds'], candidate_hold_seconds=pair['candidate_hold_seconds'],
        condition=q, overlap_seconds=pair['overlap_seconds'], physical=physical, departure_gap_seconds=gap,
        dt=reference['dt'], label_delta=delta if valid else None,
        label_delta_per_passenger={key: value/15079 for key, value in delta.items()} if valid else None)


def _design(pairs, readouts):
    service_rows = [dict(line_id=pair['line_id'], service=pair['service'],
        ref_hold=pair['reference_hold_seconds'], cand_hold=pair['candidate_hold_seconds'])
        for pair, row in zip(pairs, readouts) if row['checks']['atomic_service'] and row['checks']['native_action_support']]
    base = make_design(service_rows) if {row['line_id'] for row in service_rows} == set(LINES) else None
    eligible = len(service_rows) == len(pairs) and all(row['condition']['valid'] for row in readouts)
    q_means, q_ranges, matrix6 = {}, {}, None
    for line in LINES:
        values = [(pair['candidate_hold_seconds']-pair['reference_hold_seconds'])/7.5
            for pair in pairs if pair['line_id'] == line]
        q = [row['condition']['value'] for row in readouts if row['line_id'] == line and row['condition']['valid']]
        q_ranges[line] = dict(roots=sum(row['line_id'] == line for row in readouts), supported=len(q),
            distinct_values=sorted(set(q)), min=min(q) if q else None, max=max(q) if q else None,
            range=max(q)-min(q) if q else None)
        if eligible and values:
            q_means[line] = sum(a*a*value for a, value in zip(values, q))/sum(a*a for a in values)
        else:
            q_means[line] = None
    if base is not None and eligible:
        added = [((pair['candidate_hold_seconds']-pair['reference_hold_seconds'])/7.5)
            *(row['condition']['value']-q_means[pair['line_id']]) for pair, row in zip(pairs, readouts)]
        matrix6 = np.column_stack((base['matrix']['conditional'], added)).tolist()
    return dict(rows=len(pairs), service_design_rows=len(service_rows),
        ranks=dict(line=base['ranks']['line'] if base else None,
            service5=base['ranks']['conditional'] if base else None,
            service_q6=int(np.linalg.matrix_rank(np.asarray(matrix6))) if matrix6 is not None else None),
        service_line_centers=base['line_means'] if base else None, q_line_centers=q_means,
        q_variation_by_line=q_ranges,
        feature_names=dict(line=['line_'+line for line in LINES],
            service5=['line_'+line for line in LINES]+['load_centered', 'margin_centered'],
            service_q6=['line_'+line for line in LINES]+['load_centered', 'margin_centered', 'corridor_q_centered']),
        matrix6=matrix6, scope='Issue covariates and executed action differences only; no target solve')


def _signals(rows):
    valid = [row for row in rows if row['valid']]
    return dict(pairs=len(rows), valid_pairs=len(valid),
        min_delta_per_passenger={key: min(row['label_delta_per_passenger'][key] for row in valid)
            if valid else None for key in COST_KEYS},
        max_delta_per_passenger={key: max(row['label_delta_per_passenger'][key] for row in valid)
            if valid else None for key in COST_KEYS},
        negative_delta_labels=sum(row['label_delta']['cost'] < 0 for row in valid),
        zero_delta_labels=sum(row['label_delta']['cost'] == 0 for row in valid),
        positive_delta_labels=sum(row['label_delta']['cost'] > 0 for row in valid),
        scope='Descriptive paired response labels; no combined-policy cost or prediction accuracy')


def _audit_checks(audit):
    return dict(source_audit=audit['all_pass'] and all(audit['checks'].values()),
        source_initializer=audit['source_initializer'] == 6207,
        fixed_external_seed=audit['external_seed'] == 716207,
        twelve_original_reference_roots=audit['reference_roots'] == 12,
        both_scene_reference_paths=set(audit['reference_source_paths']) == set(SCENARIOS),
        frozen_native_thresholds=audit['thresholds'] == CUTS,
        eleven_fixed_old_pairs=audit['reused_pairs'] == 11 and len(audit['old_pairs']) == 11,
        twenty_two_old_labels=audit['reused_labels'] == 22,
        history_separate=audit['historical_native_seconds'] == 314090,
        zero_source_native_seconds=audit['native_seconds'] == 0,
        zero_source_new_fits=audit['new_fits'] == 0,
        zero_source_neural_updates=audit['new_neural_updates'] == 0,
        zero_source_suffixes=audit['new_suffixes'] == 0,
        zero_source_new_labels=audit['new_labels'] == 0,
        v87_truth_sealed=audit['v87_truth_used'] == 0)


def _protocol_checks(protocol):
    budget = protocol['budget']
    expected = dict(max_native_simulation_seconds=36900, historical_native_seconds=314090,
        cumulative_cap=350990, new_prefixes=1, new_pairs=1, new_suffixes=2, new_labels=2,
        reused_pairs=11, reused_labels=22, logical_pairs=12, logical_labels=24,
        new_fits=0, new_neural_updates=0, v87_truth_used=0)
    return dict(fixed_source=protocol['source_initializer'] == 6207 and protocol['external_seed'] == 716207
            and protocol['scenario'] == 'local_speed_drop',
        fixed_load_thresholds=protocol['thresholds'] == CUTS,
        frozen_completion_window=protocol['issue_window'] == [12000, 12900],
        fixed_label_clocks=protocol['terminal'] == 24000 and protocol['label_available_at'] == 24030,
        fixed_new_and_historical_budget=all(budget[key] == value for key, value in expected.items()))


def analyze(collector, protocol, audit):
    old = audit['old_pairs']
    old_cases = [(pair['scenario'], pair['line_id']+'/'+pair['load_bin']) for pair in old]
    if len(set(old_cases)) != len(old_cases) or set(old_cases)-CASES or NEW_CASE in old_cases:
        raise ValueError('The immutable old manifest must not duplicate or replace the completion slot')
    new = collector['pairs']
    if len(new) > 1 or any((pair['scenario'], pair['line_id']+'/'+pair['load_bin']) != NEW_CASE for pair in new):
        raise ValueError('Only the one declared shock705X lighter completion may be queried')
    coverage = collector['coverage']
    if set(coverage)-{NEW_CASE[1]}:
        raise ValueError('Only the completion covariate slot may be selected')
    records = list(old)
    for pair in new:
        row = dict(pair)
        selected = coverage.get(NEW_CASE[1])
        row['context'] = selected['context'] if selected is not None else pair['context']
        records.append(row)
    rows = [_pair_readout(pair, 'old_v90_fixed' if i < len(old) else 'new_completion')
        for i, pair in enumerate(records)]
    budget = collector['budget']
    audit_checks = _audit_checks(audit)
    protocol_checks = _protocol_checks(protocol)
    identities = [tuple(row['physical_root_id']) for row in rows]
    all_cases = {(row['scenario'], row['slot']) for row in rows}
    old_shock = {pair['line_id']+'/'+pair['load_bin']: pair for pair in old if pair['scenario'] == 'local_speed_drop'}
    reproduced = collector['reuse_checks']
    old_reproduction = (set(reproduced) == set(old_shock) and len(old_shock) == 5
        and all(bool(checks) and all(checks.values()) for checks in reproduced.values()))
    replay = collector['prefix_reproduction']
    new_root_matches = bool(new) and NEW_CASE[1] in coverage and (
        new[0]['root'] == coverage[NEW_CASE[1]]['root']
        and new[0]['service'] == coverage[NEW_CASE[1]]['service']
        and new[0]['condition'] == coverage[NEW_CASE[1]]['condition'])
    gate = collector['first_pair_gate']
    first_checks = dict(one_new_pair=len(new) == 1,
        physical_gate_passed=gate['all_pass'] is True)
    if new:
        first_checks.update(same_pair_checks=gate['checks'] == new[0]['checks'],
            all_physical_gate_checks=all(gate['checks'].values()),
            no_label_gate=gate['labels_used_for_gate'] is False,
            no_cost_sign_gate=gate['cost_sign_used_for_gate'] is False)
    design = _design(records, rows)
    checks = dict(frozen_old_manifest=all(audit_checks.values()), frozen_completion_protocol=all(protocol_checks.values()),
        collector_complete=collector['status'] == 'complete',
        collector_runtime_integrity=collector['integrity']['all_pass'] and all(collector['integrity']['checks'].values()),
        fixed_initializer=collector['seed'] == 6207,
        fixed_external_seed_and_scene=collector['external_seed'] == 716207 and collector['scenario'] == 'local_speed_drop',
        fixed_frozen_reference=_reference_valid(collector),
        same_source_actor=collector['reference_snapshot']['source_actor_path'] == audit['reference_source_paths']['local_speed_drop'],
        reference_actual_seed=collector['reference_snapshot']['heldout_external_seed'] == 716207,
        original_eleven_cases=set(old_cases) == CASES-{NEW_CASE},
        old_source_paths_present=all(bool(pair['source_pair_path']) and bool(pair['outcome_source_pair_path']) for pair in old),
        old_source_paths_unique=len({pair['source_pair_path'] for pair in old}) == 11,
        old_shock_five_roots_reproduced=old_reproduction,
        exact_extended_reference_prefix=replay['all_pass'] and all(replay['checks'].values())
            and replay['observed_issues'] == replay['matched_issues'] > 0,
        new_covariate_root_exact=new_root_matches,
        one_new_pair_physical_preflight=all(first_checks.values()),
        twelve_distinct_physical_roots=len(rows) == 12 and len(set(identities)) == 12,
        logical_cohort_complete=all_cases == CASES,
        all_twelve_valid_pairs=len(rows) == 12 and all(row['valid'] for row in rows),
        all_twelve_q_supported=len(rows) == 12 and all(row['condition']['valid'] for row in rows),
        line_rank_three=design['ranks']['line'] == 3,
        service_rank_five=design['ranks']['service5'] == 5,
        service_q_rank_six=design['ranks']['service_q6'] == 6,
        original_history_not_new_budget=budget['reused_pairs'] == 11 and budget['reused_labels'] == 22,
        only_new_native_units=budget['prefixes'] == 1 and budget['suffixes'] == 2 and budget['labels_observed'] == 2,
        actual_prefix_12900=budget['native_prefix_seconds'] == 12900,
        known_native_spend=budget['unknown_suffix_costs'] == 0,
        native_budget_identity=budget['total_native_seconds'] == budget['native_prefix_seconds']+budget['suffix_native_seconds'],
        new_native_budget=budget['total_native_seconds'] <= 36900,
        cumulative_budget=budget['total_native_seconds']+audit['historical_native_seconds'] <= 350990,
        zero_training_fits_or_feedback=all(budget[key] == 0 for key in ZERO_BUDGET))
    passed = all(checks.values())
    return dict(status='complete' if passed else 'hold', integrity=dict(all_pass=passed, checks=checks),
        audit_checks=audit_checks, protocol_checks=protocol_checks, first_pair_checks=first_checks,
        pairs=rows, covariate_design=design,
        label_signals=dict(overall=_signals(rows),
            by_scenario={scene: _signals([row for row in rows if row['scenario'] == scene]) for scene in SCENARIOS},
            by_line={line: _signals([row for row in rows if row['line_id'] == line]) for line in LINES}),
        budget=dict(new=budget, historical_native_seconds=audit['historical_native_seconds'],
            cumulative_native_seconds=budget['total_native_seconds']+audit['historical_native_seconds']
                if budget['unknown_suffix_costs'] == 0 else None,
            new_cap=36900, cumulative_cap=350990, old_logical_pairs=len(old), new_pairs=len(new),
            observed_logical_pairs=len(rows), valid_logical_pairs=sum(row['valid'] for row in rows)),
        exceptions=dict(missing_cases=[dict(scenario=scene, slot=slot) for scene, slot in sorted(CASES-all_cases)],
            invalid_pairs=[dict(scenario=row['scenario'], slot=row['slot'], checks=row['checks']) for row in rows if not row['valid']],
            unsupported_q=[dict(scenario=row['scenario'], slot=row['slot'], condition=row['condition'])
                for row in rows if not row['condition']['valid']],
            zero_duration=[dict(scenario=row['scenario'], slot=row['slot']) for row in rows if row['dt'] == 0]),
        completion_gate=dict(passed=passed, decision='The fixed twelve-pair cohort and historical condition are complete; decide any fit separately'
            if passed else 'HOLD completion; preserve missing or invalid cases without replacement'),
        interpretation=dict(stage='completion_collection_only', model_fits=budget['new_fits'],
            neural_updates=budget['new_neural_updates'], policy_cost_evaluated=False,
            rank_scope='Ranks3/5/6 and weighted centers use covariates/actions only; no least-squares target solve',
            old_v90_hold_unchanged=True, v87_truth_sealed=True, no_automatic_fitting_or_control_takeover=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collector', 'audit', 'protocol', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = analyze(json.loads(args.collector.read_text()), json.loads(args.protocol.read_text()),
        json.loads(args.audit.read_text()))
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 64000:
        raise ValueError('Completion summary exceeds64KB; raw source records remain resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()), completion=value['completion_gate']['passed'])))


if __name__ == '__main__':
    main()
