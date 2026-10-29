"""Audit the fixed twelve-pair service cohort without fitting or policy scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_fresh_response_v86_20261027.analysis import _reference_valid
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import (
    action_support, make_design, service_state,
)

SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
BINS = ('lighter', 'heavier')
SLOTS = tuple(line+'/'+group for line in LINES for group in BINS)
CUTS = {'7X': .13, '102X': .5, '705X': .5}
REUSED_SLOT = ('stable', '7X/heavier')
GUARDS = ('clock_unchanged', 'ledger_unchanged', 'rng_unchanged',
          'issue_clock_exact', 'context_before_action')
COST_KEYS = ('wait', 'ride', 'cost')
BUDGET_FIELDS = ('native_prefix_seconds', 'suffix_native_seconds', 'total_native_seconds', 'prefixes',
    'suffixes', 'labels_observed', 'new_initialization_updates', 'prefix_training_updates', 'suffix_updates',
    'labels_fed_back', 'new_fits', 'new_neural_updates', 'v87_truth_used', 'reused_pairs', 'reused_labels')
HISTORY = dict(v88_native_seconds=36598, v89_native_seconds=12600)


def _slot_readout(scenario, key, row):
    line, group = key.split('/')
    root, binding = row['root'], row['binding']
    service = service_state(row['context'])
    supported = action_support(row['ref_hold'], row['cand_hold'],
        row['sim_step_seconds'], row['action_step_seconds'])
    load = service['features'][0] if service['observed'] else None
    actual_bin = ('lighter' if load < CUTS[line] else 'heavier') if load is not None else None
    checks = dict(declared_line=row['line_id'] == binding['line_id'] == line,
        declared_new_bin=row['load_bin'] == actual_bin == group,
        recomputed_native_service=row['service'] == service,
        atomic_visible_context=service['observed'] and service['atomic_at_issue'],
        fixed_issue_window=12000 <= root['started_at'] < 12600,
        complete_issue_binding=binding['episode_id'] == 0 and binding['sequence'] == root['sequence']
            and binding['trip_id'] == root['trip_id'] and binding['station_id'] == root['station_id'],
        issue_clock_binding=binding['issue_clock'] == service['issue_clock'] == root['started_at'],
        event_clock_binding=binding['event_clock'] == service['event_clock']
            and binding['event_clock'] <= binding['issue_clock'],
        capture_does_not_change_native_state=all(row['observation_checks'][key] is True for key in GUARDS),
        old_executed_action_support=supported['supported'])
    return dict(scenario=scenario, slot=key, root=root, service=service,
        load_bin=actual_bin, legacy_half_capacity_bin=service['load_bin'],
        ref_hold=row['ref_hold'], cand_hold=row['cand_hold'],
        valid=all(checks.values()), checks=checks)


def _pair_readout(scene, pair, slots):
    scenario = scene['scenario']
    key = pair['line_id']+'/'+pair['load_bin']
    slot = slots.get(key)
    reference, candidate = (pair['remaining'][role] for role in ('reference', 'candidate'))
    physical = pair['physical']
    delta = {name: candidate[name]-reference[name] for name in COST_KEYS}
    departures = [physical[role]['departure_clock'] for role in ('reference', 'candidate')]
    gap = abs(departures[1]-departures[0]) if all(value is not None for value in departures) else None
    expected_reuse = (scenario, key) == REUSED_SLOT
    checks = dict(runtime_pair_integrity=pair['all_pass'] and all(pair['checks'].values()),
        coverage_present_and_valid=slot is not None and slot['valid'],
        same_root_as_selected=slot is not None and pair['root'] == slot['root'],
        same_service_as_selected=slot is not None and pair['service'] == slot['service'],
        same_executed_actions=slot is not None and pair['reference_hold_seconds'] == slot['ref_hold']
            and pair['candidate_hold_seconds'] == slot['cand_hold'],
        fixed_reuse_slot=pair['reused'] is expected_reuse,
        source_path_for_reuse=bool(pair.get('source_pair_path')) if expected_reuse else True,
        fixed_new_threshold=pair['new_stratum_threshold'] == CUTS[pair['line_id']],
        original_source_bin=pair.get('source_load_bin') == 'low' if expected_reuse else pair.get('source_load_bin') is None,
        physical_clocks_complete=all(physical[role][field] is not None for role in ('reference', 'candidate')
            for field in ('departure_clock', 'next_arrival_clock')),
        physical_departure_separation=gap is not None and gap >= pair['sim_step_seconds'],
        actual_pair_parallel=pair['overlap_seconds'] > 0,
        real_remaining_duration=reference['dt'] == candidate['dt'] == 24000-pair['root']['started_at'] > 0,
        real_terminal_maturity=all(row['observed_at'] == 24000 and row['available_at'] == 24030
            for row in (reference, candidate)),
        outcome_cost_identities=all(row['cost'] == 2*row['wait']+row['ride'] for row in (reference, candidate)),
        paired_delta_identity=pair['delta'] == delta and delta['cost'] == 2*delta['wait']+delta['ride'])
    valid = pair['status'] == 'complete' and all(checks.values())
    return dict(scenario=scenario, external_seed=scene['external_seed'], slot=key,
        line_id=pair['line_id'], load_bin=pair['load_bin'], root=pair['root'],
        physical_root_id=[scenario, scene['external_seed'], pair['root']['sequence']],
        reused=pair['reused'], source_pair_path=pair.get('source_pair_path'),
        valid=valid, checks=checks, service=pair['service'],
        reference_hold_seconds=pair['reference_hold_seconds'], candidate_hold_seconds=pair['candidate_hold_seconds'],
        overlap_seconds=pair['overlap_seconds'], physical=physical, departure_gap_seconds=gap,
        dt=reference['dt'], observed_at=reference['observed_at'], available_at=reference['available_at'],
        label_delta=delta if valid else None,
        label_delta_per_passenger={name: value/15079 for name, value in delta.items()} if valid else None)


def _label_statistics(rows):
    valid = [row for row in rows if row['valid']]
    return dict(pairs=len(rows), valid_pairs=len(valid), physical_outcome_labels=2*len(valid),
        min_delta_per_passenger={key: min(row['label_delta_per_passenger'][key] for row in valid)
            if valid else None for key in COST_KEYS},
        max_delta_per_passenger={key: max(row['label_delta_per_passenger'][key] for row in valid)
            if valid else None for key in COST_KEYS},
        mean_delta_per_passenger={key: sum(row['label_delta_per_passenger'][key] for row in valid)/len(valid)
            if valid else None for key in COST_KEYS},
        negative_delta_labels=sum(row['label_delta']['cost'] < 0 for row in valid),
        zero_delta_labels=sum(row['label_delta']['cost'] == 0 for row in valid),
        positive_delta_labels=sum(row['label_delta']['cost'] > 0 for row in valid),
        scope='Descriptive paired training-label signals; neither policy cost nor model prediction accuracy')


def _audit_checks(audit):
    return dict(resident_audit=audit['all_pass'] and all(audit['checks'].values()),
        fixed_initializer=audit['source_initializer'] == 6207,
        fixed_external_seed=audit['external_seed'] == 716207,
        two_reference_scenarios=set(audit['reference_source_paths']) == set(SCENARIOS),
        twelve_original_reference_roots=audit['reference_roots'] == 12,
        frozen_thresholds=audit['thresholds'] == CUTS,
        exactly_two_reused_labels=audit['reused_labels'] == 2,
        no_fits=audit['new_fits'] == 0, no_neural_updates=audit['new_neural_updates'] == 0,
        no_native_audit_seconds=audit['native_seconds'] == 0,
        no_audit_suffixes=audit['new_suffixes'] == 0, no_audit_labels=audit['new_labels'] == 0,
        no_v87_truth=audit['v87_truth_used'] == 0)


def _protocol_checks(protocol):
    budget, collection = protocol['budget'], protocol['collection']
    expected = dict(native_prefixes=2, native_suffixes=22, native_pairs=11, physical_outcome_labels=22,
        reused_pairs=1, reused_labels=2, logical_cohort_pairs=12, logical_cohort_labels=24,
        max_native_simulation_seconds=289200, initialization_steps=0, closed_form_fits=0,
        neural_updates=0, feedback_admission=0, v87_truth_used=0,
        cumulative_native_upper_bound_through_v90=338398)
    return dict(fixed_initializer=protocol['source_initializer'] == 6207,
        fixed_source_seed_and_scenarios=protocol['external_seed'] == 716207 and protocol['scenarios'] == list(SCENARIOS),
        fixed_issue_window=protocol['issue_window'] == [12000, 12600],
        fixed_terminal_maturity=protocol['terminal'] == 24000 and protocol['label_available_at'] == 24030,
        fixed_thresholds=protocol['thresholds'] == protocol['features']['load_bin_threshold'] == CUTS,
        original_two_features=protocol['features']['names'] == ['onboard_capacity_ratio', 'next_stop_capacity_margin'],
        original_action_domain=protocol['action_support'] == dict(reference_min=3., reference_max_exclusive=3.75,
            candidate=7.5, candidate_tolerance=1e-5, minimum_gap_native_steps=2),
        fixed_collection_slots=collection['pairs'] == 12 and collection['slots_per_scenario'] == list(SLOTS),
        unique_reuse_slot=(collection['reuse']['scenario'], collection['reuse']['slot']) == REUSED_SLOT,
        fixed_current_budget=all(budget[key] == value for key, value in expected.items()),
        fixed_scene_caps=budget['scenario_caps'] == dict(stable=132600, local_speed_drop=156600),
        historical_budget_separate=budget['historical_separate'] == dict(v88_actual_native_seconds=36598,
            v89_actual_native_seconds=12600, total_actual_before_v90=49198),
        future_validation_not_started=protocol['future_validation']['not_started'] is True)


def _scene_readout(scene, audit):
    scenario = scene['scenario']
    if set(scene['coverage'])-set(SLOTS) or set(scene['selection']) != set(SLOTS):
        raise ValueError('Only the six declared lighter/heavier slots are supported')
    slots = {key: _slot_readout(scenario, key, scene['coverage'][key]) for key in SLOTS if key in scene['coverage']}
    keys = [pair['line_id']+'/'+pair['load_bin'] for pair in scene['pairs']]
    if len(set(keys)) != len(keys) or set(keys)-set(SLOTS):
        raise ValueError('Duplicate or undeclared native pair slot')
    pairs = [_pair_readout(scene, pair, slots) for pair in scene['pairs']]
    reused = [row for row in pairs if row['reused']]
    budget = scene['budget']
    expected_reuse = int(scenario == 'stable')
    checks = dict(complete_collector=scene['status'] == 'complete',
        runtime_integrity=scene['integrity']['all_pass'] and all(scene['integrity']['checks'].values()),
        fixed_initializer=scene['seed'] == 6207, fixed_external_seed=scene['external_seed'] == 716207,
        fixed_denominator=scene['departed_passengers'] == 15079,
        frozen_prior_reference=_reference_valid(scene),
        same_source_actor=scene['reference_snapshot']['source_actor_path'] == audit['reference_source_paths'][scenario],
        reference_actual_seed=scene['reference_snapshot']['heldout_external_seed'] == 716207,
        all_six_selected_slots=set(slots) == set(SLOTS),
        all_six_visible_contexts=len(slots) == 6 and all(row['valid'] for row in slots.values()),
        all_six_pairs=set(keys) == set(SLOTS),
        all_pairs_valid=len(pairs) == 6 and all(row['valid'] for row in pairs),
        expected_reuse_count=len(reused) == expected_reuse,
        native_prefix_end=budget['native_prefix_seconds'] == 12600,
        known_native_spend=budget['unknown_suffix_costs'] == 0,
        native_budget_identity=budget['total_native_seconds'] == budget['native_prefix_seconds']+budget['suffix_native_seconds'],
        scene_native_budget=budget['total_native_seconds'] <= (132600 if scenario == 'stable' else 156600),
        one_native_prefix=budget['prefixes'] == 1,
        only_new_suffixes=budget['suffixes'] == 2*(6-expected_reuse),
        only_new_outcome_labels=budget['labels_observed'] == 2*(6-expected_reuse),
        reused_budget=budget['reused_pairs'] == expected_reuse and budget['reused_labels'] == 2*expected_reuse,
        zero_initialization=budget['new_initialization_updates'] == 0,
        zero_prefix_training=budget['prefix_training_updates'] == 0,
        zero_suffix_training=budget['suffix_updates'] == 0,
        zero_feedback=budget['labels_fed_back'] == 0,
        zero_fits=budget['new_fits'] == 0,
        zero_neural_updates=budget['new_neural_updates'] == 0,
        v87_truth_sealed=budget['v87_truth_used'] == 0,
        training_not_started=scene['training_started'] is False and scene['models_fitted'] is False,
        no_predictions_or_labels_for_sampling=scene['predictions_used_for_selection'] is False
            and scene['labels_used_for_selection'] is False)
    if scenario == 'stable':
        checks['source_root_reuse_exact'] = (len(reused) == 1
            and reused[0]['root'] == audit['reuse_source_root']
            and reused[0]['source_pair_path'] == audit['reuse_source_pair_path']
            and bool(scene['reuse_checks']) and all(scene['reuse_checks'].values()))
        replay = scene['prefix_reproduction']
        checks['exact_old_parent_prefix_and_eof'] = (replay['all_pass'] and all(replay['checks'].values())
            and replay['observed_issues'] == replay['matched_issues'] > 0
            and replay['checks']['reference_exhausted'] is True)
    return dict(integrity=dict(all_pass=all(checks.values()), checks=checks), slots=slots,
        pairs=pairs, budget={key: budget[key] for key in BUDGET_FIELDS},
        missing_slots=[key for key in SLOTS if key not in slots],
        missing_pairs=[key for key in SLOTS if key not in keys])


def _first_pair_checks(scenes, rows):
    stable = scenes.get('stable')
    if stable is None or 'local_speed_drop' not in scenes:
        return dict(two_scene_first_gate_receipts=False)
    gate = stable['first_pair_gate']
    new = sorted([row for row in rows if row['scenario'] == 'stable' and not row['reused']],
        key=lambda row: (row['root']['started_at'], row['root']['sequence']))
    selected = new[0] if new else None
    original = next((pair for pair in stable['pairs'] if selected is not None
        and pair['root'] == selected['root']), None)
    return dict(first_pair_gate_passed=gate['status'] == 'passed' and gate['all_pass']
            and all(gate['checks'].values()),
        same_gate_for_both_collectors=scenes['local_speed_drop']['first_pair_gate'] == gate,
        first_new_native_pair=selected is not None and gate['scenario'] == 'stable'
            and gate['slot'] == selected['slot'] and gate['root'] == selected['root']
            and selected['valid'],
        first_gate_checks_match_native_pair=original is not None and gate['checks'] == original['checks'],
        first_pair_is_in_formal_cohort=gate['new_pair_in_formal_cohort'] is True,
        no_labels_for_gate=gate['labels_used_for_gate'] is False,
        no_cost_sign_for_gate=gate['cost_sign_used_for_gate'] is False)


def analyze(scenes, protocol, audit):
    indexed = {}
    for scene in scenes:
        scenario = scene['scenario']
        if scenario not in SCENARIOS or scenario in indexed:
            raise ValueError('Use at most one collector for each fixed scenario')
        indexed[scenario] = scene
    reports = {scenario: _scene_readout(scene, audit) for scenario, scene in indexed.items()}
    rows = [row for value in reports.values() for row in value['pairs']]
    usable = [indexed[scenario]['coverage'][key] for scenario, value in reports.items()
        for key, slot in value['slots'].items() if slot['valid']]
    design = make_design(usable) if {row['line_id'] for row in usable} == set(LINES) else None
    budget = {key: sum(value['budget'][key] for value in reports.values()) for key in BUDGET_FIELDS}
    reused = [row for row in rows if row['reused']]
    roots = [tuple(row['physical_root_id']) for row in rows]
    audit_checks = _audit_checks(audit)
    protocol_checks = _protocol_checks(protocol)
    first_checks = _first_pair_checks(indexed, rows)
    checks = dict(fixed_collection_protocol=all(protocol_checks.values()),
        frozen_collection_audit=all(audit_checks.values()),
        both_declared_scenarios=set(indexed) == set(SCENARIOS),
        both_scene_integrity=len(reports) == 2 and all(value['integrity']['all_pass'] for value in reports.values()),
        twelve_unique_physical_pairs=len(rows) == 12 and len(set(roots)) == 12,
        all_twelve_native_pairs=len(rows) == 12 and all(row['valid'] for row in rows),
        one_old_pair_only=len(reused) == 1 and (reused[0]['scenario'], reused[0]['slot']) == REUSED_SLOT,
        eleven_new_pairs=sum(not row['reused'] for row in rows) == 11,
        two_new_prefixes=budget['prefixes'] == 2,
        twenty_two_new_suffixes=budget['suffixes'] == 22,
        twenty_two_new_labels=budget['labels_observed'] == 22,
        two_old_labels=budget['reused_pairs'] == 1 and budget['reused_labels'] == 2,
        current_new_budget=budget['total_native_seconds'] <= 289200,
        cumulative_budget=budget['total_native_seconds']+sum(HISTORY.values()) <= 338398,
        first_new_pair_preflight=all(first_checks.values()),
        potential_line_rank_three=design is not None and design['ranks']['line'] == 3,
        potential_conditional_rank_five=design is not None and design['ranks']['conditional'] == 5)
    passed = all(checks.values())
    complete_budget = len(reports) == 2 and all(
        value['integrity']['checks']['known_native_spend'] for value in reports.values())
    return dict(status='complete' if passed else 'hold',
        integrity=dict(all_pass=passed, checks=checks), audit_checks=audit_checks, protocol_checks=protocol_checks,
        first_pair_checks=first_checks,
        coverage={scenario: value['slots'] for scenario, value in reports.items()},
        pairs=rows, potential_design=design,
        label_signals=dict(overall=_label_statistics(rows),
            by_scenario={scenario: _label_statistics([row for row in rows if row['scenario'] == scenario]) for scenario in SCENARIOS},
            by_line={line: _label_statistics([row for row in rows if row['line_id'] == line]) for line in LINES}),
        budget=dict(current_new=budget, historical=HISTORY,
            current_new_complete=complete_budget,
            cumulative_native_seconds=budget['total_native_seconds']+sum(HISTORY.values()) if complete_budget else None,
            current_new_cap=289200, cumulative_cap=338398,
            declared_cohort_pairs=12, declared_cohort_physical_labels=24,
            observed_cohort_pairs=len(rows), valid_cohort_pairs=sum(row['valid'] for row in rows),
            observed_new_pairs=sum(not row['reused'] for row in rows), observed_reused_pairs=len(reused)),
        scene_integrity={scenario: value['integrity'] for scenario, value in reports.items()},
        exceptions=dict(missing_scenarios=[scenario for scenario in SCENARIOS if scenario not in indexed],
            missing_slots=[dict(scenario=scenario, slot=key) for scenario, value in reports.items() for key in value['missing_slots']],
            missing_native_pairs=[dict(scenario=scenario, slot=key) for scenario, value in reports.items() for key in value['missing_pairs']],
            invalid_contexts=[dict(scenario=scenario, slot=key, checks=row['checks']) for scenario, value in reports.items()
                for key, row in value['slots'].items() if not row['valid']],
            invalid_native_pairs=[dict(scenario=row['scenario'], slot=row['slot'], checks=row['checks']) for row in rows if not row['valid']],
            zero_duration=[dict(scenario=row['scenario'], slot=row['slot']) for row in rows if row['dt'] == 0]),
        collection_gate=dict(passed=passed, decision='Fixed paired collection complete; separately decide fitting after reviewing signal coverage'
            if passed else 'HOLD fixed collection; retain missing or invalid roots without replacement'),
        interpretation=dict(stage='paired_collection_only', fits=budget['new_fits'],
            neural_updates=budget['new_neural_updates'], policy_cost_evaluated=False,
            no_labels_or_signs_for_sampling=True,
            service_features='Original absolute native load/capacity and clipped queue-seat margin; old service.load_bin is legacy metadata only',
            rank_scope='Design feasibility from issue covariates/actions only; no coefficient fitting',
            old_v88_gate_unchanged=True, v87_truth_sealed=True,
            no_automatic_fitting_or_control_takeover=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stable', 'shock', 'protocol', 'audit', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    scenes = [json.loads(path.read_text()) for path in (args.stable, args.shock) if path.exists()]
    value = analyze(scenes, json.loads(args.protocol.read_text()), json.loads(args.audit.read_text()))
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 64000:
        raise ValueError('Collection summary exceeds64KB; raw native records must remain resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()), collection=value['collection_gate']['passed'])))


if __name__ == '__main__':
    main()
