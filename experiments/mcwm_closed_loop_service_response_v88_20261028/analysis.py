"""Score v88 native coverage and label feasibility; no fitting or control scoring."""
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

LINES = ('7X', '102X', '705X')
BINS = ('low', 'high')
SLOTS = tuple(line+'/'+group for line in LINES for group in BINS)
PAIR_SLOTS = ('7X/low', '7X/high')
OBSERVATION_CHECKS = ('clock_unchanged', 'ledger_unchanged', 'rng_unchanged',
                      'issue_clock_exact', 'context_before_action')
BUDGET_FIELDS = ('native_prefix_seconds', 'suffix_native_seconds', 'total_native_seconds',
                 'prefixes', 'suffixes', 'labels_observed', 'new_initialization_updates',
                 'prefix_training_updates', 'suffix_updates', 'labels_fed_back', 'new_fits')
COST_KEYS = ('wait', 'ride', 'cost')


def _protocol_checks(protocol):
    preflight, budget = protocol['preflight'], protocol['budget']
    expected_budget = dict(native_prefixes=1, native_suffixes=4, native_pairs=2,
        physical_outcome_labels=4, max_native_simulation_seconds=60600,
        initialization_steps=0, closed_form_fits=0, neural_updates=0,
        feedback_admission=0, v87_truth_used=0)
    return dict(source_initializer=protocol['source_initializer'] == 6207,
        fixed_preflight_seed_and_scenario=preflight['external_seed'] == 716207 and preflight['scenario'] == 'stable',
        fixed_window=preflight['issue_window'] == [12000, 12600],
        fixed_slots=preflight['lines'] == list(LINES) and preflight['load_bins'] == list(BINS)
            and preflight['expected_covariate_slots'] == 6,
        fixed_native_pairs=preflight['paired_line'] == '7X'
            and preflight['paired_slots'] == list(PAIR_SLOTS) and preflight['expected_pairs'] == 2,
        fixed_label_clocks=preflight['terminal'] == 24000 and preflight['label_available_at'] == 24030,
        fixed_features=protocol['features']['names'] == ['onboard_capacity_ratio', 'next_stop_capacity_margin']
            and protocol['features']['load_bin_threshold'] == .5,
        fixed_action_domain=protocol['action_support'] == dict(reference_min=3., reference_max_exclusive=3.75,
            candidate=7.5, candidate_tolerance=1e-5, minimum_gap_native_steps=2),
        fixed_budget=all(budget[key] == value for key, value in expected_budget.items()),
        future_validation_not_started=protocol['future_validation']['not_started'] is True)


def _audit_checks(audit):
    return dict(resident_reference_audit=audit['all_pass'] and all(audit['checks'].values()),
        same_initializer=audit['source_initializer'] == 6207,
        declared_external_seed=audit['external_seed'] == 716207,
        stable_scenario=audit['scenario'] == 'stable',
        twelve_original_reference_roots=audit['reference_roots'] == 12,
        two_scene_reference_sources=set(audit['reference_source_paths']) == {'stable', 'local_speed_drop'},
        zero_fits=audit['new_fits'] == 0,
        zero_neural_updates=audit['new_neural_updates'] == 0,
        v87_truth_not_used=audit['v87_truth_used'] == 0,
        audit_has_no_native_simulation=audit['native_seconds'] == 0)


def _slot_readout(key, row):
    line, group = key.split('/')
    root, binding = row['root'], row['binding']
    observed = service_state(row['context'])
    support = action_support(row['ref_hold'], row['cand_hold'],
        simulation_step=row['sim_step_seconds'], action_step=row['action_step_seconds'])
    service = row['service']
    checks = dict(declared_line=row['line_id'] == binding['line_id'] == line,
        declared_load_bin=row['load_bin'] == group,
        service_recomputed_exact=service == observed,
        observed_service=service['observed'] is True,
        issue_atomic=service['atomic_at_issue'] is True,
        expected_load_bin=service['load_bin'] == group,
        real_issue_window=12000 <= root['started_at'] < 12600,
        issue_binding=binding['episode_id'] == 0 and binding['sequence'] == root['sequence']
            and binding['trip_id'] == root['trip_id'] and binding['station_id'] == root['station_id'],
        issue_clock_binding=binding['issue_clock'] == service['issue_clock'] == root['started_at'],
        event_clock_binding=binding['event_clock'] == service['event_clock']
            and binding['event_clock'] <= binding['issue_clock'],
        observational_capture=all(row['observation_checks'][name] is True for name in OBSERVATION_CHECKS),
        original_positive_action_support=support['supported'] is True)
    return dict(slot=key, line_id=line, load_bin=group, root=root,
        valid=all(checks.values()), checks=checks, service=service,
        binding=binding, ref_hold=row['ref_hold'], cand_hold=row['cand_hold'],
        sim_step_seconds=row['sim_step_seconds'], action_step_seconds=row['action_step_seconds'],
        action_support=support)


def _pair_readout(pair, slots):
    key = pair['line_id']+'/'+pair['load_bin']
    selected = slots.get(key)
    reference, candidate = (pair['remaining'][role] for role in ('reference', 'candidate'))
    physical = pair['physical']
    delta = {name: candidate[name]-reference[name] for name in COST_KEYS}
    departures = [physical[role]['departure_clock'] for role in ('reference', 'candidate')]
    complete_clocks = all(physical[role][field] is not None for role in ('reference', 'candidate')
        for field in ('departure_clock', 'next_arrival_clock'))
    departure_gap = abs(departures[1]-departures[0]) if all(value is not None for value in departures) else None
    checks = dict(declared_native_slot=key in PAIR_SLOTS,
        selected_coverage_present=selected is not None,
        selected_coverage_valid=selected is not None and selected['valid'],
        selected_root_exact=selected is not None and pair['root'] == selected['root'],
        selected_service_exact=selected is not None and pair['service'] == selected['service'],
        selected_actions_exact=selected is not None
            and pair['reference_hold_seconds'] == selected['ref_hold']
            and pair['candidate_hold_seconds'] == selected['cand_hold'],
        runtime_pair_integrity=pair['all_pass'] and all(pair['checks'].values()),
        actual_parallel=pair['overlap_seconds'] > 0,
        physical_clocks_complete=complete_clocks,
        physical_departure_separation=departure_gap is not None and departure_gap >= pair['sim_step_seconds'],
        nonzero_real_duration=reference['dt'] == candidate['dt'] == 24000-pair['root']['started_at'] > 0,
        real_terminal_and_maturity=all(row['observed_at'] == 24000 and row['available_at'] == 24030
            for row in (reference, candidate)),
        label_cost_identity=all(row['cost'] == 2*row['wait']+row['ride'] for row in (reference, candidate)),
        paired_delta_identity=delta == pair['delta'] and delta['cost'] == 2*delta['wait']+delta['ride'])
    valid = pair['status'] == 'complete' and all(checks.values())
    return dict(slot=key, root=pair['root'], valid=valid, checks=checks,
        overlap_seconds=pair['overlap_seconds'], physical=physical, departure_gap_seconds=departure_gap,
        reference_hold_seconds=pair['reference_hold_seconds'], candidate_hold_seconds=pair['candidate_hold_seconds'],
        dt=reference['dt'], observed_at=reference['observed_at'], available_at=reference['available_at'],
        label_delta=delta if valid else None,
        label_scope='Recorded paired remaining-day cost; sign is not a preflight admission condition')


def analyze(collector, protocol, audit):
    coverage = collector['coverage']
    if set(coverage)-set(SLOTS) or set(collector['selection']) != set(SLOTS):
        raise ValueError('Only the six fixed line/load slots are declared')
    slots = {key: _slot_readout(key, coverage[key]) for key in SLOTS if key in coverage}
    pair_keys = [pair['line_id']+'/'+pair['load_bin'] for pair in collector['pairs']]
    if len(set(pair_keys)) != len(pair_keys) or set(pair_keys)-set(PAIR_SLOTS):
        raise ValueError('Only one native pair per declared 7X load slot is allowed')
    pairs = [_pair_readout(pair, slots) for pair in collector['pairs']]
    usable = [coverage[key] for key, row in slots.items() if row['valid']]
    design = make_design(usable) if {row['line_id'] for row in usable} == set(LINES) else None
    budget = {name: collector['budget'][name] for name in BUDGET_FIELDS}
    audit_checks = _audit_checks(audit)
    protocol_checks = _protocol_checks(protocol)
    checks = dict(fixed_preflight_protocol=all(protocol_checks.values()),
        frozen_reference_audit=all(audit_checks.values()),
        collector_complete=collector['status'] == 'complete',
        runtime_collector_integrity=collector['integrity']['all_pass'],
        fixed_initializer=collector['seed'] == 6207,
        fixed_external_seed=collector['external_seed'] == 716207,
        fixed_scenario=collector['scenario'] == 'stable',
        same_prior_reference=_reference_valid(collector),
        same_source_reference_path=collector['reference_snapshot']['source_actor_path']
            == audit['reference_source_paths']['stable'],
        reference_actual_seed=collector['reference_snapshot']['heldout_external_seed'] == 716207,
        all_six_covariate_slots=set(slots) == set(SLOTS),
        all_covariate_slots_valid=len(slots) == 6 and all(row['valid'] for row in slots.values()),
        each_line_two_load_states=len(slots) == 6 and all(
            slots[line+'/low']['service']['features'] != slots[line+'/high']['service']['features'] for line in LINES),
        potential_line_rank_three=design is not None and design['ranks']['line'] == 3,
        potential_conditional_rank_five=design is not None and design['ranks']['conditional'] == 5,
        two_declared_native_pairs=set(pair_keys) == set(PAIR_SLOTS),
        both_native_pairs_valid=len(pairs) == 2 and all(row['valid'] for row in pairs),
        native_prefix_bound=0 <= budget['native_prefix_seconds'] <= 12600,
        native_suffix_bound=0 <= budget['suffix_native_seconds'] <= 48000,
        native_budget_bound=0 <= budget['total_native_seconds'] <= 60600,
        native_budget_identity=budget['total_native_seconds']
            == budget['native_prefix_seconds']+budget['suffix_native_seconds'],
        one_prefix_four_suffixes=budget['prefixes'] == 1 and budget['suffixes'] == 4,
        four_observed_labels=budget['labels_observed'] == 4,
        zero_new_initialization=budget['new_initialization_updates'] == 0,
        zero_prefix_training=budget['prefix_training_updates'] == 0,
        zero_suffix_training=budget['suffix_updates'] == 0,
        zero_truth_feedback=budget['labels_fed_back'] == 0,
        zero_fits=budget['new_fits'] == 0)
    passed = all(checks.values())
    return dict(status='complete' if passed else 'hold',
        integrity=dict(all_pass=passed, checks=checks), audit_checks=audit_checks, protocol_checks=protocol_checks,
        coverage=slots, pairs=pairs, budget=budget,
        potential_design=design,
        exceptions=dict(missing_covariate_slots=[key for key in SLOTS if key not in slots],
            invalid_covariate_slots=[dict(slot=key, checks=row['checks']) for key, row in slots.items() if not row['valid']],
            missing_native_pairs=[key for key in PAIR_SLOTS if key not in pair_keys],
            invalid_native_pairs=[dict(slot=row['slot'], checks=row['checks']) for row in pairs if not row['valid']],
            support_selection=collector['selection']),
        preflight=dict(passed=passed, decision='Native coverage and label construction are feasible; formal collection/training needs a separate decision'
            if passed else 'HOLD preflight; retain missing or physically inseparable slots without replacement'),
        interpretation=dict(stage='native_preflight_only', model_fits=budget['new_fits'],
            neural_updates=sum(budget[key] for key in ('new_initialization_updates', 'prefix_training_updates', 'suffix_updates')),
            v87_truth_used=audit['v87_truth_used'], control_performance_evaluated=False,
            rank_scope='Potential issue-covariate design rank only; no coefficient solve or in-sample prediction scoring',
            label_scope='Paired labels are checked for real endpoint, maturity and identity; their signs never enter the gate',
            no_automatic_formal_collection_or_training=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collector', 'protocol', 'audit', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = analyze(json.loads(args.collector.read_text()), json.loads(args.protocol.read_text()),
        json.loads(args.audit.read_text()))
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 64000:
        raise ValueError('Preflight summary exceeds64KB; native records must remain resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()), preflight=value['preflight']['passed'])))


if __name__ == '__main__':
    main()
