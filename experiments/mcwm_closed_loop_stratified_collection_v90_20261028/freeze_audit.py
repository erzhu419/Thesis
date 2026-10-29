"""Audit the fixed sampling contract and one reusable prior native pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference, V84
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support, service_state
import numpy as np

V85 = HERE.parent/'mcwm_closed_loop_cost_advantage_v85_20261027'
V88 = HERE.parent/'mcwm_closed_loop_service_response_v88_20261028'
THRESHOLDS = {'7X': .13, '102X': .5, '705X': .5}
SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
ROOT_KEYS = ('started_at', 'sequence', 'trip_id', 'station_id')


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    protocol, strata = read(HERE/'protocol.json'), read(HERE/'strata_input.json')
    prior = read(V88/'freeze_audit.json')
    checks = dict(fixed_initializer=protocol['source_initializer'] == prior['source_initializer'] == 6207,
        fixed_external_seed=protocol['external_seed'] == 716207,
        fixed_scenarios=protocol['scenarios'] == list(SCENARIOS),
        fixed_issue_window=protocol['issue_window'] == [12000, 12600],
        fixed_label_clocks=protocol['terminal'] == 24000 and protocol['label_available_at'] == 24030,
        fixed_thresholds=protocol['thresholds'] == THRESHOLDS
            and all(strata['lines'][line]['threshold'] == cut for line, cut in THRESHOLDS.items()),
        native_strata_calibration=strata['calibration'] == dict(source_initializer=6207,
            external_seed=716207, scenario='stable', issue_window=[12000, 12600], exact_v88_prefix=True),
        old_preflight_gate_unchanged=strata['old_v88_gate_unchanged'] is True,
        declared_reuse_is_first_heavier=strata['old_v88_pair_is_first_new_heavier'] is True,
        strata_not_outcome_fitted=strata['admitted_labels'] == strata['new_fits'] == 0,
        prior_actor_audit=prior['all_pass'] and all(prior['checks'].values()))
    references, controllers, reference_snapshots, roots = {}, {}, {}, 0
    for scene in SCENARIOS:
        controller, mdp, stations, metadata = load_reference(scene)
        references[scene], controllers[scene] = metadata['source_actor_path'], controller
        metadata['heldout_external_seed'] = 716207
        reference_snapshots[scene] = metadata
        checks[scene+'/reference_source'] = (references[scene] == prior['reference_source_paths'][scene]
            and metadata['source_training_seed'] == 6207 and metadata['actor_frozen'] is True)
        for version, path in (('v84', V84), ('v85', V85)):
            for line in LINES:
                row = read(path/'runs'/scene/'pairs'/line/'root.json')
                actual = controller(np.asarray(row['root']['s'], np.float32))
                checks[version+'/'+scene+'/'+line] = bool(
                    np.array_equal(actual, np.asarray(row['choice']['reference_action'], np.float32))
                    and np.array_equal(controller.last_proposal,
                                       np.asarray(row['root']['reference_proposal'], np.float32)))
                roots += 1
    checks['twelve_original_reference_inputs'] = roots == 12

    source = V88/'runs'/'preflight'/'pairs'/'low'/'7X'
    declaration, pair = read(source/'root.json'), read(source/'pair_summary.json')
    root, choice = declaration['root'], declaration['choice']
    identity = {key: root[key] for key in ROOT_KEYS}
    service = service_state(declaration['context'])
    earliest = strata['earliest_7x_issue']['heavier']
    support = action_support(choice['reference_hold_seconds'], choice['candidate_hold_seconds'],
                             declaration['sim_step_seconds'], declaration['action_step_seconds'])
    current = controllers['stable'](np.asarray(root['s'], np.float32))
    reuse_checks = dict(source_pair_integrity=pair['status'] == 'complete' and pair['all_pass']
            and all(pair['checks'].values()),
        source_reference_snapshot=declaration['reference'] == reference_snapshots['stable'],
        source_pair_identity=pair['line_id'] == '7X' and pair['load_bin'] == 'low'
            and pair['root'] == identity,
        source_complete_context=service['observed'] and service['atomic_at_issue']
            and root['service'] == pair['service'] == service
            and all(declaration['observation_checks'].values()),
        first_new_heavier_root=all(identity[left] == earliest[right] for left, right in
            (('started_at', 'issue_clock'), ('sequence', 'sequence'), ('trip_id', 'trip_id'), ('station_id', 'station_id')))
            and service['features'][0] == earliest['load'] >= THRESHOLDS['7X'],
        original_reference_action=bool(np.array_equal(current, np.asarray(choice['reference_action'], np.float32))
            and np.array_equal(controllers['stable'].last_proposal, np.asarray(root['reference_proposal'], np.float32))),
        original_command_support=support['supported'],
        before_suffix_context=declaration['before_suffixes'] is True and pair['before_suffixes'] is True,
        paired_actions=pair['reference_hold_seconds'] == choice['reference_hold_seconds']
            and pair['candidate_hold_seconds'] == choice['candidate_hold_seconds'],
        previous_actual_parallel=pair['overlap_seconds'] > 0)
    children = {}
    for branch, role in (('low', 'reference'), ('high', 'candidate')):
        child = read(source/branch/'summary.json')
        children[role] = child
        remaining, native = child['remaining'], child['physical']['native_action']
        reuse_checks[role+'/complete_child'] = (child['status'] == 'complete' and child['integrity']['all_pass']
            and all(child['integrity']['checks'].values()))
        reuse_checks[role+'/same_native_entry'] = (child['root'] == root
            and child['reference'] == declaration['reference'] and child['external_seed'] == 716207
            and child['role'] == role)
        reuse_checks[role+'/actual_command'] = (child['executed_action'] == choice[role+'_action']
            and child['executed_hold_seconds'] == choice[role+'_hold_seconds'] and native is not None
            and abs(native['applied_hold_seconds']-choice[role+'_hold_seconds']) <= 1e-5)
        reuse_checks[role+'/real_terminal_maturity'] = (child['final_clock'] == 24000
            and remaining['observed_at'] == 24000 and remaining['available_at'] == 24030
            and remaining['dt'] == 24000-root['started_at'] > 0)
        reuse_checks[role+'/zero_training_feedback'] = child['suffix_updates'] == child['suffix_admitted_labels'] == 0
        reuse_checks[role+'/cost_identity'] = (remaining == pair['remaining'][role]
            and remaining['cost'] == 2*remaining['wait']+remaining['ride'])
    physical = {role: dict(departure_clock=child['physical']['departure']['clock_seconds'],
        next_arrival_clock=child['physical']['next_same_vehicle_arrival']['clock_seconds'])
        for role, child in children.items()}
    clocks_complete = all(value is not None for row in physical.values() for value in row.values())
    gap = abs(physical['candidate']['departure_clock']-physical['reference']['departure_clock']) if clocks_complete else None
    reuse_checks['complete_physical_clocks'] = clocks_complete
    reuse_checks['actual_departure_separation'] = gap is not None and gap >= declaration['sim_step_seconds']
    reuse_checks['physical_summary_exact'] = all(
        pair['physical'][role][key] == value for role, row in physical.items() for key, value in row.items())
    delta = {key: children['candidate']['remaining'][key]-children['reference']['remaining'][key]
             for key in ('wait', 'ride', 'cost')}
    reuse_checks['paired_label_identity'] = pair['delta'] == delta and delta['cost'] == 2*delta['wait']+delta['ride']
    checks['reusable_native_pair'] = all(reuse_checks.values())
    return dict(all_pass=all(checks.values()), checks=checks, reuse_checks=reuse_checks,
        source_initializer=6207, external_seed=716207, thresholds=THRESHOLDS,
        reference_source_paths=references, reference_roots=roots,
        source_strata_path=str(HERE/'strata_input.json'),
        reuse_source_pair_path=str(source/'pair_summary.json'), reuse_source_root=identity,
        reuse_source_root_path=str(source/'root.json'), reused_labels=2,
        reuse_physical=physical, reuse_departure_gap_seconds=gap,
        reuse_overlap_seconds=pair['overlap_seconds'],
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0,
        v87_truth_used=0, native_seconds=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'freeze_audit.json')
    args = parser.parse_args()
    value = audit()
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 16000:
        raise ValueError('Freeze/reuse audit exceeds16KB')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(raw, end='')
    if not value['all_pass']:
        raise SystemExit('Fixed source or reusable pair failed its audit')


if __name__ == '__main__':
    main()
