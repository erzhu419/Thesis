"""Read-only source audit and thin manifest of eleven fixed reusable pairs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.native import load_reference, V84
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import service_state, action_support
import numpy as np
import torch

V85 = HERE.parent/'mcwm_closed_loop_cost_advantage_v85_20261027'
V91 = HERE.parent/'mcwm_closed_loop_issue_conditions_v91_20261028'
SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
CUTS = {'7X': .13, '102X': .5, '705X': .5}
ROOT_KEYS = ('sequence', 'started_at', 'trip_id', 'station_id')
COST_KEYS = ('wait', 'ride', 'cost')


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    torch.set_num_threads(1)
    source = read(HERE/'source_protocol.json')
    readout = read(V91/'resident_summary.json')
    contract = source['collection']
    declarations = contract['reused_roots']
    checks = dict(fixed_initializer=source['reference']['initializer_seed'] == 6207,
        fixed_external_seed=source['reference']['training_external_seed'] == contract['external_seed'] == 716207,
        fixed_thresholds=contract['native_load_thresholds'] == CUTS,
        fixed_native_window=contract['issue_window'] == [12000, 12900],
        fixed_real_label_clocks=contract['terminal'] == 24000 and contract['label_available_at'] == 24030,
        exactly_eleven_reused_pairs=contract['planned_reused_pairs'] == len(declarations) == 11,
        fixed_missing_slot=contract['new_slot'] == dict(scenario='local_speed_drop', line_id='705X', stratum='lighter'),
        original_readout_passed=readout['all_pass'] is True,
        readout_zero_new_budget=all(readout['budget'][key] == 0 for key in
            ('new_native_seconds', 'new_branches', 'new_labels', 'new_fits', 'neural_updates', 'v87_truth_used')),
        frozen_history_budget=source['budgets']['historical_native_seconds_through_v90'] == 314090,
        frozen_new_budget=source['budgets']['next_collection']['max_new_native_seconds'] == 36900,
        no_fitting_authorized=source['budgets']['next_collection']['fits'] == 0)
    observed = {row['scenario']: row for row in readout['scenarios']}
    controllers, metadata, paths, reference_roots = {}, {}, {}, 0
    for scenario in SCENARIOS:
        controller, mdp, stations, reference = load_reference(scenario)
        reference['heldout_external_seed'] = 716207
        controllers[scenario], metadata[scenario], paths[scenario] = controller, reference, reference['source_actor_path']
        checks[scenario+'/readout_reference_actions'] = (observed[scenario]['replay']['all_actions_exact']
            and observed[scenario]['replay']['reference_overlap_exact']
            and observed[scenario]['replay']['all_event_clocks_visible']
            and observed[scenario]['replay']['condition_historical_only'])
        for version, directory in (('v84', V84), ('v85', V85)):
            for line in LINES:
                row = read(directory/'runs'/scenario/'pairs'/line/'root.json')
                actual = controller(np.asarray(row['root']['s'], np.float32))
                checks[version+'/'+scenario+'/'+line] = bool(
                    np.array_equal(actual, np.asarray(row['choice']['reference_action'], np.float32))
                    and np.array_equal(controller.last_proposal, np.asarray(row['root']['reference_proposal'], np.float32)))
                reference_roots += 1
    checks['twelve_original_reference_inputs'] = reference_roots == 12
    rows = []
    for declared in declarations:
        scenario, line, group = declared['scenario'], declared['line_id'], declared['stratum']
        directory = HERE.parent/declared['source']
        declaration, pair = read(directory/'root.json'), read(directory/'pair_summary.json')
        root, choice = declaration['root'], declaration['choice']
        identity = {key: root[key] for key in ROOT_KEYS}
        state = service_state(declaration['context'])
        condition_rows = [row for row in observed[scenario]['root_conditions']
            if row['line_id'] == line and row['stratum'] == group and row['root'] == identity]
        if len(condition_rows) != 1:
            raise ValueError('A fixed old root lacks its unique historical condition record')
        old_condition = condition_rows[0]
        condition = deepcopy(old_condition['condition'])
        issue = root['started_at']
        support = action_support(choice['reference_hold_seconds'], choice['candidate_hold_seconds'],
                                 declaration['sim_step_seconds'], declaration['action_step_seconds'])
        local = dict(fixed_declared_root=identity == declared['root'],
            fixed_pair_slot=pair['line_id'] == line and pair['load_bin'] == group and pair['root'] == identity,
            original_pair_integrity=pair['status'] == 'complete' and pair['all_pass'] and all(pair['checks'].values()),
            original_reference_snapshot=declaration['reference'] == metadata[scenario],
            complete_native_service=state['observed'] and state['atomic_at_issue']
                and state == root['service'] == pair['service']
                and state['features'] == old_condition['local_service_features']
                and all(declaration['observation_checks'].values()),
            native_stratum=('lighter' if state['features'][0] < CUTS[line] else 'heavier') == group,
            legal_original_command=support['supported'],
            same_pair_actions=pair['reference_hold_seconds'] == choice['reference_hold_seconds']
                and pair['candidate_hold_seconds'] == choice['candidate_hold_seconds'],
            condition_value_frozen=condition['value'] == declared['q'],
            condition_supported=condition['status'] == 'supported' and condition['historical_only'] is True,
            condition_strictly_before_issue=condition['issue_clock'] == issue
                and condition['latest_completed_at'] < issue and condition['latest_observed_at'] < issue,
            fixed_history_baseline=condition['baseline_last_observed_at'] < 9000
                and set(condition['baseline_counts']) == set(LINES)
                and min(condition['baseline_counts'].values()) >= 3,
            fresh_condition_history=0 <= condition['latest_age_seconds'] <= 900,
            actual_parallel=pair['overlap_seconds'] > 0)
        actual = controllers[scenario](np.asarray(root['s'], np.float32))
        local['original_actor_F_exact'] = bool(
            np.array_equal(actual, np.asarray(choice['reference_action'], np.float32))
            and np.array_equal(controllers[scenario].last_proposal, np.asarray(root['reference_proposal'], np.float32)))
        source_pair = Path(pair['source_pair_path']) if pair.get('reused') else directory/'pair_summary.json'
        original = read(source_pair)
        local['original_outcomes_unchanged'] = all(pair[key] == original[key]
            for key in ('root', 'remaining', 'delta', 'physical', 'reference_hold_seconds', 'candidate_hold_seconds'))
        children = {}
        for branch, role in (('low', 'reference'), ('high', 'candidate')):
            child = read(source_pair.parent/branch/'summary.json')
            children[role] = child
            remaining, native = child['remaining'], child['physical']['native_action']
            local[role+'/child_integrity'] = (child['status'] == 'complete'
                and child['integrity']['all_pass'] and all(child['integrity']['checks'].values()))
            local[role+'/same_entry_reference_seed'] = (child['root'] == root
                and child['reference'] == declaration['reference'] and child['external_seed'] == 716207)
            local[role+'/real_command'] = (child['role'] == role
                and child['executed_action'] == choice[role+'_action']
                and child['executed_hold_seconds'] == choice[role+'_hold_seconds'] and native is not None
                and abs(native['applied_hold_seconds']-choice[role+'_hold_seconds']) <= 1e-5)
            local[role+'/terminal_maturity'] = (child['final_clock'] == 24000
                and remaining['observed_at'] == 24000 and remaining['available_at'] == 24030
                and remaining['dt'] == 24000-issue > 0)
            local[role+'/zero_training_feedback'] = child['suffix_updates'] == child['suffix_admitted_labels'] == 0
            local[role+'/factual_cost_identity'] = (remaining == pair['remaining'][role]
                and remaining['cost'] == 2*remaining['wait']+remaining['ride'])
        physical = {role: dict(departure_clock=child['physical']['departure']['clock_seconds'],
            next_arrival_clock=child['physical']['next_same_vehicle_arrival']['clock_seconds'])
            for role, child in children.items()}
        complete = all(value is not None for row in physical.values() for value in row.values())
        local['physical_motion'] = complete and abs(physical['candidate']['departure_clock']
            -physical['reference']['departure_clock']) >= declaration['sim_step_seconds']
        local['original_physical_clocks'] = all(pair['physical'][role][key] == value
            for role, row in physical.items() for key, value in row.items())
        delta = {key: children['candidate']['remaining'][key]-children['reference']['remaining'][key] for key in COST_KEYS}
        local['paired_label_identity'] = pair['delta'] == delta and delta['cost'] == 2*delta['wait']+delta['ride']
        key = scenario+'/'+line+'/'+group
        checks[key] = all(local.values())
        rows.append(dict(scenario=scenario, external_seed=716207, line_id=line, load_bin=group,
            root=identity, context=declaration['context'], service=state, condition=condition,
            reference_hold_seconds=choice['reference_hold_seconds'], candidate_hold_seconds=choice['candidate_hold_seconds'],
            remaining=pair['remaining'], delta=pair['delta'], physical=pair['physical'],
            overlap_seconds=pair['overlap_seconds'], sim_step_seconds=declaration['sim_step_seconds'],
            action_step_seconds=declaration['action_step_seconds'], source_pair_path=str(directory/'pair_summary.json'),
            outcome_source_pair_path=str(source_pair), all_pass=all(local.values()),
            checks=dict(source_root_context_condition=all(value for name, value in local.items() if '/' not in name),
                native_children=all(value for name, value in local.items() if '/' in name))))
    identities = [(row['scenario'], row['root']['sequence']) for row in rows]
    checks['eleven_unique_old_roots'] = len(identities) == len(set(identities)) == 11
    return dict(all_pass=all(checks.values()), checks=checks, source_initializer=6207, external_seed=716207,
        reference_source_paths=paths, reference_roots=reference_roots, thresholds=CUTS,
        source_protocol_path=str(HERE/'source_protocol.json'), source_readout_path=str(V91/'resident_summary.json'),
        reused_pairs=11, reused_labels=22, historical_native_seconds=314090, old_pairs=rows,
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0, v87_truth_used=0, native_seconds=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'freeze_audit.json')
    args = parser.parse_args()
    value = audit()
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Thin eleven-pair audit exceeds48KB; keep full native states resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(raw, end='')
    if not value['all_pass']:
        raise SystemExit('A frozen old pair or historical condition failed its audit')


if __name__ == '__main__':
    main()
