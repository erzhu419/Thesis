"""Read existing v38 resident records; emit bounded review evidence only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


V38 = 'mcwm_closed_loop_multievent_credit_v38_20261024'
SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
ARMS = ('low', 'high')
POINT_FIELDS = ('reward_scaled_sum', 'final_value_contribution',
    'final_sac_q_contribution', 'final_entropy_contribution', 'final_bootstrap',
    'stitched_return', 'mixed_vs_recorded_target_q_gap_sum', 'target_lag_gap_sum',
    'native_td_min_sum', 'native_addition_roundoff_sum')
FIRST_FIELDS = ('index0', 'transition_number', 'prior_zero_dt', 'prior_positive_dt',
    'q_path_coefficient', 'started_at', 'completed_at', 'dt', 'reward')
PHYSICAL_FIELDS = ('status', 'physical_separable', 'departure_gap_seconds',
    'next_same_vehicle_arrival_gap_seconds', 'low_departure_status', 'high_departure_status',
    'low_next_arrival_status', 'high_next_arrival_status')


def read(path):
    return json.loads(Path(path).read_text())


def checks(value):
    values = value.get('checks', {})
    return dict(all_pass=value.get('all_pass'),
        failed=[key for key, passed in values.items() if passed is not True])


def first_reward_index(path, identity):
    """Locate the exact semantic witness without exporting the event stream."""
    target = tuple(str(value) for value in identity[:-1])
    occurrence = int(identity[-1])
    seen = defaultdict(int)
    zero = positive = 0
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            event = json.loads(line)['transition']
            base = tuple(str(event[key]) for key in
                ('line_id', 'vehicle_id', 'trip_id', 'station_id'))
            number = seen[base]
            seen[base] += 1
            if base == target and number == occurrence:
                return dict(index0=index, transition_number=index+1,
                    prior_zero_dt=zero, prior_positive_dt=positive,
                    q_path_coefficient=.9**index,
                    started_at=event['started_at'], completed_at=event['completed_at'],
                    dt=event['dt'], reward=event['r'])
            zero += event['dt'] == 0.
            positive += event['dt'] > 0.
    raise ValueError('Saved semantic reward witness absent from '+str(path))


def point_table(analysis, name):
    value = analysis['models']['shared_level'][name]
    if value is None:
        return None
    return [*value['clocks'], *[value['high_minus_low'][key] for key in POINT_FIELDS]]


def review(root):
    cases = {}
    for scenario in SCENARIOS:
        run = root/'runs'/scenario
        native, model = read(run/'native/summary.json'), read(run/'model_gate.json')
        scored = read(run/'scored/summary.json')
        arms_proof = {}
        for arm, value in model['arms'].items():
            arms_proof[arm] = dict(gate=checks(value), old_issues=value['old_issues'],
                old_predictions_exact=value['old_predictions_exact'],
                components_exact=value['variants_and_components_exact'],
                crn_exact=value['common_random_number_targets_exact'])
        case = dict(status=read(run/'summary.json')['status'], prefix=native['prefix'],
            native_checks=checks(native['integrity']), wall=native['wall'],
            model_gate=dict(all_pass=model['all_pass'],
                initializer=model['source_initializer_updates'],
                slots=model['source_feedback_slots'], updates=model['source_updates'],
                new_updates=model['new_updates'], new_feedback=model['new_feedback'],
                old12_truth_read=model['old_twelve_state_truth_read'], arms=arms_proof),
            scored=dict(status=scored['status'], suffix_logs=scored['suffix_logs'],
                model_scored_rows=scored['model_scored_rows'], new_updates=scored['new_updates'],
                new_feedback=scored['new_feedback']), budget=native['budget'], states=[])
        for line in LINES:
            for window in ('early', 'late'):
                name = line+'_'+window
                state_root = run/'native/states'/name
                state = read(state_root/'state_summary.json')
                analysis = read(root/'analysis'/scenario/(name+'.json'))['analysis']
                branch = {arm: read(state_root/arm/'summary.json') for arm in ARMS}
                receipt = state['pair_receipt']
                witness = analysis['semantic_alignment']['first_differences'].get('r')
                first = None
                if witness is not None:
                    first = {key: witness[key] for key in
                        ('identity', 'issue_clocks', 'completion_clocks', 'high_minus_low')}
                    first['arms'] = {arm: first_reward_index(
                        state_root/arm/'events.jsonl', witness['identity']) for arm in ARMS}
                    for arm in ARMS:
                        value = first['arms'][arm]
                        position = ARMS.index(arm)
                        if (value['started_at'] != witness['issue_clocks'][position]
                                or value['completed_at'] != witness['completion_clocks'][position]):
                            raise ValueError('Saved semantic witness clocks do not match its native event')
                    if first['arms']['high']['reward']-first['arms']['low']['reward'] != witness['high_minus_low']:
                        raise ValueError('Saved semantic witness reward contrast changed')
                    first['arms'] = [[first['arms'][arm][field] for field in FIRST_FIELDS] for arm in ARMS]
                actual_actions, arm_values = [], []
                for arm in ARMS:
                    value, a = branch[arm], branch[arm]['execution']
                    application = a['native_application']
                    actual_actions.append([a['requested_hold_seconds'], a['executed_hold_seconds'],
                        a['native_applied_hold_seconds'], application['assigned_action'],
                        application['real_clock_before'], application['real_clock_after']])
                    point = analysis['arms'][arm]
                    tail = point['physical_300_to_endpoint']
                    arm_values.append([point['events'], point['zero_duration_events'],
                        value['global_transition']['dt'], point['endpoint_overrun_seconds'],
                        *[tail[key] for key in ('wait', 'ride', 'cost')],
                        max(result['maximum_identity_error'] for result in point['models'].values())])
                endpoint_delta = analysis['models']['shared_level']['endpoint']['high_minus_low']['reward_scaled_sum']
                fixed_delta = .0001*analysis['fixed300']['benefit_low_minus_high']
                tail_delta = -.0001*(arm_values[1][6]-arm_values[0][6])
                case['states'].append(dict(slot=line+'/'+window,
                    integrity=all(value['integrity']['all_pass'] for value in branch.values()),
                    old_early=checks(state['old_early_replay']) if state['old_early_replay'] else None,
                    feasibility=[state['native_feasibility'][key] for key in
                        ('at_stop', 'simulation_step', 'action_step', 'feasible')],
                    physical=[state['comparison']['physical_separation'][key] for key in PHYSICAL_FIELDS],
                    parallel=dict(all_pass=receipt['all_pass'], child_pids=receipt['child_pids'],
                        overlap=receipt['overlap_seconds'], parent_rng=receipt['parent_rng_unchanged']),
                    actions=actual_actions, arms=arm_values, first_reward=first,
                    endpoint=point_table(analysis, 'endpoint'),
                    last_before300=point_table(analysis, 'last_legal_before300'),
                    credit_clock=[endpoint_delta, fixed_delta, tail_delta,
                        endpoint_delta-fixed_delta-tail_delta]))
        cases[scenario] = case
    return dict(schema=dict(arm_order=list(ARMS),
        actions=['requested_hold', 'executed_hold', 'native_applied_hold', 'native_assigned_hold',
            'real_clock_before', 'real_clock_after'],
        arms=['events', 'zero_dt', 'first_dt', 'endpoint_overrun', 'tail_wait', 'tail_ride',
            'tail_cost', 'max_identity_error_all_models'],
        point=['clock_low', 'clock_high', *POINT_FIELDS],
        point_differences='high minus low; shared_level only; endpoints may differ in clock',
        credit_clock=['endpoint_scaled_reward_delta', 'fixed300_scaled_reward_delta',
            'extra_tail_scaled_reward_delta', 'closure_error'],
        physical=list(PHYSICAL_FIELDS), feasibility=['at_stop', 'simulation_step', 'action_step', 'feasible'],
        first_reward_arm=list(FIRST_FIELDS),
        first_reward='index0 is zero based; prior counts exclude reward-bearing row; '
            'q_path_coefficient=.9**index0 is a frozen mixed-operator Q-path coefficient, '
            'not a measured training gradient'), cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/V38)
    parser.add_argument('--max-bytes', type=int, default=18000)
    args = parser.parse_args()
    output = json.dumps(review(args.root), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(output.encode())+1 > args.max_bytes:
        raise ValueError('Review output exceeds requested bounded size: '+str(len(output.encode())+1))
    print(output)
