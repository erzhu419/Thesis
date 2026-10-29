"""Small, physical-cost readout of the six fixed v84 action pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
COUNTS = dict(critic=296, value=296, actor=296, alpha=296)
BUDGET_FIELDS = ('native_prefix_seconds', 'suffix_native_seconds', 'total_native_seconds',
                 'restored_covariates', 'restored_complete_outcomes',
                 'restored_updates_per_optimizer', 'suffix_updates', 'labels_fed_back',
                 'new_initialization_updates', 'prefixes', 'suffixes', 'labels_observed')


def _direction(delta):
    return 'candidate_better' if delta < 0 else 'reference_better' if delta > 0 else 'tie'


def pair_readout(scene, pair):
    ref, candidate = pair['remaining']['reference'], pair['remaining']['candidate']
    delta = {key: candidate[key]-ref[key] for key in ('wait', 'ride', 'cost')}
    holding_delta = pair['candidate_hold_seconds']-pair['reference_hold_seconds']
    denominator = scene['departed_passengers']
    departure = {arm: pair['physical'][arm]['departure_clock'] for arm in ('reference', 'candidate')}
    arrival = {arm: pair['physical'][arm]['next_arrival_clock'] for arm in ('reference', 'candidate')}
    clocks_complete = all(value is not None for value in (*departure.values(), *arrival.values()))
    dt_valid = ref['dt'] > 0 and candidate['dt'] > 0
    maturity_valid = all(row['observed_at'] == 24000 and row['available_at'] == 24030
                         for row in (ref, candidate))
    identity_valid = (delta == pair['delta']
                      and delta['cost'] == 2*delta['wait']+delta['ride'])
    valid = bool(pair['status'] == 'complete' and pair['all_pass'] and dt_valid
                 and maturity_valid and identity_valid and clocks_complete and holding_delta != 0)
    result = dict(scenario=scene['scenario'], line_id=pair['line_id'], root=pair['root'],
        valid_for_signal=valid, runtime_all_pass=pair['all_pass'], runtime_checks=pair['checks'],
        checks=dict(nonzero_response_duration=dt_valid, label_maturity=maturity_valid,
                    saved_delta_consistent=identity_valid, physical_clocks_complete=clocks_complete),
        actions=dict(reference_hold_seconds=pair['reference_hold_seconds'],
                     candidate_hold_seconds=pair['candidate_hold_seconds'],
                     candidate_minus_reference_hold_seconds=holding_delta),
        physical=dict(departure_clocks=departure, next_same_vehicle_arrival_clocks=arrival,
            candidate_minus_reference_departure_seconds=(departure['candidate']-departure['reference'])
                if all(value is not None for value in departure.values()) else None,
            candidate_minus_reference_next_arrival_seconds=(arrival['candidate']-arrival['reference'])
                if all(value is not None for value in arrival.values()) else None,
            missing={arm: pair['physical'][arm]['missing'] for arm in ('reference', 'candidate')}),
        overlap_seconds=pair['overlap_seconds'], remaining=pair['remaining'],
        candidate_minus_reference=delta,
        candidate_minus_reference_per_departed_passenger={key:value/denominator for key,value in delta.items()},
        truth_direction=_direction(delta['cost']),
        two_action_regret=dict(reference=max(0., -delta['cost']), candidate=max(0., delta['cost'])),
        state_level_and_response=dict(reference_remaining_cost=ref['cost'],
            absolute_response_cost=abs(delta['cost']),
            absolute_response_percent_of_reference=(100*abs(delta['cost'])/abs(ref['cost']))
                if ref['cost'] != 0 else None),
        more_holding_minus_less_holding_cost=(delta['cost'] if holding_delta > 0 else -delta['cost'])
            if holding_delta != 0 else None,
        cost_difference_per_extra_holding_second=delta['cost']/holding_delta if holding_delta != 0 else None)
    return result


def _group(rows):
    usable = [row for row in rows if row['valid_for_signal']]
    n = len(usable)
    directions = {name:sum(row['truth_direction'] == name for row in usable)
                  for name in ('candidate_better', 'reference_better', 'tie')}
    return dict(states=len(rows), valid_states=n, directions=directions,
        mean_single_root_difference={key:sum(row['candidate_minus_reference'][key] for row in usable)/n
            if n else None for key in ('wait','ride','cost')},
        mean_single_root_difference_per_departed_passenger={key:sum(
            row['candidate_minus_reference_per_departed_passenger'][key] for row in usable)/n
            if n else None for key in ('wait','ride','cost')},
        sum_two_action_regret={arm:sum(row['two_action_regret'][arm] for row in usable)
            if n else None for arm in ('reference','candidate')},
        more_holding_improves=sum(row['more_holding_minus_less_holding_cost'] < 0 for row in usable),
        less_holding_improves=sum(row['more_holding_minus_less_holding_cost'] > 0 for row in usable))


def analyze(scenes):
    indexed = {row['scenario']:row for row in scenes}
    if set(indexed) != set(SCENARIOS) or len(scenes) != 2:
        raise ValueError('Read exactly the two fixed scenario summaries')
    rows, missing, support_rejections = [], [], []
    for scenario in SCENARIOS:
        scene = indexed[scenario]
        by_line = {row['line_id']:row for row in scene['pairs']}
        if len(by_line) != len(scene['pairs']) or set(by_line)-set(LINES):
            raise ValueError('Duplicate or undeclared line in fixed collection')
        for line in LINES:
            selection = scene['selection'][line]
            support_rejections.append(dict(scenario=scenario, line_id=line,
                attempted=selection['attempted'], status=selection['status'], reasons=selection['reasons']))
            pair = by_line.get(line)
            if pair is None:
                missing.append(dict(scenario=scenario, line_id=line,
                    status='selected_pair_incomplete' if selection['selected'] is not None else 'missing',
                    selection=selection))
                continue
            rows.append(pair_readout(scene, pair))
    budget = {key:sum(row['budget'][key] for row in scenes) for key in BUDGET_FIELDS}
    reference_checks = []
    for scene in scenes:
        reference = scene['reference_snapshot']
        reference_checks.append(reference['source'] == 'v75' and reference['source_arm'] == 'fresh'
            and reference['seed'] == 6207 and reference['scenario'] == scene['scenario']
            and reference['packets'] == 37 and reference['covariates'] == 185
            and reference['complete_outcomes'] == 185 and reference['counts'] == COUNTS
            and reference['data_deadline'] == 12000 and reference['max_available_at'] <= 12000
            and reference['deterministic'] is True and reference['actor_frozen'] is True
            and reference['future_training_used'] is False
            and all(reference['published_clock'] <= pair['root']['started_at'] for pair in scene['pairs']))
    checks = dict(two_complete_scenarios=all(row['status'] == 'complete' for row in scenes),
        runtime_scene_integrity=all(row['integrity']['all_pass'] for row in scenes),
        fixed_seed=all(row['seed'] == 6207 for row in scenes),
        six_declared_pairs=len(rows) == 6 and not missing,
        six_valid_pairs=len(rows) == 6 and all(row['valid_for_signal'] for row in rows),
        issue_preceding_reference=all(reference_checks),
        fixed_issue_window=all(12000 <= row['root']['started_at'] < 12600 for row in rows),
        original_updates_compared=all(row['source_update_rows'] == 74 for row in scenes),
        original_prefix_observed=all(row['source_prefix_rows'] > 0 for row in scenes),
        actual_paired_parallel=len(rows) == 6 and all(row['overlap_seconds'] > 0 for row in rows),
        twelve_observed_labels=budget['labels_observed'] == 12,
        fixed_native_budget=budget['total_native_seconds'] <= 169200,
        native_budget_identity=budget['total_native_seconds'] == (
            budget['native_prefix_seconds']+budget['suffix_native_seconds']),
        restoration_counts=budget['restored_covariates'] == 370
            and budget['restored_complete_outcomes'] == 370
            and budget['restored_updates_per_optimizer'] == 592,
        expected_prefixes_suffixes=budget['prefixes'] == 2 and budget['suffixes'] == 12,
        zero_suffix_training=budget['suffix_updates'] == 0,
        zero_truth_feedback=budget['labels_fed_back'] == 0,
        zero_new_initializer=budget['new_initialization_updates'] == 0)
    valid = [row for row in rows if row['valid_for_signal']]
    nonzero = sum(row['candidate_minus_reference']['cost'] != 0 for row in valid)
    ready = all(checks.values()) and nonzero > 0
    return dict(status='complete' if all(checks.values()) else 'hold',
        integrity=dict(all_pass=all(checks.values()), checks=checks), seed=6207,
        primary_target='G_candidate-G_reference; G=2*remaining_wait+remaining_ride',
        pairs=rows, overall=_group(rows),
        by_scenario={name:_group([row for row in rows if row['scenario'] == name]) for name in SCENARIOS},
        by_line={name:_group([row for row in rows if row['line_id'] == name]) for name in LINES},
        budget=budget,
        exceptions=dict(missing=missing,
            invalid_pair_roots=[dict(scenario=row['scenario'], line_id=row['line_id'],
                root=row['root'], checks=row['checks'], runtime_checks=row['runtime_checks'])
                for row in rows if not row['valid_for_signal']],
            zero_duration=[dict(scenario=row['scenario'], line_id=row['line_id']) for row in rows
                if not row['checks']['nonzero_response_duration']],
            zero_cost_response=[dict(scenario=row['scenario'], line_id=row['line_id']) for row in valid
                if row['candidate_minus_reference']['cost'] == 0],
            support_rejections=support_rejections),
        training_readout=dict(target_data_ready=ready, nonzero_valid_state_differences=nonzero,
            response_training_updates=0, truth_labels_fed_back=budget['labels_fed_back'],
            decision='Ready to freeze a learner and independent-holdout contract; no training authorized by this readout'
                if ready else 'Hold training: incomplete/ineligible collection or no observed action signal'),
        interpretation=dict(independent_training_seeds=1, source_prefixes=2,
            planned_query_states=6, queried_states_not_independent_replicates=len(rows),
            grouped_differences='Query summaries of separate single-root interventions; sums are not joint-policy costs',
            regret_scope='Oracle regret among only the two queried actions; no learned policy has been evaluated',
            state_level_scope='Reference remaining-cost magnitude versus observed action contrast, not evidence that a fitted model disentangles them',
            claim='No demonstrated fresh learning gain, repeated-control gain or independent confirmation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable', type=Path, required=True)
    parser.add_argument('--shock', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = analyze([json.loads(path.read_text()) for path in (args.stable, args.shock)])
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Summary exceeds48KB; raw branch data must remain resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()),
        native_seconds=value['budget']['total_native_seconds'],
        target_data_ready=value['training_readout']['target_data_ready'])))


if __name__ == '__main__':
    main()
