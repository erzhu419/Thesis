"""Score sealed D-readout choices against new native paired truth only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
MODELS = ('line', 'state')
BUDGET_FIELDS = ('native_prefix_seconds', 'suffix_native_seconds', 'total_native_seconds',
                 'prefixes', 'suffixes', 'labels_observed', 'new_initialization_updates',
                 'prefix_training_updates', 'suffix_updates', 'labels_fed_back')


def _direction(value):
    return 'candidate_better' if value < 0 else 'reference_better' if value > 0 else 'tie'


def pair_readout(scene, pair):
    reference, candidate = (pair['remaining'][role] for role in ('reference', 'candidate'))
    delta = {key:candidate[key]-reference[key] for key in ('wait', 'ride', 'cost')}
    denominator = scene['departed_passengers']
    target = delta['cost']/denominator
    prediction = pair['prediction']
    ref_hold, cand_hold = pair['reference_hold_seconds'], pair['candidate_hold_seconds']
    expected_support = 3.0 <= ref_hold <= 3.6 and abs(cand_hold-7.5) <= 1e-5 and cand_hold > ref_hold
    physical = pair['physical']
    clocks_complete = all(physical[role][key] is not None
                          for role in ('reference', 'candidate')
                          for key in ('departure_clock', 'next_arrival_clock'))
    checks = dict(
        runtime_pair_integrity=pair['all_pass'],
        nonzero_duration=reference['dt'] == candidate['dt'] > 0,
        real_terminal_maturity=all(value['observed_at'] == 24000 and value['available_at'] == 24030
                                   for value in (reference, candidate)),
        delta_identity=delta == pair['delta'] and delta['cost'] == 2*delta['wait']+delta['ride'],
        physical_clocks_complete=clocks_complete,
        pre_issue_prediction=pair['before_suffixes'] is True
            and pair['prediction_recorded_at'] == pair['root']['started_at'],
        action_support_declaration=prediction['supported'] == expected_support,
        actual_parallel=pair['overlap_seconds'] > 0)
    models = {}
    for name in MODELS:
        saved = prediction['models'][name]
        estimate, choice = saved['predicted_delta'], saved['choice']
        if prediction['supported']:
            consistent = isinstance(estimate, (int,float)) and math.isfinite(estimate)
            consistent = consistent and choice == ('candidate' if estimate < 0 else 'reference')
        else:
            consistent = estimate is None and choice == 'reference' and prediction['reason'] is not None
        checks[name+'_prediction_choice_consistent'] = consistent
        selected = delta if choice == 'candidate' else dict(wait=0., ride=0., cost=0.)
        regret = selected['cost']-min(0., delta['cost'])
        models[name] = dict(predicted_delta=estimate, choice=choice,
            fallback_reason=prediction['reason'] if not prediction['supported'] else None,
            selected_minus_reference=selected,
            selected_minus_reference_per_passenger={key:value/denominator for key,value in selected.items()},
            two_action_regret=regret, two_action_regret_per_passenger=regret/denominator,
            action_correct=regret == 0,
            predicted_direction=_direction(estimate) if estimate is not None else None,
            direction_correct=(_direction(estimate) == _direction(target)) if estimate is not None else None,
            absolute_prediction_error=abs(estimate-target) if estimate is not None else None,
            squared_prediction_error=(estimate-target)**2 if estimate is not None else None)
    valid = pair['status'] == 'complete' and all(checks.values())
    return dict(scenario=scene['scenario'], line_id=pair['line_id'], root=pair['root'],
        valid_for_evaluation=valid, checks=checks, runtime_checks=pair['checks'],
        reference_hold_seconds=ref_hold, candidate_hold_seconds=cand_hold,
        supported=prediction['supported'], support_reason=prediction['reason'], z=prediction['z'],
        actual_delta=delta, actual_delta_per_passenger={key:value/denominator for key,value in delta.items()},
        truth_direction=_direction(target), remaining=pair['remaining'], physical=physical,
        reference_full_endpoint=pair['reference_full_endpoint'], overlap_seconds=pair['overlap_seconds'],
        reference_two_action_regret=max(0., -delta['cost']),
        reference_two_action_regret_per_passenger=max(0., -delta['cost'])/denominator,
        models=models)


def _group(rows):
    valid = [row for row in rows if row['valid_for_evaluation']]
    n = len(valid)
    result = dict(states=len(rows), valid_states=n,
        supported_states=sum(row['supported'] for row in valid),
        out_of_support_states=sum(not row['supported'] for row in valid),
        reference=dict(sum_two_action_regret=sum(row['reference_two_action_regret'] for row in valid) if n else None,
            mean_two_action_regret_per_passenger=sum(
                row['reference_two_action_regret_per_passenger'] for row in valid)/n if n else None), models={})
    for name in MODELS:
        model_rows = [row['models'][name] for row in valid]
        scored = [row for row in model_rows if row['predicted_delta'] is not None]
        prediction_n = len(scored)
        choice_costs = [row['selected_minus_reference_per_passenger']['cost'] for row in model_rows]
        result['models'][name] = dict(
            candidate_choices=sum(row['choice'] == 'candidate' for row in model_rows),
            reference_choices=sum(row['choice'] == 'reference' for row in model_rows),
            fallback_choices=sum(row['fallback_reason'] is not None for row in model_rows),
            prediction_count=prediction_n,
            direction_correct=sum(row['direction_correct'] for row in scored),
            direction_accuracy=sum(row['direction_correct'] for row in scored)/prediction_n if prediction_n else None,
            mae=sum(row['absolute_prediction_error'] for row in scored)/prediction_n if prediction_n else None,
            rmse=math.sqrt(sum(row['squared_prediction_error'] for row in scored)/prediction_n) if prediction_n else None,
            action_correct=sum(row['action_correct'] for row in model_rows),
            action_accuracy=sum(row['action_correct'] for row in model_rows)/n if n else None,
            sum_selected_minus_reference={key:sum(row['selected_minus_reference'][key] for row in model_rows)
                if n else None for key in ('wait','ride','cost')},
            mean_selected_minus_reference_per_passenger={key:sum(
                row['selected_minus_reference_per_passenger'][key] for row in model_rows)/n
                if n else None for key in ('wait','ride','cost')},
            sum_two_action_regret=sum(row['two_action_regret'] for row in model_rows) if n else None,
            mean_two_action_regret_per_passenger=sum(row['two_action_regret_per_passenger']
                for row in model_rows)/n if n else None,
            improved_queries=sum(value < 0 for value in choice_costs),
            harmed_queries=sum(value > 0 for value in choice_costs),
            unchanged_queries=sum(value == 0 for value in choice_costs),
            worst_selected_minus_reference_per_passenger=max(choice_costs) if n else None)
    return result


def _reference_valid(scene):
    snapshot = scene['reference_snapshot']
    old = snapshot['source_snapshot_metadata']
    return bool(snapshot['prior_run_reference'] is True
        and snapshot['available_before_heldout_episode'] is True
        and old['source'] == 'v75' and old['source_arm'] == 'fresh'
        and old['seed'] == 6207 and old['scenario'] == scene['scenario']
        and old['packets'] == 37 and old['covariates'] == old['complete_outcomes'] == 185
        and old['counts'] == dict(critic=296,value=296,actor=296,alpha=296)
        and old['max_available_at'] <= old['data_deadline'] == 12000
        and old['deterministic'] is True and old['actor_frozen'] is True)


def _single_intervention(rows, denominator):
    """Reconstruct one sealed intervention, then the native reference suffix.

    All opportunities are reached on the baseline before the first change.
    The chosen candidate suffix already returns to pi_ref, so no combination
    of separately queried action effects is used to reconstruct this policy.
    """
    ordered = sorted(rows, key=lambda row:(row['root']['started_at'],row['root']['sequence']))
    complete = len(ordered) == 3 and all(row['valid_for_evaluation'] for row in ordered)
    endpoint_equal = complete and all(row['reference_full_endpoint'] == ordered[0]['reference_full_endpoint']
                                      for row in ordered)
    usable = complete and endpoint_equal
    oracle_delta = min([0.]+[row['actual_delta']['cost'] for row in ordered]) if usable else None
    full_reference = ordered[0]['reference_full_endpoint'] if usable else None
    result = dict(status='complete' if usable else 'incomplete',
        opportunity_order=[dict(line_id=row['line_id'],root=row['root']) for row in ordered],
        oracle_best_single_root_delta_cost=oracle_delta,
        oracle_scope='Evaluation-only minimum of no action and the three queried single-root interventions; future truth is not available to the deployed policy',
        models={})
    for name in ('reference',*MODELS):
        selected = next((row for row in ordered if row['supported']
            and row['models'][name]['predicted_delta'] is not None
            and row['models'][name]['predicted_delta'] < 0),None) if name in MODELS else None
        if usable:
            delta = selected['actual_delta'] if selected is not None else dict(wait=0.,ride=0.,cost=0.)
            full_selected = {key:full_reference[key]+delta[key] for key in ('wait','ride','cost')}
        else:
            delta = full_selected = None
        result['models'][name] = dict(status='complete' if usable else 'incomplete',
            selected_root=dict(line_id=selected['line_id'],**selected['root']) if selected is not None else None,
            interventions=int(selected is not None) if usable else None,
            selected_minus_reference=delta,
            selected_minus_reference_per_passenger={key:value/denominator for key,value in delta.items()}
                if delta is not None else None,
            full_reference_endpoint=full_reference,full_selected_endpoint=full_selected,
            full_selected_cost_per_passenger=full_selected['cost']/denominator if full_selected is not None else None,
            single_opportunity_oracle_regret=delta['cost']-oracle_delta if usable else None,
            single_opportunity_oracle_regret_per_passenger=(delta['cost']-oracle_delta)/denominator if usable else None)
    line,state = (result['models'][name] for name in MODELS)
    result['state_minus_line_selected_cost'] = state['selected_minus_reference']['cost']-line['selected_minus_reference']['cost'] if usable else None
    result['state_minus_line_selected_cost_per_passenger'] = result['state_minus_line_selected_cost']/denominator if usable else None
    return result


def analyze(scenes, artifact):
    indexed = {row['scenario']:row for row in scenes}
    if set(indexed) != set(SCENARIOS) or len(scenes) != 2:
        raise ValueError('Score exactly the two predeclared new heldout scenarios')
    rows, missing, selections, endpoint_audit = [], [], [], {}
    for scenario in SCENARIOS:
        scene = indexed[scenario]
        by_line = {row['line_id']:row for row in scene['pairs']}
        if len(by_line) != len(scene['pairs']) or set(by_line)-set(LINES):
            raise ValueError('Duplicate or undeclared heldout line')
        scene_rows = []
        for line in LINES:
            selection = scene['selection'][line]
            selections.append(dict(scenario=scenario,line_id=line,selection=selection))
            if line not in by_line:
                missing.append(dict(scenario=scenario,line_id=line,selection=selection))
                continue
            row = pair_readout(scene, by_line[line])
            rows.append(row);scene_rows.append(row)
        complete = len(scene_rows) == 3
        spread = {key:max(row['reference_full_endpoint'][key] for row in scene_rows)
            -min(row['reference_full_endpoint'][key] for row in scene_rows) if complete else None
            for key in ('wait','ride','cost')}
        endpoint_audit[scenario] = dict(complete=complete,max_minus_min=spread,
            all_equal=complete and all(value == 0 for value in spread.values()))
    budget = {key:sum(row['budget'][key] for row in scenes) for key in BUDGET_FIELDS}
    scope = artifact['training_scope']
    expected_cases = [scenario+'/'+line for scenario in SCENARIOS for line in LINES]
    training_checks = dict(
        training_gate_passed=artifact['gate']['all_pass'], two_fixed_fits=artifact['fits'] == 2,
        fixed_model_ranks=artifact['models']['line']['design_rank'] == 3
            and artifact['models']['state']['design_rank'] == 4,
        fixed_ridge=artifact['ridge'] == 1.,
        fixed_target_scale=artifact['denominator'] == 15079 and artifact['hold_scale'] == 7.5,
        only_old_training_pairs=scope['source_scope'] == 'v84_train' and scope['pairs'] == 6
            and scope['physical_outcome_labels'] == 12 and scope['source_seed'] == 6207
            and scope['external_seed'] == 106207 and scope['cases'] == expected_cases,
        training_labels_mature=scope['label_available_at'] == 24030,
        no_heldout_truth_in_fit=scope['heldout_labels_used'] == 0,
        no_original_model_updates=all(artifact[key+'_updates'] == 0
            for key in ('actor','critic','value','B','world','trust')))
    checks = dict(two_complete_scenarios=all(row['status'] == 'complete' for row in scenes),
        runtime_scene_integrity=all(row['integrity']['all_pass'] for row in scenes),
        fixed_new_external_seed=all(row['external_seed'] == 216207 for row in scenes),
        fixed_source_initializer=all(row['seed'] == 6207 for row in scenes),
        new_prefixes=all(bool(row['prefix_id']) for row in scenes)
            and len({row['prefix_id'] for row in scenes}) == 2,
        six_declared_heldout_pairs=len(rows) == 6 and not missing,
        six_valid_pairs=len(rows) == 6 and all(row['valid_for_evaluation'] for row in rows),
        same_prior_run_reference=all(_reference_valid(row) for row in scenes),
        fixed_issue_window=all(12000 <= row['root']['started_at'] < 12600 for row in rows),
        reference_full_day_endpoints_equal=all(row['all_equal'] for row in endpoint_audit.values()),
        actual_paired_parallel=len(rows) == 6 and all(row['overlap_seconds'] > 0 for row in rows),
        twelve_observed_labels=budget['labels_observed'] == 12,
        expected_native_units=budget['prefixes'] == 2 and budget['suffixes'] == 12,
        fixed_native_budget=budget['total_native_seconds'] <= 169200,
        native_budget_identity=budget['total_native_seconds'] == budget['native_prefix_seconds']+budget['suffix_native_seconds'],
        zero_initializer_training=budget['new_initialization_updates'] == 0,
        zero_prefix_training=budget['prefix_training_updates'] == 0,
        zero_suffix_training=budget['suffix_updates'] == 0,
        zero_heldout_feedback=budget['labels_fed_back'] == 0,
        frozen_training_contract=all(training_checks.values()))
    overall = _group(rows)
    by_scenario = {scenario:_group([row for row in rows if row['scenario'] == scenario]) for scenario in SCENARIOS}
    single = {scenario:_single_intervention([row for row in rows if row['scenario'] == scenario],
                                           indexed[scenario]['departed_passengers']) for scenario in SCENARIOS}
    state, line = overall['models']['state'], overall['models']['line']
    enough = overall['valid_states'] == 6
    promotion = dict(technical_integrity=all(checks.values()),
        state_actually_changes_an_action=state['candidate_choices'] > 0,
        state_beats_reference=enough and state['sum_selected_minus_reference']['cost'] < 0,
        state_beats_line_baseline=enough and state['sum_selected_minus_reference']['cost']
            < line['sum_selected_minus_reference']['cost'],
        stable_non_degradation=by_scenario['stable']['valid_states'] == 3
            and by_scenario['stable']['models']['state']['sum_selected_minus_reference']['cost'] <= 0,
        shock_non_degradation=by_scenario['local_speed_drop']['valid_states'] == 3
            and by_scenario['local_speed_drop']['models']['state']['sum_selected_minus_reference']['cost'] <= 0,
        single_state_non_degradation=all(value['status'] == 'complete'
            and value['models']['state']['selected_minus_reference']['cost'] <= 0 for value in single.values()))
    passed = all(promotion.values())
    return dict(status='complete' if all(checks.values()) else 'hold',
        integrity=dict(all_pass=all(checks.values()),checks=checks), training_checks=training_checks,
        overall=overall, by_scenario=by_scenario,
        by_line={line:_group([row for row in rows if row['line_id'] == line]) for line in LINES},
        pairs=rows, budget=budget, reference_endpoint_audit=endpoint_audit,
        single_intervention_shadow=dict(
            policy='Sort the three first-eligible opportunities by root time and sequence; change only the first supported negative-prediction action, then permanently follow pi_ref',
            by_scenario=single,
            reconstructed_from_existing_native_suffixes=True,new_native_simulation_seconds=0,
            interpretation='These costs reconstruct at most one intervention per scenario. Query means and sums do not represent this policy.'),
        exceptions=dict(missing=missing,
            invalid_pairs=[dict(scenario=row['scenario'],line_id=row['line_id'],checks=row['checks'])
                           for row in rows if not row['valid_for_evaluation']],
            outside_action_support=[dict(scenario=row['scenario'],line_id=row['line_id'],root=row['root'],reason=row['support_reason'])
                                    for row in rows if not row['supported']],
            zero_duration=[dict(scenario=row['scenario'],line_id=row['line_id']) for row in rows
                           if not row['checks']['nonzero_duration']],
            support_selection=selections),
        promotion=dict(passed=passed,checks=promotion,
            decision='Eligible for a separately declared single-intervention control validation'
                if passed else 'HOLD state readout; do not retune on heldout truth or substitute the line baseline'),
        interpretation=dict(primary_model='state',comparison_models=['line','reference'],
            source_training_pairs=6,new_heldout_states=len(rows),new_external_seeds=1,
            policy_cost_scope='Means and sums of separate single-root paired queries, not repeated-intervention full-day policy costs',
            direction_scope='Prediction accuracy uses supported states only; all valid states count in selector cost and regret, including explicit reference fallback',
            truth_scope='New labels score frozen issue-time predictions only; no heldout label enters fitting',
            no_automatic_control_takeover=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable',type=Path,required=True)
    parser.add_argument('--shock',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    value=analyze([json.loads(path.read_text()) for path in (args.stable,args.shock)],
                  json.loads(args.model.read_text()))
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Heldout summary exceeds48KB; keep raw branch records resident')
    with args.out.open('x') as handle:handle.write(raw)
    print(json.dumps(dict(status=value['status'],bytes=len(raw.encode()),promotion=value['promotion']['passed'])))


if __name__=='__main__':
    main()
