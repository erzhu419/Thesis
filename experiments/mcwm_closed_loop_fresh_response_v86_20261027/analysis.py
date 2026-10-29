"""Score frozen stale/fresh readouts as one-intervention full-day policies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCENARIOS = ('stable', 'local_speed_drop')
LINES = ('7X', '102X', '705X')
MODELS = ('stale', 'fresh', 'legacy_narrow')
BUDGET_FIELDS = ('native_prefix_seconds', 'suffix_native_seconds', 'total_native_seconds',
                 'prefixes', 'suffixes', 'labels_observed', 'new_initialization_updates',
                 'prefix_training_updates', 'suffix_updates', 'labels_fed_back')
COST_KEYS = ('wait', 'ride', 'cost')


def _direction(value):
    return 'candidate_better' if value < 0 else 'reference_better' if value > 0 else 'tie'


def pair_readout(scene, pair):
    reference, candidate = (pair['remaining'][role] for role in ('reference', 'candidate'))
    delta = {key: candidate[key]-reference[key] for key in COST_KEYS}
    denominator = scene['departed_passengers']
    target = delta['cost']/denominator
    prediction = pair['prediction']
    ref_hold, cand_hold = pair['reference_hold_seconds'], pair['candidate_hold_seconds']
    physical = pair['physical']
    checks = dict(
        runtime_pair_integrity=pair['all_pass'],
        nonzero_duration=reference['dt'] == candidate['dt'] > 0,
        real_terminal_maturity=all(value['observed_at'] == 24000 and value['available_at'] == 24030
                                   for value in (reference, candidate)),
        delta_identity=delta == pair['delta'] and delta['cost'] == 2*delta['wait']+delta['ride'],
        full_reference_cost_identity=pair['reference_full_endpoint']['cost']
            == 2*pair['reference_full_endpoint']['wait']+pair['reference_full_endpoint']['ride'],
        physical_clocks_complete=all(physical[role][key] is not None
            for role in ('reference', 'candidate') for key in ('departure_clock', 'next_arrival_clock')),
        pre_issue_prediction=pair['before_suffixes'] is True
            and pair['prediction_recorded_at'] == pair['root']['started_at'],
        actual_parallel=pair['overlap_seconds'] > 0)
    models = {}
    for name in MODELS:
        saved = prediction['models'][name]
        estimate, choice, supported = saved['predicted_delta'], saved['choice'], saved['supported']
        action_support = (3.0 <= ref_hold <= 3.6 if name == 'legacy_narrow'
                          else 3.0 <= ref_hold < 3.75)
        action_support = action_support and abs(cand_hold-7.5) <= 1e-5 and cand_hold > ref_hold
        # Nonfinite state is the only issue-state fallback in addition to the fixed action domain.
        expected_support = action_support and prediction['z'] is not None
        checks[name+'_support_declaration'] = supported == expected_support
        if supported:
            consistent = isinstance(estimate, (int, float)) and math.isfinite(estimate)
            consistent = consistent and saved['reason'] is None
            consistent = consistent and choice == ('candidate' if estimate < 0 else 'reference')
        else:
            consistent = estimate is None and choice == 'reference' and saved['reason'] is not None
        checks[name+'_prediction_choice_consistent'] = consistent
        selected = delta if choice == 'candidate' else dict(wait=0., ride=0., cost=0.)
        regret = selected['cost']-min(0., delta['cost'])
        models[name] = dict(supported=supported, predicted_delta=estimate, choice=choice,
            fallback_reason=saved['reason'], selected_minus_reference=selected,
            selected_minus_reference_per_passenger={key: value/denominator for key, value in selected.items()},
            two_action_regret=regret, two_action_regret_per_passenger=regret/denominator,
            action_correct=regret == 0,
            predicted_direction=_direction(estimate) if estimate is not None else None,
            direction_correct=_direction(estimate) == _direction(target) if estimate is not None else None,
            absolute_prediction_error=abs(estimate-target) if estimate is not None else None,
            squared_prediction_error=(estimate-target)**2 if estimate is not None else None)
    checks['common_main_support'] = models['stale']['supported'] == models['fresh']['supported']
    checks['legacy_reuses_stale_prediction'] = (not models['legacy_narrow']['supported']
        or models['legacy_narrow']['predicted_delta'] == models['stale']['predicted_delta'])
    valid = pair['status'] == 'complete' and all(checks.values())
    return dict(scenario=scene['scenario'], line_id=pair['line_id'], root=pair['root'],
        valid_for_evaluation=valid, checks=checks, runtime_checks=pair['checks'],
        reference_hold_seconds=ref_hold, candidate_hold_seconds=cand_hold, z=prediction['z'],
        actual_delta=delta, actual_delta_per_passenger={key: value/denominator for key, value in delta.items()},
        truth_direction=_direction(target), remaining=pair['remaining'], physical=physical,
        reference_full_endpoint=pair['reference_full_endpoint'], overlap_seconds=pair['overlap_seconds'],
        reference_two_action_regret=max(0., -delta['cost']),
        reference_two_action_regret_per_passenger=max(0., -delta['cost'])/denominator, models=models)


def _group(rows):
    valid = [row for row in rows if row['valid_for_evaluation']]
    n = len(valid)
    result = dict(states=len(rows), valid_states=n, models={},
        reference=dict(sum_two_action_regret=sum(row['reference_two_action_regret'] for row in valid) if n else None))
    for name in MODELS:
        model_rows = [row['models'][name] for row in valid]
        scored = [row for row in model_rows if row['predicted_delta'] is not None]
        prediction_n = len(scored)
        costs = [row['selected_minus_reference_per_passenger']['cost'] for row in model_rows]
        result['models'][name] = dict(
            supported_states=sum(row['supported'] for row in model_rows),
            out_of_support_states=sum(not row['supported'] for row in model_rows),
            candidate_choices=sum(row['choice'] == 'candidate' for row in model_rows),
            reference_choices=sum(row['choice'] == 'reference' for row in model_rows),
            prediction_count=prediction_n,
            direction_correct=sum(row['direction_correct'] for row in scored),
            direction_accuracy=sum(row['direction_correct'] for row in scored)/prediction_n if prediction_n else None,
            mae=sum(row['absolute_prediction_error'] for row in scored)/prediction_n if prediction_n else None,
            rmse=math.sqrt(sum(row['squared_prediction_error'] for row in scored)/prediction_n) if prediction_n else None,
            sum_selected_minus_reference={key: sum(row['selected_minus_reference'][key] for row in model_rows)
                if n else None for key in COST_KEYS},
            sum_two_action_regret=sum(row['two_action_regret'] for row in model_rows) if n else None,
            mean_two_action_regret_per_passenger=sum(row['two_action_regret_per_passenger']
                for row in model_rows)/n if n else None,
            improved_queries=sum(value < 0 for value in costs), harmed_queries=sum(value > 0 for value in costs),
            unchanged_queries=sum(value == 0 for value in costs))
    return result


def _reference_valid(scene):
    snapshot = scene['reference_snapshot']
    old = snapshot['source_snapshot_metadata']
    return bool(snapshot['prior_run_reference'] is True and snapshot['available_before_heldout_episode'] is True
        and old['source'] == 'v75' and old['source_arm'] == 'fresh'
        and old['seed'] == 6207 and old['scenario'] == scene['scenario']
        and old['packets'] == 37 and old['covariates'] == old['complete_outcomes'] == 185
        and old['counts'] == dict(critic=296, value=296, actor=296, alpha=296)
        and old['max_available_at'] <= old['data_deadline'] == 12000
        and old['deterministic'] is True and old['actor_frozen'] is True)


def _single_intervention(rows, denominator):
    """Use the first negative supported issue prediction, then its real pi_ref suffix."""
    ordered = sorted(rows, key=lambda row: (row['root']['started_at'], row['root']['sequence']))
    usable = len(ordered) == 3 and all(row['valid_for_evaluation'] for row in ordered)
    usable = usable and all(row['reference_full_endpoint'] == ordered[0]['reference_full_endpoint'] for row in ordered)
    full_reference = ordered[0]['reference_full_endpoint'] if usable else None
    oracle_delta = min([0.]+[row['actual_delta']['cost'] for row in ordered]) if usable else None
    result = dict(status='complete' if usable else 'incomplete', models={},
        opportunity_order=[dict(line_id=row['line_id'], root=row['root']) for row in ordered],
        oracle_best_single_root_delta_cost=oracle_delta,
        oracle_scope='Evaluation-only best of no intervention and three single-root native suffixes')
    for name in ('reference', *MODELS):
        selected = next((row for row in ordered if row['models'][name]['supported']
            and row['models'][name]['predicted_delta'] < 0), None) if name in MODELS else None
        delta = (selected['actual_delta'] if selected else dict(wait=0., ride=0., cost=0.)) if usable else None
        endpoint = {key: full_reference[key]+delta[key] for key in COST_KEYS} if usable else None
        result['models'][name] = dict(status='complete' if usable else 'incomplete',
            selected_root=dict(line_id=selected['line_id'], **selected['root']) if selected else None,
            interventions=int(selected is not None) if usable else None,
            selected_minus_reference=delta,
            selected_minus_reference_per_passenger={key: value/denominator for key, value in delta.items()}
                if usable else None,
            full_reference_endpoint=full_reference, full_selected_endpoint=endpoint,
            full_selected_cost_per_passenger=endpoint['cost']/denominator if usable else None,
            single_opportunity_oracle_regret=delta['cost']-oracle_delta if usable else None,
            single_opportunity_oracle_regret_per_passenger=(delta['cost']-oracle_delta)/denominator if usable else None)
    comparisons = {}
    for label, first, second in (('fresh_minus_stale', 'fresh', 'stale'),
                                 ('stale_minus_legacy_narrow', 'stale', 'legacy_narrow'),
                                 ('fresh_minus_reference', 'fresh', 'reference')):
        comparisons[label] = {key: result['models'][first]['full_selected_endpoint'][key]
            -result['models'][second]['full_selected_endpoint'][key] for key in COST_KEYS} if usable else None
    result['comparisons'] = comparisons
    result['comparisons_per_passenger'] = {label: {key: value/denominator for key, value in delta.items()}
        if delta is not None else None for label, delta in comparisons.items()}
    return result


def _training_checks(artifact):
    expected_cases = [scenario+'/'+line for scenario in SCENARIOS for line in LINES]
    checks = dict(training_gate_passed=artifact['gate']['all_pass'], four_fixed_fits=artifact['fits'] == 4,
        fixed_ridge=artifact['ridge'] == 1.,
        fixed_target_scale=artifact['denominator'] == 15079 and artifact['hold_scale'] == 7.5,
        common_action_domain=artifact['action_support'] == dict(reference_min=3., reference_max_exclusive=3.75,
            candidate=7.5, candidate_tolerance=1e-5),
        old_action_domain=artifact['legacy_action_support'] == dict(reference_min=3., reference_max_inclusive=3.6,
            candidate=7.5, candidate_tolerance=1e-5),
        no_original_model_updates=all(artifact[key+'_updates'] == 0
            for key in ('actor', 'critic', 'value', 'B', 'world', 'trust')))
    for name, version, seed in (('stale', 'v84_stale', 106207), ('fresh', 'v85_fresh', 216207)):
        arm = artifact['arms'][name]
        scope = arm['training_scope']
        checks[name+'_two_fixed_fits'] = arm['fits'] == 2 and arm['gate']['all_pass']
        checks[name+'_fixed_ranks'] = arm['models']['line']['design_rank'] == 3 and arm['models']['state']['design_rank'] == 4
        checks[name+'_six_training_pairs'] = (scope['source_scope'] == version and scope['pairs'] == 6
            and scope['physical_outcome_labels'] == 12 and scope['source_seed'] == 6207
            and scope['external_seed'] == seed and scope['cases'] == expected_cases)
        checks[name+'_mature_training_labels'] = scope['label_available_at'] == 24030
        checks[name+'_no_heldout_truth'] = scope['heldout_labels_used'] == 0
    return checks


def analyze(scenes, artifact):
    indexed = {row['scenario']: row for row in scenes}
    if set(indexed) != set(SCENARIOS) or len(scenes) != 2:
        raise ValueError('Score exactly the two predeclared new heldout scenarios')
    rows, missing, selections, endpoint_audit = [], [], [], {}
    for scenario in SCENARIOS:
        scene = indexed[scenario]
        by_line = {row['line_id']: row for row in scene['pairs']}
        if len(by_line) != len(scene['pairs']) or set(by_line)-set(LINES):
            raise ValueError('Duplicate or undeclared heldout line')
        scene_rows = []
        for line in LINES:
            selection = scene['selection'][line]
            selections.append(dict(scenario=scenario, line_id=line, selection=selection))
            if line not in by_line:
                missing.append(dict(scenario=scenario, line_id=line, selection=selection))
                continue
            row = pair_readout(scene, by_line[line])
            rows.append(row)
            scene_rows.append(row)
        complete = len(scene_rows) == 3
        spread = {key: max(row['reference_full_endpoint'][key] for row in scene_rows)
            -min(row['reference_full_endpoint'][key] for row in scene_rows) if complete else None for key in COST_KEYS}
        endpoint_audit[scenario] = dict(complete=complete, max_minus_min=spread,
            all_equal=complete and all(value == 0 for value in spread.values()))
    budget = {key: sum(row['budget'][key] for row in scenes) for key in BUDGET_FIELDS}
    training_checks = _training_checks(artifact)
    checks = dict(two_complete_scenarios=all(row['status'] == 'complete' for row in scenes),
        runtime_scene_integrity=all(row['integrity']['all_pass'] for row in scenes),
        fixed_new_external_seed=all(row['external_seed'] == 316207 for row in scenes),
        fixed_source_initializer=all(row['seed'] == 6207 for row in scenes),
        fixed_denominator=all(row['departed_passengers'] == 15079 for row in scenes),
        new_prefixes=all(bool(row['prefix_id']) for row in scenes) and len({row['prefix_id'] for row in scenes}) == 2,
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
    single = {scenario: _single_intervention([row for row in rows if row['scenario'] == scenario],
        indexed[scenario]['departed_passengers']) for scenario in SCENARIOS}
    complete = all(value['status'] == 'complete' for value in single.values())
    primary_models = {}
    for name in ('reference', *MODELS):
        values = [single[scenario]['models'][name] for scenario in SCENARIOS]
        primary_models[name] = dict(
            mean_full_day_cost_per_passenger=sum(value['full_selected_cost_per_passenger'] for value in values)/2 if complete else None,
            mean_selected_minus_reference_per_passenger={key: sum(value['selected_minus_reference_per_passenger'][key]
                for value in values)/2 for key in COST_KEYS} if complete else None,
            interventions=sum(value['interventions'] for value in values) if complete else None)
    comparisons = {label: {key: sum(single[scenario]['comparisons_per_passenger'][label][key]
        for scenario in SCENARIOS)/2 for key in COST_KEYS} if complete else None
        for label in ('fresh_minus_stale', 'stale_minus_legacy_narrow', 'fresh_minus_reference')}
    fresh, stale, reference = (primary_models[name] for name in ('fresh', 'stale', 'reference'))
    promotion = dict(technical_integrity=all(checks.values()), complete_full_day_policies=complete,
        fresh_actually_changes_an_action=complete and fresh['interventions'] > 0,
        fresh_mean_beats_stale=complete and fresh['mean_full_day_cost_per_passenger'] < stale['mean_full_day_cost_per_passenger'],
        fresh_mean_beats_reference=complete and fresh['mean_full_day_cost_per_passenger'] < reference['mean_full_day_cost_per_passenger'])
    for scenario in SCENARIOS:
        value = single[scenario]
        promotion[scenario+'_fresh_not_worse_than_reference'] = (value['status'] == 'complete'
            and value['comparisons']['fresh_minus_reference']['cost'] <= 0)
        promotion[scenario+'_fresh_not_worse_than_stale'] = (value['status'] == 'complete'
            and value['comparisons']['fresh_minus_stale']['cost'] <= 0)
    passed = all(promotion.values())
    return dict(status='complete' if all(checks.values()) else 'hold',
        integrity=dict(all_pass=all(checks.values()), checks=checks), training_checks=training_checks,
        primary=dict(policy='First supported negative prediction in (issue time, sequence) order; then permanently pi_ref',
            by_scenario=single, models=primary_models, mean_comparisons_per_passenger=comparisons,
            reconstructed_from_existing_native_suffixes=True, new_native_simulation_seconds=0),
        overall=_group(rows), by_scenario={scenario: _group([row for row in rows if row['scenario'] == scenario]) for scenario in SCENARIOS},
        by_line={line: _group([row for row in rows if row['line_id'] == line]) for line in LINES},
        pairs=rows, budget=budget, reference_endpoint_audit=endpoint_audit,
        exceptions=dict(missing=missing,
            invalid_pairs=[dict(scenario=row['scenario'], line_id=row['line_id'], checks=row['checks'])
                for row in rows if not row['valid_for_evaluation']],
            outside_action_support=[dict(scenario=row['scenario'], line_id=row['line_id'], root=row['root'],
                model=name, reason=row['models'][name]['fallback_reason']) for row in rows for name in MODELS
                if not row['models'][name]['supported']],
            zero_duration=[dict(scenario=row['scenario'], line_id=row['line_id']) for row in rows
                if not row['checks']['nonzero_duration']], support_selection=selections),
        promotion=dict(passed=passed, checks=promotion,
            decision='Eligible for a separately declared control validation' if passed
                else 'HOLD this fixed fresh-response candidate; no heldout retuning or automatic control takeover'),
        interpretation=dict(primary_model='fresh', comparison_models=['stale', 'reference', 'legacy_narrow'],
            query_metrics='Auxiliary separate single-root evaluations; their sums are not policy costs',
            guard_contribution='stale minus legacy_narrow: identical stale coefficients, broader action support only',
            fresh_contribution='fresh minus stale: identical model and action support, different six-pair supervision',
            line_model_scope='Auxiliary training fit only; no heldout line-model selector evaluation',
            heldout_truth_scope='New native labels only score predictions sealed before branching; no fitting',
            previous_v85_failure_unchanged=True, no_automatic_control_takeover=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stable', 'shock', 'model', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = analyze([json.loads(path.read_text()) for path in (args.stable, args.shock)], json.loads(args.model.read_text()))
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 64000:
        raise ValueError('Heldout summary exceeds64KB; keep raw branch records resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=value['status'], bytes=len(raw.encode()), promotion=value['promotion']['passed'])))


if __name__ == '__main__':
    main()
