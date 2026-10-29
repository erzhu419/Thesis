"""Audit the one reserved V seed and its two paired-policy scenarios."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
from experiments.mcwm_closed_loop_completion_v92_20261028.analysis import _condition_readout
from experiments.mcwm_closed_loop_fresh_response_v86_20261027.analysis import _reference_valid
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import action_support, service_state
from experiments.mcwm_closed_loop_confidence_validation_v100_20261028.selection import model_support
from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import predict as confidence_predict

LINES = ('7X', '102X', '705X')
SCENES = ('stable', 'local_speed_drop')
SEEDS = (5696843,)
MODELS = ('line3', 'service5', 'service_history6')
COSTS = ('wait', 'ride', 'cost')
ZERO = ('new_initialization_updates', 'prefix_training_updates', 'suffix_updates',
        'labels_fed_back', 'new_fits', 'new_neural_updates', 'v87_truth_used')
BUDGET = ('prefixes', 'suffixes', 'native_prefix_seconds', 'suffix_native_seconds',
          'total_native_seconds', 'unknown_suffix_costs', 'labels_observed', *ZERO)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def history(condition, now):
    if condition['status'] != 'supported':
        return dict(valid=False, status=condition['status'], reason=condition['reason'],
                    value=condition.get('value'), latest_age_seconds=condition.get('latest_age_seconds'))
    return _condition_readout(condition, now, 'new_v100_native')


def predictions(model, line, features, q, reference_hold, candidate_hold):
    a = (candidate_hold-reference_hold)/7.5
    center = model['design']['line_centers'][line]
    design = [a*float(line == other) for other in LINES]
    design += [a*(features[0]-center[0]), a*(features[1]-center[1]), a*(q-center[2])]
    return {name: float(np.asarray(design[:len(model['models'][name]['coef'])])
                        @np.asarray(model['models'][name]['coef'])) for name in MODELS}


def technical_pass(checks):
    return bool(checks) and all(value for key, value in checks.items()
                               if key != 'physical_departure_separation')


def policy_scores(m, q, frozen_bias, calibration):
    forward = confidence_predict(frozen_bias['bias_model'], q, m)
    bound = forward['center']+calibration['C']*forward['support_scale']
    independent = m+calibration['state_independent_C']
    return dict(calibrated_candidate=dict(m=m,b=forward['bias'],u=forward['support_scale'],
        C=calibration['C'],U=bound,accept=m < 0. and bound < 0.),
        state_independent=dict(m=m,C=calibration['state_independent_C'],U=independent,
                              accept=m < 0. and independent < 0.))


def scores_equal(saved, expected):
    return saved is not None and expected is not None and set(saved) == set(expected) and all(
        set(saved[name]) == set(values) and all(
            saved[name][key] is value if isinstance(value, bool)
            else finite(saved[name][key]) and abs(saved[name][key]-value) <= 1e-12
            for key,value in values.items()) for name,values in expected.items())


def tail_readout(pair, delta):
    """Score the saved actual18300 boundary only; it never enters calibration."""
    boundary = pair.get('tail_boundary')
    observed = bool(boundary) and all(boundary.get(role) is not None
        and boundary[role]['time_seconds'] == 18300.
        and all(finite(boundary[role][key]) for key in ('accrued_wait_seconds', 'accrued_in_vehicle_seconds'))
        for role in ('reference', 'candidate'))
    result = dict(observed=observed, delta_per_passenger=None,
                  reason=None if observed else 'missing_exact18300')
    if observed:
        ref, cand = boundary['reference'], boundary['candidate']
        wait = delta['wait']-(cand['accrued_wait_seconds']-ref['accrued_wait_seconds'])
        ride = delta['ride']-(cand['accrued_in_vehicle_seconds']-ref['accrued_in_vehicle_seconds'])
        result['delta_per_passenger'] = dict(wait=wait/15079, ride=ride/15079, cost=(2*wait+ride)/15079)
    return result


def pair_readout(scene, pair, slot, protocol, model):
    root, line = pair['root'], pair['line_id']
    now = root['started_at']
    q = history(pair['condition'], now)
    service = service_state(slot['context'])
    training_support = model_support(model, line, q['value'])
    ref, cand = pair['remaining']['reference'], pair['remaining']['candidate']
    delta = {key: cand[key]-ref[key] for key in COSTS}
    physical = pair['physical']
    departures = [physical[role]['departure_clock'] for role in ('reference', 'candidate')]
    gap = abs(departures[1]-departures[0]) if all(finite(value) for value in departures) else None
    support = action_support(pair['reference_hold_seconds'], pair['candidate_hold_seconds'],
                             pair['sim_step_seconds'], pair['action_step_seconds'])
    expected = predictions(model, line, service['features'], q['value'],
        pair['reference_hold_seconds'], pair['candidate_hold_seconds']) if service['observed'] and q['valid'] else None
    expected_scores = policy_scores(pair['predictions']['service_history6'], q['value'], scene['frozen_bias'],
        scene['frozen_calibration']) if expected is not None and scene.get('frozen_bias') is not None \
        and scene.get('frozen_calibration') is not None else None
    endpoint = pair.get('reference_full_endpoint')
    checks = dict(fixed_seed=pair['external_seed'] == scene['external_seed'] in protocol['external_seeds'],
        fixed_scene=pair['scenario'] == scene['scenario'] in protocol['scenarios'],
        issue_window=12000 <= now < 12600,
        runtime_integrity=pair['all_pass'] and technical_pass(pair['checks']),
        same_prefix_and_rng=all(pair['checks'].get(key) is True for key in
            ('same_complete_root','same_fixed_reference','same_external_seed','native_fork')),
        zero_suffix_learning=all(pair['checks'].get(key) is True for key in
            ('zero_suffix_training','zero_truth_feedback')),
        fixed_bias_binding=scene.get('frozen_bias') is not None
            and pair.get('source_fit') == scene['frozen_bias']['source_fit']
            and pair['checks'].get('same_frozen_bias') is True,
        fixed_calibration_binding=scene.get('frozen_calibration') is not None
            and pair.get('source_calibration') == scene['frozen_calibration']['source_calibration']
            and pair['checks'].get('same_frozen_calibration') is True,
        locked_root=pair['root'] == slot['root'],
        locked_observation=pair['service'] == slot['service'] == service and pair['condition'] == slot['condition'],
        atomic_native_service=service['observed'] and service['atomic_at_issue'] and service['issue_clock'] == now,
        unchanged_native_observation=bool(slot['observation_checks']) and all(slot['observation_checks'].values()),
        strict_historical_q=q['valid'],
        action_support=support['supported'],
        actual_parallel=pair['overlap_seconds'] > 0,
        physical_clocks=all(finite(physical[role][key]) for role in ('reference', 'candidate')
            for key in ('departure_clock', 'next_arrival_clock')),
        physical_separation=gap is not None and gap >= pair['sim_step_seconds'],
        positive_duration=ref['dt'] == cand['dt'] == 24000-now > 0,
        true_terminal=all(value['observed_at'] == 24000 and value['available_at'] == 24030 for value in (ref, cand)),
        cost_identity=all(value['cost'] == 2*value['wait']+value['ride'] for value in (ref, cand)),
        delta_identity=pair['delta'] == delta and delta['cost'] == 2*delta['wait']+delta['ride'],
        reference_full_cost=endpoint is not None and all(finite(endpoint[key]) for key in COSTS)
            and endpoint['cost'] == 2*endpoint['wait']+endpoint['ride'],
        predictions_pre_issue=pair['before_suffixes'] is True and pair['prediction_recorded_at'] == now,
        saved_prediction_exact=expected is not None and set(pair['predictions']) == set(MODELS)
            and all(finite(pair['predictions'][name]) and abs(pair['predictions'][name]-expected[name]) <= 1e-12 for name in MODELS),
        slot_prediction_exact=pair['predictions'] == slot['predictions'],
        saved_policy_scores_exact=scores_equal(pair.get('policy_scores'), expected_scores),
        slot_policy_scores_exact=pair.get('policy_scores') == slot.get('policy_scores'))
    # Realized departure separation is an outcome descriptor, not a label admission gate.
    valid = pair.get('status') == 'complete' and all(value for key, value in checks.items()
        if key != 'physical_separation')
    target = delta['cost']/15079
    return dict(external_seed=scene['external_seed'], scenario=scene['scenario'], line_id=line,
        root=root, physical_root_id=[scene['external_seed'], scene['scenario'], root['sequence']],
        valid=valid, checks=checks, reference_hold_seconds=pair['reference_hold_seconds'],
        candidate_hold_seconds=pair['candidate_hold_seconds'], departure_gap_seconds=gap,
        physical_noop=gap is not None and gap < pair['sim_step_seconds'],
        physical=physical, overlap_seconds=pair['overlap_seconds'], dt=ref['dt'], condition=q,
        training_support=training_support,
        tail_boundary=pair.get('tail_boundary'),tail_boundary_status=pair.get('tail_boundary_status'),
        tail=tail_readout(pair, delta),
        reference_full_endpoint=endpoint,policy_scores=pair.get('policy_scores'),
        service_features=service['features'], predictions=pair['predictions'],
        observed_delta=delta, observed_delta_per_passenger={key:value/15079 for key,value in delta.items()},
        label_delta_per_passenger={key:value/15079 for key,value in delta.items()} if valid else None,
        primary_prediction_error=pair['predictions']['service_history6']-target if valid else None,
        primary_direction_correct=((pair['predictions']['service_history6'] > 0)-(pair['predictions']['service_history6'] < 0)
                                  == (target > 0)-(target < 0)) if valid else None)


def signals(rows):
    valid = [row for row in rows if row['valid']]
    n = len(valid)
    return dict(pairs=len(rows), valid_pairs=n,
        mean_delta={key:sum(row['label_delta_per_passenger'][key] for row in valid)/n if n else None for key in COSTS},
        negative=sum(row['observed_delta']['cost'] < 0 for row in valid),
        zero=sum(row['observed_delta']['cost'] == 0 for row in valid),
        positive=sum(row['observed_delta']['cost'] > 0 for row in valid),
        primary_mae=sum(abs(row['primary_prediction_error']) for row in valid)/n if n else None,
        primary_direction_correct=sum(row['primary_direction_correct'] for row in valid),
        scope='Descriptive paired labels, never a sum of control-policy gains')


def normalize_failure(scene):
    """Keep the collector's supported failure-only receipt visible to analysis."""
    if scene['status'] != 'failed' or ('pairs' in scene and 'budget' in scene):
        return scene
    failure = scene.get('failure') or {}
    spent = failure.get('consumed_budget')
    budget = {key:0 for key in BUDGET}
    if spent is not None:
        budget.update(spent)
    else:
        # Zero is only a known-spend lower bound, never an assertion of no work.
        budget['unknown_suffix_costs'] = 1
    budget['total_native_seconds_is_lower_bound'] = bool(
        budget.get('total_native_seconds_is_lower_bound') or budget['unknown_suffix_costs'])
    return dict(scene,seed=6207,pairs=[],coverage={},budget=budget,
        selection={line:dict(selected=None,status='missing',reasons=dict(native_failure=1)) for line in LINES},
        integrity=dict(all_pass=False,checks=dict(native_collection_failed=False)),
        reference_snapshot=None,first_pair_gate=None,training_started=False,models_fitted=False,
        labels_used_for_selection=scene.get('labels_used_for_selection'),
        predictions_used_for_selection=scene.get('predictions_used_for_selection'))


def no_selection_feedback(scene):
    flags = [scene.get(key) for key in ('labels_used_for_selection', 'predictions_used_for_selection')]
    if any(value is True for value in flags):
        return False
    return True if all(value is False for value in flags) else None


def analyze(collectors, protocol, audit, model):
    collectors = [normalize_failure(scene) for scene in collectors]
    seeds, scenes = tuple(protocol['external_seeds']), tuple(protocol['scenarios'])
    expected = {(seed,scene) for seed in seeds for scene in scenes}
    expected_pairs = len(expected)*len(LINES)
    declared_biases = [scene['frozen_bias'] for scene in collectors if scene.get('frozen_bias') is not None]
    frozen_bias = declared_biases[0] if declared_biases else None
    declared_calibrations = [scene['frozen_calibration'] for scene in collectors
                            if scene.get('frozen_calibration') is not None]
    frozen_calibration = declared_calibrations[0] if declared_calibrations else None
    keys = [(row['external_seed'],row['scenario']) for row in collectors]
    if len(keys) != len(set(keys)) or set(keys)-expected:
        raise ValueError('Duplicate or undeclared collector; never substitute another seed or scene')
    rows, slots, missing, scene_checks = [], [], [], {}
    for scene in collectors:
        key = scene['external_seed'],scene['scenario']
        pairs = {pair['line_id']:pair for pair in scene['pairs']}
        if len(pairs) != len(scene['pairs']) or set(pairs)-set(LINES):
            raise ValueError('Each line has at most one declared first-root pair')
        for line in LINES:
            selected = scene['coverage'].get(line)
            slots.append(dict(external_seed=key[0],scenario=key[1],line_id=line,
                slot=selected, selection=scene['selection'][line]))
            if line not in pairs:
                missing.append(dict(external_seed=key[0],scenario=key[1],line_id=line,
                    root=selected['root'] if selected else None,selection=scene['selection'][line]))
                continue
            if selected is None:
                raise ValueError('A queried pair must have an issue-locked slot')
            rows.append(pair_readout(scene,pairs[line],selected,protocol,model))
        complete = scene['status'] == 'complete'
        started = scene['status'] != 'not_started'
        reference = scene.get('reference_snapshot')
        native_error = scene.get('error') is not None or scene.get('failure') is not None
        scene_checks[str(key[0])+'/'+key[1]] = dict(complete=complete,
            collector_integrity=(scene['integrity']['all_pass'] and all(scene['integrity']['checks'].values())) if started else None,
            source_initializer=scene['seed'] == 6207 if started else None,
            frozen_reference=_reference_valid(scene) if started and reference is not None else None,
            original_actor=reference['source_actor_path'] == audit['source_reference_paths'][key[1]]
                if started and reference is not None else None,
            actual_external_seed=reference['heldout_external_seed'] == key[0]
                if started and reference is not None else None,
            parent_to12600=scene['budget']['native_prefix_seconds'] == 12600
                if started and scene['budget']['prefixes'] else None,
            scenario_budget=not scene['budget'].get('total_native_seconds_is_lower_bound',False)
                and scene['budget']['unknown_suffix_costs'] == 0 and scene['budget']['total_native_seconds'] <= 84600,
            zero_learning=all(scene['budget'][name] == 0 for name in ZERO),
            never_used_labels=no_selection_feedback(scene),
            never_fitted=scene.get('training_started') is False and scene.get('models_fitted') is False,
            no_native_error=not native_error and scene['status'] != 'failed')
        scene_checks[str(key[0])+'/'+key[1]]['frozen_bias_matches'] = (
            scene.get('frozen_bias') == frozen_bias and frozen_bias is not None) if started else None
        scene_checks[str(key[0])+'/'+key[1]]['frozen_calibration_matches'] = (
            scene.get('frozen_calibration') == frozen_calibration and frozen_calibration is not None) if started else None
    for seed,scene in sorted(expected-set(keys)):
        missing.extend(dict(external_seed=seed,scenario=scene,line_id=line,root=None,selection=None) for line in LINES)
    total = {key:sum(scene['budget'][key] for scene in collectors) for key in BUDGET}
    total['total_native_seconds_is_lower_bound'] = bool(total['unknown_suffix_costs']
        or any(scene['budget'].get('total_native_seconds_is_lower_bound',False) for scene in collectors))
    total['native_seconds_scope'] = 'known_lower_bound' if total['total_native_seconds_is_lower_bound'] else 'actual'
    good = [row for row in rows if row['valid']]
    qlo,qhi = protocol['q_range_for_scaling_only']
    matrix = np.asarray([[1.,2*(row['condition']['value']-qlo)/(qhi-qlo)-1.] for row in good],dtype=float).reshape(-1,2)
    rank = int(np.linalg.matrix_rank(matrix)) if len(good) else 0
    gates = [scene['first_pair_gate'] for scene in collectors if scene.get('first_pair_gate')]
    gate = gates[0] if gates else None
    gate_matches = sum(row['external_seed'] == gate.get('external_seed') and row['scenario'] == gate.get('scenario')
        and row['root'] == gate.get('root') for row in rows) if gate else 0
    first_collector = next((value for value in collectors if value['external_seed'] == seeds[0]
                            and value['scenario'] == 'stable'), None)
    first_pair = min(first_collector['pairs'],key=lambda value:(value['root']['started_at'],value['root']['sequence'])) \
        if first_collector and first_collector['pairs'] else None
    checks = dict(two_declared_collectors=set(keys) == expected and len(expected) == 2,
        frozen_protocol=protocol['source_initializer'] == 6207 and seeds == SEEDS
            and scenes == SCENES and protocol['budget']['total_cap'] == 169200
            and protocol['budget']['expected_collectors'] == len(expected)
            and protocol['budget']['expected_pairs'] == expected_pairs == 6
            and protocol['budget']['expected_physical_labels'] == 2*expected_pairs,
        original_model_support=protocol['frozen_common_support'] == model['common_support'],
        source_audit=audit['all_pass'] and all(audit['checks'].values()) and audit['source_initializer'] == 6207,
        no_source_computation=all(audit[key] == 0 for key in ('new_fits','new_neural_updates','new_suffixes','new_labels','v87_truth_used','native_seconds')),
        all_collectors_pass=bool(scene_checks) and all(all(check.values()) for check in scene_checks.values()),
        six_distinct_roots=len(rows) == expected_pairs
            and len({tuple(row['physical_root_id']) for row in rows}) == expected_pairs,
        six_valid_mature_pairs=len(good) == expected_pairs and not missing,
        first_pair_reused_once=gate is not None and gate_matches == 1 and gate['all_pass']
            and gate.get('new_pair_in_formal_cohort') is True
            and gate.get('labels_used_for_gate') is False and gate.get('cost_sign_used_for_gate') is False,
        exact_first_physical_gate=gate is not None and first_pair is not None
            and gate.get('external_seed') == seeds[0] and gate.get('scenario') == 'stable'
            and gate['root'] == first_pair['root'] and gate['checks'] == first_pair['checks']
            and technical_pass(gate['checks']),
        all_collector_gates_same=bool(gates) and all(value == gate for value in gates),
        all_labels_observed=total['labels_observed'] == 2*expected_pairs,
        no_unknown_budget=total['unknown_suffix_costs'] == 0 and not total['total_native_seconds_is_lower_bound'],
        budget_identity=total['total_native_seconds'] == total['native_prefix_seconds']+total['suffix_native_seconds'],
        within_budget=not total['total_native_seconds_is_lower_bound']
            and total['total_native_seconds'] <= protocol['budget']['total_cap'],
        zero_training_fitting_or_feedback=all(total[key] == 0 for key in ZERO))
    passed = all(checks.values())
    unsupported = [dict(external_seed=value['external_seed'],scenario=value['scenario'],line_id=value['line_id'],
        root=value['slot']['root'],status=value['slot'].get('status'),reason=value['slot'].get('reason'),
        condition=value['slot']['condition']) for value in slots if value['slot'] and value['slot'].get('status') == 'unsupported']
    q_coverage = {}
    for line in LINES:
        observed = [value['slot'] for value in slots if value['line_id'] == line and value['slot']]
        conditions = [history(slot['condition'], slot['root']['started_at']) for slot in observed]
        valid_conditions = [condition for condition in conditions if condition['valid']]
        support = [model_support(model, line, condition['value']) for condition in valid_conditions]
        labeled = [row for row in good if row['line_id'] == line]
        q_coverage[line] = dict(observed_roots=len(observed),valid_history_roots=len(valid_conditions),
            in_training_range_roots=sum(value['in_training_range'] is True for value in support),
            extrapolation_roots=sum(value['extrapolation'] is True for value in support),
            completed_pairs=sum(pair['line_id'] == line and pair['status'] == 'complete'
                for scene in collectors for pair in scene['pairs']),valid_pairs=len(labeled),
            observed_values=sorted({condition['value'] for condition in valid_conditions}),
            labeled_values=sorted({row['condition']['value'] for row in labeled}))
    return dict(status='complete' if passed else 'hold', integrity=dict(all_pass=passed,checks=checks),
        scene_checks=scene_checks, pairs=rows, budget=total,frozen_bias=frozen_bias,
        frozen_calibration=frozen_calibration,
        covariate_design=dict(rows=len(good),rank=rank,columns=['1','fixed_scaled_historical_q'],
            target_solves=0,rank_is_admission_gate=False),
        signals=dict(overall=signals(rows),
            by_seed={str(seed):signals([row for row in rows if row['external_seed'] == seed]) for seed in seeds},
            by_seed_scenario={str(seed)+'/'+scene:signals([row for row in rows if row['external_seed'] == seed
                and row['scenario'] == scene]) for seed in seeds for scene in scenes},
            by_scenario={scene:signals([row for row in rows if row['scenario'] == scene]) for scene in scenes},
            by_line={line:signals([row for row in rows if row['line_id'] == line]) for line in LINES}),
        q_coverage=q_coverage,
        exceptions=dict(missing=missing,unsupported=unsupported,
            native_failures=[dict(external_seed=scene['external_seed'],scenario=scene['scenario'],
                reason=scene.get('reason'),error=scene.get('error'),failure=scene.get('failure'),budget=scene['budget'])
                for scene in collectors if scene['status'] == 'failed'
                    or scene.get('error') is not None or scene.get('failure') is not None],
            invalid_pairs=[dict(external_seed=row['external_seed'],scenario=row['scenario'],line_id=row['line_id'],checks=row['checks']) for row in rows if not row['valid']],
            nonseparable=[row['physical_root_id'] for row in rows if not row['checks']['physical_separation']],
            zero_duration=[row['physical_root_id'] for row in rows if row['dt'] == 0],
            truncated=[row['physical_root_id'] for row in rows if not row['checks']['true_terminal']]),
        collection_gate=dict(passed=passed,decision='Complete V full-day labels ready for the fixed policy evaluation; tail completeness checked there'
            if passed else 'HOLD: preserve unsupported/missing/invalid slots without replacement'),
        interpretation=dict(model_fits=0,optimizer_updates=0,policy_cost_evaluated=False,
            v87_truth_used=0,reserved816207_used=0,calibration_in_this_readout=False,
            tail_is_scoring_only=True,independent_external_seeds=1,no_further_collection=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collectors',type=Path,nargs='+',required=True)
    for name in ('protocol','audit','model','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    report = analyze([json.loads(path.read_text()) for path in args.collectors],
        protocol,json.loads(args.audit.read_text()),json.loads(args.model.read_text()))
    raw = json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode()) > protocol['retrieval']['final_summary_bytes']:
        raise ValueError('Collection summary exceeds its declared small-summary byte limit')
    with args.out.open('x') as handle: handle.write(raw)
    print(json.dumps(dict(status=report['status'],bytes=len(raw.encode()),fits=0)))


if __name__ == '__main__':
    main()
