"""Audit the two-seed confidence-training collection; never fit a response model."""
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

LINES = ('7X', '102X', '705X')
SCENES = ('stable', 'local_speed_drop')
SEEDS = (309177151, 1476154684)
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
    return _condition_readout(condition, now, 'new_v96_native')


def predictions(model, line, features, q, reference_hold, candidate_hold):
    a = (candidate_hold-reference_hold)/7.5
    center = model['design']['line_centers'][line]
    design = [a*float(line == other) for other in LINES]
    design += [a*(features[0]-center[0]), a*(features[1]-center[1]), a*(q-center[2])]
    return {name: float(np.asarray(design[:len(model['models'][name]['coef'])])
                        @np.asarray(model['models'][name]['coef'])) for name in MODELS}


def pair_readout(scene, pair, slot, protocol, model):
    root, line = pair['root'], pair['line_id']
    now = root['started_at']
    q = history(pair['condition'], now)
    service = service_state(slot['context'])
    lo, hi = protocol['frozen_common_support']['q_training_ranges'][line]
    q_in_range = q['valid'] and lo <= q['value'] <= hi
    ref, cand = pair['remaining']['reference'], pair['remaining']['candidate']
    delta = {key: cand[key]-ref[key] for key in COSTS}
    physical = pair['physical']
    departures = [physical[role]['departure_clock'] for role in ('reference', 'candidate')]
    gap = abs(departures[1]-departures[0]) if all(finite(value) for value in departures) else None
    support = action_support(pair['reference_hold_seconds'], pair['candidate_hold_seconds'],
                             pair['sim_step_seconds'], pair['action_step_seconds'])
    expected = predictions(model, line, service['features'], q['value'],
        pair['reference_hold_seconds'], pair['candidate_hold_seconds']) if service['observed'] and q['valid'] else None
    checks = dict(fixed_seed=pair['external_seed'] == scene['external_seed'] in SEEDS,
        fixed_scene=pair['scenario'] == scene['scenario'] in SCENES,
        issue_window=12000 <= now < 12600,
        runtime_integrity=pair['all_pass'] and bool(pair['checks']) and all(pair['checks'].values()),
        same_prefix_and_rng=all(pair['checks'].get(key) is True for key in
            ('same_complete_root','same_fixed_reference','same_external_seed','native_fork')),
        zero_suffix_learning=all(pair['checks'].get(key) is True for key in
            ('zero_suffix_training','zero_truth_feedback')),
        locked_root=pair['root'] == slot['root'],
        locked_observation=pair['service'] == slot['service'] == service and pair['condition'] == slot['condition'],
        atomic_native_service=service['observed'] and service['atomic_at_issue'] and service['issue_clock'] == now,
        unchanged_native_observation=bool(slot['observation_checks']) and all(slot['observation_checks'].values()),
        strict_historical_q=q['valid'], original_line_q_range=q_in_range,
        action_support=support['supported'],
        actual_parallel=pair['overlap_seconds'] > 0,
        physical_clocks=all(finite(physical[role][key]) for role in ('reference', 'candidate')
            for key in ('departure_clock', 'next_arrival_clock')),
        physical_separation=gap is not None and gap >= pair['sim_step_seconds'],
        positive_duration=ref['dt'] == cand['dt'] == 24000-now > 0,
        true_terminal=all(value['observed_at'] == 24000 and value['available_at'] == 24030 for value in (ref, cand)),
        cost_identity=all(value['cost'] == 2*value['wait']+value['ride'] for value in (ref, cand)),
        delta_identity=pair['delta'] == delta and delta['cost'] == 2*delta['wait']+delta['ride'],
        predictions_pre_issue=pair['before_suffixes'] is True and pair['prediction_recorded_at'] == now,
        saved_prediction_exact=expected is not None and set(pair['predictions']) == set(MODELS)
            and all(finite(pair['predictions'][name]) and abs(pair['predictions'][name]-expected[name]) <= 1e-12 for name in MODELS),
        slot_prediction_exact=pair['predictions'] == slot['predictions'])
    valid = pair.get('status') == 'complete' and all(checks.values())
    target = delta['cost']/15079
    return dict(external_seed=scene['external_seed'], scenario=scene['scenario'], line_id=line,
        root=root, physical_root_id=[scene['external_seed'], scene['scenario'], root['sequence']],
        valid=valid, checks=checks, reference_hold_seconds=pair['reference_hold_seconds'],
        candidate_hold_seconds=pair['candidate_hold_seconds'], departure_gap_seconds=gap,
        physical=physical, overlap_seconds=pair['overlap_seconds'], dt=ref['dt'], condition=q,
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
        labels_used_for_selection=False,predictions_used_for_selection=False)


def analyze(collectors, protocol, audit, model):
    collectors = [normalize_failure(scene) for scene in collectors]
    expected = {(seed,scene) for seed in SEEDS for scene in SCENES}
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
        scene_checks[str(key[0])+'/'+key[1]] = dict(complete=complete,
            collector_integrity=complete and scene['integrity']['all_pass'] and all(scene['integrity']['checks'].values()),
            source_initializer=scene['seed'] == 6207,
            frozen_reference=complete and _reference_valid(scene),
            original_actor=complete and scene['reference_snapshot']['source_actor_path'] == audit['source_reference_paths'][key[1]],
            actual_external_seed=complete and scene['reference_snapshot']['heldout_external_seed'] == key[0],
            parent_to12600=scene['budget']['native_prefix_seconds'] == 12600,
            scenario_budget=not scene['budget'].get('total_native_seconds_is_lower_bound',False)
                and scene['budget']['unknown_suffix_costs'] == 0 and scene['budget']['total_native_seconds'] <= 84600,
            zero_learning=all(scene['budget'][name] == 0 for name in ZERO),
            never_used_labels=scene.get('labels_used_for_selection') is False and scene.get('predictions_used_for_selection') is False,
            never_fitted=scene.get('training_started') is False and scene.get('models_fitted') is False)
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
    first_collector = next((value for value in collectors if value['external_seed'] == SEEDS[0]
                            and value['scenario'] == 'stable'), None)
    first_pair = min(first_collector['pairs'],key=lambda value:(value['root']['started_at'],value['root']['sequence'])) \
        if first_collector and first_collector['pairs'] else None
    checks = dict(four_declared_collectors=set(keys) == expected,
        frozen_protocol=protocol['source_initializer'] == 6207 and protocol['external_seeds'] == list(SEEDS)
            and protocol['scenarios'] == list(SCENES) and protocol['budget']['total_cap'] == 338400,
        original_model_support=protocol['frozen_common_support'] == model['common_support'],
        source_audit=audit['all_pass'] and all(audit['checks'].values()) and audit['source_initializer'] == 6207,
        no_source_computation=all(audit[key] == 0 for key in ('new_fits','new_neural_updates','new_suffixes','new_labels','v87_truth_used','native_seconds')),
        all_collectors_pass=bool(scene_checks) and all(all(check.values()) for check in scene_checks.values()),
        twelve_distinct_roots=len(rows) == 12 and len({tuple(row['physical_root_id']) for row in rows}) == 12,
        twelve_valid_mature_pairs=len(good) == 12 and not missing,
        first_pair_reused_once=gate is not None and gate_matches == 1 and gate['all_pass']
            and gate.get('new_pair_in_formal_cohort') is True
            and gate.get('labels_used_for_gate') is False and gate.get('cost_sign_used_for_gate') is False,
        exact_first_physical_gate=gate is not None and first_pair is not None
            and gate.get('external_seed') == SEEDS[0] and gate.get('scenario') == 'stable'
            and gate['root'] == first_pair['root'] and gate['checks'] == first_pair['checks']
            and all(gate['checks'].values()),
        all_collector_gates_same=bool(gates) and all(value == gate for value in gates),
        future_bias_design_rank2=rank == 2,
        all_labels_observed=total['labels_observed'] == 24,
        no_unknown_budget=total['unknown_suffix_costs'] == 0 and not total['total_native_seconds_is_lower_bound'],
        budget_identity=total['total_native_seconds'] == total['native_prefix_seconds']+total['suffix_native_seconds'],
        within_budget=not total['total_native_seconds_is_lower_bound'] and total['total_native_seconds'] <= 338400,
        zero_training_fitting_or_feedback=all(total[key] == 0 for key in ZERO))
    passed = all(checks.values())
    unsupported = [dict(external_seed=value['external_seed'],scenario=value['scenario'],line_id=value['line_id'],
        root=value['slot']['root'],status=value['slot'].get('status'),reason=value['slot'].get('reason'),
        condition=value['slot']['condition']) for value in slots if value['slot'] and value['slot'].get('status') == 'unsupported']
    return dict(status='complete' if passed else 'hold', integrity=dict(all_pass=passed,checks=checks),
        scene_checks=scene_checks, pairs=rows, budget=total,
        future_bias_design=dict(rows=len(good),rank=rank,columns=['1','fixed_scaled_historical_q'],target_solves=0),
        signals=dict(overall=signals(rows),
            by_seed={str(seed):signals([row for row in rows if row['external_seed'] == seed]) for seed in SEEDS},
            by_seed_scenario={str(seed)+'/'+scene:signals([row for row in rows if row['external_seed'] == seed
                and row['scenario'] == scene]) for seed in SEEDS for scene in SCENES},
            by_scenario={scene:signals([row for row in rows if row['scenario'] == scene]) for scene in SCENES},
            by_line={line:signals([row for row in rows if row['line_id'] == line]) for line in LINES}),
        q_coverage={line:dict(roots=sum(row['line_id'] == line for row in rows),
            valid=sum(row['line_id'] == line for row in good),
            values=sorted(set(row['condition']['value'] for row in good if row['line_id'] == line))) for line in LINES},
        exceptions=dict(missing=missing,unsupported=unsupported,
            native_failures=[dict(external_seed=scene['external_seed'],scenario=scene['scenario'],
                reason=scene.get('reason'),failure=scene.get('failure'),budget=scene['budget'])
                for scene in collectors if scene['status'] == 'failed'],
            invalid_pairs=[dict(external_seed=row['external_seed'],scenario=row['scenario'],line_id=row['line_id'],checks=row['checks']) for row in rows if not row['valid']],
            nonseparable=[row['physical_root_id'] for row in rows if not row['checks']['physical_separation']],
            zero_duration=[row['physical_root_id'] for row in rows if row['dt'] == 0],
            truncated=[row['physical_root_id'] for row in rows if not row['checks']['true_terminal']]),
        collection_gate=dict(passed=passed,decision='Collection ready for separate analysis/fit decision; no automatic fit or calibration'
            if passed else 'HOLD: preserve unsupported/missing/invalid slots without replacement'),
        interpretation=dict(model_fits=0,optimizer_updates=0,policy_cost_evaluated=False,
            v87_truth_used=0,reserved816207_used=0,no_automatic_calibration_or_validation=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collectors',type=Path,nargs='+',required=True)
    for name in ('protocol','audit','model','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    report = analyze([json.loads(path.read_text()) for path in args.collectors],
        json.loads(args.protocol.read_text()),json.loads(args.audit.read_text()),json.loads(args.model.read_text()))
    raw = json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode()) > 64000:
        raise ValueError('Keep the collection summary below64KB; raw branches remain resident')
    with args.out.open('x') as handle: handle.write(raw)
    print(json.dumps(dict(status=report['status'],bytes=len(raw.encode()),fits=0)))


if __name__ == '__main__':
    main()
