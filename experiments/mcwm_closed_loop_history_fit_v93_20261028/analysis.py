"""Read the three fixed ridge fits; all errors and regrets are training evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

MODELS = ('line3', 'service5', 'service_history6')
ABLATION = 'history6_without_q'
READOUTS = (*MODELS, ABLATION)
LINES = ('7X', '102X', '705X')
SCENARIOS = ('stable', 'local_speed_drop')
BINS = ('lighter', 'heavier')
DIMENSIONS = dict(line3=3, service5=5, service_history6=6)
TOLERANCE = 1e-12


def _direction(value):
    return 'candidate_better' if value < 0 else 'reference_better' if value > 0 else 'tie'


def _score(prediction, target, parts):
    selected = target if prediction < 0 else 0.
    regret = selected-min(0., target)
    return dict(prediction=float(prediction), choice='candidate' if prediction < 0 else 'reference',
        predicted_direction=_direction(prediction), direction_correct=_direction(prediction) == _direction(target),
        absolute_error=abs(prediction-target), squared_error=(prediction-target)**2,
        selected_delta_per_passenger=selected, two_action_regret_per_passenger=regret,
        correct_action=regret == 0, contributions=parts)


def _metrics(rows, name):
    values = [row['models'][name] for row in rows]
    n = len(values)
    return dict(training_roots=n,
        mae=sum(row['absolute_error'] for row in values)/n if n else None,
        rmse=math.sqrt(sum(row['squared_error'] for row in values)/n) if n else None,
        direction_correct=sum(row['direction_correct'] for row in values),
        direction_accuracy=sum(row['direction_correct'] for row in values)/n if n else None,
        correct_actions=sum(row['correct_action'] for row in values),
        candidate_choices=sum(row['choice'] == 'candidate' for row in values),
        wrong_candidate_choices=sum(value['choice'] == 'candidate' and source['target'] > 0
            for value, source in zip(values, rows)),
        missed_benefit=sum(value['choice'] == 'reference' and source['target'] < 0
            for value, source in zip(values, rows)),
        mean_two_action_regret_per_passenger=sum(row['two_action_regret_per_passenger'] for row in values)/n if n else None,
        worst_two_action_regret_per_passenger=max((row['two_action_regret_per_passenger'] for row in values), default=None))


def _group(rows):
    return {name: _metrics(rows, name) for name in READOUTS}


def _contribution_summary(rows):
    keys = ('line_delta', 'load_refit_delta', 'margin_refit_delta', 'q_direct', 'total_6_minus_5')
    n = len(rows)
    return {key: dict(mean=sum(row['six_minus_five'][key] for row in rows)/n if n else None,
        mean_absolute=sum(abs(row['six_minus_five'][key]) for row in rows)/n if n else None,
        min=min((row['six_minus_five'][key] for row in rows), default=None),
        max=max((row['six_minus_five'][key] for row in rows), default=None)) for key in keys}


def analyze(artifact):
    training = artifact['training_rows']
    models, matrices = artifact['models'], artifact['design']['matrices']
    centers = artifact['design']['line_centers']
    expected_rows = []
    for row in training:
        a = row['a']
        centered = np.asarray(row['features'])-np.asarray(centers[row['line_id']])
        expected_rows.append(np.r_[a*np.asarray([float(row['line_id'] == line) for line in LINES]), a*centered])
    expected = np.asarray(expected_rows)
    weighted_centers = {}
    for line in LINES:
        chosen = [row for row in training if row['line_id'] == line]
        denominator = sum(row['a']**2 for row in chosen)
        weighted_centers[line] = [sum(row['a']**2*row['features'][j] for row in chosen)/denominator
            for j in range(3)] if chosen and denominator else None
    model_checks, line_differences = {}, {}
    line_coef = np.asarray(models['line3']['coef'])
    for index, name in enumerate(MODELS, 1):
        model = models[name]
        matrix, coef, prediction = (np.asarray(value) for value in (matrices[name], model['coef'], model['prediction']))
        width = DIMENSIONS[name]
        model_checks[name] = dict(fixed_solve_index=model['solve_index'] == index,
            fixed_width=matrix.shape == (12, width) and coef.shape == (width,) and prediction.shape == (12,),
            full_design_rank=model['design_rank'] == np.linalg.matrix_rank(matrix) == width,
            finite_fit=np.isfinite(coef).all() and np.isfinite(prediction).all(),
            frozen_feature_wiring=np.allclose(matrix, expected[:, :width], rtol=0., atol=TOLERANCE),
            saved_prediction_exact=np.allclose(prediction, matrix@coef, rtol=0., atol=TOLERANCE))
        model_checks[name] = {key: bool(value) for key, value in model_checks[name].items()}
        line_differences[name] = float(np.max(abs(coef[:3]-line_coef)))
    budget = artifact['budget']
    cases = {(row['scenario'], row['line_id'], row['load_bin']) for row in training}
    expected_cases = {(scene, line, group) for scene in SCENARIOS for line in LINES for group in BINS}
    checks = dict(source_audit=artifact['source_audit']['all_pass'] and all(artifact['source_audit']['checks'].values()),
        fit_integrity=artifact['integrity']['all_pass'] and all(artifact['integrity']['checks'].values()),
        three_frozen_models=set(models) == set(MODELS),
        twelve_training_rows=len(training) == 12 and cases == expected_cases,
        twelve_distinct_ids=len({row['id'] for row in training}) == 12,
        twelve_distinct_physical_roots=len({(row['scenario'], row['external_seed'], row['root']['sequence'])
            for row in training}) == 12,
        no_heldout_training_data=all(row['external_seed'] == 716207 for row in training),
        finite_positive_action_anchor=all(math.isfinite(row['a']) and row['a'] > 0 for row in training),
        target_units=all(row['target'] == row['raw_delta_person_seconds']/15079 for row in training),
        fixed_ridge_and_units=artifact['ridge'] == 1. and artifact['denominator'] == 15079 and artifact['hold_scale'] == 7.5,
        same_weighted_line_centers=all(weighted_centers[line] is not None
            and np.allclose(centers[line], weighted_centers[line], rtol=0., atol=TOLERANCE) for line in LINES),
        all_model_numerics=all(all(value.values()) for value in model_checks.values()),
        shared_line_coefficients=max(line_differences.values()) <= TOLERANCE,
        exactly_three_solves=budget['model_solves'] == 3,
        fixed_training_information=budget['training_pairs'] == 12 and budget['training_physical_labels'] == 24,
        no_new_native_or_labels=budget['new_native_seconds'] == budget['new_labels'] == 0,
        no_initializer_or_optimizer_updates=budget['initializer_updates'] == budget['optimizer_updates'] == 0,
        v87_truth_sealed=budget['v87_truth_used'] == 0,
        prior_native_budget_separate=budget['historical_native_seconds'] == 349310)
    rows = []
    decomposition_errors = []
    for index, source in enumerate(training):
        line = LINES.index(source['line_id'])
        centered = np.asarray(source['features'])-np.asarray(centers[source['line_id']])
        a, target = source['a'], source['target']
        readouts = {}
        for name in MODELS:
            model = models[name]
            coef = model['coef']
            parts = dict(line=a*coef[line], load=a*coef[3]*centered[0] if len(coef) >= 5 else 0.,
                margin=a*coef[4]*centered[1] if len(coef) >= 5 else 0.,
                q=a*coef[5]*centered[2] if len(coef) == 6 else 0.)
            parts['sum'] = sum(parts.values())
            prediction = float(model['prediction'][index])
            decomposition_errors.append(abs(prediction-parts['sum']))
            readouts[name] = _score(prediction, target, parts)
        full, service = (readouts[name]['contributions'] for name in ('service_history6', 'service5'))
        ablated = dict(full, q=0.)
        ablated['sum'] = full['sum']-full['q']
        readouts[ABLATION] = _score(readouts['service_history6']['prediction']-full['q'], target, ablated)
        increment = dict(line_delta=full['line']-service['line'], load_refit_delta=full['load']-service['load'],
            margin_refit_delta=full['margin']-service['margin'], q_direct=full['q'])
        increment['total_6_minus_5'] = readouts['service_history6']['prediction']-readouts['service5']['prediction']
        decomposition_errors.append(abs(sum(increment[key] for key in ('line_delta', 'load_refit_delta', 'margin_refit_delta', 'q_direct'))
            -increment['total_6_minus_5']))
        rows.append(dict(id=source['id'], scenario=source['scenario'], line_id=source['line_id'],
            load_bin=source['load_bin'], root=source['root'], features=source['features'], a=a, target=target,
            truth_direction=_direction(target), models=readouts, six_minus_five=increment))
    checks['contribution_identities'] = bool(max(decomposition_errors, default=0.) <= TOLERANCE)
    aliases = []
    for line, group in (('7X', 'lighter'), ('705X', 'heavier')):
        pair = [next(row for row in rows if row['scenario'] == scene and row['line_id'] == line
            and row['load_bin'] == group) for scene in SCENARIOS]
        aliases.append(dict(line_id=line, load_bin=group,
            equal_local_service_features=pair[0]['features'][:2] == pair[1]['features'][:2],
            history_values=[row['features'][2] for row in pair], targets=[row['target'] for row in pair],
            models={name: dict(predictions=[row['models'][name]['prediction'] for row in pair],
                choices=[row['models'][name]['choice'] for row in pair],
                predicted_signs_differ=_direction(pair[0]['models'][name]['prediction'])
                    != _direction(pair[1]['models'][name]['prediction']),
                both_training_directions_correct=all(row['models'][name]['direction_correct'] for row in pair))
                for name in READOUTS}))
    overall = _group(rows)
    full, ablated = overall['service_history6'], overall[ABLATION]
    passed = all(checks.values())
    return dict(status='complete' if passed else 'hold',
        numerical_gate=dict(passed=passed, checks=checks), model_numerical_checks=model_checks,
        line_coefficient_max_differences=line_differences,
        normal_equation_residual_max={name: models[name].get('normal_equation_residual_max') for name in MODELS},
        training=dict(overall=overall,
            by_scenario={scene: _group([row for row in rows if row['scenario'] == scene]) for scene in SCENARIOS},
            by_line={line: _group([row for row in rows if row['line_id'] == line]) for line in LINES},
            by_load_bin={group: _group([row for row in rows if row['load_bin'] == group]) for group in BINS},
            by_line_load_bin={line+'/'+group: _group([row for row in rows if row['line_id'] == line
                and row['load_bin'] == group]) for line in LINES for group in BINS}, rows=rows,
            exact_local_aliases=aliases),
        increment_6_minus_5=dict(overall=_contribution_summary(rows),
            by_scenario={scene: _contribution_summary([row for row in rows if row['scenario'] == scene]) for scene in SCENARIOS},
            interpretation='Net6-minus5 includes the re-estimated load/margin terms. q_direct alone is the fixed-weight deletion effect.'),
        fixed_weight_q_deletion=dict(new_solves=0,
            changed_training_choices=sum(row['models']['service_history6']['choice'] != row['models'][ABLATION]['choice'] for row in rows),
            mean_training_regret_full_minus_deleted=full['mean_two_action_regret_per_passenger']
                -ablated['mean_two_action_regret_per_passenger']),
        holdout_assessment=dict(numerical_contract_passed=passed,
            history_changes_choices_vs_line=sum(row['models']['service_history6']['choice'] != row['models']['line3']['choice'] for row in rows),
            history_changes_choices_vs_service=sum(row['models']['service_history6']['choice'] != row['models']['service5']['choice'] for row in rows),
            mean_training_regret_history_minus_line=full['mean_two_action_regret_per_passenger']
                -overall['line3']['mean_two_action_regret_per_passenger'],
            mean_training_regret_history_minus_service=full['mean_two_action_regret_per_passenger']
                -overall['service5']['mean_two_action_regret_per_passenger'],
            decision='Review frozen training contrasts before separately authorizing the reserved816207 holdout'
                if passed else 'HOLD numerical contract; no holdout or refit automatically'),
        budget=budget,
        interpretation=dict(scope='Training-only fit errors, same-root two-action regrets, and coefficient attribution',
            actual_control_performance_evaluated=False, independent_holdout_evaluated=False,
            fixed_weight_deletion_is_not_an_additional_fit=True,
            no_training_screen_threshold_added=True, no_automatic_native_or_refitting=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.model.read_text()))
    raw = json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > 64000:
        raise ValueError('Training readout exceeds64KB; keep the model artifact resident')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=report['status'], bytes=len(raw.encode()), numerical_gate=report['numerical_gate']['passed'])))


if __name__ == '__main__':
    main()
