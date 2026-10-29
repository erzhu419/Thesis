"""Read-only, prequential diagnosis of the v21/v23 service auxiliary task.

Uses only existing local small summaries. No fitting candidate, simulation,
parameter selection, old-heldout selection, or retrospective gate revision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LINES = ('7X', '102X', '705X')
MODELS = ('fixed_7p5', 'past_line_mean', 'shared', 'history')


def sign(value):
    return (value > 0)-(value < 0)


def history_bin(value):
    return 'below_minus_hold' if value < -7.5 else 'above_hold' if value > 7.5 else 'within_hold'


def prequential(rows):
    """Use each issue's saved admitted IDs, never issue order as label availability."""
    indexed = {row['id']: row for row in rows}
    result = []
    for row in rows:
        past = [indexed[key] for key in row['admitted_ids']]
        if any(old['issue_at'] >= row['issue_at'] or not old['admitted'] for old in past):
            raise ValueError('Admission prefix contains current/future/unadmitted state')
        own = [old for old in past if old['line_id'] == row['line_id']]
        means = []
        for endpoint in range(6):
            available = [old['truth'][endpoint] for old in own if old['truth'][endpoint] is not None]
            means.append(float(np.mean(available)) if available else None)
        predictions = dict(fixed_7p5=[7.5]*6, past_line_mean=means, **row['prediction'])
        valid = [i for i, truth in enumerate(row['truth']) if truth is not None and all(
            predictions[model][i] is not None for model in MODELS)]
        negatives = [i for i in valid if row['truth'][i] < 0]
        known_negative = [i for i in range(6) if any(
            old['truth'][i] is not None and old['truth'][i] < 0 for old in own)]
        result.append(dict(id=row['id'], line=row['line_id'], issue_at=row['issue_at'],
            admitted_own_line_ids=[old['id'] for old in own], scored=valid,
            truth=row['truth'], predictions=predictions,
            first_negative_endpoints=[i for i in negatives if i not in known_negative],
            repeated_negative_endpoints=[i for i in negatives if i in known_negative],
            after_negative_nonnegative_endpoints=[i for i in valid if i in known_negative
                and row['truth'][i] >= 0]))
    return result


def metrics(rows):
    result = {}
    for model in MODELS:
        state_errors, correct, negative, hits, endpoints = [], 0, 0, 0, 0
        for row in rows:
            valid = row['scored']
            if valid:
                state_errors.append(float(np.mean([abs(row['predictions'][model][i]-row['truth'][i]) for i in valid])))
            for i in valid:
                truth, estimate = row['truth'][i], row['predictions'][model][i]
                endpoints += 1
                correct += sign(truth) == sign(estimate)
                negative += truth < 0
                hits += truth < 0 and estimate < 0
        result[model] = dict(states=len(state_errors), endpoints=endpoints,
            mae_seconds=float(np.mean(state_errors)) if state_errors else None,
            sign_correct=correct, negative_truth_endpoints=negative,
            negative_hits=hits if negative else None)
    return result


def support_diagnosis(gate):
    selected = {row['id']: row for row in gate['sampling']['selected']}
    rows = []
    for row in gate['rows']:
        source = selected[row['id']]['stratum']
        mean_bin = history_bin(row['history_mean_seconds'])
        rows.append(dict(id=row['id'], line=row['line_id'],
            headway_sign=source['headway_sign'], history_mean_seconds=row['history_mean_seconds'],
            model_mean_bin=mean_bin, original_third_bin=source['third_residual_bin'],
            bins_agree=mean_bin == source['third_residual_bin']))
    by_line = {}
    for line in LINES:
        own = [row for row in rows if row['line'] == line]
        by_line[line] = dict(selected_states=len(own),
            original_joint_groups=len({(row['headway_sign'], row['original_third_bin']) for row in own}),
            model_mean_joint_groups=len({(row['headway_sign'], row['model_mean_bin']) for row in own}),
            mean_seconds_range=[min(row['history_mean_seconds'] for row in own),
                max(row['history_mean_seconds'] for row in own)],
            headway_signs=sorted({row['headway_sign'] for row in own}))
    fallback = [{key: row[key] for key in ('id', 'issued_at', 'selection_reason')}
        | dict(eligible_examined=gate['sampling']['eligible_issues'][row['id']],
               candidates_after_selection='not_observed')
        for row in gate['sampling']['selected'] if row['fallback']]
    return dict(rows=rows, by_line=by_line, bins_agree=sum(row['bins_agree'] for row in rows),
        selected_states=len(rows), fallback=fallback,
        limit='Selected states only. The same +/-7.5-second bins applied to the actual three-segment mean are descriptive, not a replacement gate or an identifiability test. Numeric headway covariates and full-window candidate states are absent from this small local summary; rank/support beyond the saved headway signs is unknown.')


def main(root=ROOT):
    path = root/'experiments/mcwm_closed_loop_stratified_pairs_v23_20261023/training_gate.json'
    gate = json.loads(path.read_text())
    v21 = json.loads((root/'experiments/mcwm_closed_loop_v21_review_20261023/history_design_diagnostic.json').read_text())
    rows = prequential(gate['rows'])
    all_metrics = metrics(rows)
    # This comparison detects accidental admission-order or endpoint changes.
    for model in ('shared', 'history'):
        saved, reproduced = gate['macro'][model], all_metrics[model]
        if abs(saved['mae_seconds']-reproduced['mae_seconds']) > 1.e-12:
            raise ValueError('Original saved service MAE did not reproduce')
        if saved['sign_correct'] != reproduced['sign_correct']:
            raise ValueError('Original saved direction count did not reproduce')
    changes = [abs(row['prediction']['history'][i]-row['prediction']['shared'][i])
        for row in gate['rows'] for i in range(6) if row['prediction']['shared'][i] is not None]
    first = [row['id'] for row in rows if row['first_negative_endpoints']]
    repeated = [row['id'] for row in rows if row['repeated_negative_endpoints']]
    nonnegative = [row['id'] for row in rows if row['after_negative_nonnegative_endpoints']]
    return dict(status='complete', source='Existing local v21 design diagnostic and v23 training gate only',
        writes_to_old_sources=0, new_native_seconds=0, new_training_labels=0, model_candidates_selected=0,
        estimator=dict(
            prediction='7.5 + xT (XTX + I)^-1 XT (delta_t - 7.5)',
            target_scale='Dividing all targets by60 and multiplying predictions by60 cancels exactly for this fixed linear ridge; this is not an input rescaling.',
            raw_history_penalty='For z=history_seconds/60 and beta=z_weight/60, ridge1 contributes3600*beta^2 in raw seconds coordinates; original four coefficient penalties stay unchanged.',
            intercept_prior='Each penalized line intercept is shrunk toward zero residual, hence toward the +7.5-second service difference prior.',
            single_line_intercept_example=dict(observed_seconds=0., samples=1, ridge=1., forecast_seconds=3.75,
                actual_shared_response_arrival3=gate['rows'][1]['prediction']['shared'][2]),
            v21_history_schur_energy=v21['design']['history_schur_energy'],
            v21_history_schur_fraction=v21['design']['schur_fraction'],
            v21_max_abs_endpoint_change_seconds=max(row['total_abs_max'] for row in v21['heldout_corrections']),
            v23_max_abs_endpoint_change_seconds=max(changes),
            v23_history_extra_mae_improvement_seconds=all_metrics['shared']['mae_seconds']-all_metrics['history']['mae_seconds'],
            conclusion='Small input energy and penalized intercepts explain strong shrinkage. No lambda sweep or new fit is performed; this does not establish that removing regularization improves future control.'),
        common_support_service_metrics=all_metrics,
        by_line={line:metrics([row for row in rows if row['line'] == line]) for line in LINES},
        prequential_rows=[dict(id=row['id'], issue_at=row['issue_at'],
            admitted_own_line_ids=row['admitted_own_line_ids'],
            truth_third_arrival=row['truth'][2], prior_mean_third_arrival=row['predictions']['past_line_mean'][2],
            shared_third_arrival=row['predictions']['shared'][2],
            scored_endpoints=len(row['scored'])) for row in rows],
        reversals=dict(first_negative_states=first, repeated_negative_after_own_line_negative_states=repeated,
            nonnegative_after_own_line_negative_states=nonnegative,
            first_negative_endpoint_count=sum(len(row['first_negative_endpoints']) for row in rows),
            repeated_negative_endpoint_count=sum(len(row['repeated_negative_endpoints']) for row in rows),
            unit='Two negative arrival/departure endpoints belong to one physical state. Only own-line, same-endpoint admitted negative labels count as prior negative supervision.'),
        sampling_support=support_diagnosis(gate),
        execution_condition=dict(
            floor_clock_seconds=[3600.,18000.], clock_interval='left_closed_right_open',
            additional_floor_conditions='forward bus present, backward bus absent, demand_ratio>0, load_ratio<=0.25',
            reserve_conditions='clock<18000 and positive proposal holding; actual increment depends on canonical forward/backward headway errors',
            selected_issue_floor='Not active at all nine selected issue states: selector required both0/7.5s to be reachable under F, whereas an active floor makes minimum holding60s.',
            selected_clock_windows={row['id']: 'pre_floor' if row['issue_at'] < 3600 else
                'reserve_floor' if row['issue_at'] < 18000 else 'tail' for row in gate['rows']},
            actual_suffix_floor_or_reserve_contribution='unknown from current small summaries',
            boundary_scope='7X bootstrap1559 and response3815 have different future-F clock regimes; both300-second suffixes stay within their own regime. Existing v19 late-to-tail interventions do not identify the3600 boundary effect.',
            decision='Keep actual actor/environment clocks and F unchanged. Future auxiliary comparisons must declare future-F regime and log condition versus actual increment separately.'),
        limits=['This is development diagnosis of training forecasts, not independent confirmation or a control-improvement result.',
            'No post-selection state census, numeric v23 headway design matrix, or actual future-F event accounting is present locally.',
            'Old v21/v23 gates and saved predictions are unchanged; future method choice must not use old heldout truth.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('diagnosis.json'))
    args = parser.parse_args()
    value = main()
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with args.out.open('x') as stream:
        stream.write(payload)
    print(json.dumps(dict(status=value['status'], bytes=len(payload.encode()), out=str(args.out))))
