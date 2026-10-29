"""Freeze one C margin on four complete blocks, with the predeclared v94 comparator."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.analysis import LINES, SCENES, SEEDS
from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import (
    calibrated_margin, predict, seed_score, select_first, upper_bound)
from experiments.mcwm_closed_loop_calibration_design_v94_20261028.calibration import (
    seed_score as independent_seed_score, select_first as independent_select_first)


def finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


def extract(summary):
    rows = []
    for pair in summary['pairs']:
        label = pair['label_delta_per_passenger'] or {}
        rows.append(dict(external_seed=pair['external_seed'], scenario=pair['scenario'],
            line_id=pair['line_id'], physical_root_id=pair['physical_root_id'],
            issue_clock=pair['root']['started_at'], sequence=pair['root']['sequence'],
            active=pair['valid'], q=pair['condition'].get('value'),
            m=pair['predictions'].get('service_history6'), predictions=deepcopy(pair['predictions']),
            delta=label.get('cost'), wait_delta=label.get('wait'), ride_delta=label.get('ride'),
            physical_noop=pair['physical_noop']))
    return rows


def validate(summary, protocol, fit_model, fit_summary):
    rows = extract(summary)
    expected = {(seed, scene, line) for seed in SEEDS for scene in SCENES for line in LINES}
    keys = [(row['external_seed'], row['scenario'], row['line_id']) for row in rows]
    bias = fit_model['bias_model']
    binding = dict(source_fit=protocol['source_fit'], primary_model=fit_model['primary_model'],
        bias_model=bias, source_summary=fit_model['source_summary'], source_protocol=fit_model['source_protocol'])
    checks = dict(
        collection_complete=summary['status'] == 'complete' and summary['collection_gate']['passed'],
        collection_integrity=summary['integrity']['all_pass'] and bool(summary['integrity']['checks'])
            and all(summary['integrity']['checks'].values()),
        four_complete_declared_blocks=len(rows) == 24 and len(set(keys)) == 24 and set(keys) == expected,
        distinct_roots=len({tuple(row['physical_root_id']) for row in rows}) == 24,
        all_pairs_valid=bool(rows) and all(pair['valid'] and bool(pair['checks'])
            and all(value for key, value in pair['checks'].items() if key != 'physical_separation')
            for pair in summary['pairs']),
        forty_eight_labels=summary['budget']['labels_observed'] == 48,
        all_required_truth_finite=bool(rows) and all(finite(row[key]) for row in rows
            for key in ('q', 'm', 'delta', 'wait_delta', 'ride_delta', 'issue_clock')),
        no_C_fit=summary['interpretation']['model_fits'] == 0 and summary['budget']['new_fits'] == 0,
        fixed_fit_ready=fit_summary['status'] == 'complete'
            and fit_summary['decision']['ready_for_calibration'] is True
            and fit_summary['validation']['all_pass'] and fit_summary['numerical']['all_pass'],
        fixed_fit_integrity=fit_model['integrity']['all_pass'] and bool(fit_model['integrity']['checks'])
            and all(fit_model['integrity']['checks'].values()),
        fixed_bias_binding=summary.get('frozen_bias') == binding,
        fixed_bias_metadata=bias['n'] == 12 and bias['rank'] == 2 and bias['ridge'] == 1.
            and bias['linear_solves'] == 1 and bias['q_range'] == protocol['q_range_for_scaling_only'],
        finite_fixed_parameters=len(bias['coef']) == 2 and len(bias['gram_inverse']) == 2
            and all(finite(value) for value in bias['coef'])
            and all(len(row) == 2 and all(finite(value) for value in row) for row in bias['gram_inverse']),
        unchanged_deployment_domain=fit_model['candidate_deployment_domain'] == protocol['candidate_deployment_domain'],
        fixed_calibration_rule=fit_model['primary_model'] == 'service_history6'
            and protocol['external_seeds'] == list(SEEDS) and protocol['scenarios'] == list(SCENES)
            and protocol['calibration']['blocks'] == 4 and protocol['calibration']['alpha'] == .2)
    return dict(all_pass=all(checks.values()), checks=checks)


def calibrate(summary, protocol, fit_model, fit_summary):
    validation = validate(summary, protocol, fit_model, fit_summary)
    report = dict(status='hold', integrity=validation, ready_for_validation=False,
        alpha=protocol['calibration']['alpha'], C=None, margin_status='not_calibrated',
        block_scores=[dict(external_seed=seed, score=None, status='not_calibrated') for seed in SEEDS],
        source_identity=dict(source_fit=protocol['source_fit'], source_fit_summary=protocol['source_fit_summary'],
            source_model=protocol['source_model'], primary_model='service_history6'),
        frozen_model=deepcopy(fit_model),
        state_independent=dict(comparison_only=True, selected_for_deployment=False, C=None,
            block_scores=[dict(external_seed=seed, score=None, status='not_calibrated') for seed in SEEDS]),
        observed_C_feasibility=None,
        decision=dict(reason='C_INCOMPLETE_OR_INVALID_OR_FIT_NOT_READY', V_kept_sealed=True),
        budget=dict(candidate_calibrations=0, comparator_calibrations=0, bias_refits=0,
            native_seconds=0, new_labels=0, optimizer_updates=0),
        interpretation=dict(independent_policy_benefit=False, no_model_selection=True,
            tail_used_for_calibration=False, no_automatic_validation=True,
            feasibility_scope='Observed C roots only; rejection does not prove whole-domain uselessness',
            coverage_scope='Exchangeability-dependent80% marginal seed-block coverage, not conditional no-harm'))
    if not validation['all_pass']:
        return report
    rows, bias = extract(summary), fit_model['bias_model']
    scores, comparison_scores = [], []
    for seed in SEEDS:
        block = [row for row in rows if row['external_seed'] == seed]
        scores.append(seed_score(block, bias))
        comparison_scores.append(independent_seed_score(block, 'service_history6'))
    margin = calibrated_margin(scores, report['alpha'])
    comparator = calibrated_margin(comparison_scores, report['alpha'])
    for entries, values in ((report['block_scores'], scores),
                            (report['state_independent']['block_scores'], comparison_scores)):
        for entry, value in zip(entries, values):
            entry.update(score=value if isfinite(value) else None,
                         status='finite' if isfinite(value) else 'nonfinite')
    report['budget'].update(candidate_calibrations=1, comparator_calibrations=1)
    report.update(C=margin if isfinite(margin) else None,
                  margin_status='finite' if isfinite(margin) else 'nonfinite')
    report['state_independent'].update(C=comparator if isfinite(comparator) else None,
        margin_status='finite' if isfinite(comparator) else 'nonfinite')
    if not isfinite(margin):
        report['decision']['reason'] = 'STOP_NONFINITE_FIXED_MARGIN'
        return report
    report['status'] = 'complete'
    for row in rows:
        forward = predict(bias, row['q'], row['m'])
        bound = upper_bound(row['m'], row['q'], bias, margin)
        row.update(b=forward['bias'], u=forward['support_scale'], U=bound,
                   accepted=row['m'] < 0. and bound < 0.)
    selected, comparison_selected = {}, {}
    for seed in SEEDS:
        for scene in SCENES:
            group = [row for row in rows if row['external_seed'] == seed and row['scenario'] == scene]
            chosen = select_first(group, bias, margin)
            other = independent_select_first(group, 'service_history6', comparator)
            key = str(seed)+'/'+scene
            selected[key] = None if chosen is None else chosen['physical_root_id']
            comparison_selected[key] = None if other is None else other['physical_root_id']
    accepted = sum(row['accepted'] for row in rows)
    report['root_readout'] = rows
    report['observed_C_feasibility'] = dict(accepted_roots=accepted,
        selected_scenes=sum(value is not None for value in selected.values()), selected_roots=selected)
    report['state_independent']['selected_roots'] = comparison_selected
    report['ready_for_validation'] = accepted > 0
    report['decision'] = dict(reason='READY_FOR_RESERVED_V_REVIEW' if accepted else 'STOP_ALL_C_ROOTS_REJECTED',
        V_kept_sealed=True, next_cohort=protocol['calibration']['V'] if accepted else None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=HERE/'summary.json')
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--fit-model', type=Path, default=HERE/'source_fit.json')
    parser.add_argument('--fit-summary', type=Path, default=HERE/'source_fit_summary.json')
    parser.add_argument('--out', type=Path, default=HERE/'calibration.json')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('The fixed calibration already exists; do not retry or overwrite it')
    load = lambda path: json.loads(path.read_text())
    protocol = load(args.protocol)
    result = calibrate(load(args.summary), protocol, load(args.fit_model), load(args.fit_summary))
    raw = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'
    if len(raw.encode()) > protocol['retrieval']['final_summary_bytes']:
        raise ValueError('Calibration exceeds the declared small-summary byte limit')
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result['status'], ready_for_validation=result['ready_for_validation'],
                          C=result['C'], decision=result['decision'], bytes=len(raw.encode()))))
    return result


if __name__ == '__main__':
    main()
