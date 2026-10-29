"""One fixed T fit and descriptive training readout; no native queries or calibration."""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import (
    fit_bias, predict, select_first)

LINES = ('7X', '102X', '705X')
SCENES = ('stable', 'local_speed_drop')


def finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


def extract(summary, protocol):
    """Keep every declared pair, including invalid rows for the integrity decision."""
    rows = []
    for pair in summary['pairs']:
        label = pair['label_delta_per_passenger'] or {}
        rows.append(dict(external_seed=pair['external_seed'], scenario=pair['scenario'],
            line_id=pair['line_id'], physical_root_id=pair['physical_root_id'],
            issue_clock=pair['root']['started_at'], sequence=pair['root']['sequence'],
            q=pair['condition'].get('value'), m=pair['predictions'].get(protocol['primary_model']),
            D=label.get('cost'), wait_delta=label.get('wait'), ride_delta=label.get('ride'),
            physical_noop=pair['physical_noop'], active=pair['valid']))
    return rows


def design(rows, q_range):
    lo, hi = q_range
    return np.asarray([[1., 2.*(row['q']-lo)/(hi-lo)-1.] for row in rows],
                      dtype=np.float64).reshape(-1, 2)


def validate(summary, protocol):
    rows = extract(summary, protocol)
    expected = {(seed, scene, line) for seed in protocol['T'] for scene in SCENES for line in LINES}
    keys = [(row['external_seed'], row['scenario'], row['line_id']) for row in rows]
    values_finite = bool(rows) and all(finite(row[key]) for row in rows
        for key in ('q', 'm', 'D', 'wait_delta', 'ride_delta', 'issue_clock'))
    lo, hi = protocol['q_range_for_scaling_only']
    scale_valid = finite(lo) and finite(hi) and hi > lo
    rank = int(np.linalg.matrix_rank(design(rows, [lo, hi]))) if values_finite and scale_valid else None
    source_checks = summary['integrity']['checks']
    checks = dict(
        source_complete=summary['status'] == 'complete' and summary['collection_gate']['passed'],
        source_integrity=summary['integrity']['all_pass'] and bool(source_checks) and all(source_checks.values()),
        twelve_declared_pairs=len(rows) == 12 and len(set(keys)) == 12 and set(keys) == expected,
        twelve_distinct_roots=len({tuple(row['physical_root_id']) for row in rows}) == 12,
        valid_mature_pairs=bool(rows) and all(pair['valid'] and bool(pair['checks'])
            and all(value for key, value in pair['checks'].items() if key != 'physical_separation')
            for pair in summary['pairs']),
        twenty_four_physical_labels=summary['budget']['labels_observed'] == 24,
        no_prior_fit=summary['future_bias_design']['target_solves'] == 0
            and summary['interpretation']['model_fits'] == 0,
        finite_rows=values_finite,
        fixed_scale=scale_valid,
        rank_two=rank == 2,
        fixed_fit=protocol['primary_model'] == 'service_history6'
            and protocol['fit']['parameters'] == 2 and protocol['fit']['ridge'] == 1.
            and protocol['fit']['penalize_intercept'] is True and protocol['fit']['solves'] == 1
            and protocol['fit']['no_model_selection'] is True)
    return dict(all_pass=all(checks.values()), checks=checks, rows=len(rows), rank=rank)


def fit(rows, q_range):
    """Call the original ridge-1 solver once, then check its normal equations."""
    model = fit_bias([dict(q=row['q'], prediction=row['m'], delta=row['D']) for row in rows], q_range)
    matrix = design(rows, q_range)
    gram = np.eye(2)+matrix.T@matrix
    rhs = matrix.T@np.asarray([row['D']-row['m'] for row in rows])
    beta, inverse = np.asarray(model['coef']), np.asarray(model['gram_inverse'])
    parameters_finite = bool(np.isfinite(beta).all() and np.isfinite(inverse).all())
    checks = dict(finite_parameters=parameters_finite,
        normal_equation=bool(parameters_finite and np.allclose(gram@beta, rhs, rtol=1e-10, atol=1e-10)),
        inverse_equation=bool(parameters_finite and np.allclose(gram@inverse, np.eye(2), rtol=1e-10, atol=1e-10)),
        fixed_metadata=model['n'] == 12 and model['rank'] == 2 and model['ridge'] == 1.
            and model['linear_solves'] == 1 and model['q_range'] == list(q_range))
    return model, dict(all_pass=all(checks.values()), checks=checks,
        normal_equation_max_abs=float(np.max(np.abs(gram@beta-rhs))) if parameters_finite else None,
        inverse_equation_max_abs=float(np.max(np.abs(gram@inverse-np.eye(2)))) if parameters_finite else None)


def error_readout(rows):
    errors = [row['m']-row['D'] for row in rows]
    return dict(n=len(rows), mean_error=float(np.mean(errors)),
                mae=float(np.mean(np.abs(errors))), rmse=float(np.sqrt(np.mean(np.square(errors)))))


def selected_readout(row):
    if row is None:
        return dict(selected_root=None, reference_fallback=True, delta=0., wait_delta=0., ride_delta=0.)
    return dict(selected_root=row['physical_root_id'], reference_fallback=False,
                physical_noop=row['physical_noop'], delta=row['D'],
                wait_delta=row['wait_delta'], ride_delta=row['ride_delta'])


def analyze(summary, protocol):
    validation = validate(summary, protocol)
    report = dict(status='hold', validation=validation, model=None, pairs=[],
        decision=dict(ready_for_calibration=False, reason='T_INCOMPLETE_OR_INVALID'),
        budget=dict(real_bias_fits=0, native_seconds=0, new_labels=0,
                    optimizer_updates=0, new_initialization_updates=0),
        interpretation=dict(independent_control_evidence=False, training_readout_only=True,
            calibrated_margin_available=False, C0_gate_scope='Observed T roots only, not the future domain',
            target='D is candidate-minus-reference remaining-day (2W+R)/15079',
            no_automatic_calibration=True))
    if not validation['all_pass']:
        return report
    rows = extract(summary, protocol)
    report['budget']['real_bias_fits'] = 1
    try:
        bias_model, numerical = fit(rows, protocol['q_range_for_scaling_only'])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        report['decision']['reason'] = 'FIXED_FIT_FAILED_NO_RETRY'
        report['fit_error'] = dict(type=type(error).__name__, message=str(error))
        return report
    report['numerical'] = numerical
    if not numerical['all_pass']:
        report['decision']['reason'] = 'FIXED_FIT_NUMERICAL_CHECK_FAILED_NO_RETRY'
        return report
    for row in rows:
        predicted = predict(bias_model, row['q'], row['m'])
        row.update(b=predicted['bias'], u=predicted['support_scale'], center=predicted['center'])
        row['C0_eligible'] = row['m'] < 0. and row['center'] < 0.
    report['pairs'] = rows
    if not all(finite(row[key]) for row in rows for key in ('b', 'u', 'center')):
        report['decision']['reason'] = 'FIXED_FORWARD_NONFINITE_NO_RETRY'
        return report
    report['status'] = 'complete'
    eligible = sum(row['C0_eligible'] for row in rows)
    report['decision'] = dict(ready_for_calibration=eligible > 0, C0_eligible_roots=eligible,
        reason='READY_FOR_C_REVIEW' if eligible else 'CLOSED_NO_T_ENTRY_AT_C0')
    report['model'] = dict(bias_model=bias_model, primary_model=protocol['primary_model'],
        source_summary=protocol['source_summary'], source_protocol=protocol['source_protocol'],
        candidate_deployment_domain=protocol['candidate_deployment_domain'],
        integrity=dict(all_pass=True, checks=dict(T_integrity=validation['all_pass'],
                                                  fixed_fit_numerical=numerical['all_pass'])),
        budget=dict(report['budget']))
    report['frozen_m_errors'] = dict(overall=error_readout(rows),
        by_seed={str(seed):error_readout([row for row in rows if row['external_seed'] == seed])
                 for seed in protocol['T']},
        by_scene={scene:error_readout([row for row in rows if row['scenario'] == scene]) for scene in SCENES},
        by_seed_scene={str(seed)+'/'+scene:error_readout([row for row in rows
            if row['external_seed'] == seed and row['scenario'] == scene])
            for seed in protocol['T'] for scene in SCENES})
    selections = {}
    for seed in protocol['T']:
        for scene in SCENES:
            ordered = sorted([row for row in rows if row['external_seed'] == seed and row['scenario'] == scene],
                             key=lambda row:(row['issue_clock'], row['sequence']))
            raw = next((row for row in ordered if row['m'] < 0.), None)
            selections[str(seed)+'/'+scene] = dict(raw_m=selected_readout(raw),
                corrected_C0=selected_readout(select_first(ordered, bias_model, 0.)))
    report['training_single_selection'] = dict(by_seed_scene=selections,
        scope='Same training labels used to fit b; descriptive single selection, not independent policy benefit')
    return report


def write_once(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with path.open('x') as handle:
        handle.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--out-dir', type=Path, default=HERE)
    args = parser.parse_args(argv)
    outputs = {name:args.out_dir/(name+'.json') for name in ('fit_attempt', 'model', 'summary')}
    if any(path.exists() for path in outputs.values()):
        parser.error('A fixed fit attempt/output already exists; do not retry or overwrite it')
    protocol = json.loads(args.protocol.read_text())
    source = Path(protocol['source_summary'])
    summary = json.loads(source.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_once(outputs['fit_attempt'], dict(status='attempt_reserved', source_summary=str(source),
        protocol=str(args.protocol), maximum_solver_calls=1, retry_allowed=False))
    result = analyze(summary, protocol)
    model = result.pop('model')
    if model is not None:
        write_once(outputs['model'], model)
    write_once(outputs['summary'], result)
    print(json.dumps(dict(status=result['status'], decision=result['decision'], budget=result['budget'])))
    return result


if __name__ == '__main__':
    main()
