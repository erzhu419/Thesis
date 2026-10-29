"""Reconstruct the saved v21 service fits; decompose the added history column."""
import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

KINDS = ('arrival', 'departure')
LINES = ('7X', '102X', '705X')
SCALE = 60.


def read(path):
    return json.loads(path.read_text())


def training_prequential(source):
    """Score saved issue forecasts against later training truth without refitting."""
    journal = read(source/'acquisitions.json')['records']
    cache = {r['id']: r for r in (read(p) for p in sorted((source/'pair_cache').glob('*.json')))}
    rows, unsupported, censored = [], [], 0
    for record in (r for r in journal if r['role'] == 'train'):
        truth = cache[record['id']]['service_targets']
        errors = {m: [] for m in ('shared', 'history')}
        signs = {m: 0 for m in errors}
        flip_correct = {m: 0 for m in errors}
        flips = 0
        for kind, i in ((k, i) for k in KINDS for i in range(3)):
            target = truth[kind][i]
            if target is None:
                censored += 1
                continue
            actual = target+7.5
            forecast = {m: record['service_prediction'][m][kind][i] for m in errors}
            if any(value is None for value in forecast.values()):
                continue
            flips += actual < 0
            for model, value in forecast.items():
                errors[model].append(abs(value-actual))
                signs[model] += int(np.sign(value) == np.sign(actual))
                flip_correct[model] += int(actual < 0 and value < 0)
        if not errors['shared']:
            unsupported.append(record['id'])
            continue
        rows.append(dict(id=record['id'], available_pairs=record['prefix']['counters']['service_shared_pairs'],
            endpoints=len(errors['shared']), mae={m:float(np.mean(e)) for m,e in errors.items()},
            signs_correct=signs, negative_truth=flips, negative_correct=flip_correct))
    return dict(status='complete', training_states_scored=len(rows), unsupported=unsupported,
        censored_endpoints=censored, macro_mae={m:float(np.mean([r['mae'][m] for r in rows])) for m in ('shared','history')},
        rows=rows, scope='Original issue-time frozen forecasts, then later original training-pair truth. No refit, future input, heldout labels or gate change. Endpoints within one state are correlated.')


def main(source):
    day = read(source/'summary.json')
    assert day['status'] == 'complete' and day['integrity']['all_pass']
    assert 'candidate' not in day['readouts']
    journal = read(source/'acquisitions.json')['records']
    cache = {r['id']: r for r in (read(p) for p in sorted((source/'pair_cache').glob('*.json')))}
    order = day['budget']['admitted_pair_ids']
    records = [cache[key] for key in order]
    assert len(records) == 9 and all(r['role'] == 'train' for r in records)
    x = np.asarray([r['features'][:4] for r in records], float)
    z = np.asarray([r['service_history']['mean_residual_scaled'] for r in records], float)
    y = np.asarray([[r['service_targets'][kind][i] for kind in KINDS for i in range(3)] for r in records], float)/SCALE
    assert np.isfinite(x).all() and np.isfinite(z).all() and np.isfinite(y).all()

    def weights(n, use_history):
        a = np.column_stack((x[:n], z[:n])) if use_history else x[:n]
        return np.linalg.solve(a.T@a+np.eye(a.shape[1]), a.T@y[:n])

    parity_error, compared, nullable = 0., 0, 0
    for row in journal:
        feature = np.asarray(row['features'][:4], float)
        for name, use_history in (('shared', False), ('history', True)):
            n = row['prefix']['counters']['service_'+name+'_pairs']
            assert 0 <= n <= len(records)
            support = sum(r['line_id'] == row['line_id'] for r in records[:n])
            a = np.append(feature, row['service_history']['mean_residual_scaled']) if use_history else feature
            prediction = a@weights(n, use_history)*SCALE+7.5
            saved = row['service_prediction'][name]
            for j, (kind, i) in enumerate((k, i) for k in KINDS for i in range(3)):
                value = saved[kind][i]
                if not support:
                    assert value is None
                    nullable += 1
                else:
                    assert value is not None and saved['support_pairs'][kind][i] == support
                    parity_error = max(parity_error, abs(float(prediction[j])-value))
                    compared += 1
    assert parity_error < 1e-10, parity_error
    w4, w5 = weights(9, False), weights(9, True)
    x5 = np.column_stack((x, z))
    z_ols = z-x@np.linalg.lstsq(x, z, rcond=None)[0]
    p = x.T@x+np.eye(4)
    schur = float(z@z-(x.T@z)@np.linalg.solve(p, x.T@z))
    corrections, examples = [], []
    for row in (r for r in journal if r['role'] == 'heldout'):
        a = np.asarray(row['features'][:4], float)
        hist = row['service_history']['mean_residual_scaled']
        direct = SCALE*hist*w5[-1]
        refit = SCALE*a@(w5[:4]-w4)
        total = direct+refit
        saved = np.asarray([row['service_prediction']['history'][k][i]-row['service_prediction']['shared'][k][i]
            for k in KINDS for i in range(3)])
        assert np.max(abs(total-saved)) < 1e-10
        corrections.append(dict(id=row['id'], direct_abs_mean=float(np.mean(abs(direct))),
            refit_abs_mean=float(np.mean(abs(refit))), total_abs_mean=float(np.mean(abs(total))),
            total_abs_max=float(np.max(abs(total)))))
        if row['line_id'] == '7X':
            examples.append(dict(id=row['id'], arrival3=[float(v[2]) for v in (direct, refit, total)],
                departure3=[float(v[5]) for v in (direct, refit, total)]))
    return dict(status='complete', saved_predictions=dict(max_abs_error=parity_error,
        finite_compared=compared, unsupported_nones_matched=nullable, acquisitions=len(journal)),
        design=dict(rows=9, rank_x4=int(np.linalg.matrix_rank(x)), rank_x5=int(np.linalg.matrix_rank(x5)),
            history_energy=float(z@z), history_ols_residual_energy=float(z_ols@z_ols),
            history_schur_energy=schur, ridge=1., schur_fraction=schur/(1.+schur),
            raw_condition_x4=float(np.linalg.cond(x)), raw_condition_x5=float(np.linalg.cond(x5)),
            ridge_condition_x4=float(np.linalg.cond(x.T@x+np.eye(4))),
            ridge_condition_x5=float(np.linalg.cond(x5.T@x5+np.eye(5)))),
        coefficients=dict(seconds_per_normalized_history=(SCALE*w5[-1]).tolist(),
            seconds_per_raw_history_second=w5[-1].tolist()),
        heldout_corrections=corrections, seven_x_third_stop=examples,
        schema=dict(endpoint_order=[k+str(i+1) for k in KINDS for i in range(3)],
            example_fields=['direct_history_seconds', 'original_four_column_refit_seconds', 'total_seconds'],
            fit='Only original nine admitted v21 training pairs, original order, ridge1 and scale60. Reproduce each issue using its saved admitted-pair count; no heldout fitting.',
            schur='zTz-zTX(XTX+I)^-1XTz; fraction=Schur/(1+Schur). OLS residual energy is design-only.',
            correction='history5-shared4 = direct added column + refit of original four columns. No new simulation, labels, parameter sweep, saved-model mutation or gate change.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = main(args.source)
    text = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(text.encode()) <= 5000
    with args.out.open('x') as f:
        f.write(text)
    print(json.dumps(dict(status='complete', bytes=len(text.encode()), parity_error=value['saved_predictions']['max_abs_error'])))
