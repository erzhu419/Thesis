"""Training-only, issue-ordered comparison of mean versus endpoint-local history."""
import argparse
import json
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

LINES = ('7X', '102X', '705X')
ENDPOINTS = tuple((kind, i) for kind in ('arrival', 'departure') for i in range(3))
MODES = ('shared', 'mean', 'local')
SCALE, PRIOR = 60., 7.5


def endpoint_design(features, history, endpoint, mode):
    values = list(features[:4])
    if mode == 'shared':
        return np.asarray(values, float)
    if history.get('status') != 'supported':
        return None
    value = history['mean_residual_scaled'] if mode == 'mean' else (
        history['segments'][endpoint]['latest']['residual']/SCALE)
    return np.asarray(values+[float(value)], float)


def fit(training, mode, endpoint):
    kind, i = ENDPOINTS[endpoint]
    rows = [(r, endpoint_design(r['features'], r['service_history'], i, mode))
            for r in training if r['service_targets'][kind][i] is not None]
    rows = [(r, x) for r, x in rows if x is not None]
    if not rows:
        return [], None
    x = np.stack([x for _, x in rows])
    y = np.asarray([r['service_targets'][kind][i] for r, _ in rows])/SCALE
    return rows, np.linalg.solve(x.T@x+np.eye(x.shape[1]), x.T@y)


def forecast(training, issue, mode):
    """The saved admitted-ID prefix controls availability even at equal clocks."""
    n = issue['prefix']['counters']['service_shared_pairs']
    assert 0 <= n <= len(training)
    past = training[:n]
    now = issue['selected']['started_at']
    for row in past:
        assert row['role'] == 'train' and row['selected_for_training'] is True
        assert row['available_at'] <= now and row['admitted_at'] <= now
        assert row['issued_at'] < now and row['id'] != issue['id']
    predictions, support = [], []
    for endpoint, (_, i) in enumerate(ENDPOINTS):
        rows, weight = fit(past, mode, endpoint)
        count = sum(r['line_id'] == issue['line_id'] for r, _ in rows)
        x = endpoint_design(issue['features'], issue['service_history'], i, mode)
        support.append(count)
        predictions.append(float(x@weight*SCALE)+PRIOR
            if count and weight is not None and x is not None else None)
    return dict(prediction=predictions, support=support,
                admitted_ids=[r['id'] for r in past])


def history_audit(history, issue):
    assert history['status'] == 'supported' and len(history['segments']) == 3
    for segment in history['segments']:
        assert segment['available']
        latest = segment['latest']
        assert latest['completed_at'] < issue and latest['observed_at'] < issue
        assert abs(latest['age']-(issue-latest['completed_at'])) < 1e-8
    residuals = [s['latest']['residual'] for s in history['segments']]
    assert abs(np.mean(residuals)/SCALE-history['mean_residual_scaled']) < 1e-10


def metrics(rows, endpoint=None):
    """Aggregate endpoints within a state first, retaining unsupported states."""
    result = {}
    for mode in MODES:
        errors, signs, negative, negative_hits = [], 0, 0, 0
        endpoints = range(6) if endpoint is None else [endpoint]
        for row in rows:
            valid = [i for i in endpoints if row['truth'][i] is not None
                     and all(row['prediction'][m][i] is not None for m in MODES)]
            if valid:
                errors.append(float(np.mean([abs(row['prediction'][mode][i]-row['truth'][i]) for i in valid])))
            signs += sum(np.sign(row['prediction'][mode][i]) == np.sign(row['truth'][i]) for i in valid)
            negative += sum(row['truth'][i] < 0 for i in valid)
            negative_hits += sum(row['truth'][i] < 0 and row['prediction'][mode][i] < 0 for i in valid)
        result[mode] = dict(mae_seconds=float(np.mean(errors)) if errors else None,
            states=len(errors), sign_correct=int(signs), negative_truth=int(negative),
            negative_hits=int(negative_hits) if negative else None)
    return result


def design_report(training):
    result = {}
    for mode, endpoint in [('shared', 0), ('mean', 0), ('local', 0), ('local', 1), ('local', 2)]:
        x = np.stack([endpoint_design(r['features'], r['service_history'], endpoint, mode) for r in training])
        report = dict(rows=len(x), columns=x.shape[1], rank=int(np.linalg.matrix_rank(x)),
            ridge_condition=float(np.linalg.cond(x.T@x+np.eye(x.shape[1]))))
        if mode != 'shared':
            base, z = x[:, :4], x[:, 4]
            schur = float(z@z-(base.T@z)@np.linalg.solve(base.T@base+np.eye(4), base.T@z))
            report.update(history_energy=float(z@z), history_schur_energy=schur,
                          ridge_schur_fraction=schur/(1.+schur))
        result[mode+str(endpoint+1)] = report
    return result


def diagnose(source):
    day = json.loads((source/'summary.json').read_text())
    assert day['status'] == 'complete' and day['integrity']['all_pass']
    # Only training rows enter any fit or evaluation in this version.
    journal = [r for r in json.loads((source/'acquisitions.json').read_text())['records'] if r['role'] == 'train']
    cache = {r['id']: r for r in (json.loads(p.read_text()) for p in sorted((source/'pair_cache').glob('*.json')))}
    order = day['budget']['admitted_pair_ids']
    metadata = {r['id']: r for r in day['training_pairs']}
    training = [cache[identifier] for identifier in order]
    assert len(training) == len(journal) == 9 and set(order) == {r['id'] for r in journal}
    for row in training:
        meta = metadata[row['id']]
        assert row['role'] == 'train' and meta['admitted']
        # Cache files precede admission; the completed day's ledger is authoritative.
        row['selected_for_training'] = True
        row['admitted_at'] = meta['admitted_at']
        history_audit(row['service_history'], row['issued_at'])
    assert day['budget']['target_labels'] == 285
    parity, finite, unsupported, rows = 0., 0, 0, []
    for issue in journal:
        n = issue['prefix']['counters']['service_shared_pairs']
        assert n == issue['prefix']['counters']['service_history_pairs']
        record = cache[issue['id']]
        assert issue['features'] == record['features'] and issue['service_history'] == record['service_history']
        history_audit(issue['service_history'], issue['selected']['started_at'])
        forecasts = {mode: forecast(training, issue, mode) for mode in MODES}
        for mode, old_name in (('shared', 'shared'), ('mean', 'history')):
            saved = [issue['service_prediction'][old_name][kind][i] for kind, i in ENDPOINTS]
            for actual, expected in zip(forecasts[mode]['prediction'], saved):
                if expected is None:
                    assert actual is None
                    unsupported += 1
                else:
                    assert actual is not None
                    parity = max(parity, abs(actual-expected))
                    finite += 1
        # Forecasts above are frozen before the later truth is scored.
        truth = [None if record['service_targets'][k][i] is None else record['service_targets'][k][i]+PRIOR
                 for k, i in ENDPOINTS]
        residuals = [s['latest']['residual'] for s in issue['service_history']['segments']]
        past_line = [r for r in training[:n] if r['line_id'] == issue['line_id']]
        extrapolation = []
        for i, value in enumerate(residuals):
            reference = [r['service_history']['segments'][i]['latest']['residual'] for r in past_line]
            extrapolation.append(None if not reference else not min(reference) <= value <= max(reference))
        decomposition = []
        for endpoint, (_, i) in enumerate(ENDPOINTS):
            _, wmean = fit(training[:n], 'mean', endpoint)
            _, wlocal = fit(training[:n], 'local', endpoint)
            if forecasts['local']['prediction'][endpoint] is None:
                decomposition.append(None)
                continue
            meanx = endpoint_design(issue['features'], issue['service_history'], i, 'mean')
            localx = endpoint_design(issue['features'], issue['service_history'], i, 'local')
            direct = float((localx-meanx)@wmean*SCALE)
            refit = float(localx@(wlocal-wmean)*SCALE)
            delta = forecasts['local']['prediction'][endpoint]-forecasts['mean']['prediction'][endpoint]
            assert abs(direct+refit-delta) < 1e-10
            decomposition.append([direct, refit, delta])
        rows.append(dict(id=issue['id'], line=issue['line_id'], window=issue['window_name'],
            issue=issue['selected']['started_at'], admitted_ids=forecasts['local']['admitted_ids'],
            residuals_seconds=residuals, history_ages=[s['latest']['age'] for s in issue['service_history']['segments']],
            history_mean_seconds=float(np.mean(residuals)), local_extrapolation=extrapolation,
            truth=truth, prediction={m: forecasts[m]['prediction'] for m in MODES},
            local_minus_mean=decomposition))
    assert parity < 1e-10
    macro = metrics(rows)
    by_line = {line: metrics([r for r in rows if r['line'] == line]) for line in LINES}
    improve = macro['local']['mae_seconds'] < macro['mean']['mae_seconds']
    no_regression = all(v['local']['mae_seconds'] <= v['mean']['mae_seconds'] for v in by_line.values())
    scored_negative = macro['local']['negative_truth']
    supported = [r for r in rows if all(v is not None for v in r['prediction']['local'])]
    return dict(status='complete', priority='high', scope='Training-only development diagnosis; no heldout evaluation.',
        source=str(source), integrity=dict(original_day_passed=True, original_budget=285,
            strict_preissue_history=True, exact_admitted_id_prefix=True, shared_mean_replay_error=parity,
            finite_saved_forecasts=finite, unsupported_saved_forecasts=unsupported, heldout_fits_or_scores=0),
        macro=macro, by_line=by_line,
        by_window={w: metrics([r for r in rows if r['window'] == w]) for w in sorted({r['window'] for r in rows})},
        by_endpoint={kind+str(i+1): metrics(rows, j) for j, (kind, i) in enumerate(ENDPOINTS)},
        coverage=dict(training_states=len(rows), scored_states=len(supported),
            cold_start=[r['id'] for r in rows if r not in supported],
            censored_truth_endpoints=sum(t is None for r in rows for t in r['truth']),
            training_negative_endpoints=sum(t is not None and t < 0 for r in rows for t in r['truth']),
            scored_negative_endpoints=scored_negative,
            extrapolated_local_segments=sum(v is True for r in supported for v in r['local_extrapolation']),
            scored_local_segments=len(supported)*3),
        design=design_report(training), rows=rows,
        decomposition_order=['fixed_mean_weights_input_change', 'all_weights_refit', 'total_local_minus_mean'],
        decision=dict(macro_improved=improve, no_line_regression=no_regression,
            fresh_service_preflight_eligible=improve and no_regression,
            reversal_assessment='unassessable' if not scored_negative else 'see_metrics',
            stage_b='closed', independent_confirmation='not_submitted'),
        compute=dict(new_training_labels=0, new_native_prefix_seconds=0, new_native_suffix_seconds=0),
        limitations='One stable day, nine training states, six supported chronological states. Same-state endpoints correlated. Scale/ridge unchanged; local versus mean differences may reflect both alignment and feature amplitude. No300-second control assessment.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = diagnose(args.source)
    output = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(output.encode()) <= 24000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(output)
    print(json.dumps(dict(status='complete', bytes=len(output.encode()), decision=value['decision'])))
