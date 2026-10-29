"""Issue-visible service covariates and two fixed paired-response regressions.

Native capture uses v80.NativeContextReader in the collector. This module reads
its returned dictionary and performs no simulation, sampling or actor update.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

LINES = ('7X', '102X', '705X')
SCENARIOS = ('stable', 'local_speed_drop')
LOAD_BINS = ('low', 'high')
DENOMINATOR = 15079.
HOLD_SCALE = 7.5
RIDGE = 1.
SERVICE_FEATURES = ('load_ratio', 'queue_seat_margin')
FEATURE_NAMES = dict(line=['line_'+line for line in LINES],
    conditional=['line_'+line for line in LINES]+['load_ratio_centered', 'queue_seat_margin_centered'])


def service_state(ctx):
    """Convert the atomic native observation without filling a missing count."""
    clocks = ctx.get('clocks', {})
    counts = deepcopy(ctx.get('counts', {}))
    reasons = list(ctx.get('reasons', []))
    issue, event = clocks.get('issue'), clocks.get('event')
    times = [issue, event, clocks.get('obs_start'), clocks.get('obs_end')]
    finite_clocks = all(value is not None and np.isfinite(value) for value in times)
    atomic = bool(finite_clocks and clocks.get('atomic_at_issue')
                  and clocks['obs_start'] == clocks['obs_end'] == issue)
    if not ctx.get('observed'):
        reasons.append('native_context_unobserved')
    if not atomic:
        reasons.append('context_not_atomic_at_issue')
    capacity = ctx.get('capacity')
    if capacity is None or not np.isfinite(capacity) or capacity <= 0:
        reasons.append('capacity_missing_or_invalid')
    for key in ('onboard', 'release_next', 'compatible_next_queue', 'seats_after_release'):
        value = counts.get(key)
        if value is None or not np.isfinite(value):
            reasons.append('native_count_missing_or_invalid:'+key)
    reasons = list(dict.fromkeys(reasons))
    features = load_bin = None
    if not reasons:
        load = float(counts['onboard']/capacity)
        margin = float(np.clip((counts['compatible_next_queue']-counts['seats_after_release'])/capacity, -1., 1.))
        features = [load, margin]
        load_bin = 'low' if load < .5 else 'high'
    return dict(observed=not reasons, reason=reasons[0] if reasons else None, reasons=reasons,
        atomic_at_issue=atomic, issue_clock=issue, event_clock=event,
        capacity=capacity, counts=counts, features=features, load_bin=load_bin)


def _domain_reason(ref_hold, cand_hold):
    if not np.isfinite([ref_hold, cand_hold]).all():
        return 'nonfinite_holding'
    if cand_hold <= ref_hold:
        return 'nonpositive_holding_difference'
    if not 3. <= ref_hold < 3.75:
        return 'reference_holding_outside_3_to_3_75'
    if abs(cand_hold-HOLD_SCALE) > 1e-5:
        return 'candidate_holding_outside_7_5'
    return None


def action_support(ref_hold, cand_hold, simulation_step, action_step):
    """Check the existing action domain and its native command separation."""
    ref_hold, cand_hold = float(ref_hold), float(cand_hold)
    reason = _domain_reason(ref_hold, cand_hold)
    gap = abs(cand_hold-ref_hold) if np.isfinite([ref_hold, cand_hold]).all() else None
    clocks = [simulation_step, action_step]
    if reason is None and not all(value is not None and np.isfinite(value) and value > 0 for value in clocks):
        reason = 'native_action_clock_unavailable'
    if reason is None and gap < 2*max(clocks):
        reason = 'insufficient_native_command_gap'
    return dict(supported=reason is None, reason=reason, gap_seconds=gap)


def _features(service):
    if not service.get('observed') or not service.get('atomic_at_issue') or service.get('features') is None:
        raise ValueError('A missing issue-visible service observation cannot enter the design')
    values = np.asarray(service['features'], dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('The two fixed service features must be finite')
    return values


def make_design(rows):
    """Use covariates/actions only; compute each line's a²-weighted centers."""
    rows = list(rows)
    x, a, lines = [], [], []
    for row in rows:
        line = row['line_id']
        if line not in LINES:
            raise ValueError('Only the three declared lines enter the design')
        reason = _domain_reason(float(row['ref_hold']), float(row['cand_hold']))
        if reason is not None:
            raise ValueError(reason)
        x.append(_features(row['service']))
        a.append((float(row['cand_hold'])-float(row['ref_hold']))/HOLD_SCALE)
        lines.append(line)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64)
    means = {}
    for line in LINES:
        mask = np.asarray([value == line for value in lines], dtype=bool)
        weights = a[mask]**2
        means[line] = ((weights[:, None]*x[mask]).sum(axis=0)/weights.sum()).tolist() if mask.any() else None
    line_matrix = a[:, None]*np.asarray([[float(line == target) for target in LINES] for line in lines],
                                      dtype=np.float64).reshape(-1, 3)
    centered = np.asarray([value-np.asarray(means[line], dtype=np.float64) for value, line in zip(x, lines)],
                          dtype=np.float64).reshape(-1, 2)
    conditional = np.column_stack((line_matrix, a[:, None]*centered))
    matrix = dict(line=line_matrix, conditional=conditional)
    return dict(line_means=means, feature_names=deepcopy(FEATURE_NAMES), rows=len(rows),
        matrix={name:value.tolist() for name, value in matrix.items()},
        ranks={name:int(np.linalg.matrix_rank(value)) if value.size else 0 for name, value in matrix.items()})


def _training_rows(rows):
    rows = list(rows)
    expected = {(scene, line, load_bin) for scene in SCENARIOS for line in LINES for load_bin in LOAD_BINS}
    if len(rows) != 12:
        raise ValueError('The fixed training cohort is twelve pairs and24 native outcome labels')
    cases, roots = [], {line:set() for line in LINES}
    for row in rows:
        if row['source_scope'] != 'v88_train' or row['external_seed'] != 716207:
            raise ValueError('Only the new v88 training cohort716207 may enter fitting')
        if (row['observed_at'] != 24000 or row['label_available_at'] != 24030
                or row['full_real_physical_pass'] is not True):
            raise ValueError('Each pair requires real24000 endpoints,24030 maturity and physical integrity')
        case = (row['scenario'], row['line_id'], row['load_bin'])
        if case not in expected or row['service']['load_bin'] != row['load_bin']:
            raise ValueError('The training case must match its issue-visible load bin')
        if not isinstance(row['physical_prefix'], str) or not isinstance(row['sequence'], int):
            raise ValueError('Each physical root is identified by its prefix and integer issue sequence')
        cases.append(case)
        roots[row['line_id']].add((row['physical_prefix'], row['sequence']))
    if set(cases) != expected or any(len(value) != 4 for value in roots.values()):
        raise ValueError('Each line requires four distinct roots covering the two scenes and two load bins')
    return sorted(rows, key=lambda row:(SCENARIOS.index(row['scenario']), LINES.index(row['line_id']),
                                       LOAD_BINS.index(row['load_bin'])))


def train(rows):
    """Future fixed-cohort fit; the current native preflight never calls this."""
    rows = _training_rows(rows)
    design = make_design(rows)
    target = np.asarray([float(row['target_delta_person_seconds'])/DENOMINATOR for row in rows], dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError('Each target must be the finite observed paired remaining-day cost difference')
    models = {}
    for name, columns in (('line', 3), ('conditional', 5)):
        matrix = np.asarray(design['matrix'][name], dtype=np.float64)
        weights = np.linalg.solve(matrix.T@matrix+RIDGE*np.eye(columns), matrix.T@target)
        prediction = matrix@weights
        models[name] = dict(coef=weights.tolist(), design_rank=design['ranks'][name],
            train_fit=dict(target_delta=target.tolist(), predicted_delta=prediction.tolist(),
                mae=float(np.mean(abs(prediction-target))), rmse=float(np.sqrt(np.mean((prediction-target)**2)))))
    checks = dict(line_rank_three=design['ranks']['line'] == 3,
        conditional_rank_five=design['ranks']['conditional'] == 5,
        line_coefficients_preserved=bool(np.allclose(models['line']['coef'], models['conditional']['coef'][:3],
                                                    rtol=0., atol=1e-12)))
    return dict(name='v88_centered_service_response', models=models, line_means=design['line_means'],
        feature_names=design['feature_names'], ridge=RIDGE, denominator=DENOMINATOR, hold_scale=HOLD_SCALE,
        objective='sum_squared_error_plus_ridge_norm_squared', target='(G_candidate-G_reference)/15079',
        action_support=dict(reference_min=3., reference_max_exclusive=3.75, candidate=7.5, candidate_tolerance=1e-5),
        decision_threshold=0., fits=2, gate=dict(all_pass=all(checks.values()), checks=checks),
        training_scope=dict(source_scope='v88_train', external_seed=716207, pairs=12,
            physical_outcome_labels=24, observed_at=24000, label_available_at=24030,
            cases=['/'.join((row['scenario'], row['line_id'], row['load_bin'])) for row in rows],
            heldout_labels_used=0), actor_updates=0, critic_updates=0, value_updates=0,
        world_updates=0, trust_updates=0)


def predict(artifact, line_id, service, ref_hold, cand_hold, *, simulation_step, action_step):
    """Both readouts require the same visible context and legal executed pair."""
    reason = None
    if not artifact['gate']['all_pass']:
        reason = 'training_gate_hold'
    elif line_id not in LINES:
        reason = 'undeclared_line'
    else:
        try:
            x = _features(service)
        except ValueError:
            reason = 'service_context_missing'
    if reason is None:
        reason = action_support(ref_hold, cand_hold, simulation_step, action_step)['reason']
    models = {name:dict(supported=reason is None, reason=reason, predicted_delta=None, choice='reference')
              for name in ('line', 'conditional')}
    if reason is None:
        a = (float(cand_hold)-float(ref_hold))/HOLD_SCALE
        base = a*np.asarray([float(line_id == line) for line in LINES], dtype=np.float64)
        conditional = np.r_[base, a*(x-np.asarray(artifact['line_means'][line_id], dtype=np.float64))]
        for name, row in (('line', base), ('conditional', conditional)):
            value = float(row@np.asarray(artifact['models'][name]['coef'], dtype=np.float64))
            models[name].update(predicted_delta=value, choice='candidate' if value < 0. else 'reference')
    return dict(supported=reason is None, reason=reason, models=models,
        ref_hold=float(ref_hold), cand_hold=float(cand_hold))
