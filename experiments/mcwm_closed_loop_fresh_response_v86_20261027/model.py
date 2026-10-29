"""Fixed stale/fresh cost-response fits; the heldout labels never enter here."""
from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.model import (
    DENOMINATOR, FEATURE_NAMES, HOLD_SCALE, LINES, RIDGE, SCENARIOS,
    design_row, state_feature,
)

COHORTS = {'stale': ('v84_stale', 106207), 'fresh': ('v85_fresh', 216207)}
ACTION_SUPPORT = dict(reference_min=3., reference_max_exclusive=3.75,
                      candidate=7.5, candidate_tolerance=1e-5)
LEGACY_ACTION_SUPPORT = dict(reference_min=3., reference_max_inclusive=3.6,
                             candidate=7.5, candidate_tolerance=1e-5)


def _support_reason(ref_hold, cand_hold, *, legacy=False):
    if not np.isfinite([ref_hold, cand_hold]).all():
        return 'nonfinite_holding'
    if cand_hold <= ref_hold:
        return 'nonpositive_holding_difference'
    if legacy:
        if not 3. <= ref_hold <= 3.6:
            return 'reference_holding_outside_3_to_3_6'
    elif not 3. <= ref_hold < 3.75:
        return 'reference_holding_outside_3_to_3_75'
    if abs(cand_hold-HOLD_SCALE) > 1e-5:
        return 'candidate_holding_outside_7_5'
    return None


def _ordered_records(records, arm):
    rows = list(records)
    scope, seed = COHORTS[arm]
    if len(rows) != 6:
        raise ValueError('Each arm requires exactly six declared training pairs')
    cases = []
    for row in rows:
        if row['source_scope'] != scope or row['external_seed'] != seed:
            raise ValueError(f'{arm} accepts only {scope} labels with external seed {seed}')
        if row['label_available_at'] != 24030:
            raise ValueError('Remaining-day training labels mature at24030')
        cases.append((row['scenario'], row['line_id']))
    expected = {(scene, line) for scene in SCENARIOS for line in LINES}
    if len(set(cases)) != 6 or set(cases) != expected:
        raise ValueError('Each arm requires each fixed scene/line exactly once')
    return sorted(rows, key=lambda row: (SCENARIOS.index(row['scenario']), LINES.index(row['line_id'])))


def _fit_arm(rows, arm):
    inputs, target, support = [], [], []
    for row in rows:
        h_ref, h_candidate = float(row['ref_hold']), float(row['cand_hold'])
        support.append(_support_reason(h_ref, h_candidate))
        inputs.append(design_row(row['line_id'], row['state'], h_ref, h_candidate))
        target.append(float(row['target_delta_person_seconds'])/DENOMINATOR)
    finite = all(row is not None and np.isfinite(row).all() for row in inputs)
    finite = bool(finite and np.isfinite(target).all())
    action_support = all(reason is None for reason in support)
    models = {name: dict(coef=None, design_rank=None, train_fit=None)
              for name in ('line', 'state')}
    fits = 0
    if finite and action_support:
        matrix, target = np.asarray(inputs, dtype=np.float64), np.asarray(target, dtype=np.float64)
        for name, columns in (('line', 3), ('state', 4)):
            design = matrix[:, :columns]
            weights = np.linalg.solve(design.T@design+RIDGE*np.eye(columns), design.T@target)
            predicted = design@weights
            models[name] = dict(coef=weights.tolist(), design_rank=int(np.linalg.matrix_rank(design)),
                train_fit=dict(target_delta=target.tolist(), predicted_delta=predicted.tolist(),
                    mae=float(np.mean(abs(predicted-target))),
                    rmse=float(np.sqrt(np.mean((predicted-target)**2)))),
                feature_names=list(FEATURE_NAMES[:columns]))
            fits += 1
    checks = dict(finite_training_inputs=finite, training_action_support=action_support,
        line_rank_three=models['line']['design_rank'] == 3,
        state_rank_four=models['state']['design_rank'] == 4,
        finite_coefficients=all(value['coef'] is not None and np.isfinite(value['coef']).all()
                                for value in models.values()),
        two_fixed_solves=fits == 2)
    scope, seed = COHORTS[arm]
    return dict(models=models, fits=fits, gate=dict(all_pass=all(checks.values()), checks=checks),
        training_scope=dict(source_scope=scope, pairs=6, physical_outcome_labels=12,
            source_seed=6207, external_seed=seed, label_available_at=24030,
            cases=[row['scenario']+'/'+row['line_id'] for row in rows],
            heldout_labels_used=0, support_reasons=support))


def train(cohorts):
    """Fit each arm's line3/state4 once, with float64 sum-SSE plus ridge1.

    ``cohorts`` has exactly ``stale`` and ``fresh`` keys. Each record supplies
    scenario, line_id, state, ref_hold, cand_hold, target_delta_person_seconds,
    source_scope, external_seed and label_available_at.
    """
    if set(cohorts) != set(COHORTS):
        raise ValueError('The two fixed cohorts are stale and fresh')
    ordered = {arm: _ordered_records(cohorts[arm], arm) for arm in COHORTS}
    arms = {arm: _fit_arm(ordered[arm], arm) for arm in COHORTS}
    fits = sum(value['fits'] for value in arms.values())
    checks = {arm+'_training_gate': value['gate']['all_pass'] for arm, value in arms.items()}
    checks['four_fixed_solves'] = fits == 4
    return dict(name='v86_fixed_stale_fresh_cost_response', arms=arms,
        feature_names=list(FEATURE_NAMES), ridge=RIDGE,
        objective='sum_squared_error_plus_ridge_norm_squared',
        target='(G_candidate-G_reference)/15079', denominator=DENOMINATOR, hold_scale=HOLD_SCALE,
        action_support=dict(ACTION_SUPPORT), legacy_action_support=dict(LEGACY_ACTION_SUPPORT),
        primary_readouts=['stale', 'fresh'], auxiliary_readouts=['legacy_narrow'],
        fits=fits, gate=dict(all_pass=all(checks.values()), checks=checks),
        trained_components=['stale_D_line', 'stale_D_state', 'fresh_D_line', 'fresh_D_state'],
        actor_updates=0, critic_updates=0, value_updates=0, B_updates=0, world_updates=0, trust_updates=0)


def predict(artifact, line_id, state, ref_hold, cand_hold):
    """Score only state4; the legacy reader changes support, never weights."""
    ref_hold, cand_hold = float(ref_hold), float(cand_hold)
    z = state_feature(state)
    reason = None
    if not artifact['gate']['all_pass']:
        reason = 'training_gate_hold'
    elif line_id not in LINES:
        reason = 'undeclared_line'
    elif z is None:
        reason = 'nonfinite_state'
    common_reason = reason or _support_reason(ref_hold, cand_hold)
    legacy_reason = reason or _support_reason(ref_hold, cand_hold, legacy=True)
    models = {}
    row = design_row(line_id, state, ref_hold, cand_hold) if common_reason is None else None
    for name in ('stale', 'fresh', 'legacy_narrow'):
        failure = legacy_reason if name == 'legacy_narrow' else common_reason
        value = None
        if failure is None:
            arm = 'stale' if name == 'legacy_narrow' else name
            value = float(row@np.asarray(artifact['arms'][arm]['models']['state']['coef'], dtype=np.float64))
        models[name] = dict(supported=failure is None, reason=failure, predicted_delta=value,
                            choice='candidate' if value is not None and value < 0. else 'reference')
    return dict(supported=common_reason is None, reason=common_reason, z=z, models=models,
        ref_hold=ref_hold, cand_hold=cand_hold,
        target='candidate_minus_reference_weighted_seconds_per_passenger')
