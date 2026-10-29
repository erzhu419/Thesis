"""Two fixed, independent cost-advantage fits using only the six v84 pairs."""
from __future__ import annotations

import numpy as np

LINES = ('7X', '102X', '705X')
SCENARIOS = ('stable', 'local_speed_drop')
DENOMINATOR = 15079.
RIDGE = 1.
HOLD_SCALE = 7.5
FEATURE_NAMES = tuple('line_'+line for line in LINES)+('headway_response',)


def state_feature(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (229,):
        raise ValueError('D uses exactly the229 issue-visible canonical values')
    if not np.isfinite(state).all():
        return None
    index = int(round(15*float(state[2])))
    if not 0 <= index < 16:
        raise ValueError('Canonical current slot is outside the16-slot state')
    event = state[5:].reshape(16, 14)[index]
    forward = float(event[11] > .5)
    backward = float(event[12] > .5)
    return float(np.clip(.5*((event[4]-1.)*backward-(event[3]-1.)*forward), -1., 1.))


def design_row(line_id, state, ref_hold, cand_hold):
    """The action difference makes the reference action an exact zero anchor."""
    if line_id not in LINES:
        raise ValueError('Only the three frozen lines are supported')
    z = state_feature(state)
    if z is None:
        return None
    delta = (float(cand_hold)-float(ref_hold))/HOLD_SCALE
    return delta*np.asarray([float(line_id == line) for line in LINES]+[z], dtype=np.float64)


def _support_reason(ref_hold, cand_hold):
    if not np.isfinite([ref_hold, cand_hold]).all():
        return 'nonfinite_holding'
    if cand_hold <= ref_hold:
        return 'nonpositive_holding_difference'
    if not 3.0 <= ref_hold <= 3.6:
        return 'reference_holding_outside_3_to_3_6'
    if abs(cand_hold-HOLD_SCALE) > 1e-5:
        return 'candidate_holding_outside_7_5'
    return None


def _ordered_records(records):
    records = list(records)
    if len(records) != 6:
        raise ValueError('Exactly six fixed v84 training pairs are required')
    cases = []
    for row in records:
        if row['source_scope'] != 'v84_train':
            raise ValueError('Heldout/new labels cannot enter the v84-only fit')
        if row['label_available_at'] != 24030:
            raise ValueError('The v84 remaining-day label matures at24030')
        cases.append((row['scenario'], row['line_id']))
    expected = {(scenario, line) for scenario in SCENARIOS for line in LINES}
    if len(set(cases)) != 6 or set(cases) != expected:
        raise ValueError('Training must contain each fixed scene/line exactly once')
    return sorted(records, key=lambda row: (SCENARIOS.index(row['scenario']), LINES.index(row['line_id'])))


def train(records):
    """Solve sum squared error plus ||coef||² once for each fixed model.

    The target is a paired physical cost difference. State cost levels, SAC
    targets and future heldout labels are not accepted by this interface.
    """
    rows = _ordered_records(records)
    inputs = []
    target = []
    support = []
    for row in rows:
        ref_hold, cand_hold = float(row['ref_hold']), float(row['cand_hold'])
        support.append(_support_reason(ref_hold, cand_hold))
        inputs.append(design_row(row['line_id'], row['state'], ref_hold, cand_hold))
        target.append(float(row['target_delta_person_seconds'])/DENOMINATOR)
    finite = all(row is not None and np.isfinite(row).all() for row in inputs)
    finite = bool(finite and np.isfinite(target).all())
    action_support = all(reason is None for reason in support)
    models = {name:dict(coef=None, design_rank=None, train_fit=None) for name in ('line', 'state')}
    fits = 0
    if finite and action_support:
        matrix = np.asarray(inputs, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
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
        finite_coefficients=all(model['coef'] is not None and np.isfinite(model['coef']).all()
                                for model in models.values()),
        two_fixed_solves=fits == 2)
    return dict(name='v85_independent_cost_advantage', models=models,
        feature_names=list(FEATURE_NAMES), ridge=RIDGE, objective='sum_squared_error_plus_ridge_norm_squared',
        target='(G_candidate-G_reference)/15079', denominator=DENOMINATOR, hold_scale=HOLD_SCALE,
        fits=fits, gate=dict(all_pass=all(checks.values()), checks=checks),
        training_scope=dict(source_scope='v84_train', pairs=6, physical_outcome_labels=12,
            source_seed=6207, external_seed=106207, label_available_at=24030,
            cases=[row['scenario']+'/'+row['line_id'] for row in rows],
            heldout_labels_used=0, support_reasons=support),
        trained_components=['independent_D_line', 'independent_D_state'],
        actor_updates=0, critic_updates=0, value_updates=0, B_updates=0, world_updates=0, trust_updates=0)


def predict(artifact, line_id, state, ref_hold, cand_hold):
    """Return a choice only inside the frozen fitted action support."""
    ref_hold, cand_hold = float(ref_hold), float(cand_hold)
    z = state_feature(state)
    reason = None
    if not artifact['gate']['all_pass']:
        reason = 'training_gate_hold'
    elif line_id not in LINES:
        reason = 'undeclared_line'
    elif z is None:
        reason = 'nonfinite_state'
    else:
        reason = _support_reason(ref_hold, cand_hold)
    models = {name:dict(predicted_delta=None, choice='reference') for name in ('line', 'state')}
    if reason is None:
        row = design_row(line_id, state, ref_hold, cand_hold)
        for name, columns in (('line', 3), ('state', 4)):
            value = float(row[:columns]@np.asarray(artifact['models'][name]['coef'], dtype=np.float64))
            models[name] = dict(predicted_delta=value, choice='candidate' if value < 0. else 'reference')
    return dict(supported=reason is None, reason=reason, z=z, models=models,
        ref_hold=ref_hold, cand_hold=cand_hold, target='candidate_minus_reference_weighted_seconds_per_passenger')
