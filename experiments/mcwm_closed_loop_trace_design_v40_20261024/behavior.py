"""Design-only RN ratios for a full stochastic Hurdle execution behavior.

The proposed new behavior is the original 50000-step actor, frozen before the
new prefix, sampled stochastically and passed through the existing F.  Its
stochastic weight is 1: there is no deterministic Dirac mixture component.
Scoring an old logged action under this proposed measure does not change the
behavior that actually generated that old trajectory.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional

from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import (
    _normal_mass, atom_geometry,
)


DESIGN = dict(reference_snapshot='original_50000_initialization',
    stochastic_hurdle_weight=1., deterministic_dirac_weight=0.,
    execution_transform='unchanged_F', counterfactual_old_log_scoring=True,
    old_log_behavior_relabeling=False)


def _state(value):
    return np.asarray(value, dtype=np.float32).reshape(229)


def _action(value):
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


@torch.no_grad()
def actor_parameters(learner, policy_input):
    """Read distribution parameters without sampling or advancing an RNG."""
    states = torch.as_tensor(_state(policy_input), device=learner.device).reshape(1, 229)
    gate, mean, log_std = learner.actor.distribution_parameters(states)
    positive = float(functional.logsigmoid(gate).exp()[0])
    return dict(positive_probability=positive, zero_probability=1.-positive,
        mean=float(mean[0]), std=float(log_std.exp()[0]),
        gate=float(gate[0]), log_std=float(log_std[0]))


def _continuous_preimage(transform, geometry, action):
    """Unique active affine F branch, excluding structural atom locations."""
    maximum = float(transform.max_hold)
    holding = .5*(action+1.)*maximum
    reserve, floor = geometry['native_reserve_seconds'], geometry['native_floor_seconds']
    if not 0. < holding < maximum or holding <= floor:
        return None
    if geometry['native_early_map'] and holding < reserve:
        parent, slope = holding, 1.
    elif not geometry['native_early_map'] or holding > reserve:
        parent, slope = 8.*holding, .125
    else:
        return None
    fraction = parent/maximum
    epsilon = float(torch.finfo(torch.float32).eps)
    if not epsilon*.5 < fraction < 1.-epsilon*.5:
        return None
    return dict(parent_seconds=parent, fraction=fraction, affine_slope=slope,
                gaussian_value=math.log(fraction)-math.log1p(-fraction))


def execution_measure(transform, execution_state, logged_action, parameters, *, geometry=None):
    """Mass at a structural atom, otherwise density in normalized executed a.

    Density includes the positive-branch logistic and active affine Jacobians.
    Ordinary float32 rounding bins are not treated as new structural atoms.
    """
    state, action = _state(execution_state), _action(logged_action)
    geometry = atom_geometry(transform, state, action) if geometry is None else geometry
    if geometry['logged_action'] != action:
        raise ValueError('Atom geometry belongs to another action')
    positive, zero = parameters['positive_probability'], parameters['zero_probability']
    intervals = [(-math.inf if x['low'] is None else x['low'],
                   math.inf if x['high'] is None else x['high'])
                 for x in geometry['positive_intervals']]
    zero_mass = zero if geometry['zero_gate_matches'] else 0.
    positive_mass = positive*sum(_normal_mass(a, b, parameters['mean'], parameters['std'])
                                  for a, b in intervals)
    structural = (zero > 0. and geometry['zero_gate_matches']) or (
        positive > 0. and bool(intervals))
    mass = min(1., max(0., zero_mass+positive_mass))
    result = dict(action=action, geometry=geometry, parameters=parameters,
        structural_atom=bool(structural), atom_mass=mass,
        zero_atom_contribution=zero_mass, positive_atom_contribution=positive_mass,
        ordinary_rounding_atoms_included=False,
        map_key=[float(transform.max_hold), geometry['native_early_map'],
                 geometry['native_reserve_seconds'], geometry['native_floor_seconds']])
    if structural:
        return dict(**result, measure_kind='atom', positive_support=True,
                    log_measure=math.log(mass) if mass > 0. else None,
                    numerical_underflow=mass == 0., continuous_preimage=None)
    preimage = _continuous_preimage(transform, geometry, action)
    if preimage is None or positive <= 0.:
        return dict(**result, measure_kind='outside_support', positive_support=False,
                    log_measure=None, numerical_underflow=False,
                    continuous_preimage=preimage)
    fraction, z = preimage['fraction'], preimage['gaussian_value']
    standard = (z-parameters['mean'])/parameters['std']
    log_density = (math.log(positive)-math.log(parameters['std'])-.5*math.log(2.*math.pi)
        -.5*standard*standard-math.log(2.*preimage['affine_slope'])
        -math.log(fraction)-math.log1p(-fraction))
    return dict(**result, measure_kind='continuous', positive_support=True,
                log_measure=log_density, numerical_underflow=False,
                continuous_preimage=preimage)


def ratio_from_measures(behavior, target):
    """Pointwise ratio of like measure parts; never density/atom mass."""
    same_map = behavior['map_key'] == target['map_key']
    bpar, tpar = behavior['parameters'], target['parameters']
    # Full Hurdle distributions on the same F have the same structural support.
    full_hurdle = all(p['positive_probability'] > 0. and p['zero_probability'] > 0.
                      and p['std'] > 0. for p in (bpar, tpar))
    result = dict(same_execution_map=same_map,
        global_support_certificate=same_map and full_hurdle,
        behavior_measure_kind=behavior['measure_kind'],
        target_measure_kind=target['measure_kind'],
        behavior_log_measure=behavior['log_measure'], target_log_measure=target['log_measure'],
        behavior_atom_mass=behavior['atom_mass'], target_atom_mass=target['atom_mass'])
    if behavior['measure_kind'] == 'outside_support':
        return dict(**result, status='unsupported', reason='outside_proposed_behavior_support',
                    c=0., log_c=None, log_ratio=None, mathematical_point_support=False)
    if behavior['measure_kind'] == 'atom' and target['measure_kind'] != 'atom':
        return dict(**result, status='known_zero_trace', reason='behavior_atom_target_continuous',
                    c=0., log_c=None, log_ratio=None, mathematical_point_support=False)
    if behavior['measure_kind'] == 'continuous' and target['measure_kind'] == 'atom':
        return dict(**result, status='unsupported', reason='target_atom_missing_from_behavior',
                    c=0., log_c=None, log_ratio=None, mathematical_point_support=False)
    if target['measure_kind'] == 'outside_support':
        return dict(**result, status='known_zero_trace', reason='outside_target_support',
                    c=0., log_c=None, log_ratio=None, mathematical_point_support=False)
    if behavior['log_measure'] is None or target['log_measure'] is None:
        return dict(**result, status='numerical_underflow', reason='positive_atom_mass_underflow',
                    c=0., log_c=None, log_ratio=None, mathematical_point_support=True)
    log_ratio = target['log_measure']-behavior['log_measure']
    log_c = min(0., log_ratio)
    coefficient = math.exp(log_c)
    return dict(**result, status='supported', reason='matching_measure_parts',
        c=coefficient, log_c=log_c, log_ratio=log_ratio,
        coefficient_underflow=coefficient == 0., mathematical_point_support=True,
        identical_distribution_at_point=(behavior['log_measure'] == target['log_measure']))


@torch.no_grad()
def rn_at(reference, learner, row, *, behavior_state=None, behavior_execution_state=None,
          behavior_state_known=True, reference_parameters=None, target_parameters=None,
          behavior_geometry=None, target_geometry=None):
    """Counterfactual design score of an old row, not off-policy authorization.

    `behavior_state` is the actual actor observation from the old atomic batch;
    `behavior_execution_state` is the actual F input.  Neither is silently
    replaced by a later post-intervention recorded state.
    """
    row = row.get('transition', row)
    state, action = _state(row['s']), _action(row['a'])
    if not behavior_state_known:
        return dict(status='unsupported', reason='actual_behavior_input_missing', c=0.,
                    log_c=None, mathematical_point_support=False,
                    global_support_certificate=False, counterfactual_only=True,
                    old_log_authorized_for_training=False)
    actual_policy = state if behavior_state is None else _state(behavior_state)
    actual_execution = actual_policy if behavior_execution_state is None else _state(
        behavior_execution_state)
    reference_parameters = actor_parameters(reference, actual_policy) if (
        reference_parameters is None) else reference_parameters
    target_parameters = actor_parameters(learner, state) if target_parameters is None else target_parameters
    behavior = execution_measure(reference.action_transform, actual_execution, action,
                                 reference_parameters, geometry=behavior_geometry)
    target = execution_measure(learner.action_transform, state, action,
                               target_parameters, geometry=target_geometry)
    result = ratio_from_measures(behavior, target)
    return dict(**result, behavior_input_differs=not np.array_equal(actual_policy, state),
        behavior_F_input_differs=not np.array_equal(actual_execution, state),
        behavior_atom_sources=behavior['geometry']['atom_sources'],
        target_atom_sources=target['geometry']['atom_sources'],
        counterfactual_only=True, old_log_authorized_for_training=False)
