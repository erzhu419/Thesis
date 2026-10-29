"""Exact conditional Hurdle overlap after the unchanged native execution F.

Structural atoms use their merged Gaussian preimages. On the remaining
one-to-one pieces F's Jacobian cancels in min(d mu_F, d pi_F), so weighted
Gaussian intersections and CDF integrals give the continuous overlap.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import (
    _normal_mass, atom_geometry,
)


def _intervals(geometry):
    return [(-math.inf if row['low'] is None else row['low'],
             math.inf if row['high'] is None else row['high'])
            for row in geometry['positive_intervals']]


def _json_interval(low, high):
    return dict(low=None if low == -math.inf else low,
                high=None if high == math.inf else high)


def _complement(intervals):
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(high, merged[-1][1])
        else:
            merged.append([low, high])
    result, left = [], -math.inf
    for low, high in merged:
        if left < low:
            result.append((left, low))
        left = high
    if left < math.inf:
        result.append((left, math.inf))
    return result


@torch.no_grad()
def geometry_at(transform, execution_state):
    """Enumerate only F flats, zero gate and explicit actor clamp atoms."""
    state = np.asarray(execution_state, dtype=np.float32).reshape(229)
    tensor = torch.as_tensor(state).reshape(1, 229)

    def native(proposal):
        return float(transform(tensor, tensor.new_tensor([[proposal]]))[0, 0])

    epsilon = float(torch.finfo(torch.float32).eps)
    zero = native(-1.)
    first = atom_geometry(transform, state, zero)
    maximum = float(transform.max_hold)
    candidates = {zero, native(-1.+epsilon), native(1.-epsilon)}
    for hold in (first['native_reserve_seconds'], first['native_floor_seconds'], maximum):
        candidates.add(float(2.*torch.tensor(hold, dtype=torch.float32)/maximum-1.))
    atoms = []
    for action in sorted(candidates):
        geometry = first if action == zero else atom_geometry(transform, state, action)
        if geometry['zero_gate_matches'] or geometry['positive_intervals']:
            atoms.append(geometry)
    continuous = _complement([interval for atom in atoms for interval in _intervals(atom)])
    return dict(map_key=[maximum, first['native_early_map'],
                        first['native_reserve_seconds'], first['native_floor_seconds']],
                atoms=atoms, continuous_intervals=[_json_interval(*x) for x in continuous],
                ordinary_rounding_atoms_included=False)


def _mass(parameters, intervals):
    return parameters['positive_probability']*sum(
        _normal_mass(low, high, parameters['mean'], parameters['std'])
        for low, high in intervals)


def _crossings(behavior, target):
    """Roots of log[p_mu phi_mu(z)] - log[p_pi phi_pi(z)]."""
    p, q = behavior['positive_probability'], target['positive_probability']
    if p == 0. or q == 0.:
        return []
    mean, other_mean = behavior['mean'], target['mean']
    std, other_std = behavior['std'], target['std']
    a = .5*(1./other_std**2-1./std**2)
    b = mean/std**2-other_mean/other_std**2
    c = math.log(p/std)-math.log(q/other_std)-.5*mean**2/std**2+.5*other_mean**2/other_std**2
    if a == 0.:
        return [] if b == 0. else [-c/b]
    discriminant = b*b-4.*a*c
    if discriminant < 0.:
        return []
    if discriminant == 0.:
        return [-b/(2.*a)]
    stable = -.5*(b+math.copysign(math.sqrt(discriminant), b))
    return sorted({stable/a, c/stable})


def _log_density(parameters, z):
    probability = parameters['positive_probability']
    if probability == 0.:
        return -math.inf
    return (math.log(probability/parameters['std'])-.5*math.log(2.*math.pi)
            -.5*((z-parameters['mean'])/parameters['std'])**2)


def _continuous_min(behavior, target, intervals):
    if min(behavior['positive_probability'], target['positive_probability']) == 0.:
        return 0., []
    roots, pieces = _crossings(behavior, target), []
    total = 0.
    for low, high in intervals:
        splits = [low]+[root for root in roots if low < root < high]+[high]
        for left, right in zip(splits[:-1], splits[1:]):
            if left == -math.inf:
                point = 0. if right == math.inf else right-max(1., abs(right))
            elif right == math.inf:
                point = left+max(1., abs(left))
            else:
                point = left+.5*(right-left)
            side = 'behavior' if _log_density(behavior, point) <= _log_density(target, point) else 'target'
            parameters = behavior if side == 'behavior' else target
            mass = _mass(parameters, [(left, right)])
            total += mass
            pieces.append(dict(**_json_interval(left, right), limiting_distribution=side,
                               overlap=mass))
    return total, pieces


def _log_c_integral(behavior, target, pieces):
    """Integral of log c under behavior; use standardized truncated moments."""
    probability = behavior['positive_probability']
    if probability == 0.:
        return 0.
    if target['positive_probability'] == 0.:
        return -math.inf if pieces else 0.
    mean, std = behavior['mean'], behavior['std']
    target_mean, target_std = target['mean'], target['std']
    a = .5*(1.-(std/target_std)**2)
    b = -(mean-target_mean)*std/target_std**2
    c = (math.log(target['positive_probability']/probability)+math.log(std/target_std)
         -.5*((mean-target_mean)/target_std)**2)
    result = 0.
    for piece in pieces:
        if piece['limiting_distribution'] != 'target':
            continue
        low = -math.inf if piece['low'] is None else piece['low']
        high = math.inf if piece['high'] is None else piece['high']
        left, right = (low-mean)/std, (high-mean)/std
        phi_left = 0. if not math.isfinite(left) else math.exp(-.5*left*left)/math.sqrt(2.*math.pi)
        phi_right = 0. if not math.isfinite(right) else math.exp(-.5*right*right)/math.sqrt(2.*math.pi)
        zero = _normal_mass(left, right, 0., 1.)
        first = phi_left-phi_right
        second = zero+(left*phi_left if math.isfinite(left) else 0.)-(
            right*phi_right if math.isfinite(right) else 0.)
        result += min(0., probability*(a*second+b*first+c*zero))
    return result


def conditional_overlap(transform, execution_state, behavior_parameters, target_parameters, *,
                        target_transform=None, target_execution_state=None, geometry=None):
    """Return overlap = E_mu[min(1, d pi_F/d mu_F)] at one fixed state.

    Different native execution maps are explicitly unsupported; this diagnostic
    does not align different action preimages or substitute state observations.
    """
    geometry = geometry_at(transform, execution_state) if geometry is None else geometry
    if target_transform is not None or target_execution_state is not None:
        target_geometry = geometry_at(transform if target_transform is None else target_transform,
            execution_state if target_execution_state is None else target_execution_state)
        if target_geometry['map_key'] != geometry['map_key']:
            return dict(status='unsupported', reason='different_execution_maps',
                        same_execution_map=False, total_overlap=None,
                        expected_c_under_behavior=None)
    intervals = [(-math.inf if row['low'] is None else row['low'],
                   math.inf if row['high'] is None else row['high'])
                  for row in geometry['continuous_intervals']]
    atom_overlap, atom_behavior, atom_target, atoms = 0., 0., 0., []
    for atom in geometry['atoms']:
        masses = []
        for parameters in (behavior_parameters, target_parameters):
            masses.append(_mass(parameters, _intervals(atom))+
                          (parameters['zero_probability'] if atom['zero_gate_matches'] else 0.))
        common = min(masses)
        atom_overlap += common
        atom_behavior += masses[0]
        atom_target += masses[1]
        atoms.append(dict(action=atom['logged_action'], sources=atom['atom_sources'],
                          behavior_mass=masses[0], target_mass=masses[1], overlap=common))
    continuous_overlap, pieces = _continuous_min(behavior_parameters, target_parameters, intervals)
    behavior_continuous = _mass(behavior_parameters, intervals)
    target_continuous = _mass(target_parameters, intervals)
    behavior_total = atom_behavior+behavior_continuous
    target_total = atom_target+target_continuous
    total = atom_overlap+continuous_overlap
    # This independent atom expression is the RN expectation on behavior atoms.
    atom_expectation = sum(atom['behavior_mass']*min(1., atom['target_mass']/atom['behavior_mass'])
                          for atom in atoms if atom['behavior_mass'] > 0.)
    expected_c = atom_expectation+continuous_overlap
    expected_log_c = _log_c_integral(behavior_parameters, target_parameters, pieces)
    if behavior_parameters['positive_probability'] > 0. and target_parameters['positive_probability'] == 0. and behavior_continuous > 0.:
        expected_log_c = -math.inf
    for atom in atoms:
        if atom['behavior_mass'] > 0.:
            expected_log_c += atom['behavior_mass']*min(0., math.log(
                atom['target_mass']/atom['behavior_mass'])) if atom['target_mass'] > 0. else -math.inf
    log_overlap = math.log(total) if total > 0. else None
    finite_log = math.isfinite(expected_log_c)
    return dict(status='supported', same_execution_map=True, map_key=geometry['map_key'],
        atom_overlap=atom_overlap, continuous_overlap=continuous_overlap, total_overlap=total,
        expected_c_under_behavior=expected_c, expectation_identity_error=abs(total-expected_c),
        expected_log_c_under_behavior=expected_log_c if finite_log else None,
        expected_log_c_negative_infinity=not finite_log, log_overlap=log_overlap,
        jensen_gap=log_overlap-expected_log_c if finite_log and log_overlap is not None else None,
        behavior_atom_mass=atom_behavior, behavior_continuous_mass=behavior_continuous,
        target_atom_mass=atom_target, target_continuous_mass=target_continuous,
        behavior_mass_total=behavior_total, target_mass_total=target_total,
        normalization_error=max(abs(behavior_total-1.), abs(target_total-1.)),
        atoms=atoms, continuous_pieces=pieces, ordinary_rounding_atoms_included=False,
        conditional_on_recorded_state=True, trajectory_product_expectation=False,
        counterfactual_only=True, old_log_authorized_for_training=False)
