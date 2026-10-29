"""Structural atom support after the existing native execution map F.

The logged continuation policy is deterministic, so its measure is a Dirac.
The target is the actual hurdle actor: an exact-zero gate and a clipped
logistic-normal positive branch.  Positive density at a logged point is not
atom mass.  Native float32 is used for action matching; ordinary roundoff
cells are not promoted into structural support.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional


def _record(row):
    return row.get('transition', row) if isinstance(row, dict) else row


def _field(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _scalar(value):
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def _normal_mass(low, high, mean, std):
    """Gaussian interval mass, retaining small upper-tail differences."""
    left, right = (low-mean)/std, (high-mean)/std
    root_two = math.sqrt(2.)
    if left >= 0.:
        value = .5*(math.erfc(left/root_two)-math.erfc(right/root_two))
    elif right <= 0.:
        value = .5*(math.erfc(-right/root_two)-math.erfc(-left/root_two))
    else:
        value = 1.-.5*math.erfc(right/root_two)-.5*math.erfc(-left/root_two)
    return max(0., min(1., value))


def _logit_fraction(fraction):
    if fraction <= 0.:
        return -math.inf
    if fraction >= 1.:
        return math.inf
    return math.log(fraction)-math.log1p(-fraction)


def _merge(intervals):
    merged = []
    for low, high, source in sorted(intervals):
        if low > high:
            continue
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(high, merged[-1][1])
            merged[-1][2].add(source)
        else:
            merged.append([low, high, {source}])
    return merged


@torch.no_grad()
def atom_geometry(transform, state, logged_action):
    """Find only positive-length structural preimages at this logged action.

    Intervals are in the positive branch's pre-sigmoid Gaussian variable.
    F's reserve/floor intervals and the actor's explicit epsilon clipping
    atoms are unioned before integrating, so their overlap is counted once.
    """
    state = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, 229)
    action = _scalar(logged_action)
    maximum = float(transform.max_hold)
    event = state[0, 5:].reshape(16, 14)[int(torch.round(state[0, 2]*15).clamp(0, 15))]
    forward, backward = bool(event[11] > .5), bool(event[12] > .5)
    forward_error = (event[3]-1.)*event[5]*transform.headway_scale
    backward_error = (event[4]-1.)*event[6]*transform.headway_scale
    reserve = float((.5*(backward_error*backward-forward_error*forward)).clamp(0., maximum))
    early = bool(state[0, 0] < .75)
    completion = bool((state[0, 0] >= .15) & (state[0, 0] < .75)) and (
        forward and not backward and bool(event[9] > 0.) and bool(event[8] <= .25))
    floor = min(60., maximum) if completion else 0.

    def native(proposal):
        return float(transform(state, torch.tensor([[proposal]], dtype=torch.float32))[0, 0])

    def normalized_level(hold):
        return float(2.*torch.tensor(hold, dtype=torch.float32)/maximum-1.)

    epsilon = float(torch.finfo(torch.float32).eps)
    min_fraction, max_fraction = epsilon*.5, 1.-epsilon*.5
    clamp_low, clamp_high = _logit_fraction(min_fraction), _logit_fraction(max_fraction)
    intervals, matched = [], []
    zero_action = native(-1.)
    zero_matches = zero_action == action
    if zero_matches:
        matched.append('zero_gate')
    # These are deliberate actor clipping intervals, not floating-point bins.
    for low, high, proposal, source in (
            (-math.inf, clamp_low, -1.+epsilon, 'positive_clamp_low'),
            (clamp_high, math.inf, 1.-epsilon, 'positive_clamp_high')):
        if native(proposal) == action:
            intervals.append((low, high, source))
            matched.append(source)

    flats = []
    if completion:
        # Before the reserve plateau F(p)=p; after it F(p)=p/8.
        ceiling = floor if floor < reserve else 8.*floor
        flats.append((0., min(maximum, ceiling), floor,
                      'cap' if floor == maximum else 'floor'))
    if early and reserve > floor and 0. < reserve < maximum:
        flats.append((reserve, min(maximum, 8.*reserve), reserve, 'reserve'))
    for lower_parent, upper_parent, level, source in flats:
        if upper_parent <= lower_parent or normalized_level(level) != action:
            continue
        low_fraction, high_fraction = lower_parent/maximum, upper_parent/maximum
        # The explicit actor clipping tails are already included above.
        low = _logit_fraction(max(min_fraction, low_fraction))
        high = _logit_fraction(min(max_fraction, high_fraction))
        if high > low:
            # Verify a point strictly inside the claimed native flat piece.
            parent = .5*(max(lower_parent, maximum*min_fraction)
                         + min(upper_parent, maximum*max_fraction))
            if native(2.*parent/maximum-1.) != action:
                raise RuntimeError('The analytic flat interval disagrees with native F')
            intervals.append((low, high, source))
            matched.append(source)

    merged = _merge(intervals)
    # JSON keeps unbounded interval ends explicit rather than non-finite numbers.
    return dict(logged_action=action, zero_execution_action=zero_action,
        zero_gate_matches=zero_matches, atom_sources=sorted(set(matched)),
        positive_intervals=[dict(low=None if low == -math.inf else low,
            high=None if high == math.inf else high, sources=sorted(sources))
            for low, high, sources in merged],
        native_reserve_seconds=reserve, native_floor_seconds=floor,
        native_early_map=early, ordinary_rounding_atoms_included=False)


@torch.no_grad()
def support_at(behavior, learner, row, *, behavior_state=None,
               behavior_state_known=True, geometry=None):
    """Return c=min(1, pi_F({logged a})) for a verified Dirac behavior.

    `behavior_state` overrides only the actual original policy input, needed
    when an intervention changed earlier actions in the same atomic batch.
    If that input cannot be reconstructed, use behavior_state_known=False.
    Target pi and F always see the genuine recorded current state.
    """
    row = _record(row)
    state = np.asarray(_field(row, 's'), dtype=np.float32).reshape(229)
    action = _scalar(_field(row, 'a'))
    if not behavior_state_known:
        return dict(status='unsupported', reason='actual_behavior_policy_input_missing',
            behavior_action_exact=False, target_has_structural_support=False,
            coefficient=0., c=0., pi_atom_mass=None, logged_action=action)
    actual_input = state if behavior_state is None else np.asarray(
        behavior_state, dtype=np.float32).reshape(229)
    proposal = behavior.act(actual_input[None], deterministic=True)[0]
    behavior_action = _scalar(behavior.action_transform.numpy(actual_input, proposal))
    exact = behavior_action == action
    result = dict(logged_action=action, behavior_action=behavior_action,
        behavior_proposal=_scalar(proposal), behavior_action_exact=exact,
        behavior_input_overridden=behavior_state is not None,
        behavior_input_differs=not np.array_equal(actual_input, state), mu_atom_mass=1.)
    if not exact:
        return dict(**result, status='unsupported', reason='logged_action_not_behavior_dirac',
            target_has_structural_support=False, coefficient=0., c=0., pi_atom_mass=None)
    if geometry is None:
        geometry = atom_geometry(learner.action_transform, state, action)
    elif geometry['logged_action'] != action:
        raise ValueError('Atom geometry belongs to another executed action')
    tensor = torch.as_tensor(state, dtype=torch.float32,
                             device=learner.device).reshape(1, 229)
    gate, mean, log_std = learner.actor.distribution_parameters(tensor)
    # Forward sampling uses Bernoulli(exp(logsigmoid(gate))). Its complement,
    # rather than an independently rounded soft-backup weight, is zero mass.
    positive_probability = float(functional.logsigmoid(gate).exp()[0])
    zero_probability = 1.-positive_probability
    mean, std = float(mean[0]), float(log_std.exp()[0])
    normal_intervals = [(-math.inf if interval['low'] is None else interval['low'],
                         math.inf if interval['high'] is None else interval['high'])
                        for interval in geometry['positive_intervals']]
    positive_mass = sum(_normal_mass(low, high, mean, std)
                        for low, high in normal_intervals)
    zero_mass = zero_probability if geometry['zero_gate_matches'] else 0.
    mass = min(1., max(0., zero_mass+positive_probability*positive_mass))
    structural = zero_mass > 0. or (positive_probability > 0. and bool(normal_intervals))
    return dict(**result, status='supported' if structural else 'known_zero_trace',
        reason='structural_atom' if structural else 'continuous_target_at_behavior_dirac',
        target_has_structural_support=structural, coefficient=mass, c=mass,
        mass_underflow=structural and mass == 0.,
        pi_atom_mass=mass, zero_gate_contribution=zero_mass,
        positive_atom_contribution=positive_probability*positive_mass,
        target_hold_probability=positive_probability,
        target_zero_probability=zero_probability,
        soft_backup_zero_weight=float(functional.logsigmoid(-gate).exp()[0]),
        target_positive_mean=mean, target_positive_std=std,
        atom_sources=geometry['atom_sources'], geometry=geometry,
        ordinary_rounding_atoms_included=False)
