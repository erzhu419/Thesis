"""Detect illegal density ratios and lost/co-counted execution-map atoms."""
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_trace_support_v39_20261024.support import (
    atom_geometry, support_at,
)


class Actor:
    def __init__(self, gate=0., mean=0., log_std=0.):
        self.gate, self.mean, self.log_std = gate, mean, log_std

    def distribution_parameters(self, states):
        return tuple(states.new_full((len(states),), value)
                     for value in (self.gate, self.mean, self.log_std))


class Behavior:
    def __init__(self, transform, proposal):
        self.action_transform, self.proposal = transform, proposal

    def act(self, states, deterministic=False):
        assert deterministic
        value = self.proposal(states) if callable(self.proposal) else self.proposal
        return np.full((len(states), 1), value, dtype=np.float32)


def case(*, time=.8, reserve=0., completion=False, maximum=60., proposal=0.,
         gate=0., mean=0., log_std=0.):
    transform = ExecutionAction(dict(max_hold_seconds=maximum,
                                    scheduled_headway_scale_seconds=720.))
    state = np.zeros(229, np.float32)
    state[0] = time
    event = state[5:].reshape(16, 14)[0]
    event[3:5] = [1., 1.+2.*reserve/720.]
    event[5:7] = 1.
    event[11:13] = [1., 0. if completion else 1.]
    if completion:
        event[9], event[8] = .1, .1
    behavior = Behavior(transform, proposal)
    learner = SimpleNamespace(actor=Actor(gate, mean, log_std),
                              action_transform=transform, device='cpu')
    row = dict(s=state.tolist(), a=transform.numpy(state, [proposal]).tolist())
    return behavior, learner, row


def cdf(value):
    return .5*math.erfc(-value/math.sqrt(2.))


def test_continuous_positive_density_does_not_support_a_dirac_logged_action():
    behavior, learner, row = case()
    result = support_at(behavior, learner, row)
    assert result['behavior_action_exact']
    assert result['status'] == 'known_zero_trace'
    assert result['c'] == result['pi_atom_mass'] == 0.
    assert result['atom_sources'] == []


def test_zero_gate_and_explicit_positive_clip_mass_are_both_counted():
    behavior, learner, row = case(proposal=-1., gate=-.7, mean=-17.)
    result = support_at(behavior, learner, row)
    probability = result['target_hold_probability']
    cutoff = math.log(torch.finfo(torch.float32).eps*.5)-math.log1p(
        -torch.finfo(torch.float32).eps*.5)
    expected = 1.-probability+probability*cdf(cutoff+17.)
    assert result['c'] == pytest.approx(expected, abs=1e-15)
    assert result['atom_sources'] == ['positive_clamp_low', 'zero_gate']
    assert not result['ordinary_rounding_atoms_included']


def test_native_reserve_flat_interval_uses_cdf_mass_not_proposal_density():
    behavior, learner, row = case(time=.3, reserve=5., gate=.2)
    result = support_at(behavior, learner, row)
    reserve = result['geometry']['native_reserve_seconds']
    low = math.log(reserve/(60.-reserve))
    high = math.log(8.*reserve/(60.-8.*reserve))
    expected = result['target_hold_probability']*(cdf(high)-cdf(low))
    assert result['c'] == pytest.approx(expected, abs=1e-15)
    assert result['atom_sources'] == ['reserve']


def test_completion_floor_cap_unions_all_positive_atoms_and_the_zero_gate():
    behavior, learner, row = case(time=.3, completion=True, gate=-.3, mean=17.)
    result = support_at(behavior, learner, row)
    assert result['c'] == 1.
    assert result['geometry']['native_floor_seconds'] == 60.
    assert result['geometry']['positive_intervals'] == [dict(
        low=None, high=None,
        sources=['cap', 'positive_clamp_high', 'positive_clamp_low'])]


def test_explicit_positive_upper_clip_is_real_atom_even_without_flat_f():
    behavior, learner, row = case(proposal=1.-torch.finfo(torch.float32).eps,
                                  gate=.7, mean=17.)
    result = support_at(behavior, learner, row)
    cutoff = math.log((1.-torch.finfo(torch.float32).eps*.5)/(
        torch.finfo(torch.float32).eps*.5))
    assert result['c'] == pytest.approx(
        result['target_hold_probability']*cdf(17.-cutoff), abs=1e-15)
    assert result['atom_sources'] == ['positive_clamp_high']


def test_behavior_policy_override_recovers_atomic_batch_input_without_relabeling():
    behavior, learner, row = case(proposal=-1.)
    behavior.proposal = lambda states: -1. if states[0, 1] == 0. else 0.
    nominal = np.asarray(row['s'], np.float32)
    row['s'][1] = 1.
    mismatch = support_at(behavior, learner, row)
    assert mismatch['status'] == 'unsupported' and mismatch['pi_atom_mass'] is None
    restored = support_at(behavior, learner, row, behavior_state=nominal)
    assert restored['behavior_action_exact'] and restored['behavior_input_differs']
    assert restored['c'] > 0.
    missing = support_at(behavior, learner, row, behavior_state_known=False)
    assert missing['status'] == 'unsupported' and missing['pi_atom_mass'] is None


def test_support_geometry_and_scoring_do_not_advance_native_rng():
    behavior, learner, row = case(time=.3, reserve=5.)
    before = torch.random.get_rng_state().clone()
    geometry = atom_geometry(learner.action_transform, row['s'], row['a'])
    support_at(behavior, learner, row, geometry=geometry)
    assert torch.equal(before, torch.random.get_rng_state())
