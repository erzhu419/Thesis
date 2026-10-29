"""Detect density/mass confusion, missed F Jacobians and fake unit ratios."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_support import case, Actor
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    DESIGN, actor_parameters, execution_measure, ratio_from_measures, rn_at,
)


def test_same_hurdle_and_F_have_exact_unit_ratios_at_atoms_and_continuous_points():
    for config in (dict(proposal=-1.), dict(proposal=0.),
                   dict(time=.3, reserve=5.), dict(time=.3, completion=True)):
        _, learner, row = case(**config)
        result = rn_at(learner, learner, row)
        assert result['c'] == 1. and result['log_c'] == result['log_ratio'] == 0.
        assert result['global_support_certificate']
        assert result['counterfactual_only'] and not result['old_log_authorized_for_training']
    assert DESIGN['stochastic_hurdle_weight'] == 1.
    assert DESIGN['deterministic_dirac_weight'] == 0.


def test_compressed_continuous_density_includes_inverse_F_jacobian():
    _, learner, row = case(time=.8, proposal=0., gate=0.)
    params = actor_parameters(learner, row['s'])
    measure = execution_measure(learner.action_transform, row['s'], row['a'], params)
    # z=0, f=.5, positive probability=.5 and dy/du=.125.
    expected_density = .5/math.sqrt(2.*math.pi)/(2.*.125*.5*.5)
    assert measure['measure_kind'] == 'continuous'
    assert math.exp(measure['log_measure']) == pytest.approx(expected_density)


def test_identity_piece_below_reserve_has_no_factor_eight():
    _, learner, row = case(time=.3, reserve=20., proposal=-.75)
    params = actor_parameters(learner, row['s'])
    measure = execution_measure(learner.action_transform, row['s'], row['a'], params)
    fraction = .125
    z = math.log(fraction/(1.-fraction))
    expected = .5*math.exp(-.5*z*z)/math.sqrt(2.*math.pi)/(2.*fraction*(1.-fraction))
    assert measure['continuous_preimage']['affine_slope'] == 1.
    assert math.exp(measure['log_measure']) == pytest.approx(expected)


def test_reserve_atom_uses_CDF_mass_ratio_instead_of_latent_density():
    _, reference, row = case(time=.3, reserve=5., mean=-1.)
    _, target, _ = case(time=.3, reserve=5., mean=1.)
    result = rn_at(reference, target, row)
    assert result['behavior_measure_kind'] == result['target_measure_kind'] == 'atom'
    assert result['c'] == pytest.approx(min(1., result['target_atom_mass']/result['behavior_atom_mass']))
    assert result['global_support_certificate']


def test_changed_actor_input_is_scored_not_forced_to_one():
    _, reference, row = case(proposal=0.)
    class ConditionalActor(Actor):
        def distribution_parameters(self, states):
            gate, mean, std = super().distribution_parameters(states)
            return gate, mean+states[:, 1], std
    reference.actor = ConditionalActor()
    actual = np.asarray(row['s'], np.float32)
    actual[1] = 1.
    result = rn_at(reference, reference, row, behavior_state=actual)
    assert result['behavior_input_differs']
    assert result['log_ratio'] == pytest.approx(.5)
    assert not result['identical_distribution_at_point']


def test_distinct_F_atoms_are_not_authorized_by_positive_continuous_density():
    _, learner, row = case(time=.3, reserve=5.)
    actual = np.asarray(row['s'], np.float32)
    actual[5+4] = 1.  # Actual behavior F has no reserve atom at target's 5 s.
    result = rn_at(learner, learner, row, behavior_state=actual)
    assert result['behavior_measure_kind'] == 'continuous'
    assert result['target_measure_kind'] == 'atom'
    assert result['status'] == 'unsupported'
    assert result['reason'] == 'target_atom_missing_from_behavior'
    assert not result['global_support_certificate']


def test_continuous_log_ratio_retains_extremely_small_but_positive_support():
    _, reference, row = case(proposal=0.)
    _, target, _ = case(proposal=0., mean=50.)
    result = rn_at(reference, target, row)
    assert result['mathematical_point_support'] and result['global_support_certificate']
    assert result['log_c'] == pytest.approx(-1250.) and result['c'] == 0.
    assert result['coefficient_underflow']


def test_full_floor_is_unit_mass_for_distinct_actors_and_does_not_advance_rng():
    _, reference, row = case(time=.3, completion=True, mean=-1.)
    _, target, _ = case(time=.3, completion=True, mean=1.)
    before = torch.random.get_rng_state().clone()
    result = rn_at(reference, target, row)
    assert result['behavior_atom_mass'] == result['target_atom_mass'] == 1.
    assert result['c'] == 1.
    assert torch.equal(before, torch.random.get_rng_state())


def test_missing_actual_policy_input_cannot_be_scored_or_relabel_old_data():
    _, learner, row = case()
    result = rn_at(learner, learner, row, behavior_state_known=False)
    assert result['status'] == 'unsupported' and not result['global_support_certificate']
    assert not result['old_log_authorized_for_training']
