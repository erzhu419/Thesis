"""Detect attribution order bias, gate rounding changes and atom/density mixing."""
import math

import pytest

from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_support import case
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    actor_parameters, execution_measure, ratio_from_measures,
)
from experiments.mcwm_closed_loop_overlap_v41_20261024.attribution import decompose_log_c


def score(config=None, changes=None):
    _, learner, row = case(**(config or {}))
    behavior = actor_parameters(learner, row['s'])
    target = dict(behavior, **(changes or {}))
    return decompose_log_c(learner.action_transform, row['s'], row['a'], behavior, target)


@pytest.mark.parametrize('config', [dict(proposal=0.), dict(proposal=-1.),
                                   dict(time=.3, reserve=5.), dict(time=.3, completion=True)])
def test_same_parameters_are_exact_zero_at_atoms_and_continuous_actions(config):
    result = score(config)
    assert result['attribution_status'] == 'complete'
    assert result['actual_log_ratio'] == result['actual_log_c'] == 0.
    assert all(value == 0. for value in result['shapley_log_c'].values())
    assert all(value == 0. for value in result['shapley_log_ratio'].values())
    assert result['closure_error'] == dict(log_c=0., log_ratio=0.)


def test_gate_only_uses_supplied_probability_pair_without_recomputing_sigmoid():
    # gate=99 is deliberately inconsistent metadata: the supplied .25/.75
    # actor probabilities, not a fresh sigmoid, define the scored distribution.
    result = score(changes=dict(positive_probability=.25, zero_probability=.75, gate=99.))
    expected = math.log(.25/.5)
    assert result['actual_log_c'] == pytest.approx(expected)
    assert result['shapley_log_c'] == pytest.approx(dict(gate=expected, mean=0., std=0.))
    assert result['continuous_fixed_path']['components']['gate'] == pytest.approx(expected)


def test_std_expansion_is_penalized_at_the_old_gaussian_mode():
    result = score(changes=dict(std=2., log_std=math.log(2.)))
    assert result['actual_log_c'] == pytest.approx(-math.log(2.))
    assert result['shapley_log_c'] == pytest.approx(dict(gate=0., mean=0., std=-math.log(2.)))


def test_clipping_interaction_is_shared_across_factor_orders_and_closes():
    # At z=0 the mean shift costs .5, while gate improvement contributes
    # log(1.5).  Clipping means that fixed gate-first attribution would hide
    # gate benefit; Shapley correctly averages both possible relative orders.
    result = score(changes=dict(positive_probability=.75, zero_probability=.25, mean=1.))
    expected_gate = .5 * math.log(1.5)
    assert result['actual_log_c'] == pytest.approx(math.log(1.5) - .5)
    assert result['shapley_log_c']['gate'] == pytest.approx(expected_gate)
    assert result['shapley_log_c']['mean'] == pytest.approx(-.5 + expected_gate)
    assert result['shapley_log_c']['std'] == 0.
    assert abs(result['closure_error']['log_c']) < 1e-15
    assert abs(result['closure_error']['log_ratio']) < 1e-15
    assert abs(result['continuous_fixed_path']['closure_error']) < 1e-15


def test_reserve_atom_uses_all_eight_cdf_mass_corners_and_no_continuous_decomposition():
    _, learner, row = case(time=.3, reserve=5., mean=-1.)
    behavior = actor_parameters(learner, row['s'])
    target = dict(behavior, positive_probability=.25, zero_probability=.75, mean=1., std=2.)
    result = decompose_log_c(learner.action_transform, row['s'], row['a'], behavior, target)
    b = execution_measure(learner.action_transform, row['s'], row['a'], behavior)
    t = execution_measure(learner.action_transform, row['s'], row['a'], target)
    native = ratio_from_measures(b, t)
    assert result['actual_log_ratio'] == native['log_ratio']
    assert result['actual_log_c'] == native['log_c']
    assert len(result['corners']) == 8
    assert all(corner['target_measure_kind'] == 'atom' for corner in result['corners'])
    assert result['continuous_fixed_path'] is None
    assert abs(result['closure_error']['log_c']) < 1e-15


def test_full_floor_atom_eliminates_all_distribution_parameter_effects():
    result = score(dict(time=.3, completion=True),
        dict(positive_probability=.2, zero_probability=.8, mean=2., std=3.))
    assert result['actual_log_c'] == 0.
    assert result['shapley_log_c'] == dict(gate=0., mean=0., std=0.)
    assert result['continuous_fixed_path'] is None


def test_unresolved_atom_underflow_is_null_attribution_instead_of_fake_zero():
    result = score(dict(time=.3, reserve=5., mean=1000.))
    assert result['status'] == 'numerical_underflow'
    assert result['actual_log_c'] is None
    assert result['attribution_status'] == 'unresolved_measure'
    assert result['shapley_log_c'] is None and result['closure_error'] is None


def test_geometry_cache_preserves_exact_point_score():
    _, learner, row = case(proposal=0.)
    behavior = actor_parameters(learner, row['s'])
    target = dict(behavior, mean=1., std=1.5)
    first = decompose_log_c(learner.action_transform, row['s'], row['a'], behavior, target)
    cached = decompose_log_c(learner.action_transform, row['s'], row['a'], behavior, target,
                             geometry=first['geometry'])
    assert first == cached


def test_visible_input_baseline_plus_parameter_shapley_closes_to_old_nominal_denominator():
    _, learner, row = case(proposal=0.)
    old_nominal = actor_parameters(learner, row['s'])
    same_visible = dict(old_nominal, mean=1.)
    target = dict(same_visible, std=2.)
    result = decompose_log_c(learner.action_transform, row['s'], row['a'], same_visible,
                             target, denominator_params=old_nominal)
    assert result['baseline_log_ratio'] == pytest.approx(-.5)
    assert result['baseline_log_c'] == pytest.approx(-.5)
    assert result['actual_log_c'] == pytest.approx(-math.log(2.) - .125)
    assert (result['baseline_log_c'] + math.fsum(result['shapley_log_c'].values())
            == pytest.approx(result['actual_log_c']))
    assert result['shapley_log_c']['mean'] == result['shapley_log_c']['gate'] == 0.
    assert abs(result['continuous_fixed_path']['closure_error']) < 1e-15
