"""Detect lost F atoms, density/mass mixing and incorrect Gaussian crossings."""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_support import case
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import (
    conditional_overlap, geometry_at,
)


def parameters(positive=.6, mean=0., std=1.):
    return dict(positive_probability=positive, zero_probability=1.-positive,
                mean=mean, std=std)


def evaluate(behavior=None, target=None, **config):
    _, learner, row = case(**config)
    return conditional_overlap(learner.action_transform, row['s'],
        parameters() if behavior is None else behavior,
        parameters() if target is None else target)


@pytest.mark.parametrize('config', [dict(), dict(time=.3, reserve=5.),
    dict(time=.3, reserve=20.), dict(time=.3, completion=True),
    dict(time=.3, completion=True, maximum=120.)])
def test_identical_distributions_normalize_all_real_F_atom_geometries(config):
    before = torch.random.get_rng_state().clone()
    result = evaluate(**config)
    assert result['total_overlap'] == pytest.approx(1., abs=3e-15)
    assert result['normalization_error'] < 3e-15
    assert result['expected_c_under_behavior'] == pytest.approx(1., abs=3e-15)
    assert result['expected_log_c_under_behavior'] == 0.
    assert result['expectation_identity_error'] < 3e-15
    assert not result['ordinary_rounding_atoms_included']
    assert torch.equal(before, torch.random.get_rng_state())


def test_equal_sigma_mean_shift_matches_closed_form_overlap_and_log_moment():
    positive, shift = .7, 2.
    result = evaluate(parameters(positive), parameters(positive, shift))
    expected = 1.-positive+positive*2.*norm.cdf(-shift/2.)
    log_expected = positive*(-shift*norm.pdf(shift/2.)-.5*shift**2*norm.cdf(shift/2.))
    assert result['total_overlap'] == pytest.approx(expected, abs=2e-14)
    assert result['expected_log_c_under_behavior'] == pytest.approx(log_expected, abs=2e-13)
    assert result['jensen_gap'] >= 0.


def test_gate_only_difference_keeps_zero_atom_separate_from_continuous_density():
    result = evaluate(parameters(.2), parameters(.8))
    assert result['total_overlap'] == pytest.approx(.4, abs=2e-14)
    assert result['atom_overlap'] == pytest.approx(.2, abs=2e-14)
    assert result['continuous_overlap'] == pytest.approx(.2, abs=2e-14)
    assert result['expected_log_c_under_behavior'] == pytest.approx(.8*math.log(.25))


def test_sigma_change_uses_both_density_intersections():
    result = evaluate(parameters(.6), parameters(.6, std=2.))
    crossing = math.sqrt(8.*math.log(2.)/3.)
    expected = .4+.6*(2.*norm.cdf(crossing/2.)-1.+2.*norm.sf(crossing))
    assert result['total_overlap'] == pytest.approx(expected, abs=2e-14)
    assert len(result['continuous_pieces']) == 3


def test_full_floor_erases_gate_mean_sigma_differences_and_unions_preimages():
    result = evaluate(parameters(.2, -17., .3), parameters(.8, 17., 2.),
                      time=.3, completion=True)
    assert result['total_overlap'] == 1.
    assert result['atom_overlap'] == 1. and result['continuous_overlap'] == 0.
    assert result['expected_log_c_under_behavior'] == 0.
    assert len(result['atoms']) == 1
    assert result['atoms'][0]['sources'] == [
        'cap', 'positive_clamp_high', 'positive_clamp_low', 'zero_gate']


@pytest.mark.parametrize('config', [dict(time=.3, reserve=5.),
    dict(time=.3, completion=True, maximum=120.), dict(time=.8)])
def test_overlap_and_expected_log_c_match_independent_numerical_integrals(config):
    behavior, target = parameters(.72, -.8, .65), parameters(.43, .6, 1.4)
    _, learner, row = case(**config)
    geometry = geometry_at(learner.action_transform, row['s'])
    result = conditional_overlap(learner.action_transform, row['s'], behavior, target,
                                 geometry=geometry)
    overlap = log_moment = 0.
    for atom in geometry['atoms']:
        masses = []
        for distribution in (behavior, target):
            mass = distribution['zero_probability'] if atom['zero_gate_matches'] else 0.
            for interval in atom['positive_intervals']:
                low = -math.inf if interval['low'] is None else interval['low']
                high = math.inf if interval['high'] is None else interval['high']
                mass += distribution['positive_probability']*(
                    norm.cdf(high, distribution['mean'], distribution['std'])-
                    norm.cdf(low, distribution['mean'], distribution['std']))
            masses.append(mass)
        overlap += min(masses)
        if masses[0] and masses[1]:
            log_moment += masses[0]*min(0., math.log(masses[1]/masses[0]))
    for interval in geometry['continuous_intervals']:
        low = -40. if interval['low'] is None else interval['low']
        high = 40. if interval['high'] is None else interval['high']
        def density(distribution, z):
            return distribution['positive_probability']*norm.pdf(
                z, distribution['mean'], distribution['std'])
        def log_density(distribution, z):
            return math.log(distribution['positive_probability'])+norm.logpdf(
                z, distribution['mean'], distribution['std'])
        overlap += quad(lambda z: min(density(behavior, z), density(target, z)),
                        low, high, epsabs=2e-12, epsrel=2e-12, limit=100)[0]
        log_moment += quad(lambda z: density(behavior, z)*min(
            0., log_density(target, z)-log_density(behavior, z)),
            low, high, epsabs=2e-12, epsrel=2e-12, limit=100)[0]
    assert result['total_overlap'] == pytest.approx(overlap, abs=2e-10)
    assert result['expected_log_c_under_behavior'] == pytest.approx(log_moment, abs=2e-10)
    assert result['jensen_gap'] >= -3e-15
    assert result['normalization_error'] < 3e-15


def test_different_F_is_explicitly_unsupported_not_silently_aligned():
    _, learner, row = case(time=.3, reserve=5.)
    other = np.asarray(row['s'], dtype=np.float32)
    other[5+4] = 1.
    result = conditional_overlap(learner.action_transform, row['s'], parameters(),
                                 parameters(), target_execution_state=other)
    assert result['status'] == 'unsupported'
    assert result['reason'] == 'different_execution_maps'
    assert not result['same_execution_map'] and result['total_overlap'] is None


def test_zero_target_continuous_gate_reports_infinite_log_loss_without_nan():
    result = evaluate(parameters(.6), parameters(0.))
    assert result['total_overlap'] == pytest.approx(.4)
    assert result['expected_log_c_under_behavior'] is None
    assert result['expected_log_c_negative_infinity']
