"""Native ensemble parity and explicit capacity wiring, without optimizer steps."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

# Establish the same resident SimpleSAC import path as the frozen v75 loader.
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import learning as v75
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_capacity_context_v78_20261027.critic import CapacityContextQEnsemble
from SimpleSAC.transit_v7.networks import TransitQEnsembleV7


def inputs():
    torch.set_num_threads(1)
    base = TransitQEnsembleV7(ensemble_size=2, hidden_dims=(8, 8))
    states = torch.as_tensor(table(.72, count=4)["s"])
    actions = torch.tensor([[-1.], [-.9375], [-.5], [.25]])
    z = torch.tensor([.75, -.25, 0., float("nan")])
    observed = torch.tensor([True, True, True, False])
    return base, states, actions, z, observed


def test_construction_keeps_base_parameters_optimizer_state_and_all_rng():
    base, _, _, _, _ = inputs()
    parameters = tuple(base.parameters())
    weights = deepcopy(base.state_dict())
    optimizer = torch.optim.Adam(parameters)
    # Populate an existing state without taking any optimizer step.
    optimizer.state[parameters[0]] = {"step": torch.tensor(3.),
        "exp_avg": torch.zeros_like(parameters[0]),
        "exp_avg_sq": torch.ones_like(parameters[0])}
    adam = optimizer.state[parameters[0]]
    groups = tuple(optimizer.param_groups[0]["params"])
    torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    candidate = CapacityContextQEnsemble(base)
    assert candidate.base is base
    assert all(old is new for old, new in zip(parameters, candidate.base.parameters()))
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in weights.items())
    assert optimizer.state[parameters[0]] is adam
    assert tuple(optimizer.param_groups[0]["params"]) == groups
    assert all(parameter is not candidate.action_weight for parameter in groups)
    assert candidate.action_weight not in optimizer.state
    assert torch.count_nonzero(candidate.action_weight) == 0
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_rng[0] == current_numpy[0]
    assert np.array_equal(numpy_rng[1], current_numpy[1])
    assert numpy_rng[2:] == current_numpy[2:]


@pytest.mark.parametrize("indices", [None, [1, 0], [1]])
def test_zero_initialization_exact_full_subset_features_min_and_action_gradients(indices):
    base, states, actions, z, observed = inputs()
    candidate = CapacityContextQEnsemble(base)
    subset = None if indices is None else torch.tensor(indices)
    features = candidate.encode_states(states)
    assert torch.equal(features, base.encode_states(states))
    native_action, new_action = actions.clone().requires_grad_(), actions.clone().requires_grad_()
    before = torch.get_rng_state()
    if subset is None:
        native = base(states, native_action)
        actual = candidate(states, new_action, z, observed)
        featured = candidate.forward_features(features, actions, z, observed)
    else:
        native = base.forward_subset(states, native_action, subset)
        actual = candidate.forward_subset(states, new_action, z, observed, subset)
        featured = candidate.forward_features_subset(features, actions, z, observed, subset)
    assert torch.equal(native, actual)
    assert torch.equal(native, featured)
    assert torch.equal(native.min(0).values, actual.min(0).values)
    assert torch.equal(torch.autograd.grad(native.sum(), native_action, retain_graph=True)[0],
                       torch.autograd.grad(actual.sum(), new_action, retain_graph=True)[0])
    assert torch.equal(torch.autograd.grad(native.min(0).values.sum(), native_action)[0],
                       torch.autograd.grad(actual.min(0).values.sum(), new_action)[0])
    assert torch.equal(before, torch.get_rng_state())


def test_nonzero_probe_changes_action_gap_by_context_and_head_without_training():
    base, states, actions, z, observed = inputs()
    candidate = CapacityContextQEnsemble(base)
    with torch.no_grad():
        candidate.action_weight.copy_(torch.tensor([4., -2.]))
    low, high = torch.full_like(actions, -1.), torch.full_like(actions, -.9375)
    native_delta = base(states, high)-base(states, low)
    actual_delta = candidate(states, high, z, observed)-candidate(states, low, z, observed)
    expected = candidate.action_weight[:, None]*torch.tensor([[.75, -.25, 0., 0.]])/32.
    torch.testing.assert_close(actual_delta-native_delta, expected, atol=1e-7, rtol=0)
    subset = torch.tensor([1, 0])
    torch.testing.assert_close(candidate.forward_subset(states, high, z, observed, subset),
        candidate(states, high, z, observed).index_select(0, subset), atol=1e-7, rtol=0)
    context = torch.tensor([1., 1., 1., 1.])
    first = candidate(states, high, context, torch.ones(4, dtype=torch.bool))
    second = candidate(states, high, -context, torch.ones(4, dtype=torch.bool))
    assert torch.all(first[0] > second[0]) and torch.all(first[1] < second[1])


def test_missing_nan_context_bypasses_nonzero_residual_and_preserves_action_gradient():
    base, states, actions, _, _ = inputs()
    candidate = CapacityContextQEnsemble(base)
    with torch.no_grad():
        candidate.action_weight.copy_(torch.tensor([4., -2.]))
    z, observed = torch.full((4,), float("nan")), torch.zeros(4, dtype=torch.bool)
    left, right = actions.clone().requires_grad_(), actions.clone().requires_grad_()
    native, actual = base(states, left), candidate(states, right, z, observed)
    assert torch.equal(native, actual)
    assert torch.equal(torch.autograd.grad(native.sum(), left)[0],
                       torch.autograd.grad(actual.sum(), right)[0])
    with pytest.raises(ValueError, match="boolean"):
        candidate(states, actions, z, torch.zeros(4))
