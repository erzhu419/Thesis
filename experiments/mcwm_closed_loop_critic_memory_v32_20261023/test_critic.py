"""Test exact native initialization, action response and explicit missing context."""
from copy import deepcopy
import random

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import (
    make_trainer, table)
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.critic import ExecutedMemoryCritic


def inputs():
    learner = make_trainer("fresh")
    data = table(.72, count=4)
    return (learner, torch.as_tensor(data["s"], dtype=torch.float32),
            torch.tensor([[-1.], [-.96875], [-.9375], [0.]], dtype=torch.float32),
            torch.tensor([[1., .5, 0.], [.25, 0., 1.], [0., 1., .5], [1., 1., 1.]]))


def test_zero_initialization_preserves_native_full_subset_and_action_gradients():
    learner, states, actions, memory = inputs()
    origin = learner.critic
    candidate = ExecutedMemoryCritic(origin, memory.shape[1])
    indices = torch.tensor([candidate.ensemble_size-1, 0], dtype=torch.long)
    for subset in (None, indices):
        left = actions.clone().requires_grad_()
        right = actions.clone().requires_grad_()
        if subset is None:
            expected, actual = origin(states, left), candidate(states, right, memory)
        else:
            expected = origin.forward_subset(states, left, subset)
            actual = candidate.forward_subset(states, right, memory, subset)
        assert torch.equal(expected, actual)
        assert torch.equal(torch.autograd.grad(expected.sum(), left)[0],
                           torch.autograd.grad(actual.sum(), right)[0])


def test_missing_source_or_history_context_stays_native_after_residual_updates():
    learner, states, actions, memory = inputs()
    candidate = ExecutedMemoryCritic(learner.critic, memory.shape[1])
    with torch.no_grad():
        candidate.level_weight.fill_(3.25)
        candidate.action_weight.fill_(-8.5)
    missing = torch.zeros_like(memory)
    indices = torch.tensor([candidate.ensemble_size-1, 0], dtype=torch.long)
    assert torch.equal(candidate(states, actions, missing), learner.critic(states, actions))
    assert torch.equal(candidate.forward_subset(states, actions, missing, indices),
                       learner.critic.forward_subset(states, actions, indices))


def test_action_interaction_changes_advantage_and_selected_target_heads():
    learner, states, actions, memory = inputs()
    candidate = ExecutedMemoryCritic(learner.target_critic, memory.shape[1])
    low, high = torch.full_like(actions, -1.), torch.full_like(actions, -.9375)
    with torch.no_grad():
        candidate.level_weight.fill_(2.)
        candidate.action_weight[:, 0] = torch.arange(candidate.ensemble_size)+1.
    native_delta = learner.target_critic(states, high)-learner.target_critic(states, low)
    delta = candidate(states, high, memory)-candidate(states, low, memory)
    expected = (memory[:, 0, None] * candidate.action_weight[:, 0][None, :]).T / 32.
    torch.testing.assert_close(delta-native_delta, expected, atol=2e-5, rtol=0)
    assert torch.any(delta != native_delta)
    indices = torch.tensor([candidate.ensemble_size-1, 0], dtype=torch.long)
    subset_delta = (candidate.forward_subset(states, high, memory, indices)
                    - candidate.forward_subset(states, low, memory, indices))
    torch.testing.assert_close(subset_delta, delta.index_select(0, indices), atol=2e-5, rtol=0)
    # A level-only residual cancels between common-state, common-context actions.
    with torch.no_grad():
        candidate.action_weight.zero_()
    level_delta = candidate(states, high, memory)-candidate(states, low, memory)
    torch.testing.assert_close(level_delta, native_delta, atol=2e-5, rtol=0)


def test_construction_and_probes_preserve_origin_and_global_rng():
    learner, states, actions, memory = inputs()
    origin_before = deepcopy(learner.critic.state_dict())
    torch_before = torch.get_rng_state().clone()
    numpy_before, python_before = np.random.get_state(), random.getstate()
    candidate = ExecutedMemoryCritic(learner.critic, memory.shape[1])
    candidate(states, actions, memory)
    candidate.forward_subset(states, actions, memory,
                             torch.tensor([candidate.ensemble_size-1, 0]))
    with torch.no_grad():
        next(candidate.base.parameters()).add_(1.)
    assert all(torch.equal(value, learner.critic.state_dict()[name])
               for name, value in origin_before.items())
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()
    after_numpy = np.random.get_state()
    assert numpy_before[0] == after_numpy[0]
    assert np.array_equal(numpy_before[1], after_numpy[1])
    assert numpy_before[2:] == after_numpy[2:]
