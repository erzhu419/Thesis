"""Native target parity and causal-memory action probes without training drift."""
from copy import deepcopy

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import memory_learner, rows
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


def packet():
    result = rows(.8)
    for index, row in enumerate(result):
        row.sequence, row.line_id = index, '7X'
        row.started_at = float(row.s[0])*24000.
        row.memory_s['observed_at'] = row.started_at
        row.memory_ns['observed_at'] = row.started_at+float(row.dt)
    return result


def nonzero_learner():
    learner = memory_learner()
    with torch.no_grad():
        for critic in (learner.critic, learner.target_critic):
            critic.level_weight.fill_(.03)
            critic.action_weight.fill_(.7)
    return learner


def test_probe_matches_actual_nonzero_memory_target_and_restores_training_rng(monkeypatch):
    learner, factual = nonzero_learner(), packet()
    native_targets = []
    original_target = learner._target

    def target(batch, *, backup):
        result = original_target(batch, backup=backup)
        native_targets.append(result.clone())
        return result

    monkeypatch.setattr(learner, '_target', target)
    before_rng = _rng()
    before_private = {name: deepcopy(getattr(learner, name).bit_generator.state)
        for name in ('rng', 'diagnostic_rng')}
    before_state = deepcopy(_state(learner))
    result = probe(learner, factual, seed=53017)
    assert len(native_targets) == 1
    np.testing.assert_array_equal([row['target'] for row in result], native_targets[0].numpy())
    assert _equal(before_rng, _rng())
    assert _equal(before_state, _state(learner))
    for name, expected in before_private.items():
        assert _equal(expected, getattr(learner, name).bit_generator.state)

    states = torch.tensor(np.asarray([row.s for row in factual]))
    proposals = torch.tensor([-1., -.75, -.5]).repeat(len(factual), 1)
    expected = learner.action_transform(states[:, None, :].expand(-1, 3, -1).reshape(-1, 229),
        proposals.reshape(-1, 1)).reshape(len(factual), 3)
    np.testing.assert_array_equal([row['executed_actions'][:3] for row in result], expected.numpy())
    # F applies to proposals only. The fourth candidate is the archived actual
    # action, which would materially change if F were accidentally applied twice.
    actual = np.asarray([row.a for row in factual]).reshape(-1)
    np.testing.assert_array_equal([row['executed_actions'][3] for row in result], actual)
    twice = learner.action_transform(states, torch.tensor(actual[:, None])).numpy().reshape(-1)
    assert np.any(actual != twice)
    for row in result:
        assert row['unique_fixed_actions'] == 3
        assert row['fixed_holding_span'] == 1.875
        assert row['context_observed_at'] == row['issue']


def test_fixed_weight_zero_context_removes_the_actual_action_memory_advantage():
    learner, factual = nonzero_learner(), packet()
    before = deepcopy(_state(learner))
    result = probe(learner, factual, seed=53019)
    for original, measured in zip(factual, result):
        advantage = np.asarray(measured['advantage_min'])
        zero = np.asarray(measured['zero_context_advantage_min'])
        actions = np.asarray(measured['executed_actions'])
        # Equal residual weights across heads preserve the selected min head.
        # The context-level term cancels; only the executed-action interaction
        # contributes to the fixed-weight action advantage difference.
        interaction = .7*np.sum(original.memory_s['vector'])
        expected = interaction*(actions-actions[0])*.5
        np.testing.assert_allclose(advantage-zero, expected, rtol=2e-5, atol=2e-6)
        assert advantage[2]-zero[2] > .1
    assert _equal(before, _state(learner))
