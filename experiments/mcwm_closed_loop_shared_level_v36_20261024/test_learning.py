"""Exact old controls, shared-level invariance and retained native Adam state."""
from copy import deepcopy

import pytest
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import attach_memory
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state
from experiments.mcwm_closed_loop_shared_level_v36_20261024 import learning


def factory(origin):
    def create(seed, *, arm):
        learner = deepcopy(origin)
        if arm == "memory":
            attach_memory(learner)
        learner.memory_arm = arm
        return learner, learner.replay_buffer, None, None, "same_native_initializer"
    return create


@pytest.mark.parametrize("arm", learning.ARMS)
def test_initial_native_full_subset_encoder_feature_parity_and_no_rng_draws(arm, monkeypatch):
    origin = initial()
    monkeypatch.setattr(learning.v33, "make_learner", factory(origin))
    before = _rng()
    learner, replay, _, _, _ = learning.make_learner(6202, arm=arm)
    assert _equal(before, _rng())
    before_state = deepcopy(_state(learner))
    proof = learning.initial_preflight(learner)
    assert proof["all_pass"] and proof["rows"] == 6
    assert _equal(before, _rng())
    assert _equal(before_state, _state(learner))
    assert replay.target_received == 0
    if arm == "shared_level":
        assert proof["level_shape"] == [1, 84]
        assert proof["action_shape"] == [2, 84]
        assert all(learner.shared_level_attachment.values())
    elif arm == "per_head_level":
        assert proof["level_shape"] == proof["action_shape"] == [2, 84]
    else:
        assert proof["level_shape"] is None


@pytest.mark.parametrize("arm,legacy", [("zero_memory", "zero_memory"), ("per_head_level", "memory")])
def test_original_two_controls_have_exact_complete_native_updates(arm, legacy, monkeypatch):
    origin = initial()
    create = factory(origin)
    monkeypatch.setattr(learning.v33, "make_learner", create)
    expected = create(6202, arm=legacy)[0]
    actual = learning.make_learner(6202, arm=arm)[0]
    assert _equal(_state(expected), _state(actual))
    for packet_index in range(2):
        packet = rows(.72+packet_index*.02)
        expected.replay_buffer.add_target(packet)
        actual.replay_buffer.add_target(packet)
        for step in range(8):
            rng = torch.get_rng_state()
            old_metrics = expected.continue_step(collect_diagnostics=step == 0)
            end = torch.get_rng_state()
            torch.set_rng_state(rng)
            metrics = actual.continue_step(collect_diagnostics=step == 0)
            assert _equal(old_metrics, metrics)
            assert torch.equal(end, torch.get_rng_state())
            assert _equal(_state(expected), _state(actual))
            assert expected.replay_buffer.last_sampling == actual.replay_buffer.last_sampling


def test_shared_level_cannot_change_same_state_head_competition_or_action_advantage():
    learner = learning.attach_shared_level(initial())
    state = torch.tensor(rows()[0].s)[None].expand(2, -1)
    memory = torch.linspace(0., 1., 84)[None].expand(2, -1)
    explicit = learning.critic_states(learner, state, memory)
    actions = torch.tensor([[-1.], [-.9375]])
    with torch.no_grad():
        learner.critic.action_weight[0].fill_(.2)
        learner.critic.action_weight[1].fill_(-.1)
        without_level = learner.critic(explicit, actions)
        learner.critic.level_weight.fill_(.03)
        with_level = learner.critic(explicit, actions)
        assert torch.equal(with_level.argmin(0), without_level.argmin(0))
        tolerance = 4.*torch.finfo(with_level.dtype).eps*max(1., float(with_level.abs().max()))
        torch.testing.assert_close(with_level[:, 1]-with_level[:, 0],
            without_level[:, 1]-without_level[:, 0], rtol=0, atol=tolerance)
        torch.testing.assert_close(with_level.min(0).values.diff(), without_level.min(0).values.diff(),
                                   rtol=0, atol=tolerance)
        # A subset containing head1 must broadcast shared level, never index its
        # singleton dimension using that head index.
        indices = torch.tensor([1])
        subset = learner.critic.forward_subset(explicit, actions, indices)
        features = learner.critic.encode_states(explicit)
        feature_subset = learner.critic.forward_features_subset(features, actions, indices)
        torch.testing.assert_close(subset, with_level[1:2], rtol=0, atol=tolerance)
        assert torch.equal(subset, feature_subset)


def test_two_native_packets_retain_native_parameters_old_adam_and_learn_shared_heads():
    learner = initial()
    parameters = tuple(learner.critic.parameters())
    old_states = {parameter: state for parameter, state in learner.critic_optimizer.state.items()}
    actor, value, F, optimizer = learner.actor, learner.value, learner.action_transform, learner.critic_optimizer
    learning.attach_shared_level(learner)
    assert all(left is right for left, right in zip(parameters, learner.critic.base.parameters()))
    assert all(optimizer.state[parameter] is state for parameter, state in old_states.items())
    assert learner.actor is actor and learner.value is value and learner.action_transform is F
    assert learner.critic_optimizer is optimizer
    shared, action = learner.critic.level_weight, learner.critic.action_weight
    assert shared not in optimizer.state and action not in optimizer.state
    last_moment = None
    for packet_index in range(2):
        learner.replay_buffer.add_target(rows(.72+packet_index*.02))
        for step in range(8):
            learner.continue_step(collect_diagnostics=step == 0)
        assert learner.critic.level_weight is shared and learner.critic.action_weight is action
        for parameter in (shared, action):
            assert int(optimizer.state[parameter]["step"]) == 8*(packet_index+1)
            assert torch.count_nonzero(parameter) > 0
        if last_moment is not None:
            assert not torch.equal(last_moment, optimizer.state[shared]["exp_avg"])
        last_moment = optimizer.state[shared]["exp_avg"].clone()
        assert learner.target_critic.level_weight.shape == (1, 84)
        assert torch.count_nonzero(learner.target_critic.level_weight) > 0
        assert all(not p.requires_grad for p in learner.target_critic.parameters())
    assert learner.total_updates == 17
