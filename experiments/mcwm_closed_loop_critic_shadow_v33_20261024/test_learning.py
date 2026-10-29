"""Exercise native update routing, retained Adam and explicit factual context."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import (
    FIELDS, make_trainer, table)
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024 import learning


def rows(offset=.72, *, zero=False):
    data = table(offset, count=5)
    return [SimpleNamespace(**{field: data[field][index] for field in FIELDS.values()},
                memory_s={"vector": (np.zeros(84) if zero else
                    np.full(84, .1+index*.02)).astype(np.float32).tolist()},
                memory_ns={"vector": (np.zeros(84) if zero else
                    np.full(84, .6+index*.03)).astype(np.float32).tolist()})
            for index in range(5)]


def initial():
    learner = make_trainer("fresh")
    learner.continue_step()  # Real nonempty Adam state must survive attachment.
    learner.replay_buffer.recent.clear()
    learner.replay_buffer.target_received = 0
    return learner


def memory_learner():
    return learning.attach_memory(initial())


def test_sampler_draws_and_current_next_context_alignment():
    native = initial()
    memory = learning.attach_memory(deepcopy(native))
    packet = rows()
    native.replay_buffer.add_target(packet)
    memory.replay_buffer.add_target(packet)
    left, right = np.random.default_rng(15), np.random.default_rng(15)
    factual, source = memory.replay_buffer.sample_mixed(8, .5, rng=left)
    old, old_source = native.replay_buffer.sample_mixed(8, .5, rng=right)
    assert memory.replay_buffer.last_sampling == native.replay_buffer.last_sampling
    assert _equal(left.bit_generator.state, right.bit_generator.state)
    for key in old:
        actual = factual[key][:, :229] if key.endswith("observations") else factual[key]
        assert torch.equal(actual, old[key])
    for key in old_source:
        actual = source[key][:, :229] if key.endswith("observations") else source[key]
        assert torch.equal(actual, old_source[key])
    assert torch.count_nonzero(factual["observations"][:2, 229:]) == 0
    assert torch.count_nonzero(source["observations"][:, 229:]) == 0
    assert torch.count_nonzero(source["next_observations"][:, 229:]) == 0
    ids = memory.replay_buffer.last_sampling["recent"]
    for key, field in (("observations", "memory_s"), ("next_observations", "memory_ns")):
        expected = torch.tensor([getattr(packet[index], field)["vector"] for index in ids])
        assert torch.equal(factual[key][2:, 229:], expected)
    covariates = memory.replay_buffer.last_covariates
    assert covariates["observations"].shape == (4, 229)
    assert torch.equal(covariates["memory_s"], factual["observations"][:, 229:])


def test_zero_memory_native_comparator_is_exact_across_two_complete_packets(monkeypatch):
    origin = initial()

    def load(seed, *, arm):
        learner = deepcopy(origin)
        return learner, learner.replay_buffer, None, None, "resident_initializer"

    monkeypatch.setattr(learning, "load_learner", load)
    zero, replay, _, _, _ = learning.make_learner(6202, arm="zero_memory")
    native = deepcopy(origin)
    assert zero.__class__ is native.__class__
    assert zero.critic.__class__ is native.critic.__class__
    for index in range(2):
        packet = rows(.72+index*.02)
        replay.add_target(packet)
        native.replay_buffer.add_target(packet)
        for step in range(8):
            state = torch.get_rng_state()
            expected = native.continue_step(collect_diagnostics=step == 0)
            end = torch.get_rng_state()
            torch.set_rng_state(state)
            actual = zero.continue_step(collect_diagnostics=step == 0)
            assert _equal(actual, expected)
            assert torch.equal(end, torch.get_rng_state())
            assert _equal(_state(native), _state(zero))
            assert native.replay_buffer.last_sampling == replay.last_sampling
    assert zero.total_updates == origin.total_updates+16


def test_attachment_preserves_native_parameter_objects_adam_and_zero_context_updates():
    native = initial()
    memory = deepcopy(native)
    parameter_ids = [id(value) for value in memory.critic_optimizer.param_groups[0]["params"]]
    native_states = {id(parameter): deepcopy(state) for parameter, state in
                     memory.critic_optimizer.state.items()}
    learning.attach_memory(memory)
    assert not memory.target_critic.training
    assert all(not parameter.requires_grad for parameter in memory.target_critic.parameters())
    assert parameter_ids == [id(value) for value in
        memory.critic_optimizer.param_groups[0]["params"][:-2]]
    assert all(_equal(native_states[id(parameter)], state) for parameter, state in
               memory.critic_optimizer.state.items())
    assert tuple(memory.critic.base.parameters()) == tuple(
        memory.critic_optimizer.param_groups[0]["params"][:-2])
    for packet_index in range(2):
        packet = rows(.72+packet_index*.02, zero=True)
        native.replay_buffer.add_target(packet)
        memory.replay_buffer.add_target(packet)
        state = torch.get_rng_state()
        native.continue_step()
        torch.set_rng_state(state)
        memory.continue_step()
        for name in ("critic", "target_critic"):
            assert _equal(getattr(native, name).state_dict(), getattr(memory, name).base.state_dict())
        for name in ("actor", "value"):
            assert _equal(getattr(native, name).state_dict(), getattr(memory, name).state_dict())
        for name in ("actor_optimizer", "value_optimizer", "alpha_optimizer"):
            assert _equal(getattr(native, name).state_dict(), getattr(memory, name).state_dict())
        for left, right in zip(native.critic.parameters(), memory.critic.base.parameters()):
            assert _equal(native.critic_optimizer.state[left], memory.critic_optimizer.state[right])
        assert torch.count_nonzero(memory.critic.level_weight) == 0
        assert torch.count_nonzero(memory.critic.action_weight) == 0


def test_nonzero_context_routes_td_iql_target_bootstrap_and_actor_q(monkeypatch):
    learner = memory_learner()
    learner.replay_buffer.add_target(rows())
    factual, source = learner.replay_buffer.sample_mixed(8, .5, rng=learner.rng)
    with torch.no_grad():
        for critic in (learner.critic, learner.target_critic):
            critic.level_weight.fill_(.03)
            critic.action_weight.fill_(.7)
    seen = {"actor": [], "value": [], "target_features": [], "actor_features": [], "td": []}
    actor_call = learner.actor.soft_action_components

    def actor(states):
        seen["actor"].append(states.detach().clone())
        assert states.shape[1] == 229
        return actor_call(states)

    monkeypatch.setattr(learner.actor, "soft_action_components", actor)
    value_call = learner.value.forward

    def value(states):
        seen["value"].append(states.detach().clone())
        assert states.shape[1] == 229
        return value_call(states)

    monkeypatch.setattr(learner.value, "forward", value)
    target_call = learner.target_critic.forward_features_subset

    def target(features, actions, indices):
        seen["target_features"].append(features.detach().clone())
        return target_call(features, actions, indices)

    monkeypatch.setattr(learner.target_critic, "forward_features_subset", target)
    state = torch.get_rng_state()
    target_actual = learner._target(factual, backup="h2oplus")
    swapped = dict(factual, next_observations=factual["next_observations"].clone())
    swapped["next_observations"][:, 229:] = 0
    torch.set_rng_state(state)
    target_zero = learner._target(swapped, backup="h2oplus")
    assert not torch.equal(target_actual, target_zero)
    expected_next = factual["next_observations"][:, 229:]
    assert torch.equal(seen["target_features"][0][:, 256:], torch.cat((expected_next, expected_next)))
    assert torch.equal(seen["value"][0], factual["next_observations"][:, :229])
    # The value learner keeps old observations while its target Q reads m_s.
    original = learner.replay_buffer.last_covariates["memory_s"].clone()
    loss, _, _ = learner._iql_value_terms(factual)
    learner.replay_buffer.last_covariates["memory_s"].zero_()
    zero_loss, _, _ = learner._iql_value_terms(factual)
    assert not torch.equal(loss, zero_loss)
    learner.replay_buffer.last_covariates["memory_s"] = original
    critic_call = learner.critic.forward_features

    def actor_q(features, actions):
        seen["actor_features"].append(features.detach().clone())
        return critic_call(features, actions)

    monkeypatch.setattr(learner.critic, "forward_features", actor_q)
    actor_states = learner._actor_states(factual, source)
    learner._update_sac_actor(actor_states)
    expected_current = actor_states[:, 229:]
    assert torch.equal(seen["actor_features"][-1][:, 256:],
                       torch.cat((expected_current, expected_current)))
    learner.critic.register_forward_pre_hook(
        lambda module, inputs: seen["td"].append(inputs[0].detach().clone()))
    learner.continue_step(collect_diagnostics=True)
    combined = seen["td"][-1]
    assert combined.shape == (8, 313)
    assert torch.count_nonzero(combined[:2, 229:]) == 0
    assert torch.count_nonzero(combined[2:4, 229:]) > 0
    assert torch.count_nonzero(combined[4:, 229:]) == 0


def test_two_packets_retain_memory_head_and_adam_and_update_target_heads():
    learner = memory_learner()
    original_parameter = learner.critic.action_weight
    optimizer = learner.critic_optimizer
    last_step = None
    for packet_index in range(2):
        learner.replay_buffer.add_target(rows(.72+packet_index*.02))
        for step in range(8):
            learner.continue_step(collect_diagnostics=step == 0)
        assert learner.critic.action_weight is original_parameter
        assert learner.critic_optimizer is optimizer
        state = optimizer.state[original_parameter]
        assert int(state["step"]) == 8*(packet_index+1)
        if last_step is not None:
            assert not torch.equal(last_step, state["exp_avg"])
        last_step = state["exp_avg"].clone()
        assert torch.count_nonzero(learner.critic.action_weight) > 0
        assert torch.count_nonzero(learner.target_critic.action_weight) > 0


def test_nonzero_action_memory_changes_actor_update_with_identical_actor_observations():
    learner = memory_learner()
    learner.replay_buffer.add_target(rows())
    factual, source = learner.replay_buffer.sample_mixed(8, .5, rng=learner.rng)
    with torch.no_grad():
        learner.critic.action_weight.fill_(4.)
    zero = deepcopy(learner)
    states = learner._actor_states(factual, source)
    zero_states = states.clone()
    zero_states[:, 229:] = 0
    assert torch.equal(states[:, :229], zero_states[:, :229])
    rng = torch.get_rng_state()
    memory_loss, _, _ = learner._update_sac_actor(states)
    torch.set_rng_state(rng)
    zero_loss, _, _ = zero._update_sac_actor(zero_states)
    assert not torch.equal(memory_loss, zero_loss)
    assert not _equal(learner.actor.state_dict(), zero.actor.state_dict())
