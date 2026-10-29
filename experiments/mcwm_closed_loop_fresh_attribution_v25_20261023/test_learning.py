"""Catch outcome leakage, hybrid transitions, RNG drift and execution mismatch."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    FIELDS, ExecutionAction, FreshOutcomeReplay, FreshOutcomeTrainer, TransitTrainerConfig)
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def table(offset, count=9):
    s = np.zeros((count, 229), np.float32)
    s[:, 0] = offset + np.arange(count)*.001
    s[:, 1] = 1/16
    s[:, 3:5] = [.01, .02]
    slots = s[:, 5:].reshape(count, 16, 14)
    slots[:, 0, :13] = [1, 0, .25, 1, 1, .5, .5, .1, .2, .1, .1, 1, 1]
    ns = s.copy()
    dt = np.arange(count, dtype=np.float32)+5
    ns[:, 0] += dt/24000
    return dict(s=s, a=np.linspace(-1, 1, count, dtype=np.float32)[:, None],
        ns=ns, r=-7*dt-100*offset, dt=dt, terminal=np.zeros(count, np.float32),
        sample_weight=np.arange(1, count+1, dtype=np.float32))


def make_replay(arm, fresh=None):
    replay = FreshOutcomeReplay(table(.1), table(.4), TransitMDPConfigV7(),
                                np.full(12, 40, np.float32), arm=arm)
    data = table(.7) if fresh is None else fresh
    replay.add_target([SimpleNamespace(**{key: data[key][i] for key in FIELDS.values()})
                       for i in range(len(data["s"]))])
    return replay


def make_trainer(arm, fresh=None):
    torch.set_num_threads(1)
    replay = make_replay(arm, fresh)
    config = TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective="iql", online_backup="h2oplus",
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order="actor-before-critic")
    return FreshOutcomeTrainer(replay, config=config, device="cpu", seed=42,
                               action_transform=ExecutionAction(TransitMDPConfigV7()))


def test_no_fresh_ingestion_never_reads_or_stores_outcomes():
    class VisibleIssue:
        s, a = table(.7)["s"][0], np.array([.3], np.float32)

        def __getattr__(self, name):
            raise AssertionError(f"no_fresh read outcome field: {name}")

    replay = make_replay("no_fresh")
    replay.add_target([VisibleIssue()])
    assert set(replay.recent[-1]) == {"s", "a"}
    replay.sample_mixed(8, .5, rng=np.random.default_rng(12))
    assert replay.summary()["fresh_outcomes_stored"] == 0
    assert replay.target_received == 10


def test_matched_draws_source_and_current_covariates_share_rng_without_outcome_access():
    fresh, ablated = make_replay("fresh"), make_replay("no_fresh")
    left, right = np.random.default_rng(18), np.random.default_rng(18)
    fresh_batch, fresh_source = fresh.sample_mixed(256, .5, rng=left)
    old_batch, old_source = ablated.sample_mixed(256, .5, rng=right)
    assert fresh.last_sampling == ablated.last_sampling
    assert left.bit_generator.state == right.bit_generator.state
    for key in fresh_source:
        torch.testing.assert_close(fresh_source[key], old_source[key], rtol=0, atol=0)
    for key in fresh.last_covariates:
        torch.testing.assert_close(fresh.last_covariates[key], ablated.last_covariates[key], rtol=0, atol=0)
    assert len(fresh_source["actions"]) == 128
    assert len(fresh_batch["actions"]) == len(old_batch["actions"]) == 128
    assert torch.all(fresh_batch["observations"][64:, 0] >= .7)
    assert torch.all(old_batch["observations"][:, 0] < .2)
    # A historical reward paired with a fresh state would violate this equality.
    replacement = ablated.last_sampling["replacement"]
    for batch_key, table_key in FIELDS.items():
        expected = torch.as_tensor(ablated.history[table_key][replacement], dtype=torch.float32)
        expected = expected.reshape_as(old_batch[batch_key][64:])
        torch.testing.assert_close(old_batch[batch_key][64:], expected, rtol=0, atol=0)


def test_actor_temperature_and_value_use_common_fresh_current_covariates():
    fresh, old = make_trainer("fresh"), make_trainer("no_fresh")
    fb, fs = fresh.replay_buffer.sample_mixed(8, .5, rng=fresh.rng)
    ob, os = old.replay_buffer.sample_mixed(8, .5, rng=old.rng)
    torch.testing.assert_close(fresh._actor_states(fb, fs), old._actor_states(ob, os), rtol=0, atol=0)
    torch.testing.assert_close(fresh._temperature_states(fb, fs), old._temperature_states(ob, os), rtol=0, atol=0)
    fresh_value, _, _ = fresh._iql_value_terms(fb)
    old_value, _, _ = old._iql_value_terms(ob)
    torch.testing.assert_close(fresh_value, old_value, rtol=0, atol=0)
    assert torch.all(old._actor_states(ob, os)[2:4, 0] >= .7)
    assert torch.all(ob["observations"][2:4, 0] < .2)


@pytest.mark.parametrize("arm,unchanged", [("no_fresh", True), ("fresh", False)])
def test_only_fresh_arm_responds_to_changed_factual_outcomes(arm, unchanged):
    altered = table(.7)
    altered["r"] += 1e6
    altered["ns"][:, 0] = .95
    left, right = make_trainer(arm), make_trainer(arm, altered)
    torch_state = torch.get_rng_state()
    left.continue_step()
    torch.set_rng_state(torch_state)
    right.continue_step()
    equal = all(torch.equal(value, right.critic.state_dict()[name])
                for name, value in left.critic.state_dict().items())
    assert equal is unchanged
    if unchanged:
        for name in ("actor", "value", "target_critic"):
            assert all(torch.equal(value, getattr(right, name).state_dict()[key])
                       for key, value in getattr(left, name).state_dict().items())
        assert torch.equal(left.log_alpha, right.log_alpha)


def test_actual_update_keeps_optimizer_order_and_full_real_td_coefficient():
    learner = make_trainer("no_fresh")
    order = []
    for name in ("value", "actor", "alpha", "critic"):
        optimizer = getattr(learner, name+"_optimizer")
        original = optimizer.step

        def step(*args, label=name, call=original, **kwargs):
            order.append(label)
            return call(*args, **kwargs)

        optimizer.step = step
    target_update = learner._soft_update_target

    def target():
        order.append("target")
        return target_update()

    learner._soft_update_target = target
    metrics = learner.continue_step(collect_diagnostics=True)
    assert order == ["value", "actor", "alpha", "critic", "target"]
    diagnostic = metrics["td_diagnostic"]
    assert diagnostic["real_scale"] == 1 and diagnostic["model_scale"] == 0
    assert diagnostic["model_batches"] == [] and learner.world is None
    assert learner.total_updates == learner.online_critic_updates == 1


def test_actor_and_bellman_target_compose_F_but_observed_actions_remain_executed():
    learner = make_trainer("fresh")
    states = torch.as_tensor(table(.7)["s"])
    captures = {}
    components_original = learner.actor.soft_action_components

    def components(s):
        values = components_original(s)
        captures["components"] = {key: value.detach().clone() for key, value in values.items()}
        return values

    learner.actor.soft_action_components = components
    target_original = learner.target_critic.forward_features_subset

    def target(features, actions, chosen):
        captures["target_actions"] = actions.detach().clone()
        return target_original(features, actions, chosen)

    learner.target_critic.forward_features_subset = target
    learner._sac_bootstrap(dict(next_observations=states))
    c = captures["components"]
    expected = torch.cat((learner.action_transform(states, c["zero_action"]),
                          learner.action_transform(states, c["positive_action"])))
    torch.testing.assert_close(captures["target_actions"], expected, rtol=0, atol=0)
    critic_original = learner.critic.forward_features

    def critic(features, actions):
        captures["actor_actions"] = actions.detach().clone()
        return critic_original(features, actions)

    learner.critic.forward_features = critic
    learner._update_sac_actor(states)
    c = captures["components"]
    expected = torch.cat((learner.action_transform(states, c["zero_action"]),
                          learner.action_transform(states, c["positive_action"])))
    torch.testing.assert_close(captures["actor_actions"], expected, rtol=0, atol=0)
    batch, _ = learner.replay_buffer.sample_mixed(8, .5, rng=learner.rng)
    ids = learner.replay_buffer.last_sampling["recent"]
    expected = torch.as_tensor(table(.7)["a"][ids])
    torch.testing.assert_close(batch["actions"][2:], expected, rtol=0, atol=0)
