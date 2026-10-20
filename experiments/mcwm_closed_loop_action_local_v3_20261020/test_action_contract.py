"""Detect execution-formula drift, missing composition, and entropy relabelling."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_diagnostic_v2_20261020 import bootstrap
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.learning import (
    ActionConsistentTrainer, ActionConsistentWorldMixin)
from experiments.mcwm_closed_loop_v1.learning import DomainReplay, OnlineWorld, TransitTrainerConfig
from bus_h2o.mdp_contract_v7 import (RAW_EVENT_STATE_DIM, RAW_EVENT_STATE_INDEX as EI,
    TransitMDPConfigV7, pack_event_batch_v7, raw_state_to_canonical_v7)
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action


def state(*, time=2000., forward=360., backward=400., fwd_present=1.,
          bwd_present=1., load=20., demand=.1, current=0):
    event = np.zeros(RAW_EVENT_STATE_DIM, np.float32)
    for name, value in dict(line_id=0, station_index=3, sim_time_seconds=time,
        forward_headway_seconds=forward, backward_headway_seconds=backward,
        target_forward_headway_seconds=360, target_backward_headway_seconds=360,
        bus_capacity=50, onboard_passengers=load, eligible_arrival_rate_per_second=demand,
        base_dwell_seconds=10, forward_bus_present=fwd_present,
        backward_bus_present=bwd_present).items():
        event[EI[name]] = value
    events = np.tile(event, (current+1, 1))
    raw = pack_event_batch_v7(events, current, np.full(current+1, -1., np.float32), TransitMDPConfigV7())
    return raw, raw_state_to_canonical_v7(raw, [40], TransitMDPConfigV7())


def table():
    states = np.stack([state(time=19000.+i*10, current=i%3)[1] for i in range(8)])
    ns = states.copy()
    ns[:, 0] += 1./24000
    return dict(s=states, a=np.zeros((8, 1), np.float32), ns=ns,
                r=-np.arange(1, 9, dtype=np.float32), dt=np.ones(8, np.float32),
                terminal=np.zeros(8, np.float32))


def trainer():
    torch.set_num_threads(1)
    contract = TransitMDPConfigV7()
    replay = DomainReplay(table(), table(), contract, np.full(12, 40, np.float32))
    replay.add_target([SimpleNamespace(**{key: value[0] for key, value in table().items()})])
    config = TransitTrainerConfig(hidden_dim=8, batch_size=8, ensemble_size=2,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        offline_objective="iql", online_backup="h2oplus", online_fraction=.5,
        online_update_order="actor-before-critic")
    return ActionConsistentTrainer(replay, config=config, device="cpu", seed=41,
                                   arm="model", action_transform=ExecutionAction(contract))


def test_native_v19_equivalence_over_supported_unclipped_states_and_slots():
    transform = ExecutionAction(TransitMDPConfigV7())
    raws, states, proposals, expected = [], [], [], []
    for time in (0., 3599., 3600., 12000., 17999., 18000., 23999.):
        for fwd, bwd, load, demand, current in ((1, 1, 20, .1, 0), (1, 0, 5, .1, 2),
             (1, 0, 20, .1, 1), (1, 0, 5, 0, 0), (0, 1, 5, .1, 3)):
            raw, s = state(time=time, fwd_present=fwd, bwd_present=bwd,
                           load=load, demand=demand, current=current)
            for proposal in (-1., -.95, -.5, 0., 1.):
                states.append(s)
                proposals.append([proposal])
                expected.append(completion_safety_action(raw, [proposal], "quarter-load-sixty", TransitMDPConfigV7()))
    actual = transform.numpy(np.stack(states), proposals)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=3e-6)
    np.testing.assert_array_equal(transform.numpy(states[0], proposals[0]), actual[0])


def test_clipped_canonical_state_has_explicit_native_difference_but_one_shared_F():
    raw, canonical = state(forward=2200., backward=2500.)
    transform = ExecutionAction(TransitMDPConfigV7())
    proposal = np.array([0.], np.float32)
    native = completion_safety_action(raw, proposal, "quarter-load-sixty", TransitMDPConfigV7())
    shared = transform.numpy(canonical, proposal)
    assert float(transform.hold_seconds(torch.tensor(native))) == pytest.approx(30.)
    assert float(transform.hold_seconds(torch.tensor(shared))) == pytest.approx(3.75)
    torch.testing.assert_close(transform(torch.tensor(canonical[None]), torch.tensor(proposal[None])),
                               torch.tensor(shared[None]), rtol=0., atol=0.)


@pytest.mark.parametrize("kwargs,proposal,slope", [
    ({"time": 19000.}, 0., .125),
    ({"forward": 360., "backward": 600.}, -.8, 1.),
    ({"forward": 360., "backward": 400.}, 0., 0.),
    ({"time": 9000., "bwd_present": 0, "load": 5.}, 0., 0.),
])
def test_differentiable_rule_matches_execution_piecewise_derivative(kwargs, proposal, slope):
    transform = ExecutionAction(TransitMDPConfigV7())
    s = torch.tensor(state(**kwargs)[1][None])
    u = torch.tensor([[proposal]], requires_grad=True)
    transform(s, u).sum().backward()
    assert float(u.grad) == pytest.approx(slope, abs=1e-6)


def test_target_and_actor_Q_receive_executed_actions_and_keep_latent_entropy():
    item = trainer()
    states = torch.tensor(table()["s"])
    captured = {}
    components_original = item.actor.soft_action_components
    def components(s):
        value = components_original(s)
        captured["components"] = {key: tensor.detach().clone() for key, tensor in value.items()}
        return value
    item.actor.soft_action_components = components
    target_original = item.target_critic.forward_features_subset
    def target(features, actions, chosen):
        captured["target_actions"] = actions.detach().clone()
        value = target_original(features, actions, chosen)
        captured["target_values"] = value.detach().clone()
        return value
    item.target_critic.forward_features_subset = target
    actual = item._sac_bootstrap(dict(next_observations=states))
    c = captured["components"]
    expected_actions = torch.cat((item.action_transform(states, c["zero_action"]),
                                  item.action_transform(states, c["positive_action"])))
    torch.testing.assert_close(captured["target_actions"], expected_actions)
    zero, positive = captured["target_values"].split(len(states), dim=1)
    expected = (c["zero_probability"]*(zero.min(0).values-item.alpha.detach()*c["zero_log_prob"])
        + c["hold_probability"]*(positive.min(0).values-item.alpha.detach()*c["positive_log_prob"]))
    torch.testing.assert_close(actual, expected)
    critic_original = item.critic.forward_features
    def critic(features, actions):
        captured["actor_actions"] = actions.detach().clone()
        return critic_original(features, actions)
    item.critic.forward_features = critic
    item._update_sac_actor(states)
    c = captured["components"]
    expected_actions = torch.cat((item.action_transform(states, c["zero_action"]),
                                  item.action_transform(states, c["positive_action"])))
    torch.testing.assert_close(captured["actor_actions"], expected_actions)
    latent_entropy = -(c["zero_probability"]*c["zero_log_prob"]
                       + c["hold_probability"]*c["positive_log_prob"]).mean()
    torch.testing.assert_close(item._last_actor_diagnostics["train/hybrid_entropy"], latent_entropy)


def test_imagination_and_trust_bootstrap_use_same_executed_action_contract():
    item = trainer()
    class World(ActionConsistentWorldMixin, OnlineWorld):
        pass
    torch.manual_seed(601)
    world = World(item.replay_buffer, seed=41, updates=1)
    world.action_transform = item.action_transform
    captures = {}
    class Trust:
        def weight(self, s, a):
            captures["weighted"] = a.clone()
            return torch.ones(len(s))
        def update(self, s, a, ns, done, errors, policy, n_iters):
            captures["trust_bootstrap"] = policy(ns).detach()
            captures["trust_states"] = ns.clone()
    world.trust = Trust()
    original = world.predict
    def predict(s, a, family=None):
        captures["model_states"], captures["model_action"] = s.clone(), a.clone()
        return original(s, a, family)
    world.predict = predict
    batch, _ = world.generate(item.actor, 8)
    expected = item.action_transform(captures["model_states"],
        item.actor(captures["model_states"], deterministic=True)[0])
    torch.testing.assert_close(batch["actions"], expected)
    torch.testing.assert_close(captures["model_action"], expected)
    torch.testing.assert_close(captures["weighted"], expected)
    world.update_trust(item.actor, 8)
    ns = captures["trust_states"]
    torch.testing.assert_close(captures["trust_bootstrap"],
                               item.action_transform(ns, item.actor(ns, deterministic=True)[0]))


def test_observed_replay_actions_are_not_transformed_twice():
    item = trainer()
    before = item.replay_buffer.history["a"].copy()
    captured = []
    forward = item.critic.forward
    def critic(states, actions):
        captured.append(actions.detach().clone())
        return forward(states, actions)
    item.critic.forward = critic
    result = item.continue_step(collect_diagnostics=True)
    np.testing.assert_array_equal(item.replay_buffer.history["a"], before)
    # Real batches contain the stored zero-normalized (30 s) action; executing
    # that proposal again after 18000 would instead produce -0.875 (3.75 s).
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], torch.zeros((8, 1)), rtol=0., atol=0.)
    assert result["td_diagnostic"]["actor_parameter_change"]["l2"] > 0.
    assert result["td_diagnostic"]["critic_parameter_change"]["l2"] > 0.
