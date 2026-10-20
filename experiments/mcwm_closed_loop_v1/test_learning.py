"""Check target-data provenance, prequential scoring, and actual learning updates."""

import numpy as np
import pytest
import torch

from learning import (ClosedLoopTrainer, DomainReplay, OnlineWorld,
                      TransitTrainerConfig, losses, row_table, targets)
from stream import Transition, TransitMDPConfigV7


def table(offset=.1, size=6):
    s = np.zeros((size, 229), np.float32)
    s[:, 0] = offset + np.arange(size)*.001
    s[:, 1] = 1/16
    s[:, 3:5] = [.01, .02]
    slots = s[:, 5:].reshape(size, 16, 14)
    slots[:, 0, :13] = [1, 0, .25, 1, 1, .5, .5, .1, .2, .1, .1, 1, 1]
    ns = s.copy()
    dt = np.arange(size, dtype=np.float32)+5
    ns[:, 0] += dt/24000
    ns[:, 3] += (np.arange(size)+1)/20000
    ns[:, 5+2] = .5
    return dict(s=s, a=np.zeros((size, 1), np.float32), ns=ns, r=-7*dt,
                dt=dt, terminal=np.zeros(size, np.float32))


def transition(data, index=0, sequence=0):
    return Transition(s=data["s"][index], a=data["a"][index], ns=data["ns"][index],
        r=float(data["r"][index]), dt=float(data["dt"][index]), terminal=False,
        episode_id=0, domain="target", started_at=10., completed_at=20., available_at=25.,
        vehicle_id="A", line_id="L", station_id="S", next_vehicle_id="B", sequence=sequence)


@pytest.fixture
def replay():
    torch.set_num_threads(1)
    torch.manual_seed(6201)
    return DomainReplay(table(.1), table(.4), TransitMDPConfigV7(),
                        np.full(12, 4, np.float32))


def test_replay_keeps_target_history_recent_and_source_as_distinct_samples(replay):
    recent = table(.7)
    replay.add_target([transition(recent, index) for index in range(6)])
    target, source = replay.sample_mixed(16, .5, rng=np.random.default_rng(12))
    times = target["observations"][:, 0].numpy()
    assert np.sum((times >= .1) & (times < .2)) == 4
    assert np.sum((times >= .7) & (times < .8)) == 4
    assert torch.all((source["observations"][:, 0] >= .4) & (source["observations"][:, 0] < .5))
    assert replay.target_received == 6


def test_delayed_labels_score_the_saved_prediction_before_any_learning(replay):
    world = OnlineWorld(replay, updates=1)
    row = transition(table(.7))
    world.register(dict(s=row.s, a=row.a, sequence=row.sequence, started_at=row.started_at))
    saved = {family: value.clone() for family, value in world.pending[0]["outputs"].items()}
    y = targets(row_table([row]), world.stations, world.device)
    expected = {family: float(losses(value, y, world.norm).mean()) for family, value in saved.items()}
    with torch.no_grad():
        for model in world.models.values():
            model.head[-1].bias.add_(100.)
    with pytest.raises(ValueError, match="before it was available"):
        world.observe([row], now=24.)
    assert 0 in world.pending and not world.scored and replay.target_received == 0
    logs = world.observe([row], now=25.)
    assert logs[0]["scores"] == pytest.approx(expected)
    assert logs[0]["model_version"] == 0 and not world.pending
    assert replay.target_received == 0  # Scoring alone must not add a label to training.


def test_target_refit_updates_adapters_while_source_prior_stays_frozen(replay):
    world = OnlineWorld(replay, updates=1, context=True)
    source_before = {key: value.clone() for key, value in world.source.state_dict().items()}
    residual_before = {key: value.clone() for key, value in world.models["residual"].head.state_dict().items()}
    replay.add_target([transition(table(.7), index) for index in range(3)])
    world.context["residual"].fill_(2.)
    assert world.refit(1)
    assert world.refits == 1 and world.version == 1
    assert all(torch.equal(value, source_before[key]) for key, value in world.source.state_dict().items())
    assert all(not parameter.requires_grad for parameter in world.source.parameters())
    assert any(not torch.equal(value, residual_before[key])
               for key, value in world.models["residual"].head.state_dict().items())
    assert torch.count_nonzero(world.context["residual"]) == 0


def test_delayed_context_uses_current_model_after_refit(replay):
    world = OnlineWorld(replay, updates=1, context=True)
    row = transition(table(.7))
    world.register(dict(s=row.s, a=row.a, sequence=0, started_at=row.started_at))
    # The outcome arrives after the model has already absorbed a previous shift.
    with torch.no_grad():
        for model in world.models.values():
            model.head[-1].bias[:4].add_(2.)
    world.version += 1
    y = targets(row_table([row]), world.stations, world.device)
    with torch.no_grad():
        expected = {family: .1*((y["g"]-world.norm.g_mean)/world.norm.g_std
                               - model(y["s"], y["a"])[:, :4])[0]
                    for family, model in world.models.items()}
    world.observe([row], now=25.)
    for family in world.models:
        assert torch.allclose(world.context[family], expected[family])
    assert world.scored[0]["model_version"] == 0


def test_available_target_model_samples_reach_critic_and_actor(replay):
    config = TransitTrainerConfig(offline_objective="iql", online_backup="h2oplus",
        use_ratio_correction=False, ensemble_size=2, hidden_dim=32, batch_size=8,
        online_actor_update_interval=1, critic_loss="huber", gradient_clip_norm=10.)
    learner = ClosedLoopTrainer(replay, config=config, seed=6201)
    learner.update(online=False, collect_metrics=False)
    replay.add_target([transition(table(.7), index) for index in range(3)])
    learner.world = OnlineWorld(replay, updates=1)
    actor_before = {key: value.clone() for key, value in learner.actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in learner.critic.state_dict().items()}
    metrics = learner.continue_step()
    assert np.isfinite(metrics["model_td_loss"]) and 0 < metrics["model_trust_mean"] <= 1
    assert learner.total_updates == 2 and learner.online_critic_updates == 1
    assert any(not torch.equal(value, actor_before[key]) for key, value in learner.actor.state_dict().items())
    assert any(not torch.equal(value, critic_before[key]) for key, value in learner.critic.state_dict().items())
