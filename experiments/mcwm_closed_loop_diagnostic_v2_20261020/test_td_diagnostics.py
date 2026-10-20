"""Verify exact objectives, mixed-gradient geometry, and observational invariance."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from td_diagnostics import DiagnosticTrainer, cosine, gradient_report
from learning import ClosedLoopTrainer, DomainReplay, TransitTrainerConfig
from stream import TransitMDPConfigV7


def table():
    states = np.zeros((8, 229), np.float32)
    states[:, 0], states[:, 1], states[:, 5] = .1, 1 / 16, 1
    states[:, 3] = np.arange(8) / 20000
    return dict(s=states, a=np.zeros((8, 1), np.float32), ns=states.copy(),
                r=-np.arange(1, 9, dtype=np.float32), dt=np.ones(8, np.float32),
                terminal=np.zeros(8, np.float32))


class FixedWorld:
    seed = 7201

    def __init__(self, replay):
        self.replay, self.trust_calls, self.generate_calls = replay, 0, 0

    def update_trust(self, actor, size):
        self.trust_calls += 1
        torch.rand(3)  # The genuine model path consumes RNG inside v1's fork.

    @torch.no_grad()
    def generate(self, actor, size):
        self.generate_calls += 1
        states = torch.as_tensor(self.replay.source["s"][:size])
        actions = actor(states, deterministic=True)[0]
        return dict(observations=states, actions=actions, next_observations=states.clone(),
                    rewards=-torch.arange(10., 10. + size), delta_t_seconds=torch.full((size,), 2.),
                    discounts=torch.ones(size), terminals=torch.zeros(size)), torch.linspace(.05, .95, size)


def trainer(arm="model"):
    torch.set_num_threads(1)
    replay = DomainReplay(table(), table(), TransitMDPConfigV7(), np.full(12, 40, np.float32))
    replay.add_target([SimpleNamespace(**{key: value[0] for key, value in table().items()})])
    config = TransitTrainerConfig(hidden_dim=8, batch_size=8, ensemble_size=2,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        offline_objective="iql", online_backup="h2oplus",
        online_fraction=.5, online_update_order="actor-before-critic")
    result = DiagnosticTrainer(replay, config=config, device="cpu", seed=41, arm=arm)
    result.world = FixedWorld(replay)
    return result


@pytest.mark.parametrize("arm,expected", [("real1", 1.), ("real075", .75), ("model", .75), ("context", .75)])
def test_objective_and_gradient_retain_absolute_confidence(arm, expected):
    item = DiagnosticTrainer.__new__(DiagnosticTrainer)
    item.arm, item._collect_td = arm, False
    item.config = SimpleNamespace(critic_loss="mse", bootstrap_probability=1.)
    parameter = torch.tensor(2., requires_grad=True)
    real_prediction = (parameter * torch.tensor([1., 2.]))[None]
    target = torch.zeros(2)
    raw_real = real_prediction.square().mean()
    # Zero trust removes the model term, while retaining the .75 real coefficient.
    item.model_loss = parameter.square() * 0. if arm in ("model", "context") else None
    actual = item._ensemble_td_loss(real_prediction, target)
    expected_loss = expected * raw_real
    torch.testing.assert_close(actual, expected_loss)
    actual_gradient, = torch.autograd.grad(actual, parameter, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected_loss, parameter)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_small_trust_is_not_normalized_away_in_the_model_objective():
    item = DiagnosticTrainer.__new__(DiagnosticTrainer)
    item.arm, item._collect_td = "model", False
    item.config = SimpleNamespace(critic_loss="mse", bootstrap_probability=1.)
    parameter = torch.tensor(2., requires_grad=True)
    predictions = parameter * torch.tensor([[1., 2.]])
    model_errors = (parameter * torch.tensor([3., 4.])).square()
    trust = torch.tensor([.01, .09])
    item.model_loss = (model_errors * trust).mean()
    actual = item._ensemble_td_loss(predictions, torch.zeros(2))
    expected = .75 * predictions.square().mean() + .25 * (model_errors * trust).mean()
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual, parameter, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, parameter)
    torch.testing.assert_close(gradient, expected_gradient)


def test_geometry_uses_total_mixed_real_gradient_and_scaled_weighted_model():
    offline, source = torch.tensor([10., 0.]), torch.tensor([0., 20.])
    real = .75 * offline + .25 * source
    model = torch.tensor([-4., 3.])
    result = gradient_report(real, model, real_scale=.75, model_scale=.25)
    assert result["real_raw_norm"] == pytest.approx(float(real.norm()))
    assert result["real_scaled_norm"] == pytest.approx(float((.75 * real).norm()))
    assert result["model_scaled_norm"] == pytest.approx(1.25)
    assert result["cosine_real_model"] == pytest.approx(float(torch.dot(real, model) / real.norm() / model.norm()))
    assert result["combined_norm"] == pytest.approx(float((.75 * real + .25 * model).norm()))


def test_long_float32_gradient_cosine_is_bounded_and_scaled_direction_is_one():
    gradient = torch.linspace(-.02, .02, 600000, dtype=torch.float32)
    same = cosine(gradient, .75 * gradient)
    opposite = cosine(gradient, -.75 * gradient)
    assert -1. <= same <= 1.
    assert same == pytest.approx(1., abs=1e-12)
    assert -1. <= opposite <= 1.
    assert opposite == pytest.approx(-1., abs=1e-12)
    assert cosine(gradient, torch.zeros_like(gradient)) is None


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("arm", ["real1", "real075", "model", "context"])
def test_diagnostics_do_not_change_rng_updates_parameters_or_optimizer_state(arm):
    observed = trainer(arm)
    ordinary = copy.deepcopy(observed)
    initial_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    ordinary.continue_step()
    ordinary_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    result = observed.continue_step(collect_diagnostics=True)
    assert torch.equal(torch.get_rng_state(), ordinary_rng)
    assert_nested_equal(observed.rng.bit_generator.state, ordinary.rng.bit_generator.state)
    assert_nested_equal(observed.diagnostic_rng.bit_generator.state, ordinary.diagnostic_rng.bit_generator.state)
    for name in ("actor", "critic", "target_critic", "value"):
        assert_nested_equal(getattr(observed, name).state_dict(), getattr(ordinary, name).state_dict())
    for name in ("actor_optimizer", "critic_optimizer", "value_optimizer", "alpha_optimizer"):
        assert_nested_equal(getattr(observed, name).state_dict(), getattr(ordinary, name).state_dict())
    assert observed.total_updates == ordinary.total_updates == 1
    record = result["td_diagnostic"]
    assert record["real_partitions"][0]["fraction"] == .5
    assert record["real_partitions"][1]["fraction"] == .5
    assert record["actor_parameter_change"]["l2"] > 0
    assert record["critic_parameter_change"]["l2"] > 0
    assert observed.world.generate_calls == (1 if arm in ("model", "context") else 0)
    if arm in ("model", "context"):
        assert record["absolute_trust"]["mean"] == pytest.approx(.5)
        assert record["effective_model_mean_weight_scale"] == pytest.approx(.125)


def test_model_arm_matches_original_v1_single_update_exactly():
    observed = trainer()
    original = copy.deepcopy(observed)
    original.__class__ = ClosedLoopTrainer
    initial_rng = torch.get_rng_state().clone()
    original.continue_step()
    original_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    observed.continue_step(collect_diagnostics=True)
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert_nested_equal(observed.critic.state_dict(), original.critic.state_dict())
    assert_nested_equal(observed.actor.state_dict(), original.actor.state_dict())
    assert_nested_equal(observed.critic_optimizer.state_dict(), original.critic_optimizer.state_dict())
