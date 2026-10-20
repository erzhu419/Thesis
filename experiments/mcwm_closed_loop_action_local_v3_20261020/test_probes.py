"""Detect a passive counterfactual probe changing the live learning trajectory."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020 import bootstrap
from experiments.mcwm_closed_loop_action_local_v3_20261020.probes import ProbedTrainer, ProbedWorld
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import trainer


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_nonzero_local_counterfactual_leaves_parameters_optimizers_and_rng_identical():
    observed = trainer()
    observed.__class__ = ProbedTrainer
    physical = SimpleNamespace(numeric_line_by_name={"7X": 0, "102X": 1, "705X": 2},
        context_delta_seconds=lambda: np.array([28., 27., 28.]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(601)
        observed.world = ProbedWorld(observed.replay_buffer, seed=41, updates=1,
            action_transform=observed.action_transform, physical=physical, enabled=True)
    # Make the probe nonvacuous: the actual calibration changes global duration,
    # cost and passenger predictions, while all sampled discrete heads agree.
    for readout in observed.world.local_readouts.values():
        readout.fill_(.05)
    ordinary = copy.deepcopy(observed)
    initial_rng = torch.get_rng_state().clone()
    ordinary.world.capture_counterfactual = False
    ordinary.continue_step(collect_diagnostics=False)
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    observed.world.capture_counterfactual = True
    metrics = observed.continue_step(collect_diagnostics=True)
    assert torch.equal(torch.get_rng_state(), final_rng)
    assert observed.local_counterfactual["applied_rows"] > 0
    assert observed.local_counterfactual["mean_absolute_target_q_delta"] > 0
    assert metrics["td_diagnostic"]["actor_parameter_change"]["l2"] > 0
    for name in ("actor", "critic", "target_critic", "value"):
        assert_same(getattr(observed, name).state_dict(), getattr(ordinary, name).state_dict())
    for name in ("actor_optimizer", "critic_optimizer", "value_optimizer", "alpha_optimizer"):
        assert_same(getattr(observed, name).state_dict(), getattr(ordinary, name).state_dict())
    assert_same(observed.log_alpha, ordinary.log_alpha)
    assert_same(observed.rng.bit_generator.state, ordinary.rng.bit_generator.state)
    assert_same(observed.diagnostic_rng.bit_generator.state, ordinary.diagnostic_rng.bit_generator.state)
    assert_same(observed.world.rng.bit_generator.state, ordinary.world.rng.bit_generator.state)
    assert_same(observed.world.generator.get_state(), ordinary.world.generator.get_state())
    for name in ("net", "target"):
        assert_same(getattr(observed.world.trust, name).state_dict(),
                    getattr(ordinary.world.trust, name).state_dict())
    assert_same(observed.world.trust.optimizer.state_dict(), ordinary.world.trust.optimizer.state_dict())
    assert observed.world.trust._tau == ordinary.world.trust._tau
    assert observed.total_updates == ordinary.total_updates == 1
