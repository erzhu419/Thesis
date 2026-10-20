"""Detect passive probes changing live updates or inventing inactive effects."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import trainer
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import (
    HEAD_NAMES, ProbedTrainer, ProbedWorld,
)


def learner(*, enabled=True):
    item = trainer()
    item.__class__ = ProbedTrainer
    physical = SimpleNamespace(numeric_line_by_name={"7X": 0, "102X": 1, "705X": 2},
        context_delta_seconds=lambda: np.array([28., 27., 28.]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(601)
        item.world = ProbedWorld(item.replay_buffer, seed=41, updates=1,
            action_transform=item.action_transform, physical=physical, enabled=enabled)
    for readout in item.world.local_readouts.values():
        readout.fill_(.05)
    return item


def test_nonzero_head_probes_preserve_parameters_optimizers_and_all_rng():
    observed = learner()
    ordinary = copy.deepcopy(observed)
    initial_rng = torch.get_rng_state().clone()
    ordinary.world.capture_counterfactual = False
    ordinary.continue_step(collect_diagnostics=False)
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    observed.world.capture_counterfactual = True
    metrics = observed.continue_step(collect_diagnostics=True)
    assert torch.equal(torch.get_rng_state(), final_rng)
    variants = observed.head_counterfactual["variants"]
    assert len(variants) == 5
    assert all(value["applied_rows"] > 0 for value in variants.values())
    assert variants["omit_logcost"]["target_q"]["mean_absolute_delta"] > 0
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


def test_matched_decoder_draws_isolate_heads_and_fixed_trust_target_components():
    item = learner()
    item.world.capture_counterfactual = True
    item.continue_step(collect_diagnostics=True)
    probe = item.head_counterfactual["variants"]
    base_ns = item.world.head_counterfactual_batches["full_base"]["next_observations"]
    for batch in item.world.head_counterfactual_batches.values():
        ns = batch["next_observations"]
        assert torch.equal(ns[:, 1:3], base_ns[:, 1:3])
        assert torch.equal(ns[:, 5:], base_ns[:, 5:])
    # Cost changes only immediate reward, not the decoded future state, so its
    # Bellman target difference is exactly the scaled reward difference.
    cost = probe["omit_logcost"]
    assert abs(cost["bootstrap_component_delta"]["mean"]) < 1e-7
    assert cost["next_proposed_hold_seconds"]["mean_absolute_delta"] == 0
    assert cost["next_executed_hold_seconds"]["mean_absolute_delta"] == 0
    # Matching discrete decoder draws leaves all unrelated decoded components
    # bitwise equal; a duration/person head must not invent a reward change.
    for name in HEAD_NAMES:
        record = probe["omit_"+name]
        if name != "logdt":
            assert record["duration_seconds"]["mean_absolute_delta"] == 0
        if name != "logcost":
            assert record["cost_passenger_seconds"]["mean_absolute_delta"] == 0
        if name != "dWaiting":
            assert record["decoded_waiting_state_persons"]["mean_absolute_delta"] == 0
        if name != "dOnboard":
            assert record["decoded_onboard_state_persons"]["mean_absolute_delta"] == 0
        assert record["td_loss_actual_model_contribution"]["mean_delta"] == (
            .25*record["td_loss_trust_weighted"]["mean_delta"])


def test_inactive_local_arm_has_zero_effect_despite_nonzero_candidate_readout():
    item = learner(enabled=False)
    item.world.capture_counterfactual = True
    item.continue_step(collect_diagnostics=True)
    for record in item.head_counterfactual["variants"].values():
        assert record["applied_rows"] == 0
        for key in ("target_q", "td_loss_unweighted", "td_loss_trust_weighted",
                    "duration_seconds", "cost_passenger_seconds",
                    "next_proposed_hold_seconds", "next_executed_hold_seconds"):
            assert record[key]["mean_absolute_delta"] == 0
    item.world.capture_counterfactual = False
    item.continue_step(collect_diagnostics=False)
    assert item.head_counterfactual is None
    assert item.world.head_counterfactual_batches is None
