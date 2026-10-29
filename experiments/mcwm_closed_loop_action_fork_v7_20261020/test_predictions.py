"""Detect unmatched draws, double execution transforms and reversed action gains."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))

from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import state
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    _learner_state, rng_state, transition_batch,
)
from experiments.mcwm_closed_loop_paired_world_v6_20261020.test_paired_world import paired_learner
from experiments.mcwm_closed_loop_action_fork_v7_20261020.predictions import (
    ActionForkPrediction, compare_records, paired_effect,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation


def factual(probe, candidate, *, dt, cost, next_vehicle="next_bus"):
    s = probe.state[0].cpu().numpy().copy()
    ns = s.copy()
    ns[0] += dt/24000
    return SimpleNamespace(s=s, ns=ns,
        a=np.asarray([probe.candidates[candidate]["executed_action"]], np.float32),
        dt=float(dt), r=-float(cost), terminal=False, sequence=17,
        started_at=float(s[0])*24000, completed_at=float(s[0])*24000+dt,
        available_at=float(s[0])*24000+dt+30,
        vehicle_id="current_bus", next_vehicle_id=next_vehicle, line_id="7X")


def test_same_draws_single_execution_transform_and_no_learner_or_rng_mutation():
    item = paired_learner()
    s = state(time=19000.)[1]
    before = copy.deepcopy(_learner_state(item))
    twin = copy.deepcopy(item.world.twin_trust.net.state_dict())
    global_before = rng_state()
    draws, received = [], []
    original_decode = ModelAugmentation.decode
    def decode(decoder, states, output):
        draws.append(decoder.generator.get_state().clone())
        return original_decode(decoder, states, output)
    family = item.world.family
    original_forward = item.world.models[family].forward
    def model(states, action):
        received.append(action.detach().clone())
        return original_forward(states, action)
    with patch.object(ModelAugmentation, "decode", side_effect=decode), patch.object(
            item.world.models[family], "forward", side_effect=model):
        probe = ActionForkPrediction(item, s, seed=70)
    assert probe.valid_pair
    assert probe.candidates["low"]["holding_seconds"] == 0
    assert probe.candidates["high"]["holding_seconds"] == pytest.approx(7.5)
    # F(s,+1) is -.75 here; applying F a second time would produce -.96875.
    assert float(received[1].item()) == pytest.approx(-.75)
    assert len(draws) == 4 and all(torch.equal(draws[0], draw) for draw in draws)
    record = probe.summary()
    assert record["branches"]["online"] == record["branches"]["frozen_common"]
    assert_same(rng_state(), global_before)
    assert_same(_learner_state(item), before)
    assert_same(item.world.twin_trust.net.state_dict(), twin)
    for candidate in ("low", "high"):
        row = factual(probe, candidate, dt=2, cost=6)
        scored = probe.score_transition(row, now=row.completed_at, candidate=candidate)
        for branch in ("online", "frozen_common"):
            observed = scored["branches"][branch]["trust_outcome"]
            assert observed["target"] == pytest.approx(observed["log_c_step"]+observed["bootstrap"])
            assert observed["prediction_minus_target"] == pytest.approx(observed["prediction"]-observed["target"])
    assert_same(rng_state(), global_before)
    assert_same(_learner_state(item), before)
    assert_same(item.world.twin_trust.net.state_dict(), twin)


def test_cost_gain_and_bellman_gain_keep_their_different_sign_conventions():
    low = dict(global_decoded=[1., 100., 2., 3.],
               bellman=dict(scaled_reward=-.01, bootstrap=-.005, target=-.015))
    high = dict(global_decoded=[2., 70., 4., 5.],
                bellman=dict(scaled_reward=-.007, bootstrap=-.010, target=-.017))
    paired = paired_effect(low, high)
    assert paired["high_minus_low"]["global_decoded"] == [1., -30., 2., 2.]
    assert paired["gains"]["next_event_cost"] == 30
    assert paired["gains"]["scaled_reward"] == pytest.approx(.003)
    assert paired["gains"]["bellman_target"] == pytest.approx(-.002)
    assert paired["directions"]["next_event_cost"] == "high"
    assert paired["directions"]["bellman_target"] == "low"


def test_zero_duration_and_different_next_vehicles_use_same_causal_anchor_evaluator():
    item = paired_learner()
    probe = ActionForkPrediction(item, state(time=14000.)[1], seed=71)
    low = factual(probe, "low", dt=0, cost=0, next_vehicle="bus_A")
    high = factual(probe, "high", dt=1, cost=3, next_vehicle="bus_B")
    with pytest.raises(ValueError, match="before completion"):
        probe.score_transition(high, now=high.completed_at-1, candidate="high")
    expected = float(probe.evaluator.target(transition_batch([high], item.device)).item())
    initial = probe.summary()
    # The live learner may continue after intervention.  Neither these changes
    # nor a later world family change may affect the fixed candidate predictions.
    with torch.no_grad():
        for parameter in item.value.parameters():
            parameter.add_(50.)
        for parameter in item.actor.parameters():
            parameter.add_(1.)
        for model in item.world.models.values():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(.1)
    before = rng_state()
    ls = probe.score_transition(low, now=low.completed_at, candidate="low")
    hs = probe.score_transition(high, now=high.completed_at, candidate="high")
    assert hs["bellman"]["target"] == expected
    assert ls["zero_duration"] and ls["fact"]["dt"] == 0
    assert ls["global_decoded"][0] == 0
    assert ls["branches"]["online"]["six_losses"] is not None
    assert_same(probe.summary(), initial)
    assert_same(rng_state(), before)
    result = compare_records(initial, ls, hs, low_cost_300=100., high_cost_300=90.)
    assert result["factual"]["gains"]["cost_300_seconds"] == 10.
    assert result["factual"]["gains"]["next_event_cost"] == -3.
    assert not result["actual_branches_have_same_next_vehicle"]
    assert result["next_global_event_time_difference_seconds"] == 1.
    assert result == probe.compare(ls, hs, low_cost_300=100., high_cost_300=90.)
    corrupted = copy.deepcopy(hs)
    corrupted["target_seed"] += 1
    with pytest.raises(ValueError, match="target_seed"):
        compare_records(initial, ls, corrupted, low_cost_300=100., high_cost_300=90.)


def test_completion_floor_collapsed_candidates_are_reported_not_counted_as_action_signal():
    item = paired_learner()
    probe = ActionForkPrediction(item, state(time=14000., bwd_present=0., load=5.)[1], seed=72)
    assert not probe.valid_pair
    assert probe.summary()["executed_hold_gap_seconds"] == 0
    assert probe.candidates["low"]["holding_seconds"] == 60
    assert probe.candidates["high"]["holding_seconds"] == 60
    assert probe.summary()["branches"]["online"]["gains"]["bellman_target"] == 0
