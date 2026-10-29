"""Detect changed prefixes/trust, inert residuals and learned dt=0 decoding."""
from pathlib import Path
import copy
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))

from experiments.mcwm_closed_loop_action_fork_v7_20261020.test_predictions import factual
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import state
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import FEATURE_NAMES
from experiments.mcwm_closed_loop_action_response_v8_20261021.predictions import ResponsePrediction
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import _learner_state, rng_state
from experiments.mcwm_closed_loop_paired_world_v6_20261020.test_paired_world import paired_learner
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation


def readouts(item):
    return {branch: {family: torch.zeros(len(FEATURE_NAMES), 4)
                     for family in item.world.models}
            for branch in ("online", "frozen_common")}


def prediction(item, states, weights):
    return ResponsePrediction(item, states, np.zeros(18, np.float32), 30., [0, 1, 2],
                              weights, seed=70)


def test_zero_readout_retains_original_boundary_predictions_and_all_training_state():
    item = paired_learner()
    weights = readouts(item)
    before = copy.deepcopy(_learner_state(item))
    twin = copy.deepcopy(item.world.twin_trust.net.state_dict())
    global_before = rng_state()
    draws = []
    original = ModelAugmentation.decode
    def decode(decoder, states, outputs):
        draws.append(decoder.generator.get_state().clone())
        return original(decoder, states, outputs)
    with patch.object(ModelAugmentation, "decode", side_effect=decode):
        probe = prediction(item, state(time=19000.)[1], weights)
    record = probe.summary()
    assert len(draws) == 8 and all(torch.equal(draws[0], draw) for draw in draws)
    for branch in ("online", "frozen_common"):
        for candidate in ("low", "high"):
            baseline = record["legacy"]["branches"][branch]["candidates"][candidate]
            actual = record["branches"][branch]["candidates"][candidate]
            assert actual["global_decoded"] == baseline["global_decoded"]
            assert actual["bellman"] == baseline["bellman"]
            assert actual["trust"] == baseline["trust"]
            assert actual["response_correction_standardized"] == [0.]*4
    assert record["candidate_actions"]["high"]["holding_seconds"] == pytest.approx(7.5)
    assert_same(rng_state(), global_before)
    assert_same(_learner_state(item), before)
    assert_same(item.world.twin_trust.net.state_dict(), twin)


def test_action_residual_changes_cost_and_bellman_with_original_base_retained():
    item = paired_learner()
    weights = readouts(item)
    for families in weights.values():
        for values in families.values():
            values[FEATURE_NAMES.index("7X_hold"), 1] = .5
    probe = prediction(item, state(time=19000.)[1], weights)
    record = probe.summary()
    for branch in ("online", "frozen_common"):
        baseline = record["legacy"]["branches"][branch]["candidates"]
        actual = record["branches"][branch]["candidates"]
        assert actual["low"]["global_raw"] == baseline["low"]["global_raw"]
        assert actual["high"]["response_correction_standardized"][1] == pytest.approx(.0625)
        assert actual["high"]["global_decoded"][1] > baseline["high"]["global_decoded"][1]
        assert actual["high"]["bellman"]["scaled_reward"] < baseline["high"]["bellman"]["scaled_reward"]
        assert actual["high"]["bellman"]["target"] != baseline["high"]["bellman"]["target"]
        assert actual["high"]["trust"] == baseline["high"]["trust"]
    low = factual(probe.legacy, "low", dt=2, cost=6)
    high = factual(probe.legacy, "high", dt=2, cost=7)
    ls = probe.legacy.score_transition(low, now=low.completed_at, candidate="low")
    hs = probe.legacy.score_transition(high, now=high.completed_at, candidate="high")
    result = probe.compare(ls, hs, low_cost_300=100., high_cost_300=120.)
    assert result["factual"]["gains"]["next_event_cost"] == -1
    assert result["factual"]["gains"]["cost_300_seconds"] == -20
    hs["target_seed"] += 1
    with pytest.raises(ValueError, match="target_seed"):
        probe.compare(ls, hs, low_cost_300=100., high_cost_300=120.)


def test_same_batch_uses_exact_state_advancement_even_with_nonzero_residual():
    item = paired_learner()
    states = state(time=19000., current=1)[1]
    states[2] = 0.  # Two same-time events, the first is the selected action.
    weights = readouts(item)
    for families in weights.values():
        for values in families.values():
            values[FEATURE_NAMES.index("7X_hold"), :] = 10.
    original = ModelAugmentation.decode
    with patch.object(ModelAugmentation, "decode", wraps=original) as decode:
        probe = prediction(item, states, weights)
    # The four legacy candidates decode; no v8 same-batch candidate draws.
    assert decode.call_count == 4
    record = probe.summary()
    for branch in ("online", "frozen_common"):
        for candidate in ("low", "high"):
            actual = record["branches"][branch]["candidates"][candidate]
            assert actual["analytic_within_batch"]
            assert actual["global_decoded"] == [0.]*4
            assert actual["bellman"]["scaled_reward"] == 0
            assert actual["decoded_next_topology"]["event_count"] == 2
            assert actual["decoded_next_topology"]["current_slot"] == 1
    assert record["branches"]["online"]["gains"]["next_event_cost"] == 0
