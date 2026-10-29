"""Detect changed v8 arithmetic, anchors, RNG, trust and ablation attribution."""
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
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.ablation_world import (
    ARM_NAMES, ARM_FEATURE_INDICES, FEATURE_GROUPS,
)
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.predictions import (
    AblationPrediction, PREDICTION_ARMS, prediction_readouts,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation


def readouts(item):
    result = {}
    for index, arm in enumerate(ARM_NAMES):
        result[arm] = {}
        for branch in ("online", "frozen_common"):
            result[arm][branch] = {}
            for family in item.world.models:
                weight = torch.zeros(len(FEATURE_NAMES), 4)
                if branch == "online":
                    values = torch.arange(len(FEATURE_NAMES)*4).reshape(-1, 4).float()
                    weight[list(ARM_FEATURE_INDICES[arm])] = (
                        values[list(ARM_FEATURE_INDICES[arm])]+index+1)/10000
                result[arm][branch][family] = weight
    return result


def prediction(item, states, weights):
    return AblationPrediction(item, states, np.linspace(.01, .18, 18, dtype=np.float32),
                              30., [0, 1, 2], weights, seed=70)


def test_full_arm_matches_v8_exactly_and_all_arms_preserve_one_anchor_trust_and_rng():
    item = paired_learner()
    states, weights = state(time=19000.)[1], readouts(item)
    before = copy.deepcopy(_learner_state(item))
    twin = copy.deepcopy(item.world.twin_trust.net.state_dict())
    global_before = rng_state()
    draws = []
    original = ModelAugmentation.decode

    def decode(decoder, canonical, outputs):
        draws.append(decoder.generator.get_state().clone())
        return original(decoder, canonical, outputs)

    with patch.object(ModelAugmentation, "decode", side_effect=decode):
        probe = prediction(item, states, weights)
    assert len(draws) == 4+4*len(PREDICTION_ARMS)
    assert all(torch.equal(draws[0], draw) for draw in draws)
    assert_same(rng_state(), global_before)
    assert_same(_learner_state(item), before)
    assert_same(item.world.twin_trust.net.state_dict(), twin)
    full = ResponsePrediction(item, states, np.linspace(.01, .18, 18, dtype=np.float32),
                              30., [0, 1, 2], weights["memory_interaction"], seed=70).summary()
    record = probe.summary()
    assert list(record["arms"]) == list(PREDICTION_ARMS)
    assert record["legacy"] == full["legacy"]
    assert record["arms"]["memory_interaction"]["branches"] == full["branches"]
    for arm in record["arms"].values():
        assert "legacy" not in arm
        for field in ("anchor_update", "world_version", "common_family", "target_seed"):
            assert arm[field] == record[field]
        for branch in ("online", "frozen_common"):
            for candidate in ("low", "high"):
                actual = arm["branches"][branch]["candidates"][candidate]
                assert actual["trust"] == record["legacy"]["branches"][branch]["candidates"][candidate]["trust"]


def test_group_contributions_sum_to_full_and_deletion_effect_without_refitting():
    item = paired_learner()
    weights = readouts(item)
    saved = copy.deepcopy(weights)
    probe = prediction(item, state(time=19000.)[1], weights)
    assert_same(weights, saved)
    record = probe.summary()
    expanded = prediction_readouts(weights)
    for branch in ("online", "frozen_common"):
        groups = record["full_group_contributions"][branch]["groups"]
        total = record["full_group_contributions"][branch]["total"]
        for key in ("low", "high", "high_minus_low"):
            np.testing.assert_allclose(sum(np.asarray(group[key]) for group in groups.values()),
                                       total[key], rtol=1e-5, atol=1e-7)
        # Issue-time memory is identical for the two candidate actions.
        assert groups["memory_level"]["high_minus_low"] == [0.]*4
        for candidate in ("low", "high"):
            full = record["arms"]["memory_interaction"]["branches"][branch]["candidates"][candidate]
            np.testing.assert_allclose(total[candidate], full["response_correction_standardized"])
            for group in FEATURE_GROUPS:
                dropped = record["arms"][f"drop_{group}"]["branches"][branch]["candidates"][candidate]
                np.testing.assert_allclose(
                    np.asarray(full["response_correction_standardized"])-dropped["response_correction_standardized"],
                    groups[group][candidate], rtol=1e-5, atol=1e-7)
        for group, columns in FEATURE_GROUPS.items():
            for family, original in weights["memory_interaction"][branch].items():
                dropped = expanded[f"drop_{group}"][branch][family]
                assert torch.count_nonzero(dropped[list(columns)]) == 0
                keep = [column for column in range(len(FEATURE_NAMES)) if column not in columns]
                assert torch.equal(dropped[keep], original[keep])


def test_same_batch_analytic_path_overrides_every_nonzero_residual_and_matches_v8():
    item = paired_learner()
    states = state(time=19000., current=1)[1]
    states[2] = 0.
    weights = readouts(item)
    with patch.object(ModelAugmentation, "decode", wraps=ModelAugmentation.decode) as decode:
        probe = prediction(item, states, weights)
    assert decode.call_count == 4  # Legacy only; seven response arms use exact advancement.
    record = probe.summary()
    full = ResponsePrediction(item, states, np.linspace(.01, .18, 18, dtype=np.float32),
                              30., [0, 1, 2], weights["memory_interaction"], seed=70).summary()
    assert record["arms"]["memory_interaction"]["branches"] == full["branches"]
    for arm in record["arms"].values():
        for branch in ("online", "frozen_common"):
            for candidate in ("low", "high"):
                value = arm["branches"][branch]["candidates"][candidate]
                assert value["analytic_within_batch"]
                assert value["global_decoded"] == [0.]*4
                assert value["bellman"]["scaled_reward"] == 0
                assert value["decoded_next_topology"]["event_count"] == 2
                assert value["decoded_next_topology"]["current_slot"] == 1


def test_truth_is_joined_once_with_all_arm_directions_and_anchor_checks():
    item = paired_learner()
    probe = prediction(item, state(time=19000.)[1], readouts(item))
    low, high = (factual(probe.legacy, candidate, dt=2, cost=cost)
                 for candidate, cost in (("low", 6), ("high", 7)))
    ls = probe.legacy.score_transition(low, now=low.completed_at, candidate="low")
    hs = probe.legacy.score_transition(high, now=high.completed_at, candidate="high")
    comparison = probe.compare(ls, hs, low_cost_300=100., high_cost_300=120.)
    assert comparison["factual"]["gains"]["next_event_cost"] == -1
    assert comparison["factual"]["gains"]["cost_300_seconds"] == -20
    assert set(comparison["arms"]) == set(PREDICTION_ARMS)
    assert set(comparison["arms"]["action_physical"]) == {"online", "frozen_common"}
    hs["target_seed"] += 1
    with pytest.raises(ValueError, match="target_seed"):
        probe.compare(ls, hs, low_cost_300=100., high_cost_300=120.)
