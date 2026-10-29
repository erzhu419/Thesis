"""Detect disguised coefficient deletion, prefix leakage and changed live state."""

from pathlib import Path
import copy
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))

from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    FEATURE_NAMES, fit_response_readout,
)
from experiments.mcwm_closed_loop_action_response_v8_20261021.response_world import ResponseWorld
from experiments.mcwm_closed_loop_action_response_v8_20261021.test_response_world import (
    issue, register, sidecar,
)
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import _learner_state, rng_state
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.ablation_world import (
    ARM_FEATURE_INDICES, ARM_NAMES, FEATURE_GROUPS, AblationResponseWorld,
    design_statistics, fit_ablation_readouts, group_contributions,
)


def test_nested_designs_are_refitted_and_full_is_original_v8_solve():
    x = torch.zeros(6, len(FEATURE_NAMES))
    action = FEATURE_GROUPS["action_physical"][0]
    memory = FEATURE_GROUPS["memory_level"][0]
    interaction = FEATURE_GROUPS["memory_interaction"][0]
    x[:, action] = torch.arange(1., 7.)
    x[:, memory] = x[:, action]
    x[:, interaction] = x[:, action]
    y = torch.stack((x[:, action], -x[:, action], x[:, action]*2, x[:, action]*.5), dim=1)
    fitted = fit_ablation_readouts(x, y)
    assert {arm: len(columns) for arm, columns in ARM_FEATURE_INDICES.items()} == {
        "action_physical": 40, "memory_level": 58, "memory_interaction": 66}
    torch.testing.assert_close(fitted["memory_interaction"], fit_response_readout(x, y), rtol=0, atol=0)
    for arm, readout in fitted.items():
        removed = sorted(set(range(66))-set(ARM_FEATURE_INDICES[arm]))
        assert not torch.count_nonzero(readout[removed])
    assert not torch.equal(fitted["action_physical"][action], fitted["memory_interaction"][action])
    assert not torch.equal(fitted["memory_level"][action], fitted["memory_interaction"][action])
    parts = group_contributions(x, fitted["memory_interaction"])
    torch.testing.assert_close(sum(parts.values()), x@fitted["memory_interaction"])
    stats = design_statistics(x)
    assert all(item["rank"] == 1 for item in stats.values())
    assert all(item["ridge_precision_condition"] > 1 for item in stats.values())


def test_passive_three_way_fit_matches_v8_and_preserves_live_state_and_rng(sidecar):
    learner, legacy = sidecar
    response = AblationResponseWorld(learner.world, legacy.physical)
    for sequence in (6, 7, 8):
        event, row = issue(sequence)
        register(legacy, event)
        register(response, event)
        if sequence == 7:
            selected = row
    # One packet, one selected label; the observed-only commands remain memory.
    for item in (legacy, response):
        item.accept([selected], [6], selected.available_at)
    before, before_rng = copy.deepcopy(_learner_state(learner)), rng_state()
    before_decoder = learner.world.generator.get_state().clone()
    physical_before = copy.deepcopy(response.physical.__dict__)
    legacy.refit()
    response.refit()
    assert_same(_learner_state(learner), before)
    assert_same(rng_state(), before_rng)
    assert torch.equal(learner.world.generator.get_state(), before_decoder)
    assert_same(response.physical.__dict__, physical_before)
    assert response.labels == len(response.records) == legacy.labels == 1
    assert set(response.pending) == {8}
    assert response.refits == legacy.refits == 1
    for family in legacy.readouts:
        torch.testing.assert_close(response.readouts[family], legacy.readouts[family], rtol=0, atol=0)
    for arm in ARM_NAMES:
        assert all(not torch.count_nonzero(value)
                   for value in response.frozen_readouts_by_arm[arm].values())
    result = response.summary(selected.available_at)
    assert result["selected_holding_coverage"]["overall"] == dict(
        n=1, hold_seconds_min=15., hold_seconds_max=15., positive_holding=1, zero_holding=0)
    assert result["selected_holding_coverage"]["by_line"]["102X"]["all"]["n"] == 0
    assert result["selected_holding_coverage"]["by_line"]["102X"]["all"]["hold_seconds_min"] is None
    assert result["designs"]["action_physical"]["rows"] == 1
    assert response.coefficients()["memory_interaction"]


def test_unreleased_label_is_not_used_and_empty_summary_retains_missing_support(sidecar):
    learner, legacy = sidecar
    response = AblationResponseWorld(learner.world, legacy.physical)
    event, row = issue()
    register(response, event)
    with pytest.raises(ValueError, match="before its feedback release"):
        response.accept([row], [], row.available_at-1)
    response.refit()
    assert response.labels == response.refits == 0
    assert row.sequence in response.pending
    result = response.summary(row.available_at)
    assert result["selected_holding_coverage"]["overall"]["n"] == 0
    assert all(value["rank"] == 0 and value["ridge_precision_condition"] == 1
               for value in result["designs"].values())
