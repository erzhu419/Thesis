"""Detect paired-label leakage, inert twin trust and changed live learning."""
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

from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    _learner_state, restore_rng, rng_state,
)
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedWorld
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_probes import learner
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_world import (
    diagnostic_world, issue_and_row,
)
from experiments.mcwm_closed_loop_paired_world_v6_20261020.paired_world import PairedWorld


def paired_learner():
    item = learner(enabled=False)
    item.replay_buffer.device = "cpu"
    for row in item.replay_buffer.recent:
        row.available_at = 1230.
    physical = diagnostic_world().physical
    physical.numeric_line_by_name = {"7X": 0, "102X": 1, "705X": 2}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(991)
        item.world = PairedWorld(item.replay_buffer, seed=41, updates=1,
            action_transform=item.action_transform, physical=physical)
    return item


def test_native_string_device_constructor_has_identical_initial_active_state_and_rng():
    item = paired_learner()
    physical = item.world.physical
    before = rng_state()
    ordinary = ProbedWorld(item.replay_buffer, seed=41, updates=1,
        action_transform=item.action_transform, physical=physical, enabled=False)
    ordinary_rng = rng_state()
    restore_rng(before)
    measured = PairedWorld(item.replay_buffer, seed=41, updates=1,
        action_transform=item.action_transform, physical=physical)
    assert_same(rng_state(), ordinary_rng)
    assert measured.device == torch.device("cpu")
    assert measured.local_enabled is False
    for name in measured.models:
        assert_same(measured.models[name].state_dict(), ordinary.models[name].state_dict())
        assert_same(measured.frozen_models[name].state_dict(), ordinary.models[name].state_dict())
    assert_same(measured.trust.net.state_dict(), ordinary.trust.net.state_dict())
    assert_same(measured.twin_trust.net.state_dict(), ordinary.trust.net.state_dict())


def test_saved_issue_predictions_use_original_families_before_later_online_fit():
    item = paired_learner()
    world = item.world
    world.family, world.frozen_family = "direct", "residual"
    issue, row = issue_and_row()
    world.register(issue)
    saved = copy.deepcopy(world.pending[row.sequence])
    with torch.no_grad():
        for parameter in world.models["direct"].parameters():
            if parameter.requires_grad:
                parameter.add_(.1)
    world.family = "residual"
    before_rng = rng_state()
    logs = world.observe([row], row.available_at)
    assert_same(rng_state(), before_rng)
    record = logs[0]["paired_world"]
    assert record["online_family"] == "direct"
    assert record["frozen_family"] == "residual"
    assert not record["same_family"]
    np.testing.assert_array_equal(record["branches"]["online"]["prediction_standardized"],
                                  saved["outputs"]["direct"][0, :4].numpy())
    np.testing.assert_array_equal(record["branches"]["frozen_common"]["prediction_standardized"],
                                  saved["paired_world"]["frozen_outputs"]["direct"][0, :4].numpy())
    np.testing.assert_array_equal(record["branches"]["frozen_own"]["prediction_standardized"],
                                  saved["paired_world"]["frozen_outputs"]["residual"][0, :4].numpy())
    summary = world.summary()["paired_world_diagnostics"]
    assert summary["selected_rows"] == summary["issue_time"]["overall"]["n"] == 1
    assert summary["frozen_parameters_unchanged"]


def test_future_packet_uses_fixed_parameters_value_standard_and_common_draws_before_training():
    item = paired_learner()
    world = item.world
    issue, row = issue_and_row()
    world.register(issue)
    world.observe([row], row.available_at)
    world.diagnose_packet(item, [row], 8400, row.available_at)
    anchor = world.paired_anchor
    captured = copy.deepcopy(anchor.models["online"][anchor.family].state_dict())
    with torch.no_grad():
        for parameter in world.models[world.family].parameters():
            if parameter.requires_grad:
                parameter.add_(.002)
        for parameter in item.value.parameters():
            parameter.add_(10.)
    rows = []
    for sequence in range(20, 25):
        issue2, row2 = issue_and_row(sequence)
        issue2["started_at"] = row2.started_at = 9040.
        row2.completed_at, row2.available_at = 9043., 9073.
        world.register(issue2)
        rows.append(row2)
    before_rng = rng_state()
    decoder_rng = world.generator.get_state().clone()
    original_observe = ProbedWorld.observe
    def observe_after_scoring(self, selected, now):
        assert len(world.paired_future) == 1
        assert world.paired_future[0]["n"] == 5
        return original_observe(self, selected, now)
    with patch.object(ProbedWorld, "observe", observe_after_scoring):
        world.observe(rows, 9073.)
    assert_same(rng_state(), before_rng)
    assert torch.equal(world.generator.get_state(), decoder_rng)
    assert_same(anchor.models["online"][anchor.family].state_dict(), captured)
    assert world.paired_anchor is None
    packet = world.paired_future[0]
    assert packet["evaluator_fixed_at_anchor"] and packet["identical_decoder_and_target_draws"]
    assert packet["scored_before_next_packet_training"]
    # No world fit occurred before this anchor, so common-family initial/online
    # snapshots coincide. Any target difference would reveal unmatched draws.
    online = packet["overall"]["branches"]["online"]
    frozen = packet["overall"]["branches"]["frozen_common"]
    assert online == frozen
    for branch in packet["overall"]["branches"].values():
        assert "bellman_target_error" in branch
        assert "scaled_reward_error" in branch and "bootstrap_error" in branch


def test_future_label_rejected_before_pairing_selector_or_live_fit_changes():
    item = paired_learner()
    world = item.world
    issue, row = issue_and_row()
    world.register(issue)
    with pytest.raises(ValueError, match="before availability"):
        world.observe([row], row.available_at-1)
    assert world.paired_issue_records == []
    assert world.paired_future == []
    assert world.local_fit_records == []
    assert row.sequence in world.pending
    assert all(not values for values in world.frozen_errors.values())


def test_twin_trust_trains_all_eight_updates_and_live_learning_is_bitwise_identical():
    measured = paired_learner()
    with torch.no_grad():
        # A real divergence is needed: identical twins would not detect a
        # frozen trust that was accidentally never fitted.
        for parameter in measured.world.models[measured.world.family].parameters():
            if parameter.requires_grad:
                parameter.add_(.02)
    ordinary = copy.deepcopy(measured)
    ordinary.world.__class__ = ProbedWorld
    before_twin = copy.deepcopy(measured.world.twin_trust.net.state_dict())
    # Processing is after the shock although the latest selected label is old;
    # trust periods must follow the update clock, not its availability time.
    measured.world.diagnose_packet(measured, [], 8400, 9031.)
    initial_rng = rng_state()
    for _ in range(8):
        ordinary.continue_step(collect_diagnostics=False)
    final_rng = rng_state()
    restore_rng(initial_rng)
    for _ in range(8):
        measured.continue_step(collect_diagnostics=False)
    assert_same(rng_state(), final_rng)
    assert_same(_learner_state(measured), _learner_state(ordinary))
    assert measured.world.trust_updates == 8
    assert len(measured.world.trust_anchor_steps[0]["steps"]) == 8
    assert any(not torch.equal(value, before_twin[name])
               for name, value in measured.world.twin_trust.net.state_dict().items())
    assert measured.world.twin_trust.optimizer.state_dict()["state"]
    assert measured.world.twin_trust._tau != measured.world.trust._tau
    for record in measured.world.trust_records:
        assert record["processed_at"] == 9031.
        assert record["phase"] == "early_after"
        assert record["identical_training_indices_and_next_actions"]
        assert record["common_family"] == measured.world.family
        for branch in ("online", "frozen"):
            assert len(record[branch]["six_loss_mean"]) == 6
            assert "actor_action_weight_mean" in record[branch]
            assert record[branch]["td_mse_before"] == pytest.approx(record[branch]["update_td_loss"])
    summary = measured.world.summary()["paired_world_diagnostics"]
    assert summary["trust_period_means"]["early_after"]["n"] == 8
    assert summary["frozen_parameters_unchanged"]


def test_shadow_disabled_diagnostics_skip_twin_updates_without_changing_active_trust():
    measured = paired_learner()
    ordinary = copy.deepcopy(measured)
    ordinary.world.__class__ = ProbedWorld
    measured.world.diagnostic_enabled = False
    initial_twin = copy.deepcopy(measured.world.twin_trust.net.state_dict())
    initial_rng = rng_state()
    ordinary.continue_step(collect_diagnostics=False)
    final_rng = rng_state()
    restore_rng(initial_rng)
    measured.continue_step(collect_diagnostics=False)
    assert_same(rng_state(), final_rng)
    assert_same(_learner_state(measured), _learner_state(ordinary))
    assert measured.world.trust_updates == 0
    assert_same(measured.world.twin_trust.net.state_dict(), initial_twin)
