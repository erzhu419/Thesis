"""Detect mismatched context branches, live-state changes and future-label leakage."""

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
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import DiagnosticWorld
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedWorld
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_world import diagnostic_world, issue_and_row, state
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.context import ContextWorld, feature_identifiability


def context_world():
    world = diagnostic_world()
    world.__class__ = ContextWorld
    world.seed = 6201
    world.generator = torch.Generator().manual_seed(311)
    world.rng = np.random.default_rng(331)
    world.norm.local_mean, world.norm.local_std = torch.zeros(8), torch.ones(8)
    world.context_probe_deadlines = [1500, 2100, 2700, 3300]
    world.context_selected, world.context_historical, world.context_future = [], [], []
    world.context_anchor = None
    return world


def observe(world, issue, row, now=None):
    world.register(issue)
    with patch.object(DiagnosticWorld, "observe", return_value=[dict(sequence=row.sequence)]):
        return world.observe([row], row.available_at if now is None else now)


def test_three_branches_hold_parameters_fixed_and_default_prediction_unchanged():
    world = context_world()
    s, a = state(), torch.tensor([[-.5]])
    before = world.predict(s, a).clone()
    base = world.predict_context(s, a, branch="base")
    saved = world.predict_context(s, a, deltas=np.array([6., 0., 0.]), branch="saved")
    world.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
    current = world.predict_context(s, a, branch="current")
    torch.testing.assert_close(saved[:, :4]-base[:, :4], torch.full((1, 4), .225))
    torch.testing.assert_close(current[:, :4]-base[:, :4], torch.full((1, 4), 2.25))
    torch.testing.assert_close(saved[:, 4:], base[:, 4:], rtol=0, atol=0)
    world.physical.context_delta_seconds = lambda: np.array([6., 0., 0.])
    torch.testing.assert_close(world.predict(s, a), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="issue-time"):
        world.predict_context(s, a, branch="saved")


def test_issue_context_is_saved_before_later_physical_update_and_labels_release():
    world = context_world()
    issue, row = issue_and_row()
    world.register(issue)
    world.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
    with patch.object(DiagnosticWorld, "observe", return_value=[dict(sequence=row.sequence)]):
        world.observe([row], row.available_at)
    metadata = world.context_selected[0][1]
    np.testing.assert_array_equal(metadata["saved_delta"], [6., 0., 0.])
    assert metadata["started_at"] == row.started_at
    assert metadata["line_id"] == "7X"


def test_historical_diagnostics_preserve_all_live_rng_parameters_fit_inputs_and_predictions():
    world = context_world()
    reference = copy.deepcopy(world)
    reference.__class__ = ProbedWorld
    issue, row = issue_and_row()
    observe(world, issue, row)
    observe(reference, issue, row)
    world.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
    reference.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
    before_torch = torch.get_rng_state().clone()
    before_generator = world.generator.get_state().clone()
    before_numpy = copy.deepcopy(world.rng.bit_generator.state)
    before_global_numpy = np.random.get_state()
    result = world.diagnose_context([row], now=9031., deadline=1500)
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert torch.equal(world.generator.get_state(), before_generator)
    assert world.rng.bit_generator.state == before_numpy
    after_global_numpy = np.random.get_state()
    assert after_global_numpy[0] == before_global_numpy[0]
    np.testing.assert_array_equal(after_global_numpy[1], before_global_numpy[1])
    assert after_global_numpy[2:] == before_global_numpy[2:]
    for key in world.local_fit_records[0]:
        torch.testing.assert_close(world.local_fit_records[0][key], reference.local_fit_records[0][key], rtol=0, atol=0)
    for family in world.local_readouts:
        torch.testing.assert_close(world.local_readouts[family], reference.local_readouts[family], rtol=0, atol=0)
    torch.testing.assert_close(world.predict(state(), torch.tensor([[-.5]])),
                               reference.predict(state(), torch.tensor([[-.5]])), rtol=0, atol=0)
    assert result["n"] == 1
    assert result["overall"]["current_minus_saved"]["prediction_standardized_absolute_mean"] == pytest.approx([2.025]*4)
    assert len(result["overall"]["branches"]["saved"]["six_loss_mean"]) == 6
    assert len(result["overall"]["branches"]["saved"]["g4_mse"]) == 4
    assert result["row_records"][0]["phase_crossing"] == "crosses_issue_completion_phase"


def test_next_packet_scores_frozen_anchor_before_observe_and_never_reads_release_current_context():
    world = context_world()
    issue, row = issue_and_row()
    observe(world, issue, row)
    world.diagnose_context([row], now=9031., deadline=1500)
    assert world.context_future == []
    frozen_delta = world.context_anchor.delta.copy()
    world.local_readouts["direct"].add_(100.)
    world.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
    issue2, row2 = issue_and_row(8)
    issue2["started_at"] = row2.started_at = 9040.
    row2.completed_at, row2.available_at = 9043., 9073.
    world.register(issue2)
    def release(*args, **kwargs):
        assert len(world.context_future) == 1
        assert world.context_future[0]["scored_before_next_packet_training"]
        return [dict(sequence=row2.sequence)]
    world.physical.context_delta_seconds = lambda: (_ for _ in ()).throw(AssertionError("release context leaked"))
    # Inherited local/head scoring uses saved issue covariates and needs no release context.
    with patch.object(DiagnosticWorld, "observe", side_effect=release):
        log = world.observe([row2], row2.available_at)[0]
    detail = log["context_next_packet"]
    np.testing.assert_array_equal(detail["current_context_seconds"], frozen_delta)
    assert detail["branches"]["base"]["prediction_standardized"] == [1., 2., 0., 0.]
    assert detail["branches"]["current"]["prediction_standardized"] == pytest.approx([1.225, 2.225, .225, .225])
    assert detail["branches"]["saved"]["prediction_standardized"] == pytest.approx([3.25, 4.25, 2.25, 2.25])
    assert len(world.context_selected) == 2 and world.context_anchor is None


def test_future_label_rejection_precedes_any_scoring_or_fit_mutation():
    world = context_world()
    issue, row = issue_and_row()
    world.register(issue)
    with pytest.raises(ValueError, match="before availability"):
        world.observe([row], row.available_at-1)
    assert world.context_selected == [] and world.context_future == []
    assert world.local_fit_records == [] and world.head_prediction_records == []
    assert row.sequence in world.pending


def test_rank_and_hold_collinearity_report_actual_repeated_context_design():
    states = state().expand(8, -1).clone()
    actions = torch.linspace(-1, 1, 8).reshape(-1, 1)
    result = feature_identifiability(states, actions, np.tile([6., 0., 0.], (8, 1)),
                                     np.array([60., 0., 0.]), [0, 1, 2])
    for branch in result["branches"].values():
        assert branch["rank"] == 2
        assert branch["condition_number_full"] is None
        assert len(branch["singular_values"]) == 8
        assert branch["hold_column_residual_norm_fraction_after_other_columns"][0] > .1
        assert branch["hold_column_residual_norm_fraction_after_other_columns"][1] is None
    assert result["executed_hold_fraction"]["minimum"] == 0.
    assert result["executed_hold_fraction"]["maximum"] == 1.


def test_future_bellman_targets_share_frozen_evaluator_and_matched_decoder_draws():
    from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import trainer
    from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import transition_batch
    world, learner = context_world(), trainer()
    world.local_readouts["direct"].zero_()
    world.local_readouts["direct"][:, 1] = 1.
    issue, row = issue_and_row()
    observe(world, issue, row)
    world.diagnose_context([row], now=9031., deadline=1500, learner=learner)
    issue2, row2 = issue_and_row(8)
    issue2["started_at"] = row2.started_at = 9040.
    row2.completed_at, row2.available_at = 9043., 9073.
    frozen_truth = world.context_anchor.evaluator.target(transition_batch([row2], world.device))
    with torch.no_grad():
        for parameter in learner.value.parameters():
            parameter.add_(10.)
    world.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
    world.register(issue2)
    before_torch, before_decoder = torch.get_rng_state().clone(), world.generator.get_state().clone()
    with patch.object(DiagnosticWorld, "observe", return_value=[dict(sequence=row2.sequence)]):
        detail = world.observe([row2], row2.available_at)[0]["context_next_packet"]
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert torch.equal(world.generator.get_state(), before_decoder)
    branches = detail["branches"]
    assert all(branch["bellman_target_truth"] == float(frozen_truth[0]) for branch in branches.values())
    for name in ("saved", "current"):
        base, branch = branches["base"], branches[name]
        target_difference = base["bellman_target_prediction"]-branch["bellman_target_prediction"]
        expected_difference = learner.config.reward_scale*(
            np.expm1(branch["prediction_standardized"][1])-np.expm1(base["prediction_standardized"][1]))
        assert target_difference == pytest.approx(expected_difference, abs=1e-5)
    future = world.context_future[0]
    assert future["evaluator_fixed_at_anchor"]
    assert future["n"] == 1
    assert "bellman_target_error" in future["overall"]["branches"]["base"]


def test_real_cpu_string_replay_supports_first_anchor_and_next_packet_without_live_changes():
    from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_probes import learner
    from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same

    item = learner()
    item.replay_buffer.device = "cpu"
    physical = diagnostic_world().physical
    physical.numeric_line_by_name = {"7X": 0, "102X": 1, "705X": 2}
    world = ContextWorld(item.replay_buffer, seed=41, updates=1,
        action_transform=item.action_transform, physical=physical, enabled=True)
    assert world.device == torch.device("cpu")
    for readout in world.local_readouts.values():
        readout.fill_(.05)
    issue, row = issue_and_row()
    world.register(issue)
    world.observe([row], row.available_at)
    physical.context_delta_seconds = lambda: np.array([60., 0., 0.])

    parameters = copy.deepcopy({
        "world": {name: model.state_dict() for name, model in world.models.items()},
        "readouts": world.local_readouts,
        "learner": {name: getattr(item, name).state_dict()
                    for name in ("actor", "critic", "target_critic", "value")},
        "optimizers": {name: getattr(item, name).state_dict() for name in
                       ("actor_optimizer", "critic_optimizer", "value_optimizer", "alpha_optimizer")},
        "trust": world.trust.net.state_dict(),
        "trust_optimizer": world.trust.optimizer.state_dict(),
    })
    torch_rng = torch.get_rng_state().clone()
    decoder_rng = world.generator.get_state().clone()
    numpy_rng = copy.deepcopy(world.rng.bit_generator.state)
    learner_rng = copy.deepcopy(item.rng.bit_generator.state)

    history = world.diagnose_context([row], now=row.available_at, deadline=8400, learner=item)
    assert history["n"] == 1 and world.context_anchor is not None
    issue2, row2 = issue_and_row(8)
    issue2["started_at"] = row2.started_at = 9040.
    row2.completed_at, row2.available_at = 9043., 9073.
    world.register(issue2)
    detail = world.observe([row2], row2.available_at)[0]["context_next_packet"]
    assert set(detail["branches"]) == {"base", "saved", "current"}
    assert all("bellman_target_signed_error" in branch for branch in detail["branches"].values())
    assert world.context_anchor is None and len(world.context_future) == 1
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(world.generator.get_state(), decoder_rng)
    assert_same(world.rng.bit_generator.state, numpy_rng)
    assert_same(item.rng.bit_generator.state, learner_rng)
    assert_same({
        "world": {name: model.state_dict() for name, model in world.models.items()},
        "readouts": world.local_readouts,
        "learner": {name: getattr(item, name).state_dict()
                    for name in ("actor", "critic", "target_critic", "value")},
        "optimizers": {name: getattr(item, name).state_dict() for name in
                       ("actor_optimizer", "critic_optimizer", "value_optimizer", "alpha_optimizer")},
        "trust": world.trust.net.state_dict(),
        "trust_optimizer": world.trust.optimizer.state_dict(),
    }, parameters)
