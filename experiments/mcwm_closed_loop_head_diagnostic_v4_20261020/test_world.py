"""Detect sign/unit mistakes, decoder mismatch and noncausal issue rescoring."""

from pathlib import Path
import copy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))

from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import DiagnosticWorld, LocalWorld
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_local_world import fake_world, state
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.world import (
    HEAD_ORDER, HeadDiagnosticWorld, UPSTREAM_STOP, compare_global_heads,
    decoder_global_projection, summarize_head_records,
)
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation


def diagnostic_world():
    world = fake_world()
    world.__class__ = HeadDiagnosticWorld
    world.head_prediction_records = []
    world.physical.shock_at = 9000.
    world.physical.source_reference_features = np.eye(3)
    world.physical.frozen_weights = np.array([110., 120., 130.])
    world.physical.online_weights = np.array([114., 120., 130.])
    world.physical.context = {"7X": 2., "102X": 0., "705X": 0.}
    return world


def issue_and_row(sequence=7):
    s = state()
    s[0, 3:5] = torch.tensor([.001, .002])
    a = np.array([-.5], np.float32)
    issue = dict(s=s[0].numpy(), a=a, started_at=8998., sequence=sequence,
                 line_id="7X", station_id=UPSTREAM_STOP)
    ns = s[0].numpy().copy()
    ns[0] += 3./24000
    ns[3] -= 2./20000
    ns[4] += 1./20000
    row = SimpleNamespace(s=s[0].numpy(), a=a, ns=ns, r=-8., dt=3., terminal=False,
        line_id="7X", station_id=UPSTREAM_STOP, sequence=sequence,
        started_at=8998., completed_at=9001., available_at=9031.)
    return issue, row


def test_signed_errors_keep_log_normalization_and_negative_count_increments():
    norm = SimpleNamespace(g_mean=torch.tensor([1., 2., 3., 4.]),
                           g_std=torch.tensor([2., 3., 4., 5.]))
    truth = torch.tensor([[np.log1p(3.), np.log1p(8.), -2., 1.]], dtype=torch.float32)
    standardized = (truth-norm.g_mean)/norm.g_std
    base = standardized+torch.tensor([[.2, -.1, .5, -.4]])
    calibrated = standardized.clone()
    result = compare_global_heads(base, calibrated, truth, norm, state(), [3., 8., -2., 1.])
    np.testing.assert_allclose(result["base"]["standardized_signed_error"], [.2, -.1, .5, -.4], atol=1e-7)
    np.testing.assert_allclose(result["base"]["physical_signed_error"][2:], [2., -2.], atol=1e-7)
    assert result["calibrated"]["physical_prediction"][2] == -2.
    np.testing.assert_allclose(result["calibrated"]["physical_signed_error"], 0., atol=1e-6)
    np.testing.assert_allclose(result["correction"]["standardized"], [-.2, .1, -.5, .4], atol=1e-7)


def test_global_projection_matches_actual_decoder_without_touching_its_rng():
    norm = SimpleNamespace(g_mean=torch.zeros(4), g_std=torch.ones(4),
                           local_mean=torch.zeros(8), local_std=torch.ones(8))
    s = state()
    s[0, 0], s[0, 3], s[0, 4] = .79, .001, 1.2499
    output = torch.zeros(1, 1124)
    output[0, :4] = torch.tensor([11., 21., -100., 100.])
    decoder = SimpleNamespace(norm=norm, stations=torch.full((12,), 46.), device=torch.device("cpu"),
                              generator=torch.Generator().manual_seed(31))
    generator_before = decoder.generator.get_state().clone()
    global_before = torch.get_rng_state().clone()
    predicted, saturated = decoder_global_projection(output[:, :4], norm, s)
    assert torch.equal(decoder.generator.get_state(), generator_before)
    assert torch.equal(torch.get_rng_state(), global_before)
    ns, rewards, dt = ModelAugmentation.decode(decoder, s, output)
    expected = [float(dt[0]), -float(rewards[0]),
                *(((ns-s)[:, 3:5])*20000)[0].tolist()]
    np.testing.assert_array_equal(predicted, expected)
    assert saturated == [True, True, True, True]


def test_zero_context_is_exact_paired_identity_and_summarizes_each_scale():
    world = diagnostic_world()
    world.physical.context_delta_seconds = lambda: np.zeros(3)
    issue, row = issue_and_row()
    world.register(issue)
    with patch.object(DiagnosticWorld, "observe", return_value=[dict(sequence=row.sequence)]):
        log = world.observe([row], 9031.)[0]["four_global_heads"]
    selected = log["families"]["direct"]
    assert selected["base"] == selected["calibrated"]
    for values in selected["correction"].values():
        assert values == [0., 0., 0., 0.]
    summary = summarize_head_records([log])
    assert summary["head_order"] == list(HEAD_ORDER)
    assert summary["overall"]["n"] == 1
    assert summary["by_phase"]["before"]["n"] == 1
    assert summary["by_phase"]["early_after"]["n"] == 0
    assert summary["overall"]["direct_upstream_records"] == 1
    assert summary["overall"]["base"] == summary["overall"]["calibrated"]


def test_saved_issue_family_context_and_head_contributions_survive_later_fit():
    world = diagnostic_world()
    issue, row = issue_and_row()
    world.register(issue)
    original = copy.deepcopy(world.pending[row.sequence])
    world.local_readouts["direct"].add_(10.)
    world.physical.online_weights += 30.
    world.physical.context["7X"] = 100.
    world.physical.parameter_version = 2
    with patch.object(DiagnosticWorld, "observe", return_value=[dict(sequence=row.sequence)]):
        result = world.observe([row], 9031.)[0]["four_global_heads"]
    assert result["selected_family_at_issue"] == "direct"
    assert result["physical_parameter_version_at_issue"] == 1
    assert result["context_at_issue"]["source_reference_nonholding_seconds"] == [110., 120., 130.]
    assert result["context_at_issue"]["online_shift_seconds"] == [4., 0., 0.]
    assert result["context_at_issue"]["local_context_seconds"] == [2., 0., 0.]
    assert result["context_at_issue"]["combined_shift_seconds"] == [6., 0., 0.]
    assert result["phase"] == "before" and result["completed_phase"] == "early_after"
    contributions = np.asarray(result["readout_feature_contributions_standardized"]["direct"])
    np.testing.assert_allclose(contributions.sum(0), result["families"]["direct"]["correction"]["standardized"], atol=1e-7)
    np.testing.assert_array_equal(result["families"]["direct"]["base"]["standardized_prediction"],
                                 original["local_covariates"]["base_g"]["direct"][0].numpy())


def test_instrumented_observe_preserves_readout_fit_rows_and_predictions():
    """Additional scoring must leave v3's real learning inputs exactly unchanged."""
    measured = diagnostic_world()
    reference = copy.deepcopy(measured)
    reference.__class__ = LocalWorld
    issue, row = issue_and_row()
    measured.register(issue)
    reference.register(issue)
    measured_rng, reference_rng = torch.get_rng_state().clone(), np.random.get_state()
    with patch.object(DiagnosticWorld, "observe", side_effect=lambda *args: [dict(sequence=row.sequence)]):
        measured_logs = measured.observe([row], 9031.)
        reference_logs = reference.observe([row], 9031.)
    assert torch.equal(torch.get_rng_state(), measured_rng)
    after_rng = np.random.get_state()
    assert reference_rng[0] == after_rng[0]
    np.testing.assert_array_equal(reference_rng[1], after_rng[1])
    assert reference_rng[2:] == after_rng[2:]
    assert measured.local_labels == reference.local_labels == 1
    for key in measured.local_fit_records[0]:
        torch.testing.assert_close(measured.local_fit_records[0][key], reference.local_fit_records[0][key], rtol=0, atol=0)
    measured.refit_local()
    reference.refit_local()
    for family in reference.local_readouts:
        torch.testing.assert_close(measured.local_readouts[family], reference.local_readouts[family], rtol=0, atol=0)
    assert {key: value for key, value in measured_logs[0].items() if key != "four_global_heads"} == reference_logs[0]
    torch.testing.assert_close(measured.predict(state(), torch.tensor([[-.5]])),
                               reference.predict(state(), torch.tensor([[-.5]])), rtol=0, atol=0)


def test_future_label_rejected_before_mutating_any_fit_or_diagnostic_record():
    world = diagnostic_world()
    issue, row = issue_and_row()
    world.register(issue)
    import pytest
    with pytest.raises(ValueError, match="before availability"):
        world.observe([row], 9030.)
    assert world.head_prediction_records == []
    assert world.local_fit_records == []
    assert world.local_labels == 0
