"""Detect incorrect physical decomposition without rejecting genuine batch zeros."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.test_endpoints import issue
from experiments.mcwm_closed_loop_response_residual_v11_20261021.test_residual import observable_context
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.diagnostics import summarize_action_differences
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import StateInteractionReadout
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.test_interaction import trained_model
from experiments.mcwm_closed_loop_v1.stream import EI


def bundle(model, state, context):
    return dict(interaction=model.predict_issue(state, context),
        interaction_diag=model.contrast_diagnostic(state, context))


def test_real_fitted_states_and_preanalytic_nonzero_batch_gain_reconstruct():
    model, _ = trained_model()
    saved = []
    for forward in (75., 525.):
        state = issue()
        state["event"][EI["forward_headway_seconds"]] = forward
        saved.append(bundle(model, state, observable_context()))
    zero_state = issue()
    zero_state["s"][1] = 2/16
    zero_state["event"][EI["forward_headway_seconds"]] = 450.
    saved.append(bundle(model, zero_state, observable_context()))
    assert saved[-1]["interaction_diag"]["endpoints"]["global_next"]["raw_total"]["cost"] != 0.
    assert saved[-1]["interaction"]["gains"]["global_next"] == 0.
    result = summarize_action_differences(saved)
    assert all(result["checks"].values())
    assert result["max_reconstruction_error"] <= 1e-6
    assert result["supported_endpoint_predictions"] == 9
    assert result["unsupported_endpoint_predictions"] == 0
    assert result["analytic_only_endpoint_predictions"] == 0
    assert result["analytic_zero_endpoint_predictions"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("corruption", ("component", "level", "gain", "deleted"))
def test_changed_physical_component_or_saved_contrast_fails_its_integrity_gate(corruption):
    model, _ = trained_model()
    saved = bundle(model, issue(), observable_context())
    info = saved["interaction_diag"]["endpoints"]["fixed300"]
    if corruption == "component":
        next(iter(info["state_interactions"].values()))["cost"] += 1.
        expected = "action_contrast_reconstruction"
    elif corruption == "level":
        info["level"]["wait"] = 1.
        expected = "state_level_terms_cancel"
    elif corruption == "gain":
        saved["interaction"]["gains"]["fixed300"] += 1.
        expected = "action_contrast_predictions_match"
    else:
        info["fixed_weight_no_state_interactions"]["cost"] += 1.
        expected = "fixed_weight_deletion_reconstruction"
    result = summarize_action_differences([saved])
    assert not result["checks"][expected]
    assert result["max_reconstruction_error"] >= 1.


def test_unsupported_and_analytic_only_predictions_have_separate_counts():
    model = StateInteractionReadout()
    state, zero_state = issue(), issue()
    zero_state["s"][1] = 2/16
    saved = [bundle(model, selected, observable_context()) for selected in (state, zero_state)]
    result = summarize_action_differences(saved)
    assert all(result["checks"].values())
    assert result["supported_endpoint_predictions"] == 0
    assert result["unsupported_endpoint_predictions"] == 5
    assert result["analytic_only_endpoint_predictions"] == 1
    assert result["max_reconstruction_error"] == 0.
    corrupted = deepcopy(saved)
    corrupted[0]["interaction"]["gains"]["fixed300"] = 0.
    assert not summarize_action_differences(corrupted)["checks"]["unsupported_contrasts_preserved"]
