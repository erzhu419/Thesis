"""Gate only real separation/receipt failures using an actual fitted model."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.test_endpoints import issue
from experiments.mcwm_closed_loop_response_residual_v11_20261021.test_residual import observable_context
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.diagnostics import summarize_separation
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.separated import BACKGROUND_NAMES, SeparatedContrastReadout
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.test_separated import fitted_model
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import INTERACTION_NAMES


def saved_prediction(model, *, zero=False):
    query, context = issue(), observable_context()
    if zero:
        query["s"][1] = 2/16
    return dict(separated=model.predict_issue(query, context),
        separated_no_cost=model.predict_issue(query, context, omit_cost_rate=True),
        separated_diag=model.contrast_diagnostic(query, context),
        separated_no_cost_diag=model.contrast_diagnostic(query, context, omit_cost_rate=True))


def test_actual_model_cost_deletions_and_crossfit_receipts_pass():
    model, _ = fitted_model(background_scale=1000.)
    result = summarize_separation([saved_prediction(model), saved_prediction(model, zero=True)], model)
    assert all(result["checks"].values())
    assert result["selected_issue_labels"] == 48
    assert result["models"]["separated"]["supported_endpoint_predictions"] == 6
    assert result["models"]["separated_no_cost"]["max_no_cost_rate_action_contribution"] == 0.
    assert BACKGROUND_NAMES[10] == "global_issue_cost_rate_per_1000"
    assert any(abs(weight[10]).max() > 0. for weight in model.no_cost_readouts.values())


def test_corrupted_cost_deletion_fails_its_specific_gate():
    model, _ = fitted_model()
    saved = saved_prediction(model)
    saved["separated_diag"]["endpoints"]["fixed300"]["raw_fixed_weight_no_global_cost_rate"]["cost"] += 1.
    result = summarize_separation([saved], model)
    assert not result["checks"]["global_cost_rate_deletion_reconstruction"]
    assert result["models"]["separated"]["max_cost_rate_deletion_error"] == 1.
    assert result["checks"]["separated_background_contrast_zero"]


def test_corrupted_refit_arm_and_oof_source_ids_fail_distinct_gates():
    model, _ = fitted_model()
    saved = saved_prediction(model)
    saved["separated_no_cost_diag"]["endpoints"]["fixed300"]["state_interactions"][INTERACTION_NAMES[4]]["cost"] = 2.
    model.nuisance_diagnostics["fixed300"]["fold_models"][0]["fit_sequence_ids"].append(999999)
    result = summarize_separation([saved], model)
    assert not result["checks"]["no_cost_refit_action_rate_zero"]
    assert not result["checks"]["crossfit_fold_sources_exact"]


def test_unsupported_early_predictions_are_preserved_including_analytic_only_zero():
    model = SeparatedContrastReadout()
    result = summarize_separation([saved_prediction(model), saved_prediction(model, zero=True)], model)
    assert all(result["checks"].values())
    assert result["models"]["separated"]["unsupported_endpoint_predictions"] == 5
    assert result["models"]["separated"]["analytic_only_endpoint_predictions"] == 1


def test_background_weight_changes_cannot_change_reported_action_contrast():
    model, _ = fitted_model()
    saved = saved_prediction(model)
    for weights in (model.readouts, model.no_cost_readouts):
        for value in weights.values():
            value[:12] += 1.e12
    changed = saved_prediction(model)
    for name in ("separated", "separated_no_cost"):
        assert saved[name]["gains"] == changed[name]["gains"]
        assert saved[name]["high_minus_low"] == changed[name]["high_minus_low"]
    assert all(summarize_separation([changed], model)["checks"].values())
