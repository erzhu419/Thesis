"""Check the separated-model contrasts and their cross-fit sample scope."""

from __future__ import annotations

from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import INTERACTION_NAMES
from experiments.mcwm_closed_loop_separated_contrast_v13_20261021.separated import (
    BACKGROUND_NAMES, assignment_fold,
)


FIELDS = ("dt", "wait", "ride", "cost")
ENDPOINTS = ("global_next", "same_vehicle_next", "fixed300")


def summarize_separation(predictions, separated_model):
    """Validate saved predictions and OOF scope without using paired truth.

    Unsupported early predictions remain unsupported. All decomposition
    comparisons use 1e-6 absolute physical units. Fold-source receipts are
    checked against the model's actual selected records, not another dataset.
    """
    checks = dict(global_cost_rate_deletion_reconstruction=True,
        separated_background_contrast_zero=True, no_cost_refit_action_rate_zero=True,
        separated_fixed300_clock_exact=True, crossfit_selected_training_scope=True,
        crossfit_assignment_blocks_present=True, crossfit_fold_sources_exact=True,
        no_cost_refit_keeps_background_cost_rate=BACKGROUND_NAMES[10] == "global_issue_cost_rate_per_1000")
    models = {name: dict(supported_endpoint_predictions=0, unsupported_endpoint_predictions=0,
        analytic_only_endpoint_predictions=0, max_cost_rate_deletion_error=0.,
        max_background_contrast=0., max_no_cost_rate_action_contribution=0.)
        for name in ("separated", "separated_no_cost")}

    def compare(check, left, right, info, metric):
        error = abs(float(left)-float(right))
        checks[check] &= error <= 1e-6
        info[metric] = max(info[metric], error)

    for saved in predictions:
        for name, diagnostic_key in (("separated", "separated_diag"),
                                     ("separated_no_cost", "separated_no_cost_diag")):
            prediction, diagnostic = saved[name], saved[diagnostic_key]
            result = models[name]
            for arm in ("low", "high"):
                value = prediction[arm]["endpoints"]["fixed300"]
                checks["separated_fixed300_clock_exact"] &= value is None or value["dt"] == 300.
            for endpoint in ENDPOINTS:
                info = diagnostic["endpoints"][endpoint]
                analytic = bool(info["analytic_zero_duration"])
                if info["raw_total"] is None:
                    result["analytic_only_endpoint_predictions" if analytic else "unsupported_endpoint_predictions"] += 1
                    checks["global_cost_rate_deletion_reconstruction"] &= (
                        info["raw_fixed_weight_no_global_cost_rate"] is None)
                    if analytic:
                        for key in FIELDS:
                            compare("global_cost_rate_deletion_reconstruction",
                                info["fixed_weight_no_global_cost_rate"][key], 0., result,
                                "max_cost_rate_deletion_error")
                    else:
                        checks["global_cost_rate_deletion_reconstruction"] &= (
                            info["fixed_weight_no_global_cost_rate"] is None)
                    continue
                result["supported_endpoint_predictions"] += 1
                cost_component = info["state_interactions"][INTERACTION_NAMES[4]]
                for key in FIELDS:
                    expected_raw = info["raw_total"][key]-cost_component[key]
                    compare("global_cost_rate_deletion_reconstruction",
                        info["raw_fixed_weight_no_global_cost_rate"][key], expected_raw, result,
                        "max_cost_rate_deletion_error")
                    compare("global_cost_rate_deletion_reconstruction",
                        info["fixed_weight_no_global_cost_rate"][key], 0. if analytic else expected_raw,
                        result, "max_cost_rate_deletion_error")
                    compare("separated_background_contrast_zero", info["level"][key], 0.,
                        result, "max_background_contrast")
                    if name == "separated_no_cost":
                        compare("no_cost_refit_action_rate_zero", cost_component[key], 0., result,
                            "max_no_cost_rate_action_contribution")
    records = separated_model.records
    checks["crossfit_selected_training_scope"] = all(row.role == "train" and row.released
        and row.status == "complete" and row.selected_for_training for row in records)
    checks["crossfit_assignment_blocks_present"] = all(
        "assignment_block_index" in row.context for row in records)
    if checks["crossfit_assignment_blocks_present"]:
        for endpoint, diagnostic in separated_model.nuisance_diagnostics.items():
            rows = separated_model._endpoint_records(endpoint)
            expected = {fold: [int(row.sequence) for row in rows if assignment_fold(row) == fold]
                        for fold in (0, 1)}
            checks["crossfit_fold_sources_exact"] &= diagnostic["fold_counts"] == {
                str(fold): len(ids) for fold, ids in expected.items()}
            if diagnostic["status"] != "fitted":
                checks["crossfit_fold_sources_exact"] &= not all(expected.values())
                continue
            folds = diagnostic["fold_models"]
            checks["crossfit_fold_sources_exact"] &= len(folds) == 2
            for fold_receipt in folds:
                fold = fold_receipt["heldout_fold"]
                checks["crossfit_fold_sources_exact"] &= (
                    fold_receipt["fit_sequence_ids"] == expected[1-fold]
                    and fold_receipt["heldout_sequence_ids"] == expected[fold]
                    and fold_receipt["fit_rows"] == len(expected[1-fold])
                    and fold_receipt["heldout_rows"] == len(expected[fold]))
    else:
        checks["crossfit_fold_sources_exact"] = False
    return dict(checks=checks, models=models, selected_issue_labels=len(records),
        background_columns=12, action_columns=9,
        no_cost_refit_action_columns=8, comparison_tolerance_physical_units=1e-6)
