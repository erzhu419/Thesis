"""Check saved physical action-gain decompositions without observing truth."""

from __future__ import annotations

from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import INTERACTION_NAMES


FIELDS = ("dt", "wait", "ride", "cost")
ENDPOINTS = ("global_next", "same_vehicle_next", "fixed300")


def summarize_action_differences(predictions):
    """Verify fitted contrasts, analytic zeros and fixed-weight deletion gains.

    The input is each saved prediction bundle, containing ``interaction`` and
    ``interaction_diag``. Unsupported fits have no invented zero components.
    An exact analytic global zero without a fitted raw response is counted
    separately. All comparisons use an absolute tolerance of 1e-6 in physical
    endpoint units; the helper never consumes a realized response.
    """
    checks = dict(action_contrast_reconstruction=True, state_level_terms_cancel=True,
        action_contrast_predictions_match=True, fixed_weight_deletion_reconstruction=True,
        analytic_zero_contrast_exact=True, unsupported_contrasts_preserved=True)
    maximum = 0.
    supported = unsupported = analytic_only = analytic_zero = 0

    def compare(check, left, right):
        nonlocal maximum
        error = abs(float(left)-float(right))
        maximum = max(maximum, error)
        checks[check] &= error <= 1e-6

    for saved in predictions:
        prediction, diagnostic = saved["interaction"], saved["interaction_diag"]
        for name in ENDPOINTS:
            info = diagnostic["endpoints"][name]
            analytic = bool(info["analytic_zero_duration"])
            if analytic:
                analytic_zero += 1
                for key in FIELDS:
                    compare("analytic_zero_contrast_exact", info["total"][key], 0.)
                    compare("analytic_zero_contrast_exact", info["fixed_weight_no_state_interactions"][key], 0.)
            if info["raw_total"] is None:
                if analytic:
                    analytic_only += 1
                else:
                    unsupported += 1
                    checks["unsupported_contrasts_preserved"] &= info["total"] is None \
                        and prediction["gains"][name] is None \
                        and prediction["high_minus_low"][name] is None
                checks["unsupported_contrasts_preserved"] &= all(info[key] is None for key in (
                    "level", "line_holding", "state_interactions", "residual_total",
                    "raw_fixed_weight_no_state_interactions", "analytic_adjustment"))
                if analytic:
                    compare("action_contrast_predictions_match", prediction["gains"][name], 0.)
                    for key in FIELDS:
                        compare("action_contrast_predictions_match", prediction["high_minus_low"][name][key], 0.)
                continue
            supported += 1
            checks["action_contrast_reconstruction"] &= set(info["state_interactions"]) == set(INTERACTION_NAMES)
            for key in FIELDS:
                state_gain = sum(info["state_interactions"][feature][key] for feature in INTERACTION_NAMES)
                residual = info["level"][key]+info["line_holding"][key]+state_gain
                compare("state_level_terms_cancel", info["level"][key], 0.)
                compare("action_contrast_reconstruction", info["residual_total"][key], residual)
                compare("action_contrast_reconstruction", info["raw_total"][key], info["baseline"][key]+residual)
                compare("action_contrast_reconstruction", info["total"][key],
                    info["raw_total"][key]+info["analytic_adjustment"][key])
                raw_deleted = info["baseline"][key]+info["line_holding"][key]
                compare("fixed_weight_deletion_reconstruction",
                    info["raw_fixed_weight_no_state_interactions"][key], raw_deleted)
                compare("fixed_weight_deletion_reconstruction", info["fixed_weight_no_state_interactions"][key],
                    0. if analytic else raw_deleted)
                compare("action_contrast_predictions_match", info["total"][key],
                    -prediction["high_minus_low"][name][key])
            compare("action_contrast_predictions_match", info["total"]["cost"], prediction["gains"][name])
    return dict(checks=checks, max_reconstruction_error=maximum,
        supported_endpoint_predictions=supported, unsupported_endpoint_predictions=unsupported,
        analytic_only_endpoint_predictions=analytic_only, analytic_zero_endpoint_predictions=analytic_zero,
        comparison_tolerance_physical_units=1e-6)
