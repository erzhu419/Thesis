"""Passive background/contrast separation on the existing controlled labels.

Two-fold nuisance fits leave each balanced assignment block together. This is
an action-difference identification diagnostic, not training on paired truths
or a causal identification claim about sequential calibration assignments.
"""

from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import (
    ENDPOINT_NAMES, LINE_NAMES, RESPONSE_FEATURE_NAMES, TARGET_SCALES, response_design,
)
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import (
    RESIDUAL_TARGET_NAMES, ResidualResponseReadout, _action_contrast, _view,
    _within_batch, endpoint_baseline,
)
from experiments.mcwm_closed_loop_state_interaction_v12_20261021.interaction import (
    FEATURE_SCALES, INTERACTION_NAMES, STATE_NAMES, _difference, _physical_component,
    _state_features,
)


LEVEL_INDICES = tuple(range(3)) + tuple(range(6, 13))
BACKGROUND_NAMES = tuple(RESPONSE_FEATURE_NAMES[i] for i in LEVEL_INDICES) + STATE_NAMES[4:]
ACTION_NAMES = tuple(RESPONSE_FEATURE_NAMES[3:6]) + INTERACTION_NAMES
FEATURE_NAMES = BACKGROUND_NAMES + ACTION_NAMES
COST_RATE_ACTION_COLUMN = 7


def separated_design(record, *, hold_seconds=None):
    original = response_design(record, hold_seconds=hold_seconds)
    state = _state_features(record, original)
    background = np.concatenate((original[list(LEVEL_INDICES)], state[4:]))
    hold = record.hold_seconds if hold_seconds is None else float(hold_seconds)
    action = np.concatenate((original[3:6], (2.*hold/7.5-1.)*state))
    return background, action


def assignment_fold(record):
    """Both issues of a line's balanced assignment block share one fold."""
    return (int(record.context["assignment_block_index"])+LINE_NAMES.index(record.line_id)) % 2


def _ridge(design, target, ridge):
    return np.linalg.solve(design.T@design+ridge*np.eye(design.shape[1]), design.T@target)


def _rank_summary(design, ridge):
    result = dict(rows=len(design), columns=int(design.shape[1]), rank=0,
                  active_columns=0, ridge=float(ridge), ridge_precision_condition=1.)
    if len(design):
        singular = np.linalg.svd(design, compute_uv=False)
        minimum = float(singular[-1]) if len(singular) == design.shape[1] else 0.
        result.update(rank=int(np.linalg.matrix_rank(design)),
            active_columns=int((design != 0).any(0).sum()),
            ridge_precision_condition=float((singular[0]**2+ridge)/(minimum**2+ridge)))
    return result


def _correlation(a, b):
    a, b = a-a.mean(), b-b.mean()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denominator) if denominator else None


def _error_metric(error, names, scales, *, absolute=False):
    physical = {name: error[:, i]*scale for i, (name, scale) in enumerate(zip(names, scales))}
    physical.setdefault("dt", np.zeros(len(error)))
    physical["cost"] = 2.*physical["wait"]+physical["ride"]
    return {name: float(np.mean(abs(value)) if absolute else np.sqrt(np.mean(value**2)))
            for name, value in physical.items()}


class SeparatedContrastReadout(ResidualResponseReadout):
    """Learn action slopes from out-of-fold residuals, then restore levels."""

    def __init__(self, *, ridge=1.):
        super().__init__(ridge=ridge)
        self.no_cost_readouts = {}
        self.nuisance_diagnostics = {}

    def refit(self):
        if not self.records:
            return
        joined = np.asarray([np.concatenate(separated_design(row)) for row in self.records])
        self.design = _rank_summary(joined, self.ridge)
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            rows = self._endpoint_records(endpoint)
            z = np.asarray([separated_design(row)[0] for row in rows]).reshape(-1, 12)
            t = np.asarray([separated_design(row)[1] for row in rows]).reshape(-1, 9)
            folds = np.asarray([assignment_fold(row) for row in rows])
            counts = {str(fold): int((folds == fold).sum()) for fold in (0, 1)}
            self.endpoint_designs[endpoint] = _rank_summary(np.concatenate((z, t), axis=1), self.ridge)
            self.endpoint_designs[endpoint].update(learned_targets=list(names), fold_counts=counts)
            if not all(counts.values()):
                self.nuisance_diagnostics[endpoint] = dict(status="requires_two_assignment_folds",
                    fold_counts=counts)
                continue
            scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
            y = np.asarray([[row.endpoints[endpoint][key]-endpoint_baseline(row, endpoint)[key]
                            for key in names] for row in rows])/scales
            m, q = np.empty_like(y), np.empty_like(t)
            fold_models = []
            for fold in (0, 1):
                train, test = folds != fold, folds == fold
                m_weight, q_weight = _ridge(z[train], y[train], self.ridge), _ridge(z[train], t[train], self.ridge)
                m[test], q[test] = z[test]@m_weight, z[test]@q_weight
                fold_models.append(dict(heldout_fold=fold, fit_rows=int(train.sum()),
                    heldout_rows=int(test.sum()),
                    fit_sequence_ids=[int(row.sequence) for row, selected in zip(rows, train) if selected],
                    heldout_sequence_ids=[int(row.sequence) for row, selected in zip(rows, test) if selected],
                    background_outcome_coefficients={name: _physical_component(value, names, scales)
                        for name, value in zip(BACKGROUND_NAMES, m_weight)}))
            e, v = y-m, t-q
            beta = _ridge(v, e, self.ridge)
            gamma = _ridge(z, y-t@beta, self.ridge)
            self.readouts[endpoint] = np.concatenate((gamma, beta), axis=0)
            keep = np.arange(9) != COST_RATE_ACTION_COLUMN
            no_cost_beta = np.zeros_like(beta)
            no_cost_beta[keep] = _ridge(v[:, keep], e, self.ridge)
            no_cost_gamma = _ridge(z, y-t@no_cost_beta, self.ridge)
            self.no_cost_readouts[endpoint] = np.concatenate((no_cost_gamma, no_cost_beta), axis=0)
            rate = t[:, COST_RATE_ACTION_COLUMN]
            residual_rate = v[:, COST_RATE_ACTION_COLUMN]
            self.nuisance_diagnostics[endpoint] = dict(status="fitted", fold_counts=counts,
                fold_models=fold_models,
                oof_background_error_rmse=_error_metric(e, names, scales),
                oof_background_error_mae=_error_metric(e, names, scales, absolute=True),
                oof_action_error_rmse={name: float(value) for name, value in
                    zip(ACTION_NAMES, np.sqrt(np.mean(v**2, axis=0)))},
                residualized_action_design=_rank_summary(v, self.ridge),
                no_cost_rate_residualized_action_design=_rank_summary(v[:, keep], self.ridge),
                global_cost_rate_action=dict(original_norm=float(np.linalg.norm(rate)),
                    oof_predicted_norm=float(np.linalg.norm(q[:, COST_RATE_ACTION_COLUMN])),
                    residualized_norm=float(np.linalg.norm(residual_rate)),
                    original_background_correlations={name: _correlation(rate, column) for name, column in zip(BACKGROUND_NAMES, z.T)},
                    residual_background_correlations={name: _correlation(residual_rate, column) for name, column in zip(BACKGROUND_NAMES, z.T)},
                    action_coefficient_physical=_physical_component(beta[COST_RATE_ACTION_COLUMN], names, scales),
                    background_coefficient_physical=_physical_component(gamma[10], names, scales)),
                orthogonal_score_error_rmse=_error_metric(e-v@beta, names, scales),
                no_cost_rate_orthogonal_score_error_rmse=_error_metric(e-v@no_cost_beta, names, scales))

    def _predict(self, record, *, hold_seconds=None, baseline_only=False, omit_cost_rate=False):
        support, coverage = self.coverage().get(record.line_id, {}), self.endpoint_coverage()
        weights = self.no_cost_readouts if omit_cost_rate else self.readouts
        design = None if baseline_only else np.concatenate(separated_design(record, hold_seconds=hold_seconds))
        endpoints, statuses, baselines = {}, {}, {}
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            base = endpoint_baseline(record, endpoint, hold_seconds=hold_seconds)
            baselines[endpoint] = base
            local = coverage[endpoint].get(record.line_id, {})
            fitted = endpoint in weights and bool(local.get("zero_holding")) and bool(local.get("high_holding"))
            if base is None:
                endpoints[endpoint], statuses[endpoint] = None, "missing_issue_baseline"
                continue
            if not baseline_only and not fitted:
                endpoints[endpoint], statuses[endpoint] = None, "unsupported"
                continue
            value = dict(base)
            if not baseline_only:
                scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
                for key, number in zip(names, (design@weights[endpoint])*scales):
                    value[key] += float(number)
            value["cost"] = 2.*value["wait"]+value["ride"]
            endpoints[endpoint], statuses[endpoint] = value, "supported"
        raw_global = None if endpoints["global_next"] is None else dict(endpoints["global_next"])
        if _within_batch(record):
            endpoints["global_next"] = dict(dt=0., wait=0., ride=0., cost=0., zero_duration_analytic=True)
            statuses["global_next"] = "analytic_zero_duration"
        return dict(status="supported" if any(value in ("supported", "analytic_zero_duration")
            for value in statuses.values()) else "unsupported", support=support,
            endpoint_support={name: coverage[name].get(record.line_id, {}) for name in ENDPOINT_NAMES},
            endpoint_status=statuses, endpoints=endpoints, raw_global_next=raw_global,
            issue_time_baselines=baselines)

    def predict_issue(self, issued, context=None, *, omit_cost_rate=False):
        record = _view(issued, context)
        result = _action_contrast(self._predict(record, hold_seconds=0., omit_cost_rate=omit_cost_rate),
                                 self._predict(record, hold_seconds=7.5, omit_cost_rate=omit_cost_rate))
        # Contrast the action model directly: the same background must cancel
        # before adding/subtracting potentially large passenger-cost levels.
        weights = self.no_cost_readouts if omit_cost_rate else self.readouts
        _, tl = separated_design(record, hold_seconds=0.)
        _, th = separated_design(record, hold_seconds=7.5)
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            if result["low"]["raw_global_next"] is None and endpoint == "global_next":
                continue
            support = result["endpoint_support"][endpoint]
            if endpoint not in weights or not (support.get("zero_holding") and support.get("high_holding")):
                continue
            low = result["low"]["issue_time_baselines"][endpoint]
            high = result["high"]["issue_time_baselines"][endpoint]
            if low is None or high is None:
                continue
            baseline = _difference(low, high)
            scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
            residual = _physical_component((tl-th)@weights[endpoint][12:], names, scales)
            raw = {key: baseline[key]+residual[key] for key in baseline}
            if endpoint == "global_next":
                result["raw_global_next_gain"] = raw["cost"]
                result["raw_global_next_high_minus_low"] = {key: -value for key, value in raw.items()}
            total = {key: 0. for key in raw} if endpoint == "global_next" and _within_batch(record) else raw
            result["gains"][endpoint] = total["cost"]
            result["high_minus_low"][endpoint] = {key: -value for key, value in total.items()}
        return result

    def contrast_diagnostic(self, issued, context=None, *, omit_cost_rate=False):
        record = _view(issued, context)
        zl, tl = separated_design(record, hold_seconds=0.)
        zh, th = separated_design(record, hold_seconds=7.5)
        difference = tl-th
        prediction = self.predict_issue(issued, context, omit_cost_rate=omit_cost_rate)
        weights = self.no_cost_readouts if omit_cost_rate else self.readouts
        zero = dict(dt=0., wait=0., ride=0., cost=0.)
        endpoints = {}
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            low, high = endpoint_baseline(record, endpoint, hold_seconds=0.), endpoint_baseline(record, endpoint, hold_seconds=7.5)
            baseline = None if low is None or high is None else _difference(low, high)
            support = prediction["endpoint_support"][endpoint]
            fitted = endpoint in weights and bool(support.get("zero_holding")) and bool(support.get("high_holding"))
            info = dict(status=prediction["endpoint_status"][endpoint], support=support, baseline=baseline,
                level=None, line_holding=None, state_interactions=None, residual_total=None,
                raw_total=None, analytic_adjustment=None, total=None,
                fixed_weight_no_state_interactions=None, raw_fixed_weight_no_state_interactions=None,
                fixed_weight_no_global_cost_rate=None, raw_fixed_weight_no_global_cost_rate=None,
                analytic_zero_duration=endpoint == "global_next" and _within_batch(record))
            if fitted and baseline is not None:
                gamma, beta = weights[endpoint][:12], weights[endpoint][12:]
                scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
                level = _physical_component((zl-zh)@gamma, names, scales)
                holding = _physical_component(difference[:3]@beta[:3], names, scales)
                interactions = {name: _physical_component(difference[i]*beta[i], names, scales)
                                for i, name in enumerate(INTERACTION_NAMES, start=3)}
                residual = _physical_component(difference@beta, names, scales)
                raw_total = {key: baseline[key]+residual[key] for key in zero}
                total = dict(zero) if info["analytic_zero_duration"] else dict(raw_total)
                raw_deleted = {key: baseline[key]+holding[key] for key in zero}
                rate_contribution = interactions[INTERACTION_NAMES[4]]
                raw_no_rate = {key: raw_total[key]-rate_contribution[key] for key in zero}
                info.update(level=level, line_holding=holding, state_interactions=interactions,
                    residual_total=residual, raw_total=raw_total, total=total,
                    analytic_adjustment={key: total[key]-raw_total[key] for key in zero},
                    raw_fixed_weight_no_state_interactions=raw_deleted,
                    fixed_weight_no_state_interactions=dict(zero) if info["analytic_zero_duration"] else raw_deleted,
                    raw_fixed_weight_no_global_cost_rate=raw_no_rate,
                    fixed_weight_no_global_cost_rate=dict(zero) if info["analytic_zero_duration"] else raw_no_rate)
            elif info["analytic_zero_duration"]:
                info.update(total=dict(zero), fixed_weight_no_state_interactions=dict(zero),
                            fixed_weight_no_global_cost_rate=dict(zero))
            endpoints[endpoint] = info
        return dict(gain_semantics="low_0_seconds_minus_high_7_5_seconds",
            arm="no_global_cost_rate_refit" if omit_cost_rate else "full",
            feature_names=list(FEATURE_NAMES), background_names=list(BACKGROUND_NAMES), action_names=list(ACTION_NAMES),
            feature_scales=dict(FEATURE_SCALES),
            feature_values=dict(low=np.concatenate((zl, tl)).tolist(), high=np.concatenate((zh, th)).tolist(),
                low_minus_high=np.concatenate((zl-zh, difference)).tolist()),
            state_values={name: float(value) for name, value in zip(STATE_NAMES, _state_features(record, response_design(record)))},
            endpoints=endpoints)

    def summary(self):
        result = super().summary()
        result.update(model="crossfit_separated_background_action_response_residual",
            feature_names=list(FEATURE_NAMES), background_names=list(BACKGROUND_NAMES), action_names=list(ACTION_NAMES),
            feature_scales=dict(FEATURE_SCALES), nuisance_diagnostics=self.nuisance_diagnostics,
            no_global_cost_rate_coefficients={name: value.tolist() for name, value in self.no_cost_readouts.items()},
            crossfit=dict(folds=2, fold_rule="(assignment_block_index + line_index) % 2",
                assignment_unit="whole balanced two-issue line block", ridge=self.ridge),
            semantics="OOF m(Z), q(Z); ridge T-q on Y-m; background refit on Y-T beta; same selected released single-action labels, no paired-truth training, unchanged analytic clocks, baseline and trust")
        return result
