"""Passive response residuals with shared, issue-state-dependent action slopes.

The first 13 columns and observable endpoint baselines are the v11 design.
Six shared holding interactions allow the action contrast to vary by state
without fitting 18 sparse route-specific slopes or changing level covariates.
"""

from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import (
    ENDPOINT_NAMES, LINE_NAMES, RESPONSE_FEATURE_NAMES, TARGET_SCALES, response_design,
)
from experiments.mcwm_closed_loop_response_residual_v11_20261021.residual import (
    RESIDUAL_TARGET_NAMES, ResidualResponseReadout, _view, _within_batch, endpoint_baseline,
)


STATE_NAMES = (
    "forward_headway_deviation", "backward_headway_deviation", "queue_capacity",
    "onboard_capacity", "global_issue_cost_rate_per_1000", "source_nonholding_per_100",
)
INTERACTION_NAMES = tuple(f"holding_x_{name}" for name in STATE_NAMES)
INTERACTION_FEATURE_NAMES = RESPONSE_FEATURE_NAMES + INTERACTION_NAMES
FEATURE_SCALES = dict(
    holding="u = 2 * executed_holding_seconds / 7.5 - 1",
    forward_headway_deviation="forward / target_forward - 1",
    backward_headway_deviation="backward / target_backward - 1",
    queue_capacity="eligible_stranded_passengers / bus_capacity",
    onboard_capacity="onboard_passengers / bus_capacity",
    global_issue_cost_rate_per_1000="(2 * global_waiting + global_onboard) / 1000",
    source_nonholding_per_100="fixed_source_nonholding_seconds / 100",
)


def _state_features(record, original):
    ledger = getattr(record, "issue_ledger", None)
    if ledger is None:
        ledger = record.context.get("issue_ledger")
    nonholding = record.context.get("nonholding_seconds")
    if ledger is None or nonholding is None:
        raise ValueError("State interactions require the issue ledger and fixed-source nonholding prediction")
    levels = np.asarray(((2.*float(ledger["waiting_passengers"])
                          + float(ledger["in_vehicle_passengers"]))/1000.,
                         float(nonholding)/100.))
    return np.concatenate((original[6:10], levels))


def state_interaction_design(record, *, hold_seconds=None):
    """Append only six action interactions observable at the issue clock."""
    original = response_design(record, hold_seconds=hold_seconds)
    state = _state_features(record, original)
    hold = record.hold_seconds if hold_seconds is None else float(hold_seconds)
    return np.concatenate((original, (2.*hold/7.5-1.)*state))


def _design_summary(design, ridge, *, learned_targets=None):
    result = dict(rows=len(design), columns=len(INTERACTION_FEATURE_NAMES),
                  rank=0, active_columns=0, ridge=float(ridge),
                  ridge_precision_condition=1.)
    if len(design):
        singular = np.linalg.svd(design, compute_uv=False)
        minimum = float(singular[-1]) if len(singular) == design.shape[1] else 0.
        result.update(rank=int(np.linalg.matrix_rank(design)),
            active_columns=int((design != 0).any(0).sum()),
            ridge_precision_condition=float((singular[0]**2+ridge)/(minimum**2+ridge)))
    if learned_targets is not None:
        result["learned_targets"] = list(learned_targets)
    return result


def _physical_component(value, names, scales):
    output = dict(dt=0., wait=0., ride=0.)
    output.update({name: float(number*scale) for name, number, scale in zip(names, value, scales)})
    output["cost"] = 2.*output["wait"]+output["ride"]
    return output


def _difference(low, high):
    return {key: float(low[key]-high[key]) for key in ("dt", "wait", "ride", "cost")}


def _contrast_design_summary(records, ridge):
    design = np.asarray([state_interaction_design(row) for row in records])
    contrast = np.asarray([state_interaction_design(row, hold_seconds=0.)
        -state_interaction_design(row, hold_seconds=7.5) for row in records])
    result = _design_summary(contrast, ridge)
    result["feature_names"] = list(INTERACTION_FEATURE_NAMES)
    result["shared_interaction_columns"] = list(INTERACTION_NAMES)
    result["shared_interaction_rank"] = int(np.linalg.matrix_rank(contrast[:, 13:])) if len(contrast) else 0
    result["rank_added_beyond_original13"] = (
        int(np.linalg.matrix_rank(design)-np.linalg.matrix_rank(design[:, :13])) if len(design) else 0)
    return result


class StateInteractionReadout(ResidualResponseReadout):
    """Fit the same selected endpoint residuals using 19 issue-time columns."""

    def refit(self):
        if not self.records:
            return
        design = np.stack([state_interaction_design(row) for row in self.records])
        self.design = _design_summary(design, self.ridge)
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            records = self._endpoint_records(endpoint)
            design = np.asarray([state_interaction_design(row) for row in records])
            self.endpoint_designs[endpoint] = _design_summary(design, self.ridge, learned_targets=names)
            if not records:
                continue
            baselines = [endpoint_baseline(row, endpoint) for row in records]
            residuals = np.asarray([[row.endpoints[endpoint][key]-base[key] for key in names]
                                    for row, base in zip(records, baselines)])
            scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
            precision = design.T@design+self.ridge*np.eye(design.shape[1])
            self.readouts[endpoint] = np.linalg.solve(precision, design.T@(residuals/scales))

    def _predict(self, record, *, hold_seconds=None, baseline_only=False):
        support, coverage = self.coverage().get(record.line_id, {}), self.endpoint_coverage()
        design = None if baseline_only else state_interaction_design(record, hold_seconds=hold_seconds)
        endpoints, statuses, baselines = {}, {}, {}
        for endpoint in ENDPOINT_NAMES:
            baseline = endpoint_baseline(record, endpoint, hold_seconds=hold_seconds)
            baselines[endpoint] = baseline
            local_support = coverage[endpoint].get(record.line_id, {})
            if baseline is None:
                endpoints[endpoint], statuses[endpoint] = None, "missing_issue_baseline"
                continue
            fitted = endpoint in self.readouts and bool(local_support.get("zero_holding")) \
                and bool(local_support.get("high_holding"))
            if not baseline_only and not fitted:
                endpoints[endpoint], statuses[endpoint] = None, "unsupported"
                continue
            value = dict(baseline)
            if not baseline_only:
                scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
                residual = (design@self.readouts[endpoint])*scales
                for key, number in zip(RESIDUAL_TARGET_NAMES[endpoint], residual):
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

    def contrast_diagnostic(self, issued, context=None):
        """Decompose low-minus-high gains before and after the exact zero override.

        Every component is in endpoint physical units. The state level terms
        have exactly zero contrast; action interactions are shared across lines.
        """
        record = _view(issued, context)
        low = state_interaction_design(record, hold_seconds=0.)
        high = state_interaction_design(record, hold_seconds=7.5)
        difference = low-high
        prediction = self.predict_issue(issued, context)
        zero = dict(dt=0., wait=0., ride=0., cost=0.)
        endpoints = {}
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            status = prediction["endpoint_status"][endpoint]
            base_low = endpoint_baseline(record, endpoint, hold_seconds=0.)
            base_high = endpoint_baseline(record, endpoint, hold_seconds=7.5)
            baseline = None if base_low is None or base_high is None else _difference(base_low, base_high)
            support = prediction["endpoint_support"][endpoint]
            fitted = endpoint in self.readouts and bool(support.get("zero_holding")) \
                and bool(support.get("high_holding"))
            info = dict(status=status, support=support, baseline=baseline,
                level=None, line_holding=None, state_interactions=None, residual_total=None,
                raw_total=None, analytic_adjustment=None, total=None,
                fixed_weight_no_state_interactions=None, raw_fixed_weight_no_state_interactions=None,
                analytic_zero_duration=endpoint == "global_next" and _within_batch(record))
            if fitted and baseline is not None:
                weight = self.readouts[endpoint]
                scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
                level_columns = list(range(3))+list(range(6, 13))
                level = _physical_component(difference[level_columns]@weight[level_columns], names, scales)
                holding = _physical_component(difference[3:6]@weight[3:6], names, scales)
                interactions = {name: _physical_component(difference[column]*weight[column], names, scales)
                                for column, name in enumerate(INTERACTION_NAMES, start=13)}
                residual = _physical_component(difference@weight, names, scales)
                raw_total = {key: baseline[key]+residual[key] for key in zero}
                total = dict(zero) if info["analytic_zero_duration"] else dict(raw_total)
                adjustment = {key: total[key]-raw_total[key] for key in zero}
                raw_deleted = {key: baseline[key]+holding[key] for key in zero}
                deleted = dict(zero) if info["analytic_zero_duration"] else dict(raw_deleted)
                info.update(level=level, line_holding=holding, state_interactions=interactions,
                    residual_total=residual, raw_total=raw_total, analytic_adjustment=adjustment, total=total,
                    fixed_weight_no_state_interactions=deleted,
                    raw_fixed_weight_no_state_interactions=raw_deleted)
            elif info["analytic_zero_duration"]:
                info["total"] = dict(zero)
                info["fixed_weight_no_state_interactions"] = dict(zero)
            endpoints[endpoint] = info
        return dict(gain_semantics="low_0_seconds_minus_high_7_5_seconds",
            feature_names=list(INTERACTION_FEATURE_NAMES), feature_scales=dict(FEATURE_SCALES),
            feature_values=dict(low=low.tolist(), high=high.tolist(), low_minus_high=difference.tolist()),
            state_values={name: float(value) for name, value in zip(STATE_NAMES, _state_features(record, low[:13]))},
            endpoints=endpoints)

    def summary(self):
        result = super().summary()
        ranges = {}
        for line in LINE_NAMES:
            rows = [row for row in self.records if row.line_id == line]
            values = np.asarray([_state_features(row, response_design(row)) for row in rows])
            ranges[line] = dict(selected_issue_labels=len(rows), states={
                name: dict(min=float(values[:, index].min()), max=float(values[:, index].max()))
                if len(rows) else dict(min=None, max=None) for index, name in enumerate(STATE_NAMES)})
        result.update(model="issue_time_baseline_shared_state_action_interaction_residual",
            feature_names=list(INTERACTION_FEATURE_NAMES), feature_scales=dict(FEATURE_SCALES),
            original_feature_columns=13, added_level_covariates=[],
            contrast_design=_contrast_design_summary(self.records, self.ridge),
            endpoint_contrast_designs={name: _contrast_design_summary(self._endpoint_records(name), self.ridge)
                                      for name in ENDPOINT_NAMES},
            perline_state_ranges=ranges,
            shared_state_interaction_columns=list(INTERACTION_NAMES),
            holding_code="u = 2 * hold_seconds / 7.5 - 1",
            semantics="Same selected controlled issue residual labels; six shared state holding interactions, exact fixed300 clock, unchanged issue-time baseline and trust")
        return result
