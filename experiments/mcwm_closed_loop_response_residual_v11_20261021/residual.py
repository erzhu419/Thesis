"""Passive, issue-time baseline plus residual fits for three native endpoints.

The fixed 300-second endpoint has no learned clock.  The other endpoints retain
an unconstrained pre-analytic response for diagnosis; only the known same-batch
global transition is replaced by its exact zero-duration/zero-cost result.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.endpoints import (
    ControlledResponseReadout, ENDPOINT_NAMES, LINE_NAMES, RESPONSE_FEATURE_NAMES,
    TARGET_SCALES, response_design,
)


RESIDUAL_TARGET_NAMES = {
    "global_next": ("dt", "wait", "ride"),
    "same_vehicle_next": ("dt", "wait", "ride"),
    "fixed300": ("wait", "ride"),
}


def endpoint_baseline(record, endpoint, *, hold_seconds=None):
    """Return physical-unit predictions from facts available at action issue.

    Waiting/onboard counts are global passenger counts at the issue clock.
    The same-car clock comes only from the fixed source physical predictor;
    no realized future duration or passenger ledger is used as a covariate.
    """
    hold = record.hold_seconds if hold_seconds is None else float(hold_seconds)
    ledger = getattr(record, "issue_ledger", None)
    if ledger is None:
        ledger = record.context.get("issue_ledger")
    if ledger is None:
        return None
    if endpoint == "fixed300":
        duration = 300.
    elif endpoint == "same_vehicle_next":
        nonholding = record.context.get("nonholding_seconds")
        if nonholding is None:
            return None
        duration = float(nonholding)+hold
    elif endpoint == "global_next":
        candidate = "low" if np.isclose(hold, 0.) else "high"
        global_baseline = record.context.get("global_baselines", {}).get(candidate)
        if global_baseline is None:
            return None
        # Positive-duration predictions use the unchanged world decoder clock.
        # Only batch-zero raw diagnostics expose the pre-analytic clock.
        duration = max(0., float(global_baseline.get("raw_dt", global_baseline["dt"]))) \
            if _within_batch(record) else float(global_baseline["dt"])
    else:
        raise ValueError(f"Unknown response endpoint: {endpoint}")
    wait = float(ledger["waiting_passengers"])*duration
    ride = float(ledger["in_vehicle_passengers"])*duration
    return dict(dt=duration, wait=wait, ride=ride, cost=2*wait+ride)


def _view(issued, context):
    return SimpleNamespace(line_id=str(issued["line_id"]), s=issued["s"],
        event=issued["event"], started_at=float(issued["started_at"]),
        hold_seconds=30.*(1.+float(np.asarray(issued["a"]).reshape(-1)[0])),
        context={} if context is None else context)


def _within_batch(record):
    return int(round(float(record.s[2])*15))+1 < int(round(float(record.s[1])*16))


def _action_contrast(low, high):
    return dict(status=low["status"], support=low["support"],
        endpoint_support=low["endpoint_support"], endpoint_status=low["endpoint_status"],
        gains={name: low["endpoints"][name]["cost"]-high["endpoints"][name]["cost"]
               if low["endpoints"][name] is not None else None for name in ENDPOINT_NAMES},
        high_minus_low={name: {key: high["endpoints"][name][key]-low["endpoints"][name][key]
                              for key in ("dt", "wait", "ride", "cost")}
                        if low["endpoints"][name] is not None else None for name in ENDPOINT_NAMES},
        raw_global_next_gain=low["raw_global_next"]["cost"]-high["raw_global_next"]["cost"]
            if low["raw_global_next"] is not None else None,
        raw_global_next_high_minus_low={key: high["raw_global_next"][key]-low["raw_global_next"][key]
                                       for key in ("dt", "wait", "ride", "cost")}
            if low["raw_global_next"] is not None else None,
        low=low, high=high)


class ResidualResponseReadout(ControlledResponseReadout):
    """Same 13-column controlled design, fitting responses around known baselines.

    The same selected issue may supply several endpoint fields but is accepted
    once.  The readout is passive: it never changes policy, world or trust.
    """

    def _endpoint_records(self, endpoint):
        return [row for row in self.records
                if (endpoint != "global_next" or row.endpoints[endpoint]["dt"] > 0.)
                and endpoint_baseline(row, endpoint) is not None]

    def endpoint_coverage(self):
        return {name: {line: dict(
            n=sum(row.line_id == line for row in self._endpoint_records(name)),
            zero_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 0.))
                for row in self._endpoint_records(name)),
            high_holding=sum(row.line_id == line and bool(np.isclose(row.hold_seconds, 7.5))
                for row in self._endpoint_records(name)))
            for line in LINE_NAMES} for name in ENDPOINT_NAMES}

    def refit(self):
        if not self.records:
            return
        x = np.stack([response_design(row) for row in self.records])
        singular = np.linalg.svd(x, compute_uv=False)
        maximum = float(singular[0])
        minimum = float(singular[-1]) if len(singular) == x.shape[1] else 0.
        self.design = dict(rows=len(x), columns=x.shape[1], rank=int(np.linalg.matrix_rank(x)),
            active_columns=int((x != 0).any(0).sum()), ridge=self.ridge,
            ridge_precision_condition=(maximum**2+self.ridge)/(minimum**2+self.ridge))
        for endpoint, names in RESIDUAL_TARGET_NAMES.items():
            records = self._endpoint_records(endpoint)
            if not records:
                self.endpoint_designs[endpoint] = dict(rows=0,
                    columns=len(RESPONSE_FEATURE_NAMES), rank=0, learned_targets=list(names))
                continue
            design = np.stack([response_design(row) for row in records])
            self.endpoint_designs[endpoint] = dict(rows=len(design), columns=design.shape[1],
                rank=int(np.linalg.matrix_rank(design)), learned_targets=list(names))
            baseline = [endpoint_baseline(row, endpoint) for row in records]
            residuals = np.asarray([[row.endpoints[endpoint][key]-base[key] for key in names]
                                    for row, base in zip(records, baseline)])
            scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
            precision = design.T@design+self.ridge*np.eye(design.shape[1])
            self.readouts[endpoint] = np.linalg.solve(precision, design.T@(residuals/scales))

    def _predict(self, record, *, hold_seconds=None, baseline_only=False):
        support, coverage = self.coverage().get(record.line_id, {}), self.endpoint_coverage()
        x = response_design(record, hold_seconds=hold_seconds)
        endpoints, statuses, baselines = {}, {}, {}
        for endpoint in ENDPOINT_NAMES:
            base = endpoint_baseline(record, endpoint, hold_seconds=hold_seconds)
            baselines[endpoint] = base
            local_support = coverage[endpoint].get(record.line_id, {})
            if base is None:
                endpoints[endpoint], statuses[endpoint] = None, "missing_issue_baseline"
                continue
            fitted = endpoint in self.readouts and bool(local_support.get("zero_holding")) \
                and bool(local_support.get("high_holding"))
            if not baseline_only and not fitted:
                endpoints[endpoint], statuses[endpoint] = None, "unsupported"
                continue
            value = dict(base)
            if not baseline_only:
                scales = TARGET_SCALES if endpoint != "fixed300" else TARGET_SCALES[1:]
                for key, residual in zip(RESIDUAL_TARGET_NAMES[endpoint], (x@self.readouts[endpoint])*scales):
                    value[key] += float(residual)
            value["cost"] = 2*value["wait"]+value["ride"]
            endpoints[endpoint], statuses[endpoint] = value, "supported"
        raw_global = None if endpoints["global_next"] is None else dict(endpoints["global_next"])
        if _within_batch(record):
            endpoints["global_next"] = dict(dt=0., wait=0., ride=0., cost=0.,
                zero_duration_analytic=True)
            statuses["global_next"] = "analytic_zero_duration"
        return dict(status="supported" if any(value in ("supported", "analytic_zero_duration")
                    for value in statuses.values()) else "unsupported", support=support,
            endpoint_support={name: coverage[name].get(record.line_id, {}) for name in ENDPOINT_NAMES},
            endpoint_status=statuses, endpoints=endpoints, raw_global_next=raw_global,
            issue_time_baselines=baselines)

    def predict(self, record, *, hold_seconds=None):
        return self._predict(record, hold_seconds=hold_seconds)

    def predict_issue(self, issued, context=None):
        return self.action_difference(_view(issued, context))

    def baseline_predict_issue(self, issued, context=None):
        record = _view(issued, context)
        return _action_contrast(self._predict(record, hold_seconds=0., baseline_only=True),
                                self._predict(record, hold_seconds=7.5, baseline_only=True))

    def summary(self):
        result = super().summary()
        result.update(model="issue_time_baseline_response_residual", fixed300_duration_seconds=300.,
            target_names={name: list(keys) for name, keys in RESIDUAL_TARGET_NAMES.items()},
            residual_targets={name: list(keys) for name, keys in RESIDUAL_TARGET_NAMES.items()},
            baseline_semantics=dict(global_next="decoded base-world duration; nonnegative preanalytic duration only for batch-zero raw diagnostics",
                same_vehicle_next="fixed-source nonholding prediction plus executed holding",
                fixed300="exact 300-second window",
                passenger_cost="issue global waiting/onboard counts times issue-time duration baseline"),
            missing_issue_baseline_counts={name: sum(endpoint_baseline(row, name) is None
                                                    for row in self.records) for name in ENDPOINT_NAMES},
            semantics="Selected controlled issue residual fields; no future baseline covariates, held-out fitting, trust changes or global-target substitution")
        return result
