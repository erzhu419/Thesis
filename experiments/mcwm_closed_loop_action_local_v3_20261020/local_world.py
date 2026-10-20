"""Causal local interval calibration as a learned global-world covariate.

The physical stop interval is not the global next-event duration.  Its learned
shift enters a zero-centred residual regression whose four targets remain the
observed global SMDP log-duration, log-cost and passenger-count increments.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from experiments.mcwm_closed_loop_diagnostic_v2_20261020.interval import (
    AFFECTED_LINES, EI, IntervalDiagnostic, is_direct_interval, serialize,
)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.world_diagnostics import DiagnosticWorld
from experiments.mcwm_closed_loop_action_local_v3_20261020.learning import ActionConsistentWorldMixin
from experiments.mcwm_closed_loop_v1.learning import row_table, targets


CONTEXT_SCALE_SECONDS = 60.
FEATURE_NAMES = tuple(f"{kind}_{line}" for kind in (
    "system_local_shift", "current_line_local_shift", "current_line_hold_local_shift")
    for line in AFFECTED_LINES)


def phase(at, shock_at=9000.):
    if at < shock_at:
        return "before"
    if at < shock_at+3000:
        return "early_after"
    if at < shock_at+6000:
        return "middle_after"
    if at < shock_at+9000:
        return "late_after"
    return "drain"


class LocalIntervalCalibration(IntervalDiagnostic):
    """The unchanged three physical predictors plus source reference features."""

    def __init__(self, source_rows, contract, **kwargs):
        rows = [serialize(row) for row in source_rows]
        super().__init__(rows, contract, **kwargs)
        selected = {line: [] for line in AFFECTED_LINES}
        self.numeric_line_by_name = {}
        for row in rows:
            if not is_direct_interval(row["line_id"], row["station_id"], row["next_station_id"]):
                continue
            line = str(row["line_id"])
            self.numeric_line_by_name[line] = int(round(row["event"][EI["line_id"]]))
            if len(selected[line]) < self.source_used[line]:
                selected[line].append(self._features(line, row["event"]))
        self.source_reference_features = np.stack([
            np.mean(selected[line], axis=0) for line in AFFECTED_LINES])

    def context_delta_seconds(self):
        """Current minus source prediction at each line's fixed source reference."""
        delta = self.source_reference_features @ (self.online_weights-self.frozen_weights)
        return delta+np.asarray([self.context[line] for line in AFFECTED_LINES])

    def summary(self):
        result = super().summary()
        result["world_covariates"] = dict(
            line_order=list(AFFECTED_LINES), numeric_line_by_name=self.numeric_line_by_name,
            source_reference_features=self.source_reference_features.tolist(),
            current_delta_seconds=self.context_delta_seconds().tolist(),
            semantics="local nonholding prediction shift at fixed per-line source mean features")
        return result


def adapter_features(states, actions, deltas, numeric_lines):
    """Known line, executed action and already learned physical context only.

    System features allow global next-event consequences on other routes to be
    learned; route and holding interactions allow different observed responses.
    There is no intercept, shock indicator, or use of the next event's identity.
    """
    slots = states[:, 5:].reshape(-1, 16, 14)
    current = (states[:, 2]*15).round().long().clamp(0, 15)
    line = slots[torch.arange(len(states), device=states.device), current, 1].round().long()
    ids = torch.as_tensor(numeric_lines, device=states.device, dtype=torch.long)
    delta = torch.as_tensor(deltas, device=states.device, dtype=states.dtype)
    if delta.ndim == 1:
        delta = delta[None].expand(len(states), -1)
    normalized = delta/CONTEXT_SCALE_SECONDS
    route = normalized*(line[:, None] == ids[None]).to(states.dtype)
    hold_fraction = (actions.reshape(-1, 1)+1.)/2.
    return torch.cat((normalized, route, route*hold_fraction), dim=1)


def fit_readout(features, residual, *, ridge=1.):
    """Fit only global-transition residuals, with a zero prior and no intercept."""
    x, y = features.double(), residual.double()
    precision = x.T@x+float(ridge)*torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return torch.linalg.solve(precision, x.T@y).to(features.dtype)


def _mean(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else None


def paired_summary(rows):
    fields = ("base_duration_mae_seconds", "calibrated_duration_mae_seconds",
              "base_cost_mae_passenger_seconds", "calibrated_cost_mae_passenger_seconds",
              "base_global_head_loss", "calibrated_global_head_loss",
              "global_correction_abs_mean_standardized", "context_abs_mean_seconds")
    def stats(values):
        return dict(n=len(values), **{key: _mean(values, key) for key in fields})
    return dict(overall=stats(rows),
        by_phase={name: stats([row for row in rows if row["phase"] == name])
                  for name in ("before", "early_after", "middle_after", "late_after", "drain")},
        by_line={line: stats([row for row in rows if row["line_id"] == line])
                 for line in sorted({row["line_id"] for row in rows})})


class LocalWorld(ActionConsistentWorldMixin, DiagnosticWorld):
    """Execution-consistent baseline world with a supervised local-context readout.

    Both enabled and passive versions fit the same readout.  ``local_enabled``
    decides whether it enters deployment/model-TD predictions.  Paired issue-time
    base/calibrated errors are retained in either case, using the family selected
    before the delayed label was available.
    """

    def __init__(self, replay, *, action_transform, physical, enabled=True,
                 seed=6201, updates=20, ridge=1.):
        self.action_transform, self.physical = action_transform, physical
        self.local_enabled, self.local_ridge = bool(enabled), float(ridge)
        self.numeric_lines = [physical.numeric_line_by_name[line] for line in AFFECTED_LINES]
        super().__init__(replay, seed=seed, updates=updates, context=False)
        self.local_readouts = {family: torch.zeros(len(FEATURE_NAMES), 4, device=self.device)
                               for family in self.models}
        self.local_fit_records, self.local_prediction_records = [], []
        self.local_refits = self.local_labels = 0
        self.local_rows_by_line = Counter()

    def local_features(self, states, actions, deltas=None):
        if deltas is None:
            deltas = self.physical.context_delta_seconds()
        return adapter_features(states, actions, deltas, self.numeric_lines)

    @torch.no_grad()
    def predict(self, s, a, family=None, calibrated=None):
        family = family or self.family
        output = super().predict(s, a, family)
        if calibrated is None:
            calibrated = self.local_enabled
        if calibrated:
            output[:, :4] += self.local_features(s, a)@self.local_readouts[family]
        return output

    @torch.no_grad()
    def register(self, issued):
        super().register(issued)
        pending = self.pending.get(issued["sequence"])
        if pending is None:
            return
        s = torch.as_tensor(issued["s"][None], device=self.device)
        a = torch.as_tensor(issued["a"].reshape(1, 1), device=self.device)
        delta = self.physical.context_delta_seconds().copy()
        features = self.local_features(s, a, delta)
        base_g = {family: self.predict(s, a, family, calibrated=False)[:, :4].clone()
                  for family in self.models}
        pending["local_covariates"] = dict(features=features.clone(), delta_seconds=delta,
            physical_parameter_version=self.physical.parameter_version,
            base_g=base_g,
            calibrated_g={family: values+features@self.local_readouts[family]
                          for family, values in base_g.items()})

    @torch.no_grad()
    def observe(self, rows, now):
        details = {}
        for row in rows:
            pending = self.pending.get(row.sequence)
            if pending is None or row.terminal or row.dt <= 0:
                continue
            if row.available_at > now or row.completed_at > now:
                raise ValueError("A local readout label was supplied before availability")
            saved = pending["local_covariates"]
            y = targets(row_table([row]), self.stations, self.device)
            standardized = (y["g"]-self.norm.g_mean)/self.norm.g_std
            family = pending["family_at_issue"]
            base, calibrated = saved["base_g"][family], saved["calibrated_g"][family]
            physical = lambda g: torch.expm1((g*self.norm.g_std+self.norm.g_mean)[0, :2].clamp(0, 20))
            base_p, calibrated_p = physical(base), physical(calibrated)
            detail = dict(phase=phase(row.started_at, self.physical.shock_at), line_id=str(row.line_id),
                base_duration_mae_seconds=abs(float(base_p[0])-row.dt),
                calibrated_duration_mae_seconds=abs(float(calibrated_p[0])-row.dt),
                base_cost_mae_passenger_seconds=abs(float(base_p[1])+row.r),
                calibrated_cost_mae_passenger_seconds=abs(float(calibrated_p[1])+row.r),
                base_global_head_loss=float(((base-standardized)**2).mean()),
                calibrated_global_head_loss=float(((calibrated-standardized)**2).mean()),
                global_correction_abs_mean_standardized=float((calibrated-base).abs().mean()),
                context_abs_mean_seconds=float(np.mean(np.abs(saved["delta_seconds"]))),
                physical_parameter_version_at_issue=saved["physical_parameter_version"])
            details[row.sequence] = detail
            self.local_prediction_records.append(detail)
            # These covariates were fixed when the action was issued.  A later
            # physical label cannot rewrite an old global training example.
            self.local_fit_records.append(dict(s=y["s"].clone(), a=y["a"].clone(),
                standardized_target=standardized.clone(), features=saved["features"].clone()))
            self.local_labels += 1
            self.local_rows_by_line[str(row.line_id)] += 1
        logs = super().observe(rows, now)
        for log in logs:
            log["local_adapter"] = details[log["sequence"]]
        return logs

    def refit(self, updates):
        updated = super().refit(updates)
        self.refit_local()
        return updated

    @torch.no_grad()
    def refit_local(self):
        if not self.local_fit_records:
            return False
        records = self.local_fit_records
        s, a, x, y = (torch.cat([row[key] for row in records]) for key in
                       ("s", "a", "features", "standardized_target"))
        for family, model in self.models.items():
            # The ordinary global model is refitted independently. Recenter
            # the local regression against that current model, using original
            # targets and the saved causal covariates for every selected label.
            residual = y-model(s, a)[:, :4]
            self.local_readouts[family] = fit_readout(x, residual, ridge=self.local_ridge)
        self.local_refits += 1
        return True

    def summary(self):
        result = super().summary()
        result["local_adapter"] = dict(enabled=self.local_enabled, ridge=self.local_ridge,
            feature_names=list(FEATURE_NAMES), scale_seconds=CONTEXT_SCALE_SECONDS,
            target_heads=["log_global_next_event_duration", "log_global_passenger_cost",
                          "system_waiting_increment", "system_onboard_increment"],
            target_semantics="unchanged global SMDP transitions; no physical interval substituted",
            fit_labels=self.local_labels, fit_rows_by_line=dict(self.local_rows_by_line),
            refits=self.local_refits,
            readouts={family: values.tolist() for family, values in self.local_readouts.items()},
            paired_prequential=paired_summary(self.local_prediction_records),
            candidate_calibration_scored_in_passive_arm=True,
            training_covariates="saved at global action issuance; physical labels already available then")
        return result
