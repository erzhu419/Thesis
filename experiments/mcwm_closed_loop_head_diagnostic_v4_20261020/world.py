"""Passive, saved-issue diagnostics for the four existing global target heads.

No target, model fit, family selector, or model-TD prediction is changed here.
The raw log-head inverses and the actual deterministic global decoder projection
are reported separately, since count clipping can hide a nonzero head correction.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import (
    AFFECTED_LINES, FEATURE_NAMES, LocalWorld, phase,
)
from experiments.mcwm_closed_loop_formal_20261020.scenarios import UPSTREAM_STOP
from experiments.mcwm_closed_loop_v1.learning import row_table, targets


HEAD_ORDER = ("logdt", "logcost", "dWaiting", "dOnboard")
PHYSICAL_UNITS = ("seconds", "passenger_seconds", "passengers", "passengers")
PHASES = ("before", "early_after", "middle_after", "late_after", "drain")


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(4)


def _raw_physical(value):
    result = value.copy()
    result[:2] = np.expm1(result[:2])
    return result


def decoder_global_projection(output, norm, state):
    """The existing decoder's global values without drawing any topology.

    Keep its tensor dtype and operations: these are the same four global values
    used by ModelAugmentation.decode, including the next-state count clipping.
    """
    values = output.reshape(1, 4)*norm.g_std+norm.g_mean
    s = state.reshape(1, -1)
    before_minimum = torch.expm1(values[:, 0].clamp(0, 10))
    horizon_cap = (1-s[:, 0])*24000-1
    duration = torch.minimum(before_minimum.clamp_min(1.), horizon_cap)
    cost = torch.expm1(values[:, 1].clamp(0, 20))
    next_counts_unclipped = s[:, 3:5]+values[:, 2:4]/20000
    next_counts = next_counts_unclipped.clamp(0, 1.25)
    increments = (next_counts-s[:, 3:5])*20000
    physical = torch.cat((duration[:, None], cost[:, None], increments), dim=1)
    saturated = torch.cat((
        (((values[:, 0] < 0) | (values[:, 0] > 10)
          | (before_minimum < 1) | (before_minimum.clamp_min(1.) > horizon_cap)))[:, None],
        ((values[:, 1] < 0) | (values[:, 1] > 20))[:, None],
        (next_counts_unclipped < 0) | (next_counts_unclipped > 1.25)), dim=1)
    return _array(physical), saturated[0].detach().cpu().tolist()


def compare_global_heads(base, calibrated, target, norm, state, observed_physical=None):
    """Score the two predictions against one label, with prediction-minus-truth sign."""
    mean, scale, truth_g = _array(norm.g_mean), _array(norm.g_std), _array(target)
    standardized_truth = (truth_g-mean)/scale
    physical_truth = (_raw_physical(truth_g) if observed_physical is None
                      else _array(observed_physical))
    result = dict(target_standardized=standardized_truth.tolist(),
                  target_physical=physical_truth.tolist())
    for name, output in (("base", base), ("calibrated", calibrated)):
        standardized = _array(output)
        raw_physical = _raw_physical(standardized*scale+mean)
        decoded_physical, saturated = decoder_global_projection(output, norm, state)
        result[name] = dict(standardized_prediction=standardized.tolist(),
            physical_prediction=raw_physical.tolist(),
            decoded_physical_prediction=decoded_physical.tolist(),
            standardized_signed_error=(standardized-standardized_truth).tolist(),
            physical_signed_error=(raw_physical-physical_truth).tolist(),
            decoded_physical_signed_error=(decoded_physical-physical_truth).tolist(),
            decode_saturated=saturated)
    result["correction"] = {
        f"{scale_name}": (np.asarray(result["calibrated"][f"{scale_name}_prediction"])
                         -np.asarray(result["base"][f"{scale_name}_prediction"])).tolist()
        for scale_name in ("standardized", "physical", "decoded_physical")}
    return result


def _error_moments(values):
    values = np.asarray(values, dtype=float)
    return dict(signed_mean=values.mean(0).tolist(), mae=np.abs(values).mean(0).tolist(),
                mse=np.square(values).mean(0).tolist())


def summarize_head_records(rows):
    """Compact aggregate vectors in HEAD_ORDER; individual issue rows stay remote."""
    def statistics(values):
        result = dict(n=len(values),
            direct_upstream_records=sum(row["direct_upstream"] for row in values),
            direct_upstream_by_line={line: sum(row["direct_upstream"] and row["line_id"] == line
                                              for row in values) for line in AFFECTED_LINES})
        if not values:
            return result
        for candidate in ("base", "calibrated"):
            result[candidate] = {
                scale: _error_moments([row["families"][row["selected_family_at_issue"]]
                                      [candidate][f"{scale}_signed_error"] for row in values])
                for scale in ("standardized", "physical", "decoded_physical")}
            result[candidate]["decode_saturation_fraction"] = np.asarray([
                row["families"][row["selected_family_at_issue"]][candidate]["decode_saturated"]
                for row in values], dtype=float).mean(0).tolist()
        result["correction"] = {f"{scale}_mean": np.asarray([
            row["families"][row["selected_family_at_issue"]]["correction"][scale]
            for row in values]).mean(0).tolist()
            for scale in ("standardized", "physical", "decoded_physical")}
        result["covariates"] = {
            key: np.asarray([row["context_at_issue"][key] for row in values]).mean(0).tolist()
            for key in ("source_reference_nonholding_seconds", "online_shift_seconds",
                        "local_context_seconds", "combined_shift_seconds")}
        result["physical_version_mean"] = float(np.mean([
            row["physical_parameter_version_at_issue"] for row in values]))
        result["model_version_mean"] = float(np.mean([row["model_version_at_issue"] for row in values]))
        return result

    return dict(head_order=list(HEAD_ORDER), physical_units=list(PHYSICAL_UNITS),
        error_sign="prediction-minus-observation",
        pairing="same issue, selected family saved before delayed label; scored before fitting",
        physical_semantics="raw expm1 inverse for logdt/logcost; signed count increments without clipping",
        decoded_physical_semantics="unchanged decoder global projection; no topology draws",
        covariate_line_order=list(AFFECTED_LINES),
        overall=statistics(rows),
        by_phase={name: statistics([row for row in rows if row["phase"] == name]) for name in PHASES},
        by_line={line: statistics([row for row in rows if row["line_id"] == line])
                 for line in sorted({row["line_id"] for row in rows})},
        by_phase_and_line={name: {
            line: statistics([row for row in rows if row["phase"] == name and
                (row["line_id"] == line if line != "other" else row["line_id"] not in AFFECTED_LINES)])
            for line in (*AFFECTED_LINES, "other")} for name in PHASES})


class HeadDiagnosticWorld(LocalWorld):
    """Instrument v3 identically in both action_model and action_local arms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_prediction_records = []

    @torch.no_grad()
    def register(self, issued):
        super().register(issued)
        pending = self.pending.get(issued["sequence"])
        if pending is None:
            return
        physical = self.physical
        reference = physical.source_reference_features
        source = reference@physical.frozen_weights
        online = reference@physical.online_weights
        context = np.asarray([physical.context[line] for line in AFFECTED_LINES])
        saved = pending["local_covariates"]
        features = saved["features"].reshape(-1, 1)
        pending["head_diagnostic"] = dict(
            line_id=str(issued["line_id"]), station_id=str(issued["station_id"]),
            context_at_issue=dict(source_reference_nonholding_seconds=source.tolist(),
                online_reference_nonholding_seconds=online.tolist(),
                online_shift_seconds=(online-source).tolist(), local_context_seconds=context.tolist(),
                combined_shift_seconds=np.asarray(saved["delta_seconds"]).tolist()),
            readout_feature_contributions_standardized={family:
                (features*weights).detach().cpu().tolist()
                for family, weights in self.local_readouts.items()})

    @torch.no_grad()
    def observe(self, rows, now):
        details = {}
        for row in rows:
            pending = self.pending.get(row.sequence)
            if pending is None or row.terminal or row.dt <= 0:
                continue
            if row.available_at > now or row.completed_at > now:
                raise ValueError("A head diagnostic label was supplied before availability")
            saved, metadata = pending["local_covariates"], pending["head_diagnostic"]
            y = targets(row_table([row]), self.stations, self.device)
            observed = [float(row.dt), -float(row.r), *y["g"][0, 2:4].cpu().tolist()]
            line, station = metadata["line_id"], metadata["station_id"]
            shock_at = self.physical.shock_at
            detail = dict(sequence=int(row.sequence), phase=phase(row.started_at, shock_at),
                completed_phase=phase(row.completed_at, shock_at),
                available_phase=phase(row.available_at, shock_at), scored_phase=phase(now, shock_at),
                issued_at=float(row.started_at), completed_at=float(row.completed_at),
                available_at=float(row.available_at), scored_at=float(now),
                direct_upstream=line in AFFECTED_LINES and station == UPSTREAM_STOP,
                selected_family_at_issue=pending["family_at_issue"],
                model_version_at_issue=pending["version"],
                physical_parameter_version_at_issue=saved["physical_parameter_version"],
                features_at_issue=saved["features"][0].cpu().tolist(),
                **metadata,
                families={family: compare_global_heads(saved["base_g"][family],
                    saved["calibrated_g"][family], y["g"], self.norm, y["s"], observed)
                    for family in self.models})
            details[row.sequence] = detail
        logs = super().observe(rows, now)
        for log in logs:
            detail = details[log["sequence"]]
            log["four_global_heads"] = detail
            self.head_prediction_records.append(detail)
        return logs

    def summary(self):
        result = super().summary()
        result["head_diagnostics"] = summarize_head_records(self.head_prediction_records)
        result["head_diagnostics"]["feature_names"] = list(FEATURE_NAMES)
        return result
