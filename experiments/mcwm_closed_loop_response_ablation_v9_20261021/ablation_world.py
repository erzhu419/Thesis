"""Three independently refitted nested residuals on the same released labels."""

from __future__ import annotations

import torch

from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    FEATURE_NAMES, LINE_NAMES, MEMORY_FEATURE_NAMES, fit_response_readout,
    response_features,
)
from experiments.mcwm_closed_loop_action_response_v8_20261021.response_world import (
    ResponseWorld,
)
from experiments.mcwm_closed_loop_formal_20261020.scenarios import UPSTREAM_STOP
from experiments.mcwm_closed_loop_v1.learning import row_table, targets


ARM_NAMES = ("action_physical", "memory_level", "memory_interaction")
_MEMORY = tuple(range(len(MEMORY_FEATURE_NAMES)))
_INTERACTION = tuple(i for i, name in enumerate(FEATURE_NAMES)
                     if name.endswith(("_hold_pending_intervals",
                                       "_hold_remaining_command_hold")))
_ACTION = tuple(i for i in range(len(FEATURE_NAMES))
                if i not in _MEMORY and i not in _INTERACTION)
FEATURE_GROUPS = dict(action_physical=_ACTION, memory_level=_MEMORY,
                      memory_interaction=_INTERACTION)
ARM_FEATURE_INDICES = dict(
    action_physical=_ACTION,
    memory_level=tuple(i for i in range(len(FEATURE_NAMES)) if i not in _INTERACTION),
    memory_interaction=tuple(range(len(FEATURE_NAMES))))


def fit_ablation_readouts(features, residuals, *, ridge=1.):
    """Refit each restricted design; removing fitted coefficients is not a refit.

    Every returned readout remains 66 by 4. The full arm calls the original v8
    solver directly, preserving its column ordering and arithmetic exactly.
    """
    readouts = {}
    for arm, columns in ARM_FEATURE_INDICES.items():
        if arm == "memory_interaction":
            readouts[arm] = fit_response_readout(features, residuals, ridge=ridge)
        else:
            readout = features.new_zeros(features.shape[1], residuals.shape[1])
            readout[list(columns)] = fit_response_readout(
                features[:, list(columns)], residuals, ridge=ridge)
            readouts[arm] = readout
    return readouts


def group_contributions(features, readout):
    """Additive standardized four-head contributions before decoder clipping."""
    return {name: features[..., list(columns)] @ readout[list(columns)]
            for name, columns in FEATURE_GROUPS.items()}


def design_statistics(features, *, ridge=1.):
    """Report effective support and conditioning of the system actually solved."""
    stats = {}
    for arm, columns in ARM_FEATURE_INDICES.items():
        x = features[:, list(columns)].double()
        values = torch.linalg.svdvals(x)
        maximum = float(values[0]) if len(values) else 0.
        tolerance = max(x.shape) * torch.finfo(x.dtype).eps * maximum
        rank = int((values > tolerance).sum())
        minimum = float(values[-1]) if len(values) == x.shape[1] else 0.
        stats[arm] = dict(rows=x.shape[0], columns=x.shape[1], rank=rank,
            active_columns=int((x != 0).any(dim=0).sum()), ridge=float(ridge),
            ridge_precision_condition=(maximum**2+ridge)/(minimum**2+ridge))
    return stats


class AblationResponseWorld(ResponseWorld):
    """Observed memory remains identical to v8; only the residual design differs.

    ``readouts`` aliases the full v8-compatible arm. Separate ablation readouts
    never modify the base world, its generator, physical learner or trust.
    """

    def __init__(self, world, physical, *, window=285, ridge=1.):
        super().__init__(world, physical, window=window, ridge=ridge)
        self.readouts_by_arm = {
            arm: {family: value.clone() for family, value in self.readouts.items()}
            for arm in ARM_NAMES}
        self.frozen_readouts_by_arm = {
            arm: {family: value.clone() for family, value in self.frozen_readouts.items()}
            for arm in ARM_NAMES}
        self.readouts = self.readouts_by_arm["memory_interaction"]
        self.frozen_readouts = self.frozen_readouts_by_arm["memory_interaction"]
        self._last_features = torch.zeros(0, len(FEATURE_NAMES), device=world.device)
        self.fit_errors = {arm: {} for arm in ARM_NAMES}

    @torch.no_grad()
    def refit(self):
        if not self.records:
            return
        records = list(self.records)
        y = targets(row_table([r["row"] for r in records]),
                    self.world.stations, self.world.device)
        x = torch.cat([response_features(
            y["s"][i:i+1], y["a"][i:i+1], r["memory_features"],
            r["nonholding_seconds"], self.numeric_lines)
            for i, r in enumerate(records)])
        target = (y["g"]-self.world.norm.g_mean)/self.world.norm.g_std
        for family, model in self.world.models.items():
            residual = target-model(y["s"], y["a"])[:, :4]
            fitted = fit_ablation_readouts(x, residual, ridge=self.ridge)
            for arm, readout in fitted.items():
                self.readouts_by_arm[arm][family] = readout
                self.fit_errors[arm][family] = dict(
                    base_mse=float(residual.square().mean()),
                    residual_mse=float((residual-x@readout).square().mean()))
        self._last_features = x.detach()
        self.refits += 1

    def holding_coverage(self):
        """Coverage of the retained selected labels, never of diagnostic truth."""
        rows = [record["row"] for record in self.records]

        def coverage(selected):
            holds = [self.physical._holding(row.a) for row in selected]
            return dict(n=len(holds),
                hold_seconds_min=min(holds) if holds else None,
                hold_seconds_max=max(holds) if holds else None,
                positive_holding=sum(hold > 0 for hold in holds),
                zero_holding=sum(hold == 0 for hold in holds))

        lines = list(LINE_NAMES)+sorted({str(row.line_id) for row in rows}-set(LINE_NAMES))
        return dict(scope="retained selected released prefix labels",
            diagnostic_station=UPSTREAM_STOP, overall=coverage(rows),
            by_line={line: dict(
                all=coverage([row for row in rows if str(row.line_id) == line]),
                at_diagnostic_station=coverage([row for row in rows
                    if str(row.line_id) == line and row.station_id == UPSTREAM_STOP]))
                for line in lines})

    def coefficients(self):
        """Optional resident artifact; compact collection needs only summary()."""
        return {arm: {family: value.detach().cpu().tolist()
                      for family, value in readouts.items()}
                for arm, readouts in self.readouts_by_arm.items()}

    def summary(self, now):
        result = super().summary(now)
        result.update(
            arms=list(ARM_NAMES),
            arm_feature_indices={name: list(columns)
                                 for name, columns in ARM_FEATURE_INDICES.items()},
            feature_groups={name: [FEATURE_NAMES[i] for i in columns]
                            for name, columns in FEATURE_GROUPS.items()},
            designs=design_statistics(self._last_features, ridge=self.ridge),
            selected_holding_coverage=self.holding_coverage(),
            fit_errors=self.fit_errors,
            readouts_by_arm=self.coefficients(),
            coefficient_group_l2={arm: {family: {
                group: float(readout[list(columns)].norm())
                for group, columns in FEATURE_GROUPS.items()}
                for family, readout in readouts.items()}
                for arm, readouts in self.readouts_by_arm.items()},
            fitting="Independent nested ridge refits on identical selected released labels; no truth training")
        return result
