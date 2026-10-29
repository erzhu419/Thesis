"""Passive world-only memory residual; only selected released labels fit it."""

from collections import deque

import numpy as np
import torch

from experiments.mcwm_closed_loop_v1.learning import row_table, targets
from experiments.mcwm_closed_loop_v1.stream import EI
from experiments.mcwm_closed_loop_formal_20261020.scenarios import UPSTREAM_STOP
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    ActionResponseMemory, FEATURE_NAMES, LINE_NAMES, fit_response_readout, response_features,
)


class ResponseWorld:
    def __init__(self, world, physical, *, window=285, ridge=1.):
        self.world, self.physical = world, physical
        self.memory = ActionResponseMemory()
        self.numeric_lines = [physical.numeric_line_by_name[line] for line in LINE_NAMES]
        self.pending, self.records = {}, deque(maxlen=window)
        self.readouts = {family: torch.zeros(len(FEATURE_NAMES), 4, device=world.device)
                         for family in world.models}
        self.frozen_readouts = {family: value.clone() for family, value in self.readouts.items()}
        self.ridge = ridge
        self.labels = self.refits = 0

    def context(self, event):
        """Observed command history and the already available physical predictor."""
        nonholding = None
        line = event["line_id"]
        if line in LINE_NAMES and event["station_id"] == UPSTREAM_STOP:
            x = self.physical._features(line, event["event"])
            nonholding = max(0., float(x @ self.physical.online_weights) + self.physical.context[line])
        return dict(memory_features=self.memory.features(event["started_at"]),
                    nonholding_seconds=nonholding)

    def register(self, event, context):
        current = int(round(float(event["s"][2]) * 15))
        count = int(round(float(event["s"][1]) * 16))
        if current + 1 == count:
            self.pending[event["sequence"]] = dict(
                s=event["s"].copy(), a=event["a"].copy(), **context)
        self.memory.register(event,
            hold_seconds=self.physical._holding(event["a"]),
            onboard_passengers=float(event["event"][EI["onboard_passengers"]]),
            nonholding_seconds=context["nonholding_seconds"])

    def accept(self, packet, discarded, now):
        for sequence in discarded:
            self.pending.pop(sequence, None)
        for row in packet:
            if row.available_at > now or row.completed_at > now:
                raise ValueError("Response target was supplied before its feedback release")
            saved = self.pending.pop(row.sequence, None)
            if saved is None:
                raise ValueError("Selected response label lacks its issue-time memory")
            if not (np.array_equal(saved["s"], row.s) and np.array_equal(saved["a"], row.a)):
                raise ValueError("Response label differs from its saved executed issue")
            self.records.append(dict(**saved, row=row))
            self.labels += 1

    @torch.no_grad()
    def refit(self):
        if not self.records:
            return
        records = list(self.records)
        y = targets(row_table([r["row"] for r in records]), self.world.stations, self.world.device)
        x = torch.cat([response_features(
            y["s"][i:i+1], y["a"][i:i+1], r["memory_features"],
            r["nonholding_seconds"], self.numeric_lines) for i, r in enumerate(records)])
        target = (y["g"] - self.world.norm.g_mean) / self.world.norm.g_std
        for family, model in self.world.models.items():
            residual = target - model(y["s"], y["a"])[:, :4]
            self.readouts[family] = fit_response_readout(x, residual, ridge=self.ridge)
        self.refits += 1

    def summary(self, now):
        return dict(labels=self.labels, retained_labels=len(self.records), refits=self.refits,
                    feature_names=list(FEATURE_NAMES), memory=self.memory.summary(now),
                    readout_l2={family: float(value.norm()) for family, value in self.readouts.items()},
                    frozen_readout_l2={family: float(value.norm()) for family, value in self.frozen_readouts.items()},
                    fitting="Selected delayed global facts only; original base models and trust unchanged",
                    scope="Passive one-step candidate with observed multi-event memory; not an imagined multi-step controller")
