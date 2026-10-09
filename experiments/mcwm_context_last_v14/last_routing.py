"""Persistent high-context routing without posterior-weight leakage."""

import numpy as np
from last_common import HERE, module

window = module("last_window", HERE.parent / "mcwm_context_filter_v9/filters.py")


class HysteresisFilter(window.WindowFilter):
    def __init__(self, model, p):
        self.high_context = p["adjusted_context"]
        self.high_coefficient = p["context_high_coefficient"]
        self.enter_blocks = p["hysteresis_enter_blocks"]
        self.enter_margin = p["hysteresis_enter_margin"]
        self.exit_margin = p["hysteresis_exit_margin"]
        super().__init__(model, [p["shared_coefficients"][c] for c in model["classes"]],
            p["window_blocks"], p["global_coefficient"], p["fallback_score_margin"])

    def reset(self):
        super().reset()
        self.active, self.high_count, self.other_count = False, 0, 0

    def observe(self, features):
        super().observe(features)
        margin = np.sort(self.belief)[-1]-np.sort(self.belief)[-2]
        high = self.model["classes"][int(np.argmax(self.belief))] == self.high_context
        self.high_count = self.high_count+1 if high and margin >= self.enter_margin else 0
        self.other_count = self.other_count+1 if not high and margin >= self.enter_margin else 0
        if self.active and (margin < self.exit_margin or self.other_count >= self.enter_blocks):
            self.active = False
        elif not self.active and self.high_count >= self.enter_blocks:
            self.active = True
        return self.coefficient

    @property
    def decision(self):
        return (self.high_coefficient if self.active else self.global_coefficient), self.active
