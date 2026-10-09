"""Keep the v10 evidence filter and isolate neutral-context soft leakage."""

import numpy as np
from mapping_common import HERE, module

window = module("mapping_window", HERE.parent / "mcwm_context_filter_v9/filters.py")


class GatedWindowFilter(window.WindowFilter):
    def __init__(self, *args, adjusted_context, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjusted_context = adjusted_context

    @property
    def decision(self):
        coefficient, accepted = super().decision
        predicted = self.model["classes"][int(np.argmax(self.belief))]
        if not accepted or predicted != self.adjusted_context:
            coefficient = self.global_coefficient
        return float(coefficient), bool(accepted)
