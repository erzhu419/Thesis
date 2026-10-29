"""Matched M8 replay: no-fresh never receives a fresh outcome or next memory."""
from __future__ import annotations

import numpy as np
import torch

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.bootstrap import formal
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import FreshOutcomeReplay
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import MemoryOutcomeReplay


class MatchedOutcomeReplay(MemoryOutcomeReplay):
    def add_target(self, rows):
        for row in rows:
            # FreshOutcomeReplay's no_fresh path reads s/a only.
            FreshOutcomeReplay.add_target(self, [row])
            fields = ('memory_s', 'memory_ns') if self.arm == 'fresh' else ('memory_s',)
            for field in fields:
                vector = np.asarray(getattr(row, field)['vector'], np.float32)
                if vector.shape != (84,):
                    raise ValueError('Expected the existing84-dimensional execution memory')
                self.recent[-1][field] = vector.copy()

    def sample_mixed(self, size, online_fraction, *, rng):
        factual, source = FreshOutcomeReplay.sample_mixed(self, size, online_fraction, rng=rng)
        ids = self.last_sampling['recent']
        zero = torch.zeros((len(ids), 84), device=self.device)
        current = torch.as_tensor(np.asarray([self.recent[i]['memory_s'] for i in ids]),
            dtype=torch.float32, device=self.device)
        for key, field in (('observations', 'memory_s'), ('next_observations', 'memory_ns')):
            # Entire replacement transitions come from history, whose memory
            # is unobserved/zero. Never attach current native context to history.
            context = torch.as_tensor(np.asarray([self.recent[i][field] for i in ids]),
                dtype=torch.float32, device=self.device) if self.arm == 'fresh' else zero
            factual[key] = torch.cat((factual[key], torch.cat((zero, context))), dim=1)
            source[key] = torch.cat((source[key], torch.zeros((len(source[key]), 84),
                dtype=source[key].dtype, device=self.device)), dim=1)
        self.last_covariates['memory_s'] = torch.cat((zero, current))
        return factual, source

    def summary(self):
        result = super().summary()
        result.update(current_memory_available=True,
            fresh_next_memory_stored=len(self.recent) if self.arm == 'fresh' else 0,
            replacement_critic_memory='zero for the entire archived transition',
            stored_fields=sorted(self.recent[0]) if self.recent else [])
        return result
