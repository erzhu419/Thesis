"""Read-only parity and coverage on the unchanged admitted factual packets."""
from collections import defaultdict

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_memory_v32_20261023.critic import ExecutedMemoryCritic
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


class ZeroInitProbe:
    def __init__(self, learner, memory_dim):
        self.candidates = {name: ExecutedMemoryCritic(getattr(learner, name), memory_dim)
                           for name in ('critic', 'target_critic')}
        self.rows = self.packets = self.full_comparisons = self.subset_comparisons = 0
        self.exact = self.rng_exact = True

    @torch.no_grad()
    def observe(self, learner, rows):
        saved_rng = _rng()
        states = torch.as_tensor(np.asarray([r.s for r in rows]), device=learner.device)
        next_states = torch.as_tensor(np.asarray([r.ns for r in rows]), device=learner.device)
        actions = torch.as_tensor(np.asarray([r.a for r in rows]), device=learner.device).reshape(-1, 1)
        next_actions = learner.action_transform(next_states,
            learner.actor(next_states, deterministic=True)[0])
        for name, candidate in self.candidates.items():
            native = getattr(learner, name)
            # The base always matches THIS packet's native parameters; it is
            # never left at the original 50000-update initializer.
            candidate.base.load_state_dict(native.state_dict())
            indices = torch.tensor([0, native.ensemble_size-1], device=learner.device)
            for s, a, key in ((states, actions, 'memory_s'),
                              (next_states, next_actions, 'memory_ns')):
                memory = torch.as_tensor(np.asarray([getattr(r, key)['vector'] for r in rows]),
                                         dtype=s.dtype, device=learner.device)
                self.exact &= torch.equal(native(s, a), candidate(s, a, memory))
                self.exact &= torch.equal(native.forward_subset(s, a, indices),
                                         candidate.forward_subset(s, a, memory, indices))
                self.full_comparisons += len(rows)
                self.subset_comparisons += len(rows)
        self.rows += len(rows)
        self.packets += 1
        self.rng_exact &= _equal(saved_rng, _rng())
        if not self.exact or not self.rng_exact:
            raise RuntimeError('Zero-initialized memory critic changed predictions or main RNG')

    def summary(self):
        return dict(all_pass=self.exact and self.rng_exact, native_values_exact=self.exact,
                    rng_unchanged=self.rng_exact, rows=self.rows, packets=self.packets,
                    full_head_row_comparisons=self.full_comparisons,
                    subset_row_comparisons=self.subset_comparisons,
                    optimizer_updates=0)


def coverage(rows):
    """Each counter is conditional on admission by the original PacketBudget."""
    groups = defaultdict(lambda: [0]*8)
    complete = 0
    ranges = {}
    for row in rows:
        left, right = getattr(row, 'memory_s', None), getattr(row, 'memory_ns', None)
        if left is None or right is None:
            continue
        complete += 1
        s, ns = np.asarray(left['vector']), np.asarray(right['vector'])
        line_ids = left['line_ids']
        if right['line_ids'] != line_ids:
            raise RuntimeError('Execution-memory line ordering changed between s and ns')
        own = line_ids.index(str(row.line_id))
        s, ns = s.reshape(len(line_ids), -1), ns.reshape(len(line_ids), -1)
        # Seven declared per-line features: presence/count/observed phase/
        # stopped/departed/apply age/observed departure age.
        if s.shape[1] != 7 or ns.shape != s.shape:
            raise RuntimeError('Unexpected v32 execution-memory schema')
        window = 'before9000' if row.started_at < 9000 else \
            '9000-12000' if row.started_at < 12000 else '12000-18000' if row.started_at < 18000 else 'tail'
        value = [1, int(s[own, 0] > 0), int(ns[own, 0] > 0),
                 int(s[own, 2] > 0), int(ns[own, 2] > 0),
                 int(s[own, 4] > 0), int(ns[own, 4] > 0), int(not np.array_equal(s, ns))]
        for key in (str(row.line_id), f'{row.line_id}/{window}'):
            groups[key] = [a+b for a, b in zip(groups[key], value)]
        if str(row.line_id) not in ranges:
            ranges[str(row.line_id)] = [s[own].copy(), s[own].copy()]
        low, high = ranges[str(row.line_id)]
        ranges[str(row.line_id)] = [np.minimum(low, s[own]), np.maximum(high, s[own])]
    return dict(admitted_rows=len(rows), rows_with_both_contexts=complete,
        columns=['rows', 'own_line_s_present', 'own_line_ns_present',
                 'own_line_s_phase_observed', 'own_line_ns_phase_observed',
                 'own_line_s_departed', 'own_line_ns_departed', 's_ns_context_changed'],
        by_line={k:v for k,v in sorted(groups.items()) if '/' not in k},
        by_line_window={k:v for k,v in sorted(groups.items()) if '/' in k},
        own_line_s_feature_ranges={k:[a.tolist(), b.tolist()] for k,(a,b) in sorted(ranges.items())})
