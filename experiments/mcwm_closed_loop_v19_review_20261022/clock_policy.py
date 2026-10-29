"""Audit the native event clock without replacing it with ledger delivery time."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.phase_stream import (
    PhasePolicy, tail_action,
)


class NativeClockPolicy(PhasePolicy):
    """Keep v19 actor/execution logic; compare to the event that built its state."""

    def __call__(self, state):
        # Original packing takes the common event detection time. SUMO advances
        # its ledger before delivering that event; delivery is a different clock.
        event_time = float(self.stream.env._current_events[0].sim_time)
        clock = np.float32(np.float32(event_time) / self.horizon)
        self.evidence['actor_real_clock'] &= bool(state[0] == clock)
        self.evidence['actor_clock_max_abs_error'] = max(
            self.evidence['actor_clock_max_abs_error'],
            abs(float(state[0]) - float(clock)) * self.horizon,
        )
        lag = self.stream.now - event_time
        stats = self.evidence.setdefault('ledger_vs_event_lag', dict(
            count=0, min_seconds=lag, max_seconds=lag))
        stats['count'] += 1
        stats['min_seconds'] = min(stats['min_seconds'], lag)
        stats['max_seconds'] = max(stats['max_seconds'], lag)
        self.last_proposal = self.learner.act(state[None], deterministic=True)[0]
        if self.stream.tail_policy_active:
            self.evidence['tail_policy_issues'] += 1
            return tail_action(self.transform, self.last_proposal)
        self.evidence['native_policy_issues'] += 1
        return self.transform.numpy(state, self.last_proposal)
