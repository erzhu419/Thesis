"""
coupling/holding_feedback.py
============================
Lower → Upper feedback: aggregate per-trip holding actions into statistics
that the upper policy uses as state input.

Core idea:
  If lower RL consistently holds at every stop for trip i →
    mean_holding(i) >> 0 → trip i departed too early → upper should shift later.
  If lower RL holding ≈ 0 everywhere →
    timetable was perfect for this trip → no adjustment needed.

The ideal steady state is mean_holding → 0 for all trips,
meaning the upper timetable exactly matches what the lower needs.
"""

import numpy as np
from collections import deque, defaultdict


class HoldingFeedback:
    """
    Tracks per-trip holding actions from the lower level and provides
    aggregated statistics for the upper policy's state vector.

    Used once per episode; call clear() at episode start.
    """

    def __init__(self, window_size=10):
        """
        Args:
            window_size: number of recent completed trips to keep per direction
                         for computing rolling statistics.
        """
        self.window_size = window_size

        # Per-trip within current episode: {trip_id: [action_1, action_2, ...]}
        self._trip_actions = defaultdict(list)

        # Rolling history across trips (per direction):
        #   deque of (trip_id, mean_hold, std_hold, n_stops)
        self._history = {
            True: deque(maxlen=window_size),   # direction=True (上行)
            False: deque(maxlen=window_size),   # direction=False (下行)
        }

        # Cross-episode persistence: EMA of mean holding per direction
        self._ema_holding = {True: 0.0, False: 0.0}
        self._ema_alpha = 0.3  # EMA decay

    def clear(self):
        """Reset per-episode data. Keep cross-episode EMA."""
        self._trip_actions.clear()

    def record_action(self, trip_id, action):
        """
        Record a holding action taken by lower RL at one stop.

        Args:
            trip_id: int, the trip (timetable entry) this bus is serving
            action: float, the holding time decided by lower policy (seconds)
        """
        self._trip_actions[trip_id].append(float(action))

    def finalize_trip(self, trip_id, direction):
        """
        Called when a trip completes. Computes summary stats and adds
        to rolling history.

        Args:
            trip_id: int
            direction: bool (True=上行, False=下行)
        """
        actions = self._trip_actions.get(trip_id, [])
        if not actions:
            return

        mean_hold = np.mean(actions)
        std_hold = np.std(actions)
        max_hold = np.max(actions)
        n_stops = len(actions)

        self._history[direction].append({
            'trip_id': trip_id,
            'mean': mean_hold,
            'std': std_hold,
            'max': max_hold,
            'n_stops': n_stops,
        })

        # Update EMA
        self._ema_holding[direction] = (
            self._ema_alpha * mean_hold +
            (1 - self._ema_alpha) * self._ema_holding[direction]
        )

    def get_trip_stats(self, trip_id):
        """
        Get holding stats for a specific trip (within current episode).
        Returns: dict with mean, std, max, n_stops, or None if no data.
        """
        actions = self._trip_actions.get(trip_id, [])
        if not actions:
            return None
        return {
            'mean': np.mean(actions),
            'std': np.std(actions),
            'max': np.max(actions),
            'n_stops': len(actions),
        }

    def get_direction_stats(self, direction):
        """
        Get rolling holding statistics for a direction (from recent trips).
        This is used to build the upper policy's state vector.

        Returns: dict with:
            rolling_mean: mean of recent trips' mean_holding
            rolling_std: std of recent trips' mean_holding
            rolling_trend: slope of recent mean_holdings (positive = getting worse)
            ema: exponential moving average of mean holding
            n_trips: number of trips in window
        """
        history = self._history[direction]
        if len(history) == 0:
            return {
                'rolling_mean': 0.0,
                'rolling_std': 0.0,
                'rolling_trend': 0.0,
                'ema': self._ema_holding[direction],
                'n_trips': 0,
            }

        means = [h['mean'] for h in history]
        rolling_mean = np.mean(means)
        rolling_std = np.std(means) if len(means) > 1 else 0.0

        # Trend: simple slope over window
        if len(means) >= 3:
            x = np.arange(len(means))
            slope = np.polyfit(x, means, 1)[0]
        else:
            slope = 0.0

        return {
            'rolling_mean': float(rolling_mean),
            'rolling_std': float(rolling_std),
            'rolling_trend': float(slope),
            'ema': float(self._ema_holding[direction]),
            'n_trips': len(history),
        }

    def compute_timetable_correction(self, trip_id):
        """
        Suggest how much to shift departure time based on holding history.

        If mean_holding > 0: bus was early → suggest positive shift (depart later)
        The magnitude is the mean holding time (in seconds).

        Returns: float, suggested δ_t in seconds (positive = depart later)
        """
        actions = self._trip_actions.get(trip_id, [])
        if not actions:
            return 0.0
        return float(np.mean(actions))

    def holding_penalty(self, trip_id):
        """
        Compute penalty for upper reward: how much lower-level intervention
        was needed for this trip. Lower is better.

        Returns: float in [0, 1], where 0 = no holding, 1 = max holding
        """
        actions = self._trip_actions.get(trip_id, [])
        if not actions:
            return 0.0
        # Normalize: max action is 60s, so mean/60 gives [0, 1]
        return float(np.mean(np.abs(actions)) / 60.0)

    @property
    def episode_summary(self):
        """Summary stats for the entire episode."""
        all_actions = []
        for actions in self._trip_actions.values():
            all_actions.extend(actions)
        if not all_actions:
            return {'mean': 0.0, 'std': 0.0, 'total_interventions': 0}
        return {
            'mean': float(np.mean(all_actions)),
            'std': float(np.std(all_actions)),
            'total_interventions': len(all_actions),
            'n_trips': len(self._trip_actions),
        }
