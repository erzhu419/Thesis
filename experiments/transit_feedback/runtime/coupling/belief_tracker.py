"""
coupling/belief_tracker.py
===========================
Bayesian Online Change Detection for lower-level non-stationarity.

Adapted from CS-BAPR (csbapr/belief/).

The upper level changes the timetable gradually, making the lower level's
MDP non-stationary. The belief tracker detects these shifts via:
  1. Reward distribution changes (upper timetable affects lower reward)
  2. Q-value ensemble disagreement spikes (stale Q estimates)

When a changepoint is detected (high surprise → short run-length):
  - Lower policy increases exploration (adaptive alpha boost)
  - Replay buffer sampling weights recent data more (implicit via belief)
"""

import numpy as np
from collections import deque


class SurpriseComputer:
    """
    Multi-signal surprise detector for lower-level non-stationarity.
    Detects when upper-level timetable changes cause distribution shift.
    """

    def __init__(self, ema_alpha=0.3, reward_window=50):
        self.ema_alpha = ema_alpha
        self.ema_surprise = 0.0
        self.reward_window = reward_window
        self.reward_history = deque(maxlen=reward_window)
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.prev_q_std = None
        # Track upper delta changes as additional signal
        self.prev_delta_mean = 0.0

    def reset(self):
        self.ema_surprise = 0.0
        self.reward_history.clear()
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.prev_q_std = None
        self.prev_delta_mean = 0.0

    def compute(self, reward_mean, q_std, delta_mean=None):
        """
        Compute surprise from lower-level signals + upper-level change rate.

        Args:
            reward_mean: float, mean reward of current episode/batch
            q_std: float, ensemble Q-value std (disagreement)
            delta_mean: float, mean upper δ_t this episode (optional)

        Returns:
            float: EMA-smoothed surprise scalar
        """
        signals = []

        # Signal 1: Reward z-score deviation
        self.reward_history.append(reward_mean)
        self.reward_ema = 0.9 * self.reward_ema + 0.1 * reward_mean
        deviation = (reward_mean - self.reward_ema) ** 2
        self.reward_var_ema = 0.9 * self.reward_var_ema + 0.1 * deviation
        reward_std = max(self.reward_var_ema ** 0.5, 1e-6)
        reward_zscore = abs(reward_mean - self.reward_ema) / reward_std
        signals.append(reward_zscore)

        # Signal 2: Q-std spike (ensemble disagreement change)
        if self.prev_q_std is not None and self.prev_q_std > 1e-6:
            q_std_change = abs(q_std - self.prev_q_std) / self.prev_q_std
            signals.append(q_std_change)
        self.prev_q_std = q_std

        # Signal 3: Upper policy change rate (timetable shift)
        if delta_mean is not None:
            delta_change = abs(delta_mean - self.prev_delta_mean)
            # Normalize: δ_t range is [-120, 120], so change of 20 is moderate
            signals.append(delta_change / 60.0)
            self.prev_delta_mean = delta_mean

        raw_surprise = max(signals) if signals else 0.0
        self.ema_surprise = (self.ema_alpha * raw_surprise +
                             (1 - self.ema_alpha) * self.ema_surprise)
        return self.ema_surprise


class BeliefTracker:
    """
    BOCD-style belief over run-lengths.
    Short run-length = recent changepoint = high non-stationarity.

    Used to modulate lower-level training:
      - adaptive_alpha_boost: increase exploration after changepoint
      - effective_window: how many recent episodes are "trustworthy"
    """

    def __init__(self, max_run_length=20, hazard_rate=0.05,
                 base_variance=0.1, variance_growth=0.05):
        self.max_H = max_run_length
        self.hazard = hazard_rate
        self.base_var = base_variance
        self.var_growth = variance_growth
        self.belief = np.ones(max_run_length) / max_run_length

    def reset(self):
        self.belief = np.ones(self.max_H) / self.max_H

    def update(self, surprise):
        """BOCD belief update."""
        L = self._compute_likelihood(surprise)
        unnorm = self.belief * L
        Z = unnorm.sum()
        if Z > 1e-10:
            self.belief = unnorm / Z
        else:
            self.belief = np.ones(self.max_H) / self.max_H

        growth_prob = self.belief * (1 - self.hazard)
        changepoint_prob = self.belief.sum() * self.hazard

        new_belief = np.zeros(self.max_H)
        new_belief[0] = changepoint_prob
        new_belief[1:] = growth_prob[:-1]
        total = new_belief.sum()
        self.belief = (new_belief / total if total > 1e-10
                       else np.ones(self.max_H) / self.max_H)

    def _compute_likelihood(self, surprise):
        variances = self.base_var + self.var_growth * np.arange(self.max_H)
        return np.exp(-surprise ** 2 / (2 * variances))

    @property
    def effective_window(self):
        """Expected run-length: how many recent episodes are 'trustworthy'."""
        return float(np.sum(np.arange(self.max_H) * self.belief))

    @property
    def changepoint_prob(self):
        """Probability that a changepoint just occurred (belief[0])."""
        return float(self.belief[0])

    @property
    def entropy(self):
        p = self.belief[self.belief > 1e-10]
        return float(-np.sum(p * np.log(p)))

    def adaptive_alpha_boost(self, base_alpha, max_boost=3.0):
        """
        Boost exploration alpha when changepoint detected.
        Short effective window → more boost → more exploration.

        Returns: float, boosted alpha value
        """
        # effective_window in [0, max_H-1]
        # When window is short (changepoint), boost is high
        window = max(self.effective_window, 1.0)
        boost = min(max_boost, self.max_H / window)
        return base_alpha * boost
