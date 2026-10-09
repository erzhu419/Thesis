"""
coupling/tap.py
===============
Temporal Advantage Propagation (TAP)

Core innovation of TransitDuet: extends RoboDuet's synchronous cross-advantage
injection to asynchronous, multi-timescale hierarchical systems.

RoboDuet (synchronous):
    loss_dog = -(A_dog + β·A_arm) · ratio_dog
    ← both advantages available at the same timestep

TAP (asynchronous):
    A_U_augmented[i] = A_U[i] + β · mean(A_L[j ∈ trip_i])
    A_L_augmented[j] = A_L[j] + β · A_U[trip_of(j)]
    ← advantages from different timescales, aggregated temporally
"""

import numpy as np
from collections import defaultdict


class TAPManager:
    """
    Manages cross-layer advantage propagation between upper (dispatch)
    and lower (holding) policies.

    Usage:
        tap = TAPManager(beta_schedule)

        # During episode:
        tap.record_upper_transition(s, a, r, trip_id)
        tap.record_lower_reward(reward, trip_id)

        # At episode end:
        aug_upper_returns = tap.compute_augmented_upper_returns(episode)
        trip_advantage_for_lower = tap.get_upper_advantage_for_trip(trip_id)
    """

    def __init__(self, beta_schedule):
        self.beta_schedule = beta_schedule
        self.clear()

    def clear(self):
        """Reset at start of each episode."""
        self._upper_transitions = []  # [(s, a, r_upper, trip_id)]
        self._lower_rewards_by_trip = defaultdict(list)  # {trip_id: [r_lower]}
        self._upper_rewards_by_trip = {}  # {trip_id: r_upper} (filled at ep end)

    def record_upper_transition(self, s_upper, a_upper, trip_id):
        """
        Record an upper-level dispatch event.
        NOTE: r_upper is not yet available — will be set at episode end.
        """
        self._upper_transitions.append({
            'state': s_upper.copy() if isinstance(s_upper, np.ndarray) else s_upper,
            'action': a_upper.copy() if isinstance(a_upper, np.ndarray) else a_upper,
            'trip_id': trip_id,
        })

    def record_lower_reward(self, reward, trip_id):
        """
        Record a lower-level reward obtained during a specific trip's operation.
        Called every time a bus (associated with trip_id) gets a reward.
        """
        self._lower_rewards_by_trip[trip_id].append(float(reward))

    def compute_augmented_upper_returns(self, episode, upper_reward_per_trip):
        """
        Compute TAP-augmented upper returns: R_U[i] + β·mean(R_L[j ∈ trip_i])

        Args:
            episode: current episode number (for beta schedule)
            upper_reward_per_trip: dict {trip_id: r_upper}
                Typically computed from measurement_vector at episode end.
                For simplicity, we distribute the total R_upper equally.

        Returns:
            list of dicts: [{trip_id, augmented_return, state, action}, ...]
        """
        beta = self.beta_schedule.get_beta(episode)

        results = []
        for trans in self._upper_transitions:
            tid = trans['trip_id']
            r_upper = upper_reward_per_trip.get(tid, 0.0)

            # TAP: inject lower-level performance during this trip
            lower_rewards = self._lower_rewards_by_trip.get(tid, [])
            if lower_rewards and beta > 0:
                cross_term = beta * np.mean(lower_rewards)
            else:
                cross_term = 0.0

            results.append({
                'trip_id': tid,
                'state': trans['state'],
                'action': trans['action'],
                'augmented_return': r_upper + cross_term,
                'raw_upper': r_upper,
                'cross_term': cross_term,
            })

        return results

    def get_upper_signal_for_trip(self, trip_id, episode, upper_reward_per_trip):
        """
        For lower-level training: get the upper advantage signal to inject
        into lower transitions belonging to `trip_id`.

        TAP reverse direction: A_L_augmented[j] = A_L[j] + β·A_U[trip_of(j)]

        Returns:
            float: β · r_upper_of_trip (to be added to lower advantage)
        """
        beta = self.beta_schedule.get_beta(episode)
        if beta == 0:
            return 0.0
        r_upper = upper_reward_per_trip.get(trip_id, 0.0)
        return beta * r_upper

    @property
    def num_upper_transitions(self):
        return len(self._upper_transitions)

    @property
    def num_lower_rewards(self):
        return sum(len(v) for v in self._lower_rewards_by_trip.values())

    def summary(self):
        return {
            'upper_transitions': self.num_upper_transitions,
            'lower_rewards_total': self.num_lower_rewards,
            'trips_with_lower_data': len(self._lower_rewards_by_trip),
        }
