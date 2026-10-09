"""
upper/measurement_proj.py
=========================
ApproPO-inspired measurement projection for fleet size constraint.

Uses θ-OGD to adaptively modulate upper reward weights.
When fleet consistently exceeds N_fleet → θ increases fleet penalty weight.

Reference: Miryoosefi et al., 2019 - "Reinforcement Learning with Convex Constraints"
           (Algorithm 2: OGD on polar cone)
"""

import numpy as np


class MeasurementProjection:
    """
    Maintains θ ∈ C° ∩ B (polar cone ∩ unit ball) and uses OGD
    to adaptively weight upper-level reward components.

    Measurement vector z = [avg_wait, peak_fleet, bunching_rate]
    Target set C = {z : z[1] ≤ N_fleet}

    NOT a full ApproPO nested loop (computationally infeasible for bus env).
    Instead, borrows the θ-OGD mechanism to auto-tune reward weights.
    """

    def __init__(self, N_fleet=12, d=3, lr=0.01):
        """
        Args:
            N_fleet: fleet size hard constraint
            d: measurement vector dimension
            lr: OGD step size
        """
        self.N_fleet = N_fleet
        self.d = d
        self.lr = lr

        # Initialize theta: bias toward penalizing wait time
        self.theta = np.zeros(d, dtype=np.float64)
        self.theta[0] = -0.5  # wait penalty
        self.theta[1] = -0.3  # fleet penalty
        self.theta[2] = -0.2  # bunching penalty
        self._normalize_theta()

        self._iter = 1
        self._z_history = []

    def _normalize_theta(self):
        """Project theta onto unit ball."""
        norm = np.linalg.norm(self.theta)
        if norm > 1.0:
            self.theta /= norm

    def update(self, z_observed):
        """
        OGD update with observed measurement vector.
        Called once per episode at episode end.

        Args:
            z_observed: np.array [avg_wait, peak_fleet, bunching_rate]
        """
        self._z_history.append(z_observed.copy())

        # Fleet violation signal: extra penalty when over limit
        fleet_violation = max(0.0, z_observed[1] - self.N_fleet)

        # Loss gradient: directions that should increase penalty
        loss_vector = np.array([
            -z_observed[0],       # higher wait → higher penalty weight
            -fleet_violation,     # fleet over limit → higher fleet penalty
            -z_observed[2],       # higher bunching → higher bunching penalty
        ])

        # OGD step with decreasing learning rate
        eta = self.lr / np.sqrt(self._iter)
        self.theta = self.theta + eta * loss_vector

        # Project to unit ball (simplified polar cone projection)
        self._normalize_theta()
        self._iter += 1

    def compute_upper_reward(self, z):
        """
        Compute θ-weighted upper-level reward from measurement vector.

        Args:
            z: np.array [avg_wait, peak_fleet, bunching_rate]
        Returns:
            float: scalar reward (higher is better)
        """
        fleet_over = max(0.0, z[1] - self.N_fleet) ** 2
        penalties = np.array([z[0], fleet_over, z[2]])
        return float(np.dot(self.theta, -penalties))

    def get_reward_weights(self):
        """Return current θ as interpretable weights (positive, sum-to-1)."""
        w = np.abs(self.theta)
        total = w.sum()
        if total < 1e-8:
            return np.ones(self.d) / self.d
        return w / total

    @property
    def fleet_penalty_weight(self):
        """Current weight on fleet size penalty (for logging)."""
        return abs(self.theta[1])

    @property
    def z_history(self):
        return self._z_history
