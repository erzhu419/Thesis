"""
upper/upper_contextual_cmaes.py
================================
Contextual CMA-ES: optimizes a state-dependent headway policy.

Instead of 3 static headway params, optimizes a small parametric policy:
  headway(state) = sigmoid(W @ state + b) * (action_high - action_low) + action_low

where W is [3, 5] and b is [3], total 18 parameters.

This gives CMA-ES's sample efficiency + state-dependent decisions.
Can react to real-time demand, fleet size, and holding pressure.
"""

import numpy as np


class ContextualPolicy:
    """Tiny linear policy: state (5D) → headway (3D) via sigmoid."""

    def __init__(self, state_dim=5, action_dim=3,
                 action_low=None, action_high=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_range = self.action_high - self.action_low

        # Parameters: W [3, 5] + b [3] = 18 total
        self.n_params = action_dim * state_dim + action_dim
        self.params = np.zeros(self.n_params)

    def set_params(self, params):
        self.params = np.array(params)

    def get_params(self):
        return self.params.copy()

    def __call__(self, state):
        """state: np.array(5,) → headway: np.array(3,)"""
        W = self.params[:self.action_dim * self.state_dim].reshape(
            self.action_dim, self.state_dim)
        b = self.params[self.action_dim * self.state_dim:]
        logit = W @ state + b
        u = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))  # sigmoid
        return self.action_low + u * self.action_range


class ContextualCMAESUpperPolicy:
    """
    CMA-ES optimizing a contextual linear policy (18 params).

    Each episode: use current policy to set headways state-dependently,
    report system reward, CMA-ES updates search distribution.
    """

    def __init__(self, state_dim=5, action_dim=3,
                 action_low=None, action_high=None,
                 pop_size=12, sigma0=0.5):
        self.policy = ContextualPolicy(
            state_dim, action_dim, action_low, action_high)
        self.dim = self.policy.n_params  # 18

        self.pop_size = pop_size
        self.mu_count = pop_size // 2
        self.sigma = sigma0

        # CMA-ES state
        self.mean = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)

        weights = np.log(self.mu_count + 0.5) - np.log(
            np.arange(1, self.mu_count + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0, np.sqrt(
            (self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma)
        self.c_c = ((4 + self.mu_eff / self.dim) /
                    (self.dim + 4 + 2 * self.mu_eff / self.dim))
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                        ((self.dim + 2) ** 2 + self.mu_eff))
        self.chi_n = (np.sqrt(self.dim) *
                      (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2)))

        self._generation = 0
        self._current_pop = []
        self._current_idx = 0
        self._current_fitness = []

        self.best_params = self.mean.copy()
        self.best_fitness = -np.inf
        self.history = []

        self._generate_population()

    def _generate_population(self):
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            L = np.eye(self.dim)

        self._current_pop = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * L @ z
            self._current_pop.append(x)
        self._current_idx = 0
        self._current_fitness = []

    def suggest(self):
        """Set policy params for next evaluation, return the policy."""
        if self._current_idx >= self.pop_size:
            self._update()
            self._generate_population()

        params = self._current_pop[self._current_idx]
        self.policy.set_params(params)
        return self.policy

    def report(self, fitness):
        self._current_fitness.append(float(fitness))
        if float(fitness) > self.best_fitness:
            self.best_fitness = float(fitness)
            self.best_params = self._current_pop[self._current_idx].copy()
        self.history.append({
            'gen': self._generation,
            'idx': self._current_idx,
            'fitness': float(fitness),
        })
        self._current_idx += 1

    def _update(self):
        if len(self._current_fitness) < self.pop_size:
            return

        indices = np.argsort(self._current_fitness)[::-1]
        pop = np.array(self._current_pop)
        selected = pop[indices[:self.mu_count]]
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        delta = (self.mean - old_mean) / self.sigma
        try:
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C))
        except np.linalg.LinAlgError:
            C_inv_sqrt = np.eye(self.dim)

        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma +
                        np.sqrt(self.c_sigma * (2 - self.c_sigma) *
                                self.mu_eff) * C_inv_sqrt @ delta)

        h_sigma = (1.0 if np.linalg.norm(self.p_sigma) /
                   np.sqrt(1 - (1 - self.c_sigma) **
                           (2 * (self._generation + 1))) <
                   (1.4 + 2 / (self.dim + 1)) * self.chi_n else 0.0)

        self.p_c = ((1 - self.c_c) * self.p_c +
                    h_sigma * np.sqrt(self.c_c * (2 - self.c_c) *
                                      self.mu_eff) * delta)

        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.zeros_like(self.C)
        for k in range(self.mu_count):
            d_k = (selected[k] - old_mean) / self.sigma
            rank_mu += self.weights[k] * np.outer(d_k, d_k)

        self.C = ((1 - self.c1 - self.c_mu) * self.C +
                  self.c1 * rank_one + self.c_mu * rank_mu)
        self.C = (self.C + self.C.T) / 2

        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) *
            (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, 1e-4, 2.0)

        self._generation += 1

    def get_best_policy(self):
        """Return the best policy found."""
        p = ContextualPolicy(
            self.policy.state_dim, self.policy.action_dim,
            self.policy.action_low.tolist(), self.policy.action_high.tolist())
        p.set_params(self.best_params)
        return p
