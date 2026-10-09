"""
upper/upper_cmaes.py
====================
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for upper-level
headway policy optimization.

Gold standard for low-dimensional (dim < 20) continuous black-box optimization.
Population-based: evaluates multiple headway configs per generation,
adapts search distribution covariance.

Reference: Hansen & Ostermeier 2001, "Completely Derandomized Self-Adaptation
in Evolution Strategies"
"""

import numpy as np


class CMAESUpperPolicy:
    """
    CMA-ES optimizer for headway parameters [H_peak, H_off, H_trans].

    Works in normalized [0,1]^d space, maps to [action_low, action_high].
    Each "evaluation" is one episode of the bus simulation.
    """

    def __init__(self, action_low=None, action_high=None,
                 pop_size=10, sigma0=0.3):
        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_range = self.action_high - self.action_low
        self.dim = len(action_low)

        # CMA-ES state
        self.pop_size = pop_size  # λ (population size)
        self.mu_count = pop_size // 2  # μ (parent count)
        self.sigma = sigma0  # step size

        # Initialize mean at center of search space
        self.mean = np.ones(self.dim) * 0.5

        # Covariance matrix (identity initially)
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)  # evolution path for σ
        self.p_c = np.zeros(self.dim)      # evolution path for C

        # Weights for recombination
        weights = np.log(self.mu_count + 0.5) - np.log(np.arange(1, self.mu_count + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # Generation tracking
        self._generation = 0
        self._current_pop = []    # current generation's candidates (normalized)
        self._current_idx = 0     # which candidate we're evaluating
        self._current_fitness = [] # fitness of evaluated candidates
        self._current_params = None

        # Best ever
        self.best_params = self.mean.copy()
        self.best_fitness = -np.inf

        # History
        self.history = []

        # Generate first population
        self._generate_population()

    def _generate_population(self):
        """Sample a new population from N(mean, sigma^2 * C)."""
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            L = np.eye(self.dim)

        self._current_pop = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * L @ z
            x = np.clip(x, 0, 1)  # stay in bounds
            self._current_pop.append(x)
        self._current_idx = 0
        self._current_fitness = []

    def _to_headway(self, normalized):
        return self.action_low + np.clip(normalized, 0, 1) * self.action_range

    def suggest(self):
        """Return next headway params to evaluate."""
        if self._current_idx >= self.pop_size:
            # All candidates evaluated, update CMA-ES and new generation
            self._update()
            self._generate_population()

        x = self._current_pop[self._current_idx]
        self._current_params = self._to_headway(x)
        return self._current_params.copy()

    def report(self, fitness):
        """Report fitness (higher = better) for the last suggested params."""
        self._current_fitness.append(float(fitness))
        self._current_idx += 1

        if float(fitness) > self.best_fitness:
            self.best_fitness = float(fitness)
            self.best_params = self._current_pop[self._current_idx - 1].copy()

        self.history.append({
            'gen': self._generation,
            'fitness': float(fitness),
            'params': self._current_params.tolist(),
        })

    def _update(self):
        """CMA-ES update step after a full generation."""
        if len(self._current_fitness) < self.pop_size:
            return

        # Sort by fitness (descending — we maximize)
        indices = np.argsort(self._current_fitness)[::-1]
        pop = np.array(self._current_pop)

        # Weighted recombination of top μ
        selected = pop[indices[:self.mu_count]]
        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        # Evolution paths
        delta = (self.mean - old_mean) / self.sigma
        try:
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C))
        except np.linalg.LinAlgError:
            C_inv_sqrt = np.eye(self.dim)

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
            np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_inv_sqrt @ delta

        h_sigma = 1.0 if np.linalg.norm(self.p_sigma) / \
            np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self._generation + 1))) < \
            (1.4 + 2 / (self.dim + 1)) * self.chi_n else 0.0

        self.p_c = (1 - self.c_c) * self.p_c + \
            h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * delta

        # Covariance matrix update
        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.zeros_like(self.C)
        for k in range(self.mu_count):
            d_k = (selected[k] - old_mean) / self.sigma
            rank_mu += self.weights[k] * np.outer(d_k, d_k)

        self.C = (1 - self.c1 - self.c_mu) * self.C + \
            self.c1 * rank_one + self.c_mu * rank_mu

        # Symmetrize
        self.C = (self.C + self.C.T) / 2

        # Step size update
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) *
            (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        self.sigma = np.clip(self.sigma, 1e-4, 2.0)

        self._generation += 1

    def get_action(self, state=None, deterministic=False):
        """Interface compatible with runner."""
        if self._current_params is None:
            self.suggest()
        return self._current_params.copy()

    def get_best(self):
        """Return best parameters found."""
        return self._to_headway(self.best_params)
