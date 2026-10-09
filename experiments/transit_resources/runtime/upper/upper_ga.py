"""
upper/upper_ga.py
=================
Genetic Algorithm for upper-level headway policy optimization.

Simple and robust: tournament selection, BLX-alpha crossover,
Gaussian mutation. Population of headway parameter vectors.
"""

import numpy as np


class GAUpperPolicy:
    """
    Genetic Algorithm for headway parameters [H_peak, H_off, H_trans].

    Each individual is a 3D vector in [action_low, action_high].
    Fitness = episode reward (higher is better).
    Evaluates one individual per episode.
    """

    def __init__(self, action_low=None, action_high=None,
                 pop_size=20, elite_ratio=0.2,
                 crossover_alpha=0.5, mutation_sigma=0.1,
                 mutation_prob=0.3):
        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_range = self.action_high - self.action_low
        self.dim = len(action_low)

        self.pop_size = pop_size
        self.n_elite = max(1, int(pop_size * elite_ratio))
        self.crossover_alpha = crossover_alpha
        self.mutation_sigma = mutation_sigma
        self.mutation_prob = mutation_prob

        # Initialize population (normalized [0,1]^d)
        self.population = [np.random.uniform(0, 1, self.dim)
                           for _ in range(pop_size)]
        self.fitness = [None] * pop_size
        self._eval_idx = 0
        self._generation = 0
        self._current_params = None

        # Best ever
        self.best_params = np.ones(self.dim) * 0.5
        self.best_fitness = -np.inf
        self.history = []

    def _to_headway(self, normalized):
        return self.action_low + np.clip(normalized, 0, 1) * self.action_range

    def suggest(self):
        """Return next headway params to evaluate."""
        if self._eval_idx >= self.pop_size:
            self._evolve()
            self._eval_idx = 0

        x = self.population[self._eval_idx]
        self._current_params = self._to_headway(x)
        return self._current_params.copy()

    def report(self, fitness):
        """Report fitness for the last suggested params."""
        self.fitness[self._eval_idx] = float(fitness)
        self._eval_idx += 1

        if float(fitness) > self.best_fitness:
            self.best_fitness = float(fitness)
            self.best_params = self.population[self._eval_idx - 1].copy()

        self.history.append({
            'gen': self._generation,
            'fitness': float(fitness),
            'params': self._current_params.tolist(),
        })

    def _evolve(self):
        """Create next generation via selection, crossover, mutation."""
        # Sort by fitness (descending)
        valid = [(i, f) for i, f in enumerate(self.fitness) if f is not None]
        if not valid:
            return
        valid.sort(key=lambda x: x[1], reverse=True)

        # Elite selection
        elite_indices = [v[0] for v in valid[:self.n_elite]]
        new_pop = [self.population[i].copy() for i in elite_indices]

        # Fill rest with crossover + mutation
        while len(new_pop) < self.pop_size:
            # Tournament selection (size=3)
            p1 = self._tournament_select(valid)
            p2 = self._tournament_select(valid)

            # BLX-alpha crossover
            child = self._blx_crossover(
                self.population[p1], self.population[p2])

            # Gaussian mutation
            child = self._mutate(child)

            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self.fitness = [None] * self.pop_size
        self._generation += 1

    def _tournament_select(self, valid_sorted, k=3):
        """Tournament selection from sorted (index, fitness) list."""
        contestants = np.random.choice(len(valid_sorted), size=min(k, len(valid_sorted)),
                                       replace=False)
        winner = min(contestants)  # lower index = higher fitness (sorted desc)
        return valid_sorted[winner][0]

    def _blx_crossover(self, p1, p2):
        """BLX-alpha crossover: sample uniformly from extended range."""
        alpha = self.crossover_alpha
        d = np.abs(p1 - p2)
        lo = np.minimum(p1, p2) - alpha * d
        hi = np.maximum(p1, p2) + alpha * d
        child = np.random.uniform(lo, hi)
        return np.clip(child, 0, 1)

    def _mutate(self, individual):
        """Gaussian mutation with per-gene probability."""
        mask = np.random.random(self.dim) < self.mutation_prob
        noise = np.random.randn(self.dim) * self.mutation_sigma
        individual = individual + mask * noise
        return np.clip(individual, 0, 1)

    def get_action(self, state=None, deterministic=False):
        """Interface compatible with runner."""
        if deterministic:
            return self._to_headway(self.best_params)
        if self._current_params is None:
            self.suggest()
        return self._current_params.copy()

    def get_best(self):
        """Return best parameters found."""
        return self._to_headway(self.best_params)
