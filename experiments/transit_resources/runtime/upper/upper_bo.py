"""
upper/upper_bo.py
=================
Bayesian Optimization for upper-level headway policy.

Optimizes a small parameter vector (3-9 params) that maps
time-of-day to target headways. Extremely sample-efficient
for low-dimensional continuous optimization.

Uses a Gaussian Process surrogate + Expected Improvement acquisition.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class GaussianProcessSimple:
    """Minimal GP with RBF kernel for Bayesian Optimization."""

    def __init__(self, length_scale=1.0, noise=0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X = None
        self.y = None

    def _rbf_kernel(self, X1, X2):
        """RBF kernel: k(x1, x2) = exp(-||x1-x2||^2 / (2*l^2))"""
        sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X_new):
        """Return mean and std of GP prediction."""
        X_new = np.atleast_2d(X_new)
        if self.X is None or len(self.X) == 0:
            return np.zeros(len(X_new)), np.ones(len(X_new))

        K = self._rbf_kernel(self.X, self.X) + self.noise ** 2 * np.eye(len(self.X))
        K_star = self._rbf_kernel(X_new, self.X)
        K_ss = self._rbf_kernel(X_new, X_new)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            v = np.linalg.solve(L, K_star.T)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(len(K))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            v = np.linalg.solve(L, K_star.T)

        mu = K_star @ alpha
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-8)
        return mu, np.sqrt(var)


def expected_improvement(X, gp, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    mu, sigma = gp.predict(X)
    imp = mu - y_best - xi
    Z = imp / (sigma + 1e-8)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


class BOUpperPolicy:
    """
    Bayesian Optimization for headway parameters.

    Parameterization: 3 values [H_peak, H_off_peak, H_transition]
    mapped to [action_low, action_high] per dimension.

    The BO treats each episode as one evaluation: run env with
    these headways → observe system reward → update GP → suggest next.
    """

    def __init__(self, action_low=None, action_high=None,
                 n_initial=10, length_scale=0.5, noise=0.1):
        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_range = self.action_high - self.action_low
        self.dim = len(action_low)
        self.n_initial = n_initial

        self.gp = GaussianProcessSimple(length_scale=length_scale, noise=noise)
        self.X_history = []  # normalized [0,1]^d
        self.y_history = []  # reward values
        self._current_params = None

    def _to_normalized(self, params):
        return (np.array(params) - self.action_low) / self.action_range

    def _to_headway(self, normalized):
        return self.action_low + np.array(normalized) * self.action_range

    def suggest(self):
        """Suggest next headway parameters to try."""
        if len(self.X_history) < self.n_initial:
            # Random initial samples (Latin hypercube-ish)
            x = np.random.uniform(0, 1, self.dim)
            self._current_params = self._to_headway(x)
            return self._current_params.copy()

        # Fit GP
        self.gp.fit(np.array(self.X_history), np.array(self.y_history))
        y_best = max(self.y_history)

        # Optimize EI via random restarts
        best_ei = -1
        best_x = None
        for _ in range(200):
            x0 = np.random.uniform(0, 1, self.dim)
            res = minimize(
                lambda x: -expected_improvement(x.reshape(1, -1), self.gp, y_best),
                x0, bounds=[(0, 1)] * self.dim, method='L-BFGS-B')
            ei_val = -res.fun
            if ei_val > best_ei:
                best_ei = ei_val
                best_x = res.x

        if best_x is None:
            best_x = np.random.uniform(0, 1, self.dim)

        self._current_params = self._to_headway(best_x)
        return self._current_params.copy()

    def report(self, reward):
        """Report the reward from the last suggested parameters."""
        if self._current_params is not None:
            self.X_history.append(self._to_normalized(self._current_params))
            self.y_history.append(float(reward))

    def get_action(self, state=None, deterministic=False):
        """Interface compatible with runner: returns current params."""
        if self._current_params is None:
            self.suggest()
        return self._current_params.copy()

    def get_best(self):
        """Return the best parameters found so far."""
        if not self.y_history:
            return self._to_headway(np.ones(self.dim) * 0.5)
        best_idx = np.argmax(self.y_history)
        return self._to_headway(self.X_history[best_idx])
