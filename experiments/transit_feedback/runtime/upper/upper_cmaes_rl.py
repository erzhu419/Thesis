"""
upper/upper_cmaes_rl.py
========================
Two-phase upper-level optimizer:
  Phase 1: CMA-ES finds good static headway params (fast, sample-efficient)
  Phase 2: RL fine-tunes a state-conditional residual on top of CMA-ES solution

The insight: CMA-ES finds the global optimum for the "average case",
but can't adapt to real-time conditions. RL learns a small residual
delta(state) that adjusts headways based on current demand/fleet/etc.

  headway(state) = CMA_ES_best + clip(RL_residual(state), -delta_max, delta_max)

This gives CMA-ES's efficiency + RL's adaptiveness, with RL only needing
to learn a small correction (much easier than learning from scratch).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random

from upper.upper_cmaes import CMAESUpperPolicy


class ResidualPolicy(nn.Module):
    """
    Small MLP that outputs a residual adjustment to CMA-ES base headways.
    Output clamped to [-delta_max, +delta_max] per dimension.
    """

    def __init__(self, state_dim=5, action_dim=3, hidden_dim=32,
                 delta_max=None):
        super().__init__()
        if delta_max is None:
            delta_max = [60., 120., 90.]  # max adjustment per period (seconds)
        self.register_buffer('delta_max',
                             torch.tensor(delta_max, dtype=torch.float32))

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Init near zero: residual starts at 0 (trust CMA-ES baseline)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, -2.0)  # small initial std

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.delta_max  # [-delta, +delta]
        log_std = torch.clamp(self.log_std(x), -5.0, -1.0)  # small std
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        delta = torch.tanh(z) * self.delta_max
        log_prob = dist.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return delta, log_prob

    def get_delta(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                delta = mean
            else:
                std = log_std.exp()
                delta = mean + std * torch.randn_like(mean)
                delta = torch.clamp(delta, -self.delta_max, self.delta_max)
        return delta.squeeze(0).cpu().numpy()


class ResidualReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, delta, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(delta, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, d, r, ns, done = zip(*batch)
        return (np.array(s), np.array(d),
                np.array(r).reshape(-1, 1),
                np.array(ns), np.array(done).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


class CMAESRLUpperPolicy:
    """
    Phase 1 (CMA-ES): optimize static headway [H_peak, H_off, H_trans].
    Phase 2 (RL residual): fine-tune with state-dependent adjustments.

    The RL only learns a SMALL residual on top of CMA-ES's solution.
    This is much easier than learning headways from scratch because:
    - The search space is bounded (±delta_max)
    - The baseline is already good (CMA-ES optimum)
    - RL only needs to learn WHEN to deviate, not the absolute values
    """

    def __init__(self, state_dim=5, action_dim=3,
                 action_low=None, action_high=None,
                 delta_max=None,
                 # CMA-ES params
                 cmaes_pop_size=10, cmaes_sigma=0.3,
                 cmaes_budget=80,
                 # RL params
                 rl_lr=3e-4, rl_gamma=0.95, rl_batch_size=64,
                 device='cpu'):

        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]
        if delta_max is None:
            delta_max = [60., 120., 90.]

        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.delta_max = delta_max
        self.cmaes_budget = cmaes_budget
        self.device = device

        # Phase 1: CMA-ES
        self.cmaes = CMAESUpperPolicy(
            action_low=action_low, action_high=action_high,
            pop_size=cmaes_pop_size, sigma0=cmaes_sigma)

        # Phase 2: RL residual
        self.residual_net = ResidualPolicy(
            state_dim, action_dim, hidden_dim=32,
            delta_max=delta_max).to(device)
        self.residual_optimizer = optim.Adam(
            self.residual_net.parameters(), lr=rl_lr)
        self.replay_buffer = ResidualReplayBuffer(capacity=10000)
        self.rl_batch_size = rl_batch_size
        self.rl_gamma = rl_gamma

        # State
        self._phase = 1  # 1=CMA-ES, 2=RL
        self._cmaes_evals = 0
        self._base_headway = np.array([360., 360., 360.])

        # For building (s, delta, r, s') transitions
        self._prev_state = None
        self._prev_delta = None

    @property
    def phase(self):
        return self._phase

    def suggest(self, state=None):
        """
        Phase 1: return CMA-ES suggestion (static).
        Phase 2: return CMA-ES_best + RL_residual(state).
        """
        if self._phase == 1:
            params = self.cmaes.suggest()
            return params

        # Phase 2: base + residual
        if state is None:
            state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        delta = self.residual_net.get_delta(state, deterministic=False)
        headway = self._base_headway + delta
        headway = np.clip(headway, self.action_low, self.action_high)
        return headway

    def suggest_with_state(self, state):
        """State-dependent suggestion (for use in dispatch callback)."""
        if self._phase == 1:
            return self.cmaes.suggest()

        delta = self.residual_net.get_delta(state, deterministic=False)
        headway = self._base_headway + delta

        # Store for replay buffer
        if self._prev_state is not None:
            # We'll get the reward later via report_dispatch()
            pass
        self._prev_state = state.copy()
        self._prev_delta = delta.copy()

        return np.clip(headway, self.action_low, self.action_high)

    def report(self, fitness):
        """Report episode-level fitness."""
        if self._phase == 1:
            self.cmaes.report(fitness)
            self._cmaes_evals += 1

            # Check if CMA-ES phase is done
            if self._cmaes_evals >= self.cmaes_budget:
                self._base_headway = self.cmaes.get_best()
                self._phase = 2
                print(f"  [Phase 2] CMA-ES done. Base headway: "
                      f"[{self._base_headway[0]:.0f}, "
                      f"{self._base_headway[1]:.0f}, "
                      f"{self._base_headway[2]:.0f}]")

    def report_dispatch(self, state, reward, next_state, done):
        """Report per-dispatch transition for RL training (Phase 2 only)."""
        if self._phase == 2 and self._prev_delta is not None:
            self.replay_buffer.push(
                self._prev_state, self._prev_delta,
                reward, next_state, done)
            self._prev_state = state.copy() if not done else None
            self._prev_delta = None

    def train_rl(self, n_updates=5):
        """Train residual RL from replay buffer (Phase 2 only)."""
        if self._phase != 2 or len(self.replay_buffer) < self.rl_batch_size:
            return {}

        metrics = {}
        for _ in range(n_updates):
            s, d, r, ns, done = self.replay_buffer.sample(self.rl_batch_size)
            s_t = torch.FloatTensor(s).to(self.device)
            d_t = torch.FloatTensor(d).to(self.device)
            r_t = torch.FloatTensor(r).to(self.device)
            ns_t = torch.FloatTensor(ns).to(self.device)
            done_t = torch.FloatTensor(done).to(self.device)

            # Simple REINFORCE-style with baseline
            # (no critic — residual is small, variance is manageable)
            _, log_prob = self.residual_net.evaluate(s_t)

            # Normalize rewards
            if r_t.std() > 1e-6:
                r_norm = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
            else:
                r_norm = r_t

            loss = -(log_prob.squeeze(-1) * r_norm.squeeze(-1)).mean()

            self.residual_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.residual_net.parameters(), 0.5)
            self.residual_optimizer.step()

            metrics = {'residual_loss': loss.item()}

        return metrics

    def get_best(self):
        """Return base headway (CMA-ES optimum)."""
        if self._phase == 1:
            return self.cmaes.get_best()
        return self._base_headway.copy()
