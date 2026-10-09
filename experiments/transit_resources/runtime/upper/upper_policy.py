"""
upper/upper_policy.py
=====================
Upper-level timetable policy πU for TransitDuet.

Independent network (not shared with lower policy).
Outputs K=3 parameters: [H_peak, H_off_peak, H_transition]
"""

import torch
import torch.nn as nn
import numpy as np


class UpperPolicy(nn.Module):
    """
    Gaussian policy for timetable headway parameters.

    Action space: K=3 continuous values
        [H_peak, H_off_peak, H_transition] in seconds

    Called once per dispatch event (~132 times per episode).
    """
    def __init__(self, state_dim=5, K=3, hidden_dim=64,
                 action_low=None, action_high=None, device='cpu'):
        super().__init__()
        self.K = K
        self.device = device

        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, K)
        self.log_std_head = nn.Linear(hidden_dim, K)

        # Init: output near sigmoid(0)=0.5 → mid-range action
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)
        Returns:
            mean: (batch, K) in [action_low, action_high]
            log_std: (batch, K) clamped
        """
        h = self.net(state)
        raw = torch.sigmoid(self.mean_head(h))
        mean = self.action_low + raw * (self.action_high - self.action_low)
        log_std = torch.clamp(self.log_std_head(h), -5.0, 0.0)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        """
        Single-sample action for rollout.

        Args:
            state: np.array of shape (state_dim,)
            deterministic: if True, return mean action
        Returns:
            action: np.array of shape (K,) in [action_low, action_high]
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                noise = torch.randn_like(mean) * std
                action = mean + noise

            action = torch.clamp(action, self.action_low, self.action_high)
        return action.squeeze(0).cpu().numpy()

    def evaluate(self, state):
        """
        For policy gradient: returns log_prob and entropy.

        Args:
            state: (batch, state_dim)
        Returns:
            action, log_prob, entropy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        # Clamp to valid range
        action = torch.clamp(raw_action, self.action_low, self.action_high)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy
