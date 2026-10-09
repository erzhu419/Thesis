"""
lower/dsac_lagrangian.py
========================
DSAC (Distributional SAC) with Lagrangian cost constraint.

Adapted from omnisafe/Holding_control/dsac_lag_bus.py.
Key additions:
  - cost_critic: Q_cost(s, a) for headway deviation
  - learnable λ (log_lambda): Lagrangian multiplier
  - policy_loss includes λ·Q_cost term

The lower policy is a parameter-shared single agent: one network
controls ALL buses (same architecture as original DSAC trainer).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class GaussianPolicy(nn.Module):
    """Gaussian policy for holding time in [0, action_range]."""

    def __init__(self, num_inputs, hidden_dim=32, action_range=60.0, init_w=3e-3):
        super().__init__()
        self.action_range = action_range

        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        action = action * self.action_range
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean) * self.action_range
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            z = dist.sample()
            action = torch.tanh(z) * self.action_range
        return action.detach().squeeze().cpu().numpy()


class TwinQNetwork(nn.Module):
    """Twin Q-network for clipped double Q-learning."""

    def __init__(self, num_inputs, num_actions, hidden_dim=32, init_w=3e-3):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class CostQNetwork(nn.Module):
    """Single Q-network for cost (headway deviation) value estimation."""

    def __init__(self, num_inputs, num_actions, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)


class DSACLagrangianTrainer:
    """
    DSAC trainer with Lagrangian cost constraint.

    Adds to standard SAC:
      - cost_critic: estimates expected cost Q_c(s,a)
      - log_lambda: learnable Lagrangian multiplier
      - policy loss = α·log_prob - Q(s,a) + λ·Q_c(s,a)
      - lambda update: maximize λ·(cost_limit - E[Q_c])
    """

    def __init__(self, state_dim, action_dim=1, hidden_dim=32,
                 action_range=60.0, cost_limit=0.15, lr=1e-5,
                 lambda_lr=1e-3, gamma=0.99, soft_tau=5e-3,
                 auto_entropy=True, maximum_alpha=0.3,
                 device='cpu'):
        self.device = device
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.cost_limit = cost_limit
        self.auto_entropy = auto_entropy

        # Networks
        self.policy_net = GaussianPolicy(state_dim, hidden_dim, action_range).to(device)

        self.q_net = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net = TwinQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.cost_q_net = CostQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_cost_q_net = CostQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_cost_q_net.load_state_dict(self.cost_q_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.cost_q_optimizer = optim.Adam(self.cost_q_net.parameters(), lr=lr)

        # Entropy temperature α
        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -1.0 * action_dim
        self.alpha = 0.1
        self.maximum_alpha = maximum_alpha

        # Lagrangian multiplier λ
        self.log_lambda = torch.zeros(1, requires_grad=True, device=device)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=lambda_lr)

    @property
    def lambda_param(self):
        return self.log_lambda.exp().item()

    def update(self, replay_buffer, batch_size, reward_scale=10.0,
               update_policy=True, tap_signal=None):
        """
        One gradient step for critic, policy, and lambda.

        Args:
            replay_buffer: CostReplayBuffer
            batch_size: int
            reward_scale: float
            update_policy: bool (False during critic warm-up)
            tap_signal: dict {trip_id: float} optional TAP upper advantage
        Returns:
            dict of training metrics
        """
        state, action, reward, cost, next_state, done, trip_ids = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device) * reward_scale
        cost = torch.FloatTensor(cost).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Apply TAP: augment reward with upper signal
        if tap_signal is not None:
            tap_bonus = torch.zeros_like(reward)
            for i, tid in enumerate(trip_ids):
                if tid in tap_signal:
                    tap_bonus[i] = tap_signal[tid] * reward_scale
            reward = reward + tap_bonus

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
            tq1, tq2 = self.target_q_net(next_state, next_action)
            target_q = torch.min(tq1, tq2) - self.alpha * next_log_prob
            target_value = reward + (1.0 - done) * self.gamma * target_q

        q1, q2 = self.q_net(state, action)
        q_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()

        # ---- Cost critic update ----
        with torch.no_grad():
            next_action_c, _, _, _, _ = self.policy_net.evaluate(next_state)
            target_cost_q = self.target_cost_q_net(next_state, next_action_c)
            target_cost_value = cost + (1.0 - done) * self.gamma * target_cost_q

        cost_q = self.cost_q_net(state, action)
        cost_q_loss = F.mse_loss(cost_q, target_cost_value)

        self.cost_q_optimizer.zero_grad()
        cost_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_q_net.parameters(), 1.0)
        self.cost_q_optimizer.step()

        # ---- Policy update ----
        metrics = {'q_loss': q_loss.item(), 'cost_q_loss': cost_q_loss.item()}

        if update_policy:
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)
            q1_new, q2_new = self.q_net(state, new_action)
            q_new = torch.min(q1_new, q2_new)
            cost_q_new = self.cost_q_net(state, new_action)

            lam = self.log_lambda.exp().detach()
            policy_loss = (self.alpha * log_prob - q_new + lam * cost_q_new).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()

            # ---- Alpha (entropy) update ----
            if self.auto_entropy:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = min(self.log_alpha.exp().item(), self.maximum_alpha)

            # ---- Lambda update ----
            lambda_loss = -self.log_lambda.exp() * (self.cost_limit - cost_q_new.mean().detach())
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
            self.log_lambda.data.clamp_(min=-10.0, max=5.0)

            metrics.update({
                'policy_loss': policy_loss.item(),
                'alpha': self.alpha,
                'lambda': self.lambda_param,
                'cost_q_mean': cost_q_new.mean().item(),
                'q_mean': q_new.mean().item(),
            })

        # ---- Soft target update ----
        for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)
        for tp, p in zip(self.target_cost_q_net.parameters(), self.cost_q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)

        return metrics

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'cost_q_net': self.cost_q_net.state_dict(),
            'log_lambda': self.log_lambda.data,
            'log_alpha': self.log_alpha.data if self.auto_entropy else None,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['q_net'])
        self.cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.target_cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.log_lambda.data = ckpt['log_lambda']
        if ckpt.get('log_alpha') is not None:
            self.log_alpha.data = ckpt['log_alpha']
