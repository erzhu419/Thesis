"""
upper/resac_upper.py
====================
RE-SAC for the adaptive target-headway upper policy.

Same ensemble Q + epistemic penalty as the lower RE-SAC, but:
  - No Lagrangian cost constraint (fleet constraint via θ-OGD instead).
  - Action dim is 1 in the main HIRO configuration (per-dispatch
    target-headway shift δ_t ∈ [-120, +120] s); the constructor still accepts
    arbitrary action_dim so the legacy 3-action [H_peak, H_off, H_trans]
    headway-triple variant remains usable for ablations.
  - Bounded action via sigmoid + affine rescale to [action_low, action_high]
    (matches paper Section IV-B; not tanh).
  - Own replay buffer for dispatch-level transitions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from collections import deque
import random


# ──────────────────── Upper Replay Buffer ────────────────────

class UpperReplayBuffer:
    """Simple replay buffer for (s, a, r, s') dispatch transitions."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a),
                np.array(r).reshape(-1, 1),
                np.array(ns), np.array(d).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


# ──────────────────── Networks ────────────────────

class BoundedGaussianPolicy(nn.Module):
    """Gaussian policy with action mapped to [action_low, action_high] via sigmoid."""

    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 action_low=None, action_high=None):
        super().__init__()
        self.action_dim = action_dim

        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.register_buffer('action_low',
                             torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high',
                             torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_range',
                             self.action_high - self.action_low)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Init near sigmoid(0) = 0.5 → mid-range
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -5.0, 0.0)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """For training: returns action, log_prob."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        # Sigmoid squashing → [0, 1] → scale to [low, high]
        u = torch.sigmoid(z)
        action = self.action_low + u * self.action_range

        # Log-prob with sigmoid correction: log_prob = log_N(z) - log(sigmoid'(z))
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = u * (1 - u)
        log_prob = dist.log_prob(z) - torch.log(u * (1.0 - u) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                u = torch.sigmoid(mean)
            else:
                std = log_std.exp()
                z = Normal(mean, std).sample()
                u = torch.sigmoid(z)
            action = self.action_low + u * self.action_range
        return action.squeeze(0).cpu().numpy()

    def log_prob(self, state, action, epsilon=1e-6):
        """Compute log-prob of a given action under the current policy.
        Used by TPC-Lower for importance-weighted lower training."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        action = action.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            std = log_std.exp()
            # Invert sigmoid: u = (action - low) / range, z = log(u/(1-u))
            u = ((action - self.action_low) / self.action_range).clamp(epsilon, 1.0 - epsilon)
            z = torch.log(u / (1.0 - u))
            dist = Normal(mean, std)
            lp = dist.log_prob(z) - torch.log(u * (1.0 - u) + epsilon)
            lp = lp.sum(-1)
        return lp.cpu().numpy().squeeze()


class CategoricalPolicy(nn.Module):
    """Categorical policy over a fixed scalar action grid."""

    is_discrete = True

    def __init__(self, state_dim, action_bins, hidden_dim=64):
        super().__init__()
        if not action_bins:
            raise ValueError("action_bins must contain at least one value")
        self.register_buffer(
            'action_bins',
            torch.tensor(action_bins, dtype=torch.float32).view(1, -1))
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, len(action_bins))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def _dist(self, state):
        return Categorical(logits=self.forward(state))

    def evaluate(self, state, epsilon=1e-8):
        dist = self._dist(state)
        idx = dist.sample()
        action = self.action_bins.squeeze(0)[idx].unsqueeze(-1)
        log_prob = dist.log_prob(idx).unsqueeze(-1)
        probs = torch.softmax(dist.logits, dim=-1).clamp_min(epsilon)
        return action, log_prob, idx, dist.logits, probs

    def evaluate_all(self, state, epsilon=1e-8):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1).clamp_min(epsilon)
        log_probs = torch.log(probs)
        actions = self.action_bins.expand(state.shape[0], -1)
        return actions, probs, log_probs

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        with torch.no_grad():
            logits = self.forward(state)
            if deterministic:
                idx = torch.argmax(logits, dim=-1)
            else:
                idx = Categorical(logits=logits).sample()
            action = self.action_bins.squeeze(0)[idx]
        return action.view(-1).cpu().numpy()

    def log_prob(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.view(1, 1)
        elif action.dim() == 1:
            action = action.view(-1, 1)
        state = state.to(next(self.parameters()).device)
        action = action.to(next(self.parameters()).device)
        with torch.no_grad():
            logits = self.forward(state)
            bins = self.action_bins.to(action.device)
            idx = torch.argmin((action - bins).abs(), dim=-1)
            lp = torch.log_softmax(logits, dim=-1).gather(
                1, idx.view(-1, 1)).squeeze(-1)
        return lp.cpu().numpy().squeeze()


class EnsembleQNetwork(nn.Module):
    """Ensemble of K Q-networks (same as lower)."""

    def __init__(self, num_inputs, num_actions, hidden_dim=64,
                 ensemble_size=10, n_layers=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_layers = n_layers

        dims = [num_inputs + num_actions] + [hidden_dim] * n_layers + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            stddev = 1.0 / np.sqrt(dims[i])
            self.weights.append(nn.Parameter(
                torch.randn(ensemble_size, dims[i], dims[i+1]) * stddev))
            self.biases.append(nn.Parameter(
                torch.zeros(ensemble_size, 1, dims[i+1])))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, w) + b
            if i < self.n_layers:
                x = F.relu(x)
        return x.squeeze(-1)  # [K, B]

    def compute_l1_norm(self):
        total = torch.zeros(self.ensemble_size, device=self.weights[0].device)
        for w, b in zip(self.weights, self.biases):
            total = total + w.abs().sum(dim=(1, 2)) + b.abs().sum(dim=(1, 2))
        return total


# ──────────────────── Trainer ────────────────────

class RESACUpperTrainer:
    """
    RE-SAC for upper-level timetable policy (no Lagrangian).
    Fleet constraint handled externally by θ-OGD reward modulation.
    """

    def __init__(self, state_dim=5, action_dim=3, hidden_dim=64,
                 action_low=None, action_high=None,
                 action_bins=None,
                 ensemble_size=10, beta=-2.0, beta_ood=0.01,
                 weight_reg=0.01,
                 lr=3e-4, gamma=0.99, soft_tau=5e-3,
                 auto_entropy=True, maximum_alpha=0.3,
                 replay_capacity=50000, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.ensemble_size = ensemble_size
        self.beta = beta
        self.beta_ood = beta_ood
        self.weight_reg = weight_reg
        self.auto_entropy = auto_entropy

        # Replay buffer for dispatch transitions
        self.replay_buffer = UpperReplayBuffer(replay_capacity)

        self.action_bins = action_bins
        self.discrete = action_bins is not None

        # Policy
        if self.discrete:
            self.policy_net = CategoricalPolicy(
                state_dim, action_bins, hidden_dim).to(device)
            action_dim = 1
        else:
            self.policy_net = BoundedGaussianPolicy(
                state_dim, action_dim, hidden_dim,
                action_low, action_high).to(device)

        # Ensemble Q
        self.q_net = EnsembleQNetwork(
            state_dim, action_dim, hidden_dim, ensemble_size).to(device)
        self.target_q_net = EnsembleQNetwork(
            state_dim, action_dim, hidden_dim, ensemble_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Entropy
        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            if self.discrete:
                self.target_entropy = 0.98 * np.log(len(action_bins))
            else:
                self.target_entropy = -float(action_dim)
        self.alpha = 0.1
        self.maximum_alpha = maximum_alpha

    def _q_values_for_bins(self, q_net, state):
        actions, probs, log_probs = self.policy_net.evaluate_all(state)
        batch_size, n_actions = actions.shape
        s_rep = state.unsqueeze(1).expand(-1, n_actions, -1).reshape(
            batch_size * n_actions, -1)
        a_rep = actions.reshape(batch_size * n_actions, 1)
        q = q_net(s_rep, a_rep).reshape(
            self.ensemble_size, batch_size, n_actions)
        return q, probs, log_probs, actions

    def update(self, batch_size=64):
        """One gradient step from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return {}

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # ── Critic update ──
        with torch.no_grad():
            if self.discrete:
                target_q_all, next_probs, next_log_probs, _ = \
                    self._q_values_for_bins(self.target_q_net, next_state)
                target_q_mean = target_q_all.mean(dim=0)  # [B, A]
                target_v = (next_probs * (
                    target_q_mean - self.alpha * next_log_probs)).sum(dim=-1)
            else:
                next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
                target_q_all = self.target_q_net(next_state, next_action)  # [K, B]
                # Shared mean target → prevents ensemble divergence
                target_v = target_q_all.mean(dim=0) - self.alpha * next_log_prob.squeeze(-1)
            r = reward.squeeze(-1)
            d = done.squeeze(-1)
            shared_target = r + (1.0 - d) * self.gamma * target_v
            shared_target = shared_target.clamp(-50.0, 50.0)

        predicted_q = self.q_net(state, action)
        target_value = shared_target.unsqueeze(0).expand(
            predicted_q.shape[0], -1)  # [K, B]
        q_mse = F.mse_loss(predicted_q, target_value)
        ood_loss = predicted_q.std(dim=0).mean()
        l1_norm = self.q_net.compute_l1_norm().mean()
        q_loss = q_mse + self.beta_ood * ood_loss + self.weight_reg * l1_norm

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 50.0)
        self.q_optimizer.step()

        # ── Policy update ──
        if self.discrete:
            q_all_bins, probs, log_probs, _ = self._q_values_for_bins(
                self.q_net, state)
            q_mean = q_all_bins.mean(dim=0)
            q_std = q_all_bins.std(dim=0)
            q_lcb = q_mean + self.beta * q_std
            policy_loss = (probs * (
                self.alpha * log_probs - q_lcb)).sum(dim=-1).mean()
            entropy_signal = (log_probs * probs).sum(
                dim=-1, keepdim=True).detach()
            q_mean_metric = q_mean.mean()
            q_std_metric = q_std.mean()
        else:
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)
            q_all = self.q_net(state, new_action)
            q_mean = q_all.mean(dim=0)
            q_std = q_all.std(dim=0)
            q_lcb = q_mean + self.beta * q_std

            policy_loss = (self.alpha * log_prob.squeeze(-1) - q_lcb).mean()
            entropy_signal = log_prob.detach()
            q_mean_metric = q_mean.mean()
            q_std_metric = q_std.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.policy_optimizer.step()

        # ── Alpha update ──
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha *
                           (entropy_signal + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = min(self.log_alpha.exp().item(), self.maximum_alpha)

        # ── Soft target update ──
        for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)

        return {
            'upper_q_mean': q_mean_metric.item(),
            'upper_q_std': q_std_metric.item(),
            'upper_policy_loss': policy_loss.item(),
            'upper_q_loss': q_loss.item(),
            'upper_q_mse': q_mse.item(),
            'upper_ood_loss': ood_loss.item(),
            'upper_alpha': self.alpha,
            'upper_pi_grad_norm': pi_grad_norm.item() if isinstance(pi_grad_norm, torch.Tensor) else float(pi_grad_norm),
            'upper_q_grad_norm': q_grad_norm.item() if isinstance(q_grad_norm, torch.Tensor) else float(q_grad_norm),
            'upper_reward_batch_mean': reward.mean().item(),
            'upper_reward_batch_std': reward.std().item(),
            'upper_action_batch_mean': action.mean().item(),
            'upper_action_batch_std': action.std().item(),
        }

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'log_alpha': self.log_alpha.data if self.auto_entropy else None,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['q_net'])
        if ckpt.get('log_alpha') is not None:
            self.log_alpha.data = ckpt['log_alpha']
