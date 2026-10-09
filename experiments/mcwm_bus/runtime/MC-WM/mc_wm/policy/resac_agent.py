"""
RE-SAC Agent (PyTorch, gymnasium-compatible)

提取自 LSTM-RL/RE-SAC/sac_ensemble_original_logging.py 的核心思想：
  - Ensemble critic (VectorizedCritic): N 个 Q 函数并行，一次前向
  - LCB policy loss:  L = -(Q_mean + β * Q_std).mean()
    β < 0 → 悲观（避免 Q 过估计）
    β > 0 → 乐观（探索）
  - OOD loss: + β_ood * Q_std（惩罚高方差的 Q 估计）
  - BC loss:  + β_bc * MSE(π(s), a_behavior)（行为克隆正则）

相比原版的简化：
  - 去掉 EmbeddingLayer（bus-specific）
  - 去掉 state_norm / reward_scaling（外部可选）
  - 去掉 argparse 全局变量，改为构造函数参数
  - 适配 gymnasium continuous action spaces

核心公式（来自原版 compute_policy_loss）：
    q_dist = ensemble_Q(s, π(s))           # shape (N_critics, B)
    q_mean = q_dist.mean(0)
    q_std  = q_dist.std(0)
    policy_loss = -(q_mean + β * q_std).mean() + β_bc * MSE(π(s), a_data)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from copy import deepcopy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Vectorized Critic（直接移植自原版）
# ─────────────────────────────────────────────

class VectorizedLinear(nn.Module):
    """N 个线性层并行，权重 shape (N, in, out)。"""
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias   = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        for i in range(ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):               # x: (N, B, in)
        return x @ self.weight + self.bias  # (N, B, out)


class EnsembleCritic(nn.Module):
    """
    Ensemble of N Q-networks, evaluated in one forward pass.
    forward(s, a) → Q values shape (N_critics, B)
    """
    def __init__(self, obs_dim, act_dim, hidden_dim, n_critics):
        super().__init__()
        self.n = n_critics
        self.net = nn.Sequential(
            VectorizedLinear(obs_dim + act_dim, hidden_dim, n_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, n_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, n_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, n_critics),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)                        # (B, obs+act)
        sa = sa.unsqueeze(0).repeat_interleave(self.n, dim=0) # (N, B, obs+act)
        return self.net(sa).squeeze(-1)                        # (N, B)


# ─────────────────────────────────────────────
# Gaussian Policy（标准 SAC actor）
# ─────────────────────────────────────────────

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, act_limit=1.0,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        nn.init.uniform_(self.mean_head.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)

    def forward(self, s):
        h = self.net(s)
        mean    = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, s, eps=1e-6):
        """Reparameterized sample + log_prob (TanhNormal)."""
        mean, log_std = self.forward(s)
        std = log_std.exp()
        z   = Normal(0, 1).sample(mean.shape).to(s.device)
        a0  = torch.tanh(mean + std * z)
        a   = a0 * self.act_limit
        log_prob = (Normal(mean, std).log_prob(mean + std * z)
                    - torch.log(1 - a0.pow(2) + eps)
                    ).sum(dim=-1)
        return a, log_prob, mean

    def get_action(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            s = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, log_std = self.forward(s)
            if deterministic:
                a = torch.tanh(mean) * self.act_limit
            else:
                std = log_std.exp()
                z = Normal(0, 1).sample(mean.shape).to(s.device)
                a = torch.tanh(mean + std * z) * self.act_limit
        return a.squeeze(0).cpu().numpy()

    def get_actions_batch(self, obs_batch: np.ndarray, deterministic=False) -> np.ndarray:
        """Batched version of get_action — avoids Python list comprehension overhead."""
        with torch.no_grad():
            device = next(self.parameters()).device
            s = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            mean, log_std = self.forward(s)
            if deterministic:
                a = torch.tanh(mean) * self.act_limit
            else:
                std = log_std.exp()
                z = Normal(0, 1).sample(mean.shape).to(s.device)
                a = torch.tanh(mean + std * z) * self.act_limit
        return a.cpu().numpy()  # (B, act_dim)


# ─────────────────────────────────────────────
# RE-SAC Agent
# ─────────────────────────────────────────────

class RESACAgent:
    """
    RE-SAC: Ensemble-critic SAC with LCB policy loss.

    Key hyperparameters:
        n_critics  : ensemble size (原版默认 10，MuJoCo 用 5 足够)
        beta       : LCB coefficient (< 0 → pessimistic，原版默认 -2)
        beta_ood   : OOD penalty on Q_std (原版默认 0.01)
        beta_bc    : behavior cloning weight (原版默认 0.001)
        critic_actor_ratio : critic 每 N 步 actor 更新一次 (原版默认 2)

    QΔ integration (optional):
        gap_fn     : callable(s_np, a_np) → gap_reward (np.ndarray, shape (N,))
                     SINDy-based dynamics gap detector. If provided, enables
                     QΔ (Residual Bellman) penalty on Q-targets.
        penalty_scale : weight of QΔ penalty on Q-target (default 0.1)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float = 1.0,
        hidden_dim: int = 256,
        n_critics: int = 5,
        beta: float = -2.0,
        beta_ood: float = 0.01,
        beta_bc: float = 0.001,
        critic_actor_ratio: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha_init: float = 0.2,
        max_alpha: float = 0.6,
        # ── BAPR v10 stability fix (mirrored from CS-BAPR P0+P1) ─────────
        # First N updates run as pure RE-SAC (no gap/QΔ trust weighting).
        # The QΔ critic and ensemble Q-std are unreliable until the critic has
        # seen enough transitions to actually disagree. Forcing trust=1 during
        # warmup avoids over-conservatism that locks bad early basins.
        bapr_warmup_iters: int = 100,
        # Alpha floor: prevent entropy collapse during early training.
        # CS-BAPR uses 0.01; RESAC's auto-alpha can otherwise go to ~0.
        min_alpha: float = 0.01,
        # Reward EMA normalization: divide reward by running std (no mean
        # shift — preserves sign). Stabilizes Q magnitudes when the env
        # reward has variable scale across episodes.
        use_reward_norm: bool = True,
        reward_ema_alpha: float = 0.01,
        # ─────────────────────────────────────────────────────────────────
        device: str = DEVICE,
        gap_fn=None,
        penalty_scale: float = 0.1,
    ):
        self.gamma = gamma
        self.tau   = tau
        self.beta  = beta
        self.beta_ood = beta_ood
        self.beta_bc  = beta_bc
        self.critic_actor_ratio = critic_actor_ratio
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.bapr_warmup_iters = bapr_warmup_iters
        self.use_reward_norm = use_reward_norm
        self.reward_ema_alpha = reward_ema_alpha
        self._reward_ema_var = 1.0
        self.device = device
        self._update_count = 0
        self._critic_updates_since_policy = 0
        self._policy_eligible_update_count = 0
        self._temperature_update_count = 0
        self._actor_update_count = 0
        self._reward_stats_update_count = 0
        self._weighted_actor_update_count = 0
        self._actor_weight_mean_sum = 0.0
        self._actor_synthetic_fraction_sum = 0.0
        self._violation_balanced_critic_update_count = 0

        self.actor = GaussianActor(obs_dim, act_dim, hidden_dim, act_limit).to(device)
        self.critic     = EnsembleCritic(obs_dim, act_dim, hidden_dim, n_critics).to(device)
        self.critic_tgt = deepcopy(self.critic)
        for p in self.critic_tgt.parameters():
            p.requires_grad_(False)

        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32, device=device))
        self.target_entropy = -act_dim

        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_alpha  = optim.Adam([self.log_alpha], lr=lr)

        # Dynamics gap penalty (optional)
        # gap_fn: callable(s_np, a_np) → gap per transition (N,)
        # Three modes:
        #   1. gap_fn=None → no penalty (baseline)
        #   2. gap_fn=callable → direct per-step penalty (method 1, recommended)
        #   3. gap_fn=QDeltaModule → pre-trained QΔ (method 2, Bellman accumulation)
        self.gap_fn = None
        self.q_delta = None
        self.use_direct_gap = False
        self.penalty_scale = penalty_scale
        if hasattr(gap_fn, 'get_penalty'):
            # Pre-trained QΔ module
            self.q_delta = gap_fn
        elif callable(gap_fn):
            # Direct gap function — per-step penalty, no Bellman
            self.gap_fn = gap_fn
            self.use_direct_gap = True

    @property
    def alpha(self):
        # Both ceiling (max_alpha) and floor (min_alpha) — floor prevents
        # entropy collapse during early training when log_prob is tiny.
        return min(self.max_alpha, max(self.min_alpha, self.log_alpha.exp().item()))

    def update(self, buf, *, update_policy: bool = True,
               update_reward_stats: bool = True,
               update_actor: bool | None = None,
               update_temperature: bool | None = None,
               actor_batch=None):
        """Run one critic update with independently controlled policy state.

        ``update_policy=False`` keeps actor and entropy-temperature updates off
        for backward compatibility. ``update_actor`` and ``update_temperature``
        can override those two permissions independently. Reward normalization
        always uses the current scale, but its running statistic is updated only
        when ``update_reward_stats`` is true. ``actor_batch`` optionally supplies
        ``(state, behavior_action, weight, synthetic_mask)`` tensors for the
        actor objective, independently of the critic replay batch.
        """
        actor_update_enabled = (
            update_policy if update_actor is None else update_actor)
        temperature_update_enabled = (
            update_policy if update_temperature is None else update_temperature)
        self._update_count += 1
        self._critic_updates_since_policy += 1
        if actor_update_enabled:
            self._policy_eligible_update_count += 1
        # Sample (s, a, r, s2, d) and optional per-sample weight ``w_buf``.
        # New buffers (Bug 2 fix) return a 6-tuple where ``w_buf`` is the
        # trust weight stored at insertion time; legacy buffers return a
        # 5-tuple, in which case we default ``w_buf=1.0`` (no weighting).
        sample = buf.sample(256)
        if len(sample) == 6:
            s, a, r, s2, d, w_buf = sample
        else:
            s, a, r, s2, d = sample
            w_buf = torch.ones_like(r)
        if getattr(buf, "sqrt_balance_violations", False):
            critic_sample = buf.sample_critic(256)
            self._violation_balanced_critic_update_count += 1
            if len(critic_sample) == 6:
                cs, ca, cr, cs2, cd, cw_buf = critic_sample
            else:
                cs, ca, cr, cs2, cd = critic_sample
                cw_buf = torch.ones_like(cr)
        else:
            cs, ca, cr, cs2, cd, cw_buf = s, a, r, s2, d, w_buf

        # ── Reward EMA normalization (BAPR v10 piece): scale-only, no shift.
        # Divides reward by running std so Q magnitude stays well-behaved
        # under variable per-episode reward scale. Skipped on first batch
        # (var≈0) and gracefully bounded below by 1e-6.
        if self.use_reward_norm:
            if update_reward_stats:
                batch_var = float(r.var().item())
                self._reward_ema_var = (
                    (1.0 - self.reward_ema_alpha) * self._reward_ema_var
                    + self.reward_ema_alpha * batch_var
                )
                self._reward_stats_update_count += 1
            reward_scale = max(self._reward_ema_var ** 0.5, 1e-6)
            r = r / reward_scale
            cr = cr / reward_scale

        # ── Alpha update
        alpha_loss_val = 0.0
        if temperature_update_enabled:
            _, log_prob, _ = self.actor.evaluate(s)
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
            self._temperature_update_count += 1
            alpha_loss_val = float(alpha_loss.detach())

        # ── Critic update
        with torch.no_grad():
            a2, lp2, _ = self.actor.evaluate(cs2)
            q_next = self.critic_tgt(cs2, a2)      # (N, B)
            q_next_min = q_next.min(0).values       # (B,)
            q_tgt_raw = cr.squeeze(-1) + self.gamma * (1 - cd.squeeze(-1)) * (q_next_min - self.alpha * lp2)
            q_tgt = q_tgt_raw

        # Compute per-transition importance weight from gap signal.
        # Warmup gate (BAPR v10 P0): for the first bapr_warmup_iters updates,
        # force iw_weights=None so the critic trains as pure RE-SAC.  This
        # avoids early-training over-conservatism when gap/QΔ signals are
        # still uncalibrated.
        gap_raw = None
        iw_weights = None  # importance weights (B,)
        warmup_done = self._update_count > self.bapr_warmup_iters
        if warmup_done and self.use_direct_gap and self.gap_fn is not None:
            with torch.no_grad():
                s_np = cs.cpu().numpy(); a_np = ca.cpu().numpy()
                gap_np = self.gap_fn(s_np, a_np)  # (B,) — rank-preserving, [0,1] normalized
                gap_raw = gap_np.copy()
                gap_t = torch.FloatTensor(gap_np).to(self.device)
                # Multiplicative: w = w_min + (1 - w_min) * (1 - gap)
                # gap=0 (no gap) → w=1.0 (full weight)
                # gap=1 (max gap) → w=w_min (reduced weight)
                w_min = self.penalty_scale  # reuse penalty_scale as w_min
                iw_weights = w_min + (1.0 - w_min) * (1.0 - gap_t)  # (B,) in [w_min, 1.0]
        elif warmup_done and self.q_delta is not None:
            with torch.no_grad():
                qd_val = self.q_delta.q_delta(cs, ca).mean(0).squeeze(-1)  # (B,)
                # Normalize QΔ to [0, 1] via sigmoid
                gap_norm = torch.sigmoid(qd_val - qd_val.mean())
                w_min = self.penalty_scale
                iw_weights = w_min + (1.0 - w_min) * (1.0 - gap_norm)

        q_pred = self.critic(cs, ca)                # (N, B)
        q_tgt_exp = q_tgt.unsqueeze(0).expand_as(q_pred)
        ood_loss = q_pred.std(0).mean()

        # Combine the in-update gap weight with the per-sample weight stored
        # at buffer-insertion time (Bug 2 fix: model-rollout transitions carry
        # exp(QΔ(s,a)) as ``w_buf``; env transitions carry 1.0).  Both
        # multiply the TD-error², leaving the reward signal physical.
        w_buf_flat = cw_buf.squeeze(-1)  # (B,)
        if iw_weights is not None:
            sample_w = iw_weights * w_buf_flat
        else:
            sample_w = w_buf_flat

        # If sample_w is the constant-one tensor, fall back to plain MSE for
        # backward-compatibility (numerically identical, slightly cheaper).
        if torch.allclose(sample_w, torch.ones_like(sample_w)):
            critic_loss = F.mse_loss(q_pred, q_tgt_exp.detach()) + self.beta_ood * ood_loss
        else:
            sw_exp = sample_w.unsqueeze(0).expand_as(q_pred)  # (N, B)
            td_err = (q_pred - q_tgt_exp.detach()) ** 2       # (N, B)
            critic_loss = (sw_exp * td_err).mean() + self.beta_ood * ood_loss

        self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

        # ── Actor update（每 critic_actor_ratio 步一次）
        actor_loss_val = 0.0
        confidence_penalty_val = 0.0
        actor_updated = False
        if (actor_update_enabled and
                self._critic_updates_since_policy >= self.critic_actor_ratio):
            actor_weight = None
            actor_synthetic_mask = None
            s_actor = s
            a_behavior = a
            if actor_batch is not None:
                (s_actor, a_behavior, actor_weight,
                 actor_synthetic_mask) = actor_batch
                actor_weight = actor_weight.reshape(-1).detach()
                actor_synthetic_mask = (
                    actor_synthetic_mask.reshape(-1).detach())

            a_new, lp_new, _ = self.actor.evaluate(s_actor)
            q_dist = self.critic(s_actor, a_new)    # (N, B)
            q_mean = q_dist.mean(0)
            q_std  = q_dist.std(0)

            policy_per_sample = -(
                q_mean + self.beta * q_std - self.alpha * lp_new)
            bc_per_sample = (a_new - a_behavior).pow(2).mean(dim=-1)
            if actor_weight is None:
                policy_loss = policy_per_sample.mean()
                bc_loss = bc_per_sample.mean()
            else:
                actor_weight_sum = actor_weight.sum().clamp_min(1e-8)
                policy_loss = (
                    actor_weight * policy_per_sample).sum() / actor_weight_sum
                bc_loss = (
                    actor_weight * bc_per_sample).sum() / actor_weight_sum
            actor_loss = policy_loss + self.beta_bc * bc_loss

            # Confidence constraint on actor: penalize exploring low-confidence regions
            # gap_fn returns 1-confidence, so high gap = low confidence.
            # Same warmup gate as critic: skip during the BAPR v10 warmup window.
            if warmup_done and self.use_direct_gap and self.gap_fn is not None:
                with torch.no_grad():
                    s_np_actor = s_actor.cpu().numpy()
                    a_np_actor = a_new.detach().cpu().numpy()
                    gap_actor = self.gap_fn(s_np_actor, a_np_actor)
                # Penalize actor for proposing actions in low-confidence regions
                # This is a soft constraint: don't go where we can't trust the sim
                confidence_penalty = torch.FloatTensor(gap_actor).to(self.device).mean()
                actor_loss = actor_loss + self.penalty_scale * confidence_penalty
                confidence_penalty_val = float(confidence_penalty)

            actor_loss_val = float(actor_loss.detach())

            self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()
            self._actor_update_count += 1
            self._critic_updates_since_policy = 0
            actor_updated = True

            if actor_weight is not None:
                effective_synthetic_fraction = (
                    (actor_weight * actor_synthetic_mask).sum()
                    / actor_weight.sum().clamp_min(1e-8)).item()
                diag_actor_weight_mean = actor_weight.mean().item()
                self._weighted_actor_update_count += 1
                self._actor_weight_mean_sum += diag_actor_weight_mean
                self._actor_synthetic_fraction_sum += (
                    effective_synthetic_fraction)

            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)

        # ── Build diagnostics dict
        diag = {
            "critic_loss": float(critic_loss.detach()),
            "alpha_loss": alpha_loss_val,
            "actor_loss": actor_loss_val,
            "alpha": self.alpha,
            "q_pred_mean": float(q_pred.mean().detach()),
            "q_pred_std": float(q_pred.std(0).mean().detach()),
            "q_tgt_mean": float(q_tgt.mean().detach()),
            "q_tgt_raw_mean": float(q_tgt_raw.mean().detach()),
            # BAPR v10 diagnostics
            "warmup_done": int(warmup_done),
            "reward_ema_std": float(self._reward_ema_var ** 0.5),
            "policy_update_enabled": int(
                actor_update_enabled and temperature_update_enabled),
            "actor_update_enabled": int(actor_update_enabled),
            "temperature_update_enabled": int(temperature_update_enabled),
            "actor_updated": int(actor_updated),
            "reward_stats_updated": int(
                self.use_reward_norm and update_reward_stats),
        }
        if iw_weights is not None:
            diag["iw_mean"] = float(iw_weights.mean())
            diag["iw_min"] = float(iw_weights.min())
            diag["iw_std"] = float(iw_weights.std())
            diag["iw_reduction"] = float(1.0 - iw_weights.mean())
        if confidence_penalty_val > 0:
            diag["conf_penalty"] = confidence_penalty_val
        if actor_updated and actor_weight is not None:
            diag["actor_weight_mean"] = diag_actor_weight_mean
            diag["actor_effective_synthetic_fraction"] = (
                effective_synthetic_fraction)
        if gap_raw is not None:
            diag["gap_mean"] = float(gap_raw.mean())
            diag["gap_std"] = float(gap_raw.std())
            diag["gap_max"] = float(gap_raw.max())
            diag["gap_min"] = float(gap_raw.min())

        return diag

    def get_update_stats(self) -> dict[str, int | float]:
        """Return auditable counts for critic, policy, and normalization updates."""
        stats = {
            "critic_updates": self._update_count,
            "policy_eligible_updates": self._policy_eligible_update_count,
            "temperature_updates": self._temperature_update_count,
            "actor_updates": self._actor_update_count,
            "reward_stats_updates": self._reward_stats_update_count,
        }
        weighted_updates = self._weighted_actor_update_count
        if weighted_updates:
            stats["weighted_actor_updates"] = weighted_updates
            stats["actor_weight_mean"] = (
                self._actor_weight_mean_sum / weighted_updates)
            stats["actor_effective_synthetic_fraction"] = (
                self._actor_synthetic_fraction_sum / weighted_updates)
        if self._violation_balanced_critic_update_count:
            stats["violation_balanced_critic_updates"] = (
                self._violation_balanced_critic_update_count)
        return stats

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.actor.get_action(obs, deterministic)

    def get_actions_batch(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Batched action sampling — one GPU call instead of N individual calls."""
        return self.actor.get_actions_batch(obs_batch, deterministic)
