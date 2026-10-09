"""
World Model Ensemble: M_sim(s,a) → (s', r).

MBPO-style probabilistic ensemble world model.
Each model outputs (mean, logvar) for Gaussian prediction.
Ensemble disagreement = epistemic uncertainty.

Two-stage usage:
  1. Pretrain M_sim on sim transitions (large data, converge)
  2. Train residual δ on small paired data: M_real = M_sim + δ
  3. Generate imagined rollouts from M_real for policy training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ProbabilisticMLP(nn.Module):
    """Single model: (s,a) → (mean_s', logvar_s', mean_r, logvar_r)."""

    def __init__(self, obs_dim, act_dim, hidden=200):
        super().__init__()
        self.obs_dim = obs_dim
        in_dim = obs_dim + act_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # Separate heads for state and reward
        self.head_s = nn.Linear(hidden, obs_dim * 2)  # mean + logvar
        self.head_r = nn.Linear(hidden, 2)             # mean + logvar

        # Bound logvar to prevent numerical issues
        self.max_logvar = nn.Parameter(torch.ones(obs_dim + 1) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(obs_dim + 1) * -10.0)

    def forward(self, s, a):
        """Returns (mean, logvar) for state and reward."""
        sa = torch.cat([s, a], dim=-1)
        h = self.trunk(sa)

        s_out = self.head_s(h)
        r_out = self.head_r(h)

        s_mean, s_logvar = s_out[:, :self.obs_dim], s_out[:, self.obs_dim:]
        r_mean, r_logvar = r_out[:, :1], r_out[:, 1:]

        mean = torch.cat([s_mean, r_mean], dim=-1)
        logvar = torch.cat([s_logvar, r_logvar], dim=-1)

        # Soft clamp logvar
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)

        return mean, logvar


class WorldModelEnsemble:
    """
    Ensemble of K probabilistic MLPs.

    Predicts next state as DELTA (s' = s + Δs), following MBPO convention.
    """

    def __init__(self, obs_dim, act_dim, K=5, hidden=200,
                 lr=1e-3, weight_decay=1e-4, device="cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K = K
        self.device = device

        self.models = [ProbabilisticMLP(obs_dim, act_dim, hidden).to(device)
                       for _ in range(K)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
                           for m in self.models]
        self._trained = False
        self._input_mean = None
        self._input_std = None
        self._target_mean = None
        self._target_std = None
        self.validation_indices = None
        self.validation_rmse = None

    def fit(self, states, actions, next_states, rewards,
            n_epochs=100, batch_size=256, val_ratio=0.1, patience=20,
            train_indices=None, validation_indices=None):
        """
        Train ensemble on transition data.
        Targets are DELTAS: Δs = s' - s.
        """
        N = len(states)
        if N < 2:
            raise ValueError("world-model fit requires at least two transitions")
        deltas = next_states - states
        targets = np.concatenate([deltas, rewards.reshape(-1, 1)], axis=-1)  # (N, obs+1)
        inputs = np.concatenate([states, actions], axis=-1)

        # An explicit split lets model-family selection reserve a calibration
        # partition that neither candidate sees during fitting or early stopping.
        if (train_indices is None) != (validation_indices is None):
            raise ValueError(
                "train_indices and validation_indices must be provided together")
        if train_indices is None:
            perm = np.random.permutation(N)
            min_val = min(100, max(1, N // 5))
            n_val = min(max(int(N * val_ratio), min_val), N - 1)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            train_idx = np.asarray(train_indices, dtype=np.int64)
            val_idx = np.asarray(validation_indices, dtype=np.int64)
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError("explicit train and validation splits must be nonempty")
            if (train_idx.min() < 0 or train_idx.max() >= N
                    or val_idx.min() < 0 or val_idx.max() >= N):
                raise ValueError("explicit split index is out of range")
            if np.intersect1d(train_idx, val_idx).size:
                raise ValueError("explicit train and validation splits must be disjoint")
        self.validation_indices = val_idx.copy()

        # Fit normalizers once and freeze them across online refits so warm-started
        # network weights keep the same coordinate system.
        if self._input_mean is None:
            self._input_mean = inputs[train_idx].mean(axis=0, keepdims=True)
            self._input_std = inputs[train_idx].std(axis=0, keepdims=True)
            self._input_std = np.where(self._input_std > 1e-6, self._input_std, 1.0)
            self._target_mean = targets[train_idx].mean(axis=0, keepdims=True)
            self._target_std = targets[train_idx].std(axis=0, keepdims=True)
            self._target_std = np.where(self._target_std > 1e-6, self._target_std, 1.0)
        inputs_norm = (inputs - self._input_mean) / self._input_std
        targets_norm = (targets - self._target_mean) / self._target_std

        inputs_t = torch.as_tensor(inputs_norm, dtype=torch.float32, device=self.device)
        s_t = inputs_t[:, :self.obs_dim]
        a_t = inputs_t[:, self.obs_dim:]
        tgt_t = torch.as_tensor(targets_norm, dtype=torch.float32, device=self.device)

        best_val_loss = [float('inf')] * self.K
        best_epoch = [0] * self.K
        best_state = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in self.models]

        for epoch in range(n_epochs):
            # Train
            for k, (model, opt) in enumerate(zip(self.models, self.optimizers)):
                model.train()
                boot_idx = np.random.choice(train_idx, len(train_idx), replace=True)
                np.random.shuffle(boot_idx)
                for i in range(0, len(boot_idx), batch_size):
                    bi = boot_idx[i:i+batch_size]
                    mean, logvar = model(s_t[bi], a_t[bi])
                    # Gaussian NLL loss
                    inv_var = torch.exp(-logvar)
                    mse = (mean - tgt_t[bi]) ** 2
                    loss = (mse * inv_var + logvar).mean()
                    # Logvar bounds regularization
                    loss += 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
                    opt.zero_grad(); loss.backward(); opt.step()
                model.eval()

            # Validate
            with torch.no_grad():
                for k, model in enumerate(self.models):
                    mean, logvar = model(s_t[val_idx], a_t[val_idx])
                    inv_var = torch.exp(-logvar)
                    mse = (mean - tgt_t[val_idx]) ** 2
                    val_loss = float((mse * inv_var + logvar).mean())
                    if val_loss < best_val_loss[k]:
                        best_val_loss[k] = val_loss
                        best_epoch[k] = epoch
                        best_state[k] = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                avg_val = np.mean(best_val_loss)
                print(f"    epoch {epoch+1:3d} | avg best_val={avg_val:.4f}")

            # Early stop if ALL models plateaued
            if all(epoch - be >= patience for be in best_epoch):
                print(f"    Early stop at epoch {epoch+1}")
                break

        # Restore best
        for m, sd in zip(self.models, best_state):
            m.load_state_dict(sd)
        self._trained = True
        with torch.no_grad():
            val_means = torch.stack(
                [model(s_t[val_idx], a_t[val_idx])[0] for model in self.models]
            ).mean(dim=0).cpu().numpy()
        val_physical = val_means * self._target_std + self._target_mean
        self.validation_rmse = float(np.sqrt(
            np.mean((val_physical - targets[val_idx]) ** 2)))
        print(f"  World model trained. Best val losses: "
              f"[{', '.join(f'{v:.4f}' for v in best_val_loss)}]")

    def predict(self, states, actions, deterministic=False):
        """
        Predict (next_state, reward) from ensemble.

        If deterministic: use ensemble mean.
        If stochastic: sample from random model, add Gaussian noise.
        Returns: next_states (N, obs), rewards (N,)
        """
        if not self._trained:
            raise RuntimeError("world model must be fit before prediction")
        inputs = np.concatenate([states, actions], axis=-1)
        inputs = (inputs - self._input_mean) / self._input_std
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        s_t = inputs_t[:, :self.obs_dim]
        a_t = inputs_t[:, self.obs_dim:]

        with torch.no_grad():
            all_means = []
            all_logvars = []
            for model in self.models:
                mean, logvar = model(s_t, a_t)
                all_means.append(mean)
                all_logvars.append(logvar)

            means = torch.stack(all_means)      # (K, N, obs+1)
            logvars = torch.stack(all_logvars)   # (K, N, obs+1)

            if deterministic:
                pred = means.mean(0)  # ensemble mean
            else:
                # Sample random model per datapoint
                N = len(states)
                k_idx = torch.randint(self.K, (N,))
                pred_mean = means[k_idx, torch.arange(N)]
                pred_logvar = logvars[k_idx, torch.arange(N)]
                std = torch.exp(0.5 * pred_logvar)
                pred = pred_mean + std * torch.randn_like(std)

        pred = pred.cpu().numpy() * self._target_std + self._target_mean
        delta_s = pred[:, :self.obs_dim]
        reward = pred[:, self.obs_dim]
        next_states = states + delta_s

        return next_states, reward

    def get_disagreement(self, states, actions):
        """Ensemble disagreement (epistemic uncertainty). Returns (N,)."""
        inputs = np.concatenate([states, actions], axis=-1)
        inputs = (inputs - self._input_mean) / self._input_std
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        s_t = inputs_t[:, :self.obs_dim]
        a_t = inputs_t[:, self.obs_dim:]
        with torch.no_grad():
            means = torch.stack([m(s_t, a_t)[0] for m in self.models])
            target_std = torch.as_tensor(
                self._target_std, dtype=means.dtype, device=means.device)
            means = means * target_std
            return means.std(0, unbiased=False).mean(dim=-1).cpu().numpy()

    def per_dim_std(self, s_t, a_t):
        """Ensemble-disagreement signature for QΔ belief-conditioning (P1).

        Args:
            s_t: (B, obs_dim) torch tensor on this ensemble's device.
            a_t: (B, act_dim) torch tensor on this ensemble's device.
        Returns:
            (B, obs_dim + 1) tensor of per-dim ensemble std (state dims +
            reward dim).  No normalization — the QΔ critic learns its own
            scaling.  Detached, no_grad.
        """
        with torch.no_grad():
            inputs = torch.cat([s_t, a_t], dim=-1)
            input_mean = torch.as_tensor(
                self._input_mean, dtype=inputs.dtype, device=inputs.device)
            input_std = torch.as_tensor(
                self._input_std, dtype=inputs.dtype, device=inputs.device)
            inputs = (inputs - input_mean) / input_std
            s_norm = inputs[:, :self.obs_dim]
            a_norm = inputs[:, self.obs_dim:]
            means = torch.stack(
                [m(s_norm, a_norm)[0] for m in self.models])  # (K, B, D+1)
            target_std = torch.as_tensor(
                self._target_std, dtype=means.dtype, device=means.device)
            means = means * target_std
            return means.std(dim=0, unbiased=False).detach()

    def freeze(self):
        """Freeze all parameters (for residual adaptation phase)."""
        for m in self.models:
            for p in m.parameters():
                p.requires_grad_(False)


class ResidualAdapter:
    """
    Small residual network δ: adapts frozen M_sim to M_real.

    M_real(s,a) = M_sim(s,a) + δ(s,a)

    δ is trained on paired (sim, real) data where we know the true residual.
    Inherits ensemble disagreement from M_sim for confidence.
    """

    def __init__(self, obs_dim, act_dim, hidden=64, lr=1e-3,
                 weight_decay=1e-4, device="cpu"):
        self.obs_dim = obs_dim
        self.device = device
        # Small network — residual should be simpler than full dynamics
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim + 1),  # Δ(Δs) + Δr correction
        ).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self._trained = False
        self._input_mean = None
        self._input_std = None
        self._target_mean = None
        self._target_std = None
        self.validation_indices = None
        self.validation_rmse = None

    def fit(self, states, actions, sim_next_states, sim_rewards,
            real_next_states, real_rewards, n_epochs=100, batch_size=256, patience=20,
            real_dones=None):
        """
        Train residual on paired data.

        Target: (real_next - sim_prediction) for state, (real_r - sim_r) for reward.
        This is the CORRECTION that M_sim needs to match M_real.
        """
        # Compute what M_sim would predict (delta format)
        sim_deltas = sim_next_states - states
        real_deltas = real_next_states - states

        # Residual target: what M_sim got wrong
        delta_correction_s = real_deltas - sim_deltas  # (N, obs)
        delta_correction_r = (real_rewards - sim_rewards).reshape(-1, 1)  # (N, 1)
        targets = np.concatenate([delta_correction_s, delta_correction_r], axis=-1)
        inputs = np.concatenate([states, actions], axis=-1)

        N = len(states)
        if N < 2:
            raise ValueError("residual fit requires at least two transitions")
        perm = np.random.permutation(N)
        min_val = min(50, max(1, N // 5))
        n_val = min(max(int(N * 0.1), min_val), N - 1)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self.validation_indices = val_idx.copy()

        if self._input_mean is None:
            self._input_mean = inputs[train_idx].mean(axis=0, keepdims=True)
            self._input_std = inputs[train_idx].std(axis=0, keepdims=True)
            self._input_std = np.where(self._input_std > 1e-6, self._input_std, 1.0)
            self._target_mean = targets[train_idx].mean(axis=0, keepdims=True)
            self._target_std = targets[train_idx].std(axis=0, keepdims=True)
            self._target_std = np.where(self._target_std > 1e-6, self._target_std, 1.0)
        inputs_norm = (inputs - self._input_mean) / self._input_std
        targets_norm = (targets - self._target_mean) / self._target_std

        inputs_t = torch.as_tensor(inputs_norm, dtype=torch.float32, device=self.device)
        s_t = inputs_t[:, :self.obs_dim]
        a_t = inputs_t[:, self.obs_dim:]
        tgt_t = torch.as_tensor(targets_norm, dtype=torch.float32, device=self.device)

        best_val = float('inf'); best_epoch = 0
        best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

        for epoch in range(n_epochs):
            self.net.train()
            perm_train = np.random.permutation(train_idx)
            for i in range(0, len(perm_train), batch_size):
                bi = perm_train[i:i+batch_size]
                sa = torch.cat([s_t[bi], a_t[bi]], dim=-1)
                pred = self.net(sa)
                loss = nn.MSELoss()(pred, tgt_t[bi])
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            self.net.eval()

            with torch.no_grad():
                sa_v = torch.cat([s_t[val_idx], a_t[val_idx]], dim=-1)
                val_loss = float(nn.MSELoss()(self.net(sa_v), tgt_t[val_idx]))
                if val_loss < best_val:
                    best_val = val_loss; best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                print(f"    residual epoch {epoch+1} | val={val_loss:.5f} best={best_val:.5f}")
            if epoch - best_epoch >= patience:
                print(f"    Residual early stop at epoch {epoch+1} (best={best_epoch+1})")
                break

        self.net.load_state_dict(best_state)
        self._trained = True
        with torch.no_grad():
            val_pred = self.net(torch.cat([s_t[val_idx], a_t[val_idx]], dim=-1))
            val_pred = val_pred.cpu().numpy() * self._target_std + self._target_mean
        self.validation_rmse = float(np.sqrt(
            np.mean((val_pred - targets[val_idx]) ** 2)))
        print(f"  Residual trained. Best val loss: {best_val:.5f}")

    def predict_correction(self, states, actions):
        """Returns (Δs_correction, Δr_correction)."""
        inputs = np.concatenate([states, actions], axis=-1)
        inputs = (inputs - self._input_mean) / self._input_std
        sa = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.net(sa).cpu().numpy()
        out = out * self._target_std + self._target_mean
        return out[:, :self.obs_dim], out[:, self.obs_dim]


class CorrectedWorldModel:
    """
    M_real = M_sim + δ.

    Generates imagined rollouts from the corrected model.
    """

    def __init__(self, world_model, residual, beta=1.0,
                 beta_delta=None, beta_qdelta=None,
                 ensemble_gate=None,
                 apply_residual_state_reward=True):
        self.wm = world_model
        self.residual = residual
        # Decoupled β: beta_delta scales residual correction in predict();
        # beta_qdelta (used by callers) scales QΔ filter aggressiveness.
        # Setting `corrected.beta = X` still sets both (backward compat).
        self._beta_delta = float(beta if beta_delta is None else beta_delta)
        self._beta_qdelta = float(beta if beta_qdelta is None else beta_qdelta)
        self.apply_residual_state_reward = bool(apply_residual_state_reward)
        # SINDy ensemble: per-sample OOD gate that multiplies δ.
        # High ensemble agreement → gate≈1 (trust correction); disagreement
        # → gate→0 (the residual is extrapolating; rollout with raw sim).
        self.ensemble_gate = ensemble_gate

    @property
    def beta(self):
        return self._beta_delta  # legacy: exposes residual-scale β

    @beta.setter
    def beta(self, value):
        v = float(value)
        self._beta_delta = v
        self._beta_qdelta = v

    @property
    def beta_delta(self):
        return self._beta_delta

    @beta_delta.setter
    def beta_delta(self, value):
        self._beta_delta = float(value)

    @property
    def beta_qdelta(self):
        return self._beta_qdelta

    @beta_qdelta.setter
    def beta_qdelta(self, value):
        self._beta_qdelta = float(value)

    def predict(self, states, actions, deterministic=False):
        """
        Predict with residual correction.

        Returns (next_states, rewards) for backward compatibility.
        Use `predict_full_tuple` for the full (next_states, rewards, dones).
        """
        next_s_sim, r_sim = self.wm.predict(states, actions, deterministic)
        if self.residual._trained and self.apply_residual_state_reward:
            ds_corr, dr_corr = self.residual.predict_correction(states, actions)
            # Per-sample ensemble gate (broadcasted across obs dims and reward)
            if self.ensemble_gate is not None:
                import numpy as _np
                _sa = _np.concatenate(
                    [_np.asarray(states), _np.asarray(actions)], axis=-1)
                _, _, gates = self.ensemble_gate.predict_batch_with_uncertainty(_sa)
                # gates: shape (N,); broadcast to (N,1) for ds_corr
                ds_corr = ds_corr * gates[:, None]
                dr_corr = dr_corr * gates
            next_s_real = next_s_sim + self._beta_delta * ds_corr
            r_real = r_sim + self._beta_delta * dr_corr
            return next_s_real, r_real
        return next_s_sim, r_sim

    def predict_full_tuple(self, states, actions, deterministic=False):
        """
        Full-tuple prediction: (next_states, rewards, done_probs).

        done_probs ∈ [0, 1]^N — P(done | s, a, s').  Callers may
        threshold at 0.5 or sample Bernoulli.  Returns 0 if residual
        doesn't have a trained done head.
        """
        import numpy as np
        ns, r = self.predict(states, actions, deterministic)
        if (self.residual._trained and
                hasattr(self.residual, "predict_done")):
            d = self.residual.predict_done(states, actions, ns)
        else:
            d = np.zeros(len(states))
        return ns, r, d

    def imagine_rollout(self, start_states, policy_fn, horizon=5, deterministic=False):
        """
        Generate imagined trajectories from corrected world model.

        Returns dict with states, actions, rewards, dones (all numpy).
        """
        B = len(start_states)
        states = [start_states.copy()]
        actions_list = []
        rewards_list = []

        s = start_states.copy()
        for t in range(horizon):
            a = policy_fn(s)
            s_next, r = self.predict(s, a, deterministic)
            states.append(s_next)
            actions_list.append(a)
            rewards_list.append(r)
            s = s_next

        return {
            "states": np.array(states[:-1]),    # (H, B, obs)
            "actions": np.array(actions_list),   # (H, B, act)
            "next_states": np.array(states[1:]), # (H, B, obs)
            "rewards": np.array(rewards_list),   # (H, B)
        }
