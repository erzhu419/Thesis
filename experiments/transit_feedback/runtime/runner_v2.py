"""
runner_v2.py
============
TransitDuet v2: bi-level bus control with genuine upper-lower coupling.

Key difference from v1:
  v1: upper outputs 3 static headway params → CMA-ES solves trivially
  v2: upper outputs per-trip departure ADJUSTMENT δ_t ∈ [-120, +120]s
      state includes lower-level holding statistics (the feedback signal)
      reward penalizes total lower-level intervention

Coupling mechanism:
  Lower holding ≫ 0 at all stops → trip departed too early
    → upper state shows high holding_mean → upper learns to shift later
    → holding drops → system converges to minimal-intervention timetable

  Ideal steady state: holding → 0 everywhere (timetable is perfect)

Usage:
    python -u runner_v2.py [--episodes 300] [--seed 42] [--gpu]
"""

import copy
import os
import sys
import argparse
import csv
import json
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from upper.resac_upper import RESACUpperTrainer
from upper.measurement_proj import MeasurementProjection
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer
from coupling.holding_feedback import HoldingFeedback
from coupling.belief_tracker import BeliefTracker, SurpriseComputer


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    """Load YAML config, supporting _extends: <parent_file>."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if '_extends' in cfg:
        parent_path = cfg.pop('_extends')
        base_dir = os.path.dirname(os.path.abspath(path))
        parent_full = os.path.join(base_dir, '..', parent_path) if not os.path.isabs(parent_path) else parent_path
        if not os.path.exists(parent_full):
            parent_full = os.path.join(base_dir, parent_path)
        if not os.path.exists(parent_full):
            # try relative to script dir
            parent_full = os.path.join(str(SCRIPT_DIR), parent_path)
        parent = load_config(parent_full)
        cfg = _deep_merge(parent, cfg)
    return cfg


# ═══════════════════════════════════════════════════════════════
#  Diagnostic helpers
# ═══════════════════════════════════════════════════════════════

def _stat(arr):
    """μ / σ / min / max for a list of floats."""
    if not arr:
        return {'mean': 0., 'std': 0., 'min': 0., 'max': 0., 'n': 0}
    a = np.asarray(arr, dtype=np.float64)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max()), 'n': len(a)}


class DiagnosticLog:
    """Collects per-episode diagnostics and writes them as CSV + JSON."""

    HEADER = [
        'ep', 'stage', 'wall_env_s', 'wall_train_s',
        # env
        'avg_wait_min', 'peak_fleet', 'headway_cv', 'ep_reward', 'ep_cost',
        'ep_steps', 'n_dispatches',
        # lower policy
        'lower_action_mean', 'lower_action_std', 'lower_action_min', 'lower_action_max',
        'lower_reward_mean', 'lower_reward_std',
        # lower training
        'lower_q_mean', 'lower_q_std', 'lower_q_loss', 'lower_q_mse',
        'lower_ood_loss', 'lower_cost_q_mean', 'lower_cost_q_loss',
        'lower_policy_loss', 'lower_pi_grad_norm', 'lower_q_grad_norm',
        'lower_alpha', 'lower_lambda',
        'lower_replay_size',
        # upper policy (only after warmup)
        'upper_delta_mean', 'upper_delta_std', 'upper_delta_min', 'upper_delta_max',
        'upper_reward_mean', 'upper_reward_std',
        # upper training
        'upper_q_mean', 'upper_q_std', 'upper_q_loss', 'upper_q_mse',
        'upper_ood_loss', 'upper_policy_loss',
        'upper_pi_grad_norm', 'upper_q_grad_norm',
        'upper_alpha', 'upper_replay_size',
        # coupling
        'hold_fb_mean', 'hold_fb_std', 'hold_fb_n_trips',
        'hold_fb_dir0_mean', 'hold_fb_dir1_mean',
        'hold_penalty_mean',
        'theta_wait', 'theta_fleet', 'theta_cv',
        # CS-BAPR belief
        'surprise', 'belief_window', 'belief_cp_prob', 'belief_entropy',
        # v2j belief-weighted MORL
        'w_wait', 'w_fleet', 'w_cv',
        # v2k elastic fleet
        'N_fleet', 'fleet_overshoot',
    ]

    def __init__(self, log_dir, resume=False):
        self.csv_path = os.path.join(log_dir, 'diagnostics.csv')
        self.json_path = os.path.join(log_dir, 'diagnostics.json')
        self._rows = []
        # Write CSV header only if not resuming or CSV missing
        if not (resume and os.path.exists(self.csv_path)):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.HEADER)

    def append(self, row_dict):
        """Append one episode row. Missing keys default to 0."""
        self._rows.append(row_dict)
        row = [row_dict.get(k, 0.) for k in self.HEADER]
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self._rows, f, indent=1, default=str)


# ═══════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════

class TransitDuetV2Runner:
    """v2 training loop: per-trip upper decisions + holding feedback coupling."""

    def __init__(self, config, device='cpu'):
        self.cfg = config
        self.device = device

        # Environment
        env_path = os.path.join(str(SCRIPT_DIR), config['env']['path'])
        self.env = env_bus(env_path, route_sigma=config['env']['route_sigma'])
        self.env.enable_plot = False
        self.env._n_fleet_target = config['upper']['N_fleet']
        self.env.demand_noise = config['env'].get('demand_noise', 0.0)

        state_dim = self.env.state_dim

        # ── Upper policy ──
        upper_cfg = config['upper']
        self.delta_max = upper_cfg.get('delta_max', 120.0)
        # v2k: elastic fleet — sample N_fleet per episode
        self.fleet_mode = upper_cfg.get('fleet_mode', 'fixed')
        self.fleet_min = upper_cfg.get('fleet_min', 8)
        self.fleet_max = upper_cfg.get('fleet_max', 16)
        self.N_fleet_default = upper_cfg['N_fleet']
        self._current_N_fleet = self.N_fleet_default  # set per-episode in elastic mode
        self.upper_state_dim = upper_cfg.get('state_dim', 10)

        self.upper_trainer = RESACUpperTrainer(
            state_dim=self.upper_state_dim, action_dim=1,
            hidden_dim=upper_cfg.get('hidden_dim', 64),
            action_low=[-self.delta_max], action_high=[self.delta_max],
            ensemble_size=upper_cfg.get('ensemble_size', 10),
            beta=upper_cfg.get('resac_beta', -2.0),
            lr=upper_cfg.get('lr', 3e-4),
            gamma=upper_cfg.get('gamma', 0.95),
            maximum_alpha=upper_cfg.get('maximum_alpha', 0.05),
            replay_capacity=upper_cfg.get('replay_capacity', 50000),
            device=device)

        # ── Lower policy ──
        lower_cfg = config['lower']
        self.replay_buffer = CostReplayBuffer(config['training']['replay_buffer_size'])
        self.lower_trainer = RESACLagrangianTrainer(
            state_dim=state_dim, action_dim=1,
            hidden_dim=lower_cfg['hidden_dim'],
            action_range=lower_cfg['action_range'],
            cost_limit=lower_cfg['cost_limit'],
            ensemble_size=lower_cfg.get('ensemble_size', 10),
            beta=lower_cfg.get('resac_beta', -2.0),
            beta_ood=lower_cfg.get('beta_ood', 0.01),
            weight_reg=lower_cfg.get('weight_reg', 0.01),
            lr=lower_cfg['lr'], lambda_lr=lower_cfg['lambda_lr'],
            gamma=lower_cfg['gamma'], soft_tau=lower_cfg['soft_tau'],
            auto_entropy=lower_cfg['auto_entropy'],
            maximum_alpha=lower_cfg['maximum_alpha'],
            device=device)

        # ── Coupling ──
        coupling_cfg = config['coupling']
        # Ablation flags (for paper experiments)
        self.ablate_holding_feedback = coupling_cfg.get('ablate_holding_feedback', False)
        self.ablate_csbapr = coupling_cfg.get('ablate_csbapr', False)
        self.ablate_hindsight_credit = coupling_cfg.get('ablate_hindsight_credit', False)
        self.ablate_morl = coupling_cfg.get('ablate_morl', False)

        self.holding_feedback = HoldingFeedback(
            window_size=coupling_cfg.get('feedback_window', 10))
        self.measurement_proj = MeasurementProjection(
            N_fleet=upper_cfg['N_fleet'],
            lr=coupling_cfg.get('measurement_lr', 0.01))
        self.alpha_holding = coupling_cfg.get('alpha_holding', 0.5)
        self.upper_warmup = coupling_cfg.get('upper_warmup_eps', 30)

        # ─── TPC-Lower (Target-Policy-Corrected lower SAC) ───
        # Mitigates the "noisy upper contaminates lower" failure mode that loses
        # to the Fixed baseline. Enabled via coupling.tpc_enable in config.
        tpc = coupling_cfg.get('tpc', {})
        self.tpc_enable = bool(tpc.get('enable', False))
        self.tpc_eps = float(tpc.get('eps_explore', 0.25))
        self.tpc_sigma_tgt = float(tpc.get('sigma_tgt', 20.0))
        self.tpc_w_max = float(tpc.get('w_max', 5.0))
        self.tpc_ema_tau = float(tpc.get('ema_tau', 0.005))
        self.tpc_warmstart_lower_from = tpc.get('warmstart_lower_from', None)
        self.target_upper_trainer = None  # initialised at end of upper_warmup
        # global_tid -> {z, delta, log_mu} for IS weight lookup
        self.dispatch_meta = {}
        # bound the metadata dict size (replay capacity / trips_per_episode + buffer)
        self._dispatch_meta_max = int(tpc.get('meta_max_size', 200_000))

        # CS-BAPR belief tracker: detect non-stationarity from upper changes
        self.surprise_computer = SurpriseComputer(
            ema_alpha=coupling_cfg.get('surprise_ema', 0.3))
        self.belief_tracker = BeliefTracker(
            max_run_length=coupling_cfg.get('belief_max_H', 20),
            hazard_rate=coupling_cfg.get('belief_hazard', 0.05))
        self.belief_alpha_boost_max = coupling_cfg.get('belief_alpha_boost', 2.0)

        # Training params
        self.batch_size = lower_cfg.get('batch_size', 512)
        self.updates_per_episode = lower_cfg.get('updates_per_episode', 30)
        self.upper_batch_size = upper_cfg.get('batch_size', 64)
        self.upper_updates = upper_cfg.get('updates_per_episode', 10)

        # Episode bookkeeping
        self._episode_upper_transitions = []
        self._prev_upper_state = None
        self._ep_lower_actions = []     # all lower actions this episode
        self._ep_lower_rewards = []     # all lower rewards this episode
        self._ep_upper_deltas = []      # all δ_t this episode
        self._ep_trip_records = []      # per-trip detail for step-level diag
        self._ep_dispatch_times = {'up': [], 'down': []}  # actual launch times per dir
        self._ep_upper_rewards = []     # all upper rewards this episode
        self._current_ep = 0

        # Logging
        seed = config.get('seed', 42)
        exp_name = config.get('_name', 'v2')
        self.log_dir = os.path.join(str(SCRIPT_DIR), 'logs', f'{exp_name}_seed{seed}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = defaultdict(list)
        self.resume_from_ep = 0  # set by maybe_resume() before train()
        self.diag = None  # created after resume decision in train()

    # ────────────────── Upper callback ──────────────────

    @staticmethod
    def compute_system_reward(z, N_fleet=12):
        """
        Default scalar system reward (fallback, unused when belief-weighted).
        """
        wait_penalty = -z[0] / 10.0
        fleet_penalty = -max(0, z[1] - N_fleet) ** 2 / N_fleet
        cv_penalty = -z[2]
        return wait_penalty + fleet_penalty + cv_penalty

    def compute_belief_weighted_reward(self, z, N_fleet=12):
        """
        BAMOR-style multi-objective scalarization with belief-aware weights.

        Three objective penalties (all negative, higher=better):
          p_wait  = -wait / 10
          p_fleet = -(fleet - N_fleet)² / N_fleet  (only counts overshoot)
          p_cv    = -cv

        Weighting policy:
          Base weights w_base from θ-OGD (long-term adaptation over episodes)
          Belief modulation (short-term shift detection):
            - cp_prob high → changepoint detected → boost fleet weight (safety)
            - window long → stable → shift weight to wait/cv (quality)

        Returns: scalar reward + weight dict for logging
        """
        wait_p = -z[0] / 10.0
        fleet_p = -max(0, z[1] - N_fleet) ** 2 / N_fleet
        cv_p = -z[2]

        # Ablation: fixed equal weights instead of belief-driven
        if self.ablate_morl:
            fixed_w = np.array([0.5, 0.25, 0.25])
            r = fixed_w[0] * wait_p + fixed_w[1] * fleet_p + fixed_w[2] * cv_p
            return float(r * 3.0), fixed_w

        # Base weights from θ-OGD (long-term adaptation, already in [0,1] sum=1)
        base_w = self.measurement_proj.get_reward_weights()  # [w_wait, w_fleet, w_cv]

        # Belief modulation
        cp_prob = self.belief_tracker.changepoint_prob
        window = self.belief_tracker.effective_window

        # Crisis modulation: if changepoint detected, boost fleet safety
        if cp_prob > 0.1:
            # Shift up to 30% mass toward fleet term
            crisis_strength = min(1.0, (cp_prob - 0.1) / 0.2)
            boost = 0.3 * crisis_strength
            adj_w = base_w.copy()
            # Take mass from wait+cv, add to fleet
            adj_w[0] *= (1 - boost)
            adj_w[2] *= (1 - boost)
            adj_w[1] += boost * (base_w[0] + base_w[2])
        # Stable modulation: if very stable, shift toward quality
        elif window > 15:
            stability = min(1.0, (window - 15) / 5)
            shift = 0.15 * stability
            adj_w = base_w.copy()
            # Take from fleet (already safe), add to quality
            adj_w[1] *= (1 - shift)
            adj_w[0] += shift * base_w[1] * 0.6
            adj_w[2] += shift * base_w[1] * 0.4
        else:
            adj_w = base_w

        # Normalize
        adj_w = adj_w / max(adj_w.sum(), 1e-6)

        # Scalarize with M=3 dimensions
        r = adj_w[0] * wait_p + adj_w[1] * fleet_p + adj_w[2] * cv_p
        # Rescale to match old magnitude (old reward range ≈ [-2, 0])
        r = r * 3.0

        return float(r), adj_w

    def _build_tpc_weight_fn(self):
        """Return a closure that maps batch trip_ids → per-sample IS weights.

        Weight per sample = clip( π_target(δ|z) / μ_behavior(δ|z), 0, w_max ),
        normalised so the batch mean ≈ 1. Samples whose trip_id has no metadata
        in self.dispatch_meta (e.g., evicted by size cap or pre-Phase-1) get 1.
        """
        if self.target_upper_trainer is None or not self.dispatch_meta:
            return None
        target_pi = self.target_upper_trainer.policy_net
        meta = self.dispatch_meta
        w_max = self.tpc_w_max

        def fn(trip_ids):
            # Vectorise where possible: gather z and δ for samples with metadata.
            n = len(trip_ids)
            w = np.ones(n, dtype=np.float32)
            zs, ds, log_mus, idx = [], [], [], []
            for i, tid in enumerate(trip_ids):
                m = meta.get(int(tid))
                if m is not None:
                    zs.append(m['z']); ds.append(m['delta']); log_mus.append(m['log_mu'])
                    idx.append(i)
            if zs:
                zs = np.stack(zs).astype(np.float32)
                ds = np.array(ds, dtype=np.float32).reshape(-1, 1)
                log_mus = np.array(log_mus, dtype=np.float32)
                # Batched log_prob under EMA target upper policy
                log_p_target = target_pi.log_prob(zs, ds)
                if np.isscalar(log_p_target):
                    log_p_target = np.array([float(log_p_target)])
                log_w = log_p_target - log_mus
                w_corr = np.clip(np.exp(log_w), 0.0, w_max)
                for j, i in enumerate(idx):
                    w[i] = float(w_corr[j])
            # Normalise to mean ≈ 1
            mean_w = w.mean()
            if mean_w > 1e-6:
                w = w / mean_w
            return w
        return fn

    def _upper_callback_v2(self, s_upper_v1, trip):
        """Per-dispatch decision: output δ_t, store (s, a, trip_id, s') without reward.
        Reward is backfilled at episode end via hindsight credit assignment."""
        s_upper = self.env._build_upper_state_v2(trip)
        if self.ablate_holding_feedback:
            # Zero out holding feedback state dims [5,6,7]
            s_upper[5:8] = 0.0

        # ─── TPC-Lower behaviour-policy sampling ───
        # During Phase 1 (after warmup, target_upper_trainer initialised), sample
        # δ_t from a mixture: ε from current upper (exploratory) + (1−ε) from
        # N(target_mean, σ_tgt) so the lower's training distribution can be
        # importance-corrected back toward the EMA "deployment" upper.
        log_mu = None
        if self.tpc_enable and self.target_upper_trainer is not None:
            target_mean_arr = self.target_upper_trainer.policy_net.get_action(
                s_upper, deterministic=True)
            target_mean = float(target_mean_arr[0])
            if np.random.random() < self.tpc_eps:
                # exploratory: sample from current π_U
                delta_t = float(self.upper_trainer.policy_net.get_action(
                    s_upper, deterministic=False)[0])
            else:
                # target: N(target_mean, σ_tgt), clipped to action range
                delta_t = float(np.clip(
                    target_mean + np.random.randn() * self.tpc_sigma_tgt,
                    -self.delta_max, self.delta_max))
            # Mixture log-prob log_mu = log( ε π_U + (1-ε) N(target_mean, σ_tgt) )
            log_p_explore = float(self.upper_trainer.policy_net.log_prob(
                s_upper, np.array([delta_t], dtype=np.float32)))
            log_p_target = (-0.5 * ((delta_t - target_mean) / self.tpc_sigma_tgt) ** 2
                            - np.log(self.tpc_sigma_tgt * np.sqrt(2 * np.pi)))
            log_mu = float(np.logaddexp(
                np.log(self.tpc_eps + 1e-12) + log_p_explore,
                np.log(1.0 - self.tpc_eps + 1e-12) + log_p_target))
        else:
            delta_t = float(self.upper_trainer.policy_net.get_action(
                s_upper, deterministic=False)[0])
        self._ep_upper_deltas.append(delta_t)

        # v2g: δ_t directly sets launch time offset (no headway cascade).
        # Store on trip object so env.step() uses direct time gating.
        if not hasattr(trip, '_original_launch'):
            trip._original_launch = trip.launch_time
        trip._delta_t = int(delta_t)

        # TPC-Lower: store dispatch metadata for IS weight lookup. Use a global
        # trip id (episode * 1000 + local tid) so cross-episode samples in the
        # replay buffer can still be reweighted (or default to 1 if pruned).
        if log_mu is not None:
            global_tid = self._current_ep * 1000 + int(trip.launch_turn)
            self.dispatch_meta[global_tid] = {
                'z': s_upper.astype(np.float32),
                'delta': float(delta_t),
                'log_mu': log_mu,
            }
            # Bound metadata size: drop oldest when over budget
            if len(self.dispatch_meta) > self._dispatch_meta_max:
                # delete the oldest 10% in one pass
                k_drop = max(1, self._dispatch_meta_max // 10)
                for old_key in sorted(self.dispatch_meta.keys())[:k_drop]:
                    del self.dispatch_meta[old_key]

        # Keep target_headway unchanged (lower still uses base headway as target)
        base_hw = trip.target_headway if hasattr(trip, 'target_headway') else 360.0

        # Store (s, a, trip_id, s', done) — reward computed at episode end
        if self._prev_upper_state is not None:
            prev_s, prev_a, prev_tid, prev_dir = self._prev_upper_state
            self._episode_upper_transitions.append({
                's': prev_s, 'a': prev_a, 'tid': prev_tid,
                'ns': s_upper.copy(), 'done': False,
            })

        self._prev_upper_state = (
            s_upper.copy(),
            np.array([delta_t], dtype=np.float32),
            trip.launch_turn, trip.direction)

        # Record dispatch info (actual launch time captured post-episode from env)
        dir_key = 'up' if trip.direction else 'down'
        self._ep_dispatch_times[dir_key].append({
            'tid': trip.launch_turn,
            'scheduled': trip._original_launch,
            'delta_t': delta_t,
            'effective_launch': trip._original_launch + int(delta_t),
        })

        # Per-trip record for step-level diagnostics
        hour = 6 + trip.launch_time // 3600
        period = 'peak' if (7 <= hour <= 9 or 17 <= hour <= 19) else (
            'off' if 9 < hour < 17 else 'trans')
        self._ep_trip_records.append({
            'tid': trip.launch_turn,
            'dir': int(trip.direction),
            'hour': hour,
            'period': period,
            'delta_t': round(delta_t, 1),
            'base_hw': round(base_hw, 0),
            'eff_hw': round(base_hw, 0),  # unchanged now
            's_hold_mean': round(s_upper[5] * 60, 1),
            's_hold_std': round(s_upper[6] * 60, 1),
        })

        return base_hw  # return original headway (δ_t already applied to launch_time)

    # ────────────────── Episode ──────────────────

    def run_episode(self, ep, training=True, N_fleet_override=None):
        t0 = time.time()
        self.env.reset()
        self.holding_feedback.clear()
        self._current_ep = ep
        self._episode_upper_transitions = []
        self._prev_upper_state = None
        self._ep_lower_actions = []
        self._ep_lower_rewards = []
        self._ep_upper_deltas = []
        self._ep_upper_rewards = []
        self._ep_trip_records = []
        self._ep_dispatch_times = {'up': [], 'down': []}

        # v2k: elastic fleet sampling per-episode
        if N_fleet_override is not None:
            self._current_N_fleet = int(N_fleet_override)
        elif self.fleet_mode == 'elastic' and training:
            self._current_N_fleet = int(np.random.randint(
                self.fleet_min, self.fleet_max + 1))
        else:
            self._current_N_fleet = self.N_fleet_default
        self.env._n_fleet_target = self._current_N_fleet

        upper_active = ep >= self.upper_warmup and training
        self.env._upper_policy_callback = (
            self._upper_callback_v2 if upper_active else None)

        state_dict, reward_dict, _ = self.env.initialize_state()
        action_dict = {k: None for k in range(self.env.max_agent_num)}
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        while not self.env.done:
            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        obs = np.array(state_dict[key][0], dtype=np.float32)
                        a = self.lower_trainer.policy_net.get_action(
                            torch.from_numpy(obs).float().to(self.device),
                            deterministic=not training)
                        action_dict[key] = a

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        state = np.array(state_dict[key][0], dtype=np.float32)
                        next_state = np.array(state_dict[key][1], dtype=np.float32)
                        reward = reward_dict[key]
                        cost = self.env.cost.get(key, 0.0)

                        # Track for diagnostics
                        act_val = float(action_dict[key]) if action_dict[key] is not None else 0.0
                        self._ep_lower_actions.append(act_val)
                        self._ep_lower_rewards.append(float(reward))

                        if training:
                            # Look up the bus's current trip_id (launch_turn) so
                            # downstream TPC IS-weight lookup can match dispatch_meta.
                            bus_id_key = int(state[0])
                            cur_tid = -1
                            for bus in self.env.bus_all:
                                if bus.bus_id == bus_id_key:
                                    cur_tid = int(getattr(bus, 'trip_id', -1))
                                    break
                            global_tid = (self._current_ep * 1000 + cur_tid
                                          if cur_tid >= 0 else int(state[0]))
                            self.replay_buffer.push(
                                state, action_dict[key], reward, cost,
                                next_state, False, global_tid)
                            # v2: record holding action for feedback
                            bus_id = int(state[0])
                            for bus in self.env.bus_all:
                                if bus.bus_id == bus_id:
                                    self.holding_feedback.record_action(
                                        bus.trip_id, action_dict[key])
                                    break

                        episode_reward += reward
                        episode_cost += cost
                        episode_steps += 1

                    state_dict[key] = state_dict[key][1:]
                    obs = np.array(state_dict[key][0], dtype=np.float32)
                    action_dict[key] = self.lower_trainer.policy_net.get_action(
                        torch.from_numpy(obs).float().to(self.device),
                        deterministic=not training)

            state_dict, reward_dict, cost_dict, done = self.env.step(
                action_dict, render=False)

        env_time = time.time() - t0

        # ── Finalize trip holdings ──
        for bus in self.env.bus_all:
            if not bus.on_route and hasattr(bus, 'applied_actions') and bus.applied_actions:
                self.holding_feedback.finalize_trip(bus.trip_id, bus.direction)

        # ── Finalize last upper transition ──
        if self._prev_upper_state is not None:
            prev_s, prev_a, prev_tid, prev_dir = self._prev_upper_state
            self._episode_upper_transitions.append({
                's': prev_s, 'a': prev_a, 'tid': prev_tid,
                'ns': prev_s, 'done': True,
            })

        # ── Hindsight Credit Assignment (v2g: gap-based, not holding-based) ──
        # Old: credit based on holding magnitude → corr(δ_t, hold)=0, BROKEN
        # New: credit based on dispatch gap uniformity → directly causal
        #   δ_t → dispatch timing → gap to neighbors → gap deviation = credit
        z = self.env.measurement_vector
        N_fleet = self._current_N_fleet  # v2k: use episode's sampled budget
        # v2j: belief-weighted multi-objective scalarization (Option 1 BAMOR)
        sys_r, adj_w = self.compute_belief_weighted_reward(z, N_fleet)
        self._last_adj_weights = adj_w

        # Compute per-trip gap deviation using ACTUAL launch times from env
        trip_gap_devs = {}
        for dir_key_bool in [True, False]:  # direction
            dir_key = 'up' if dir_key_bool else 'down'
            # Get actual launch times from timetable objects
            launched = [(tt.launch_turn, tt._actual_launch_time)
                        for tt in self.env.timetables
                        if tt.launched and tt.direction == dir_key_bool
                        and hasattr(tt, '_actual_launch_time')]
            if len(launched) < 2:
                continue
            launched.sort(key=lambda x: x[1])  # sort by actual time
            tids = [l[0] for l in launched]
            times = [l[1] for l in launched]
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            if not gaps:
                continue
            mean_gap = np.mean(gaps)
            std_gap = max(np.std(gaps), 1.0)
            # Assign deviation to each trip (trip i created gap[i-1])
            for i in range(len(tids)):
                if i == 0:
                    dev = abs(gaps[0] - mean_gap) / std_gap if gaps else 0.0
                elif i < len(gaps):
                    dev = (abs(gaps[i-1] - mean_gap) + abs(gaps[i] - mean_gap)) / (2 * std_gap)
                else:
                    dev = abs(gaps[-1] - mean_gap) / std_gap
                trip_gap_devs[tids[i]] = dev

        # Normalize gap devs to zero-mean credit
        if trip_gap_devs:
            devs = np.array(list(trip_gap_devs.values()))
            dev_mean = devs.mean()
            dev_std = max(devs.std(), 1e-6)
        else:
            dev_mean, dev_std = 0.0, 1.0

        backfilled = []
        for trans in self._episode_upper_transitions:
            tid = trans['tid']
            if self.ablate_hindsight_credit:
                # Ablation: all transitions get same episode reward
                credit = 0.0
            else:
                gap_dev = trip_gap_devs.get(tid, dev_mean)
                credit = -(gap_dev - dev_mean) / dev_std * 0.5
            r = sys_r + credit
            backfilled.append(
                (trans['s'], trans['a'], r, trans['ns'], trans['done']))
            self._ep_upper_rewards.append(r)
        self._episode_upper_transitions = backfilled

        # ── Enrich per-trip records with holding + gap deviation ──
        for rec in self._ep_trip_records:
            tid = rec['tid']
            stats = self.holding_feedback.get_trip_stats(tid)
            if stats:
                rec['hold_mean'] = round(stats['mean'], 1)
                rec['hold_std'] = round(stats['std'], 1)
                rec['hold_max'] = round(stats['max'], 1)
                rec['hold_n'] = stats['n_stops']
            else:
                rec['hold_mean'] = 0.0
                rec['hold_std'] = 0.0
                rec['hold_max'] = 0.0
                rec['hold_n'] = 0
            gap_dev = trip_gap_devs.get(tid, 0.0)
            rec['gap_dev'] = round(gap_dev, 3)
            rec['penalty'] = round(gap_dev, 3)  # now gap-based
            credit = -(gap_dev - dev_mean) / dev_std * 0.5
            rec['reward'] = round(sys_r + credit, 3)

        # ══════════════ CS-BAPR: Belief Update ══════════════
        # Detect non-stationarity from upper-level timetable changes
        ep_reward_mean = episode_reward / max(episode_steps, 1)
        # Use Q_std from previous episode's training (history), or 0 if first ep
        prev_q_stds = self.history.get('lower_q_std', [])
        ep_q_std = prev_q_stds[-1] if prev_q_stds else 0.0
        ep_delta_mean = (np.mean(self._ep_upper_deltas)
                         if self._ep_upper_deltas else 0.0)

        surprise = self.surprise_computer.compute(
            ep_reward_mean, ep_q_std, ep_delta_mean)
        self.belief_tracker.update(surprise)

        # Adaptive alpha: boost exploration after detected changepoint
        base_alpha = self.lower_trainer.alpha
        boosted_alpha = self.belief_tracker.adaptive_alpha_boost(
            base_alpha, max_boost=self.belief_alpha_boost_max)
        # Temporarily set alpha for this episode's training
        # (auto-entropy will correct it over time, this just gives a nudge)
        if not self.ablate_csbapr and surprise > 0.5 and upper_active:
            self.lower_trainer.alpha = min(boosted_alpha,
                                           self.lower_trainer.maximum_alpha)

        # ══════════════ Training ══════════════
        t1 = time.time()
        lower_m = {}
        upper_m = {}

        # ─── TPC: lazy-init EMA target upper at start of Phase 1 ───
        # We snapshot the current upper at end of warmup; subsequent Polyak
        # averaging keeps this "deployment" copy as a slow-moving anchor for
        # importance reweighting on the lower SAC.
        if (self.tpc_enable and upper_active
                and self.target_upper_trainer is None):
            self.target_upper_trainer = copy.deepcopy(self.upper_trainer)
            print(f"  [TPC] initialised EMA target upper at ep {ep}")

        # Build per-sample IS weight function for lower SAC
        weight_fn = self._build_tpc_weight_fn() if self.tpc_enable else None

        # Lower
        if training and len(self.replay_buffer) > self.batch_size:
            for _ in range(self.updates_per_episode):
                lower_m = self.lower_trainer.update(
                    self.replay_buffer, self.batch_size, reward_scale=1.0,
                    weight_fn=weight_fn)

        # Upper
        if upper_active:
            for (s, a, r, ns, d) in self._episode_upper_transitions:
                self.upper_trainer.replay_buffer.push(s, a, r, ns, d)
            if len(self.upper_trainer.replay_buffer) > self.upper_batch_size:
                for _ in range(self.upper_updates):
                    upper_m = self.upper_trainer.update(self.upper_batch_size)

            # ─── TPC: Polyak update EMA target after each upper training step ───
            if self.tpc_enable and self.target_upper_trainer is not None:
                with torch.no_grad():
                    for p_t, p in zip(self.target_upper_trainer.policy_net.parameters(),
                                      self.upper_trainer.policy_net.parameters()):
                        p_t.data.mul_(1.0 - self.tpc_ema_tau).add_(
                            p.data, alpha=self.tpc_ema_tau)

        # Measurement projection (z already computed above for upper reward)
        self.measurement_proj.update(z)
        theta_w = self.measurement_proj.get_reward_weights()

        train_time = time.time() - t1

        # ══════════════ Diagnostics ══════════════
        stage = "Warmup" if ep < self.upper_warmup else "BiLevel"
        hold_summary = self.holding_feedback.episode_summary
        hold_dir0 = self.holding_feedback.get_direction_stats(False)
        hold_dir1 = self.holding_feedback.get_direction_stats(True)
        la_stat = _stat(self._ep_lower_actions)
        lr_stat = _stat(self._ep_lower_rewards)
        ud_stat = _stat(self._ep_upper_deltas)
        ur_stat = _stat(self._ep_upper_rewards)

        # Holding penalties across all trips this episode
        hold_pens = []
        for tid in self.holding_feedback._trip_actions:
            hold_pens.append(self.holding_feedback.holding_penalty(tid))
        hp_stat = _stat(hold_pens)

        row = {
            'ep': ep, 'stage': stage,
            'wall_env_s': round(env_time, 1),
            'wall_train_s': round(train_time, 1),
            # env
            'avg_wait_min': round(z[0], 3),
            'peak_fleet': int(z[1]),
            'headway_cv': round(z[2], 4),
            'ep_reward': round(episode_reward, 3),
            'ep_cost': round(episode_cost, 3),
            'ep_steps': episode_steps,
            'n_dispatches': len(self._ep_upper_deltas) if upper_active else 0,
            # lower policy
            'lower_action_mean': round(la_stat['mean'], 2),
            'lower_action_std': round(la_stat['std'], 2),
            'lower_action_min': round(la_stat['min'], 2),
            'lower_action_max': round(la_stat['max'], 2),
            'lower_reward_mean': round(lr_stat['mean'], 4),
            'lower_reward_std': round(lr_stat['std'], 4),
            # lower training
            'lower_q_mean': lower_m.get('q_mean', 0.),
            'lower_q_std': lower_m.get('q_std', 0.),
            'lower_q_loss': lower_m.get('q_loss', 0.),
            'lower_q_mse': lower_m.get('q_mse', 0.),
            'lower_ood_loss': lower_m.get('ood_loss', 0.),
            'lower_cost_q_mean': lower_m.get('cost_q_mean', 0.),
            'lower_cost_q_loss': lower_m.get('cost_q_loss', 0.),
            'lower_policy_loss': lower_m.get('policy_loss', 0.),
            'lower_pi_grad_norm': lower_m.get('pi_grad_norm', 0.),
            'lower_q_grad_norm': lower_m.get('q_grad_norm', 0.),
            'lower_alpha': lower_m.get('alpha', 0.),
            'lower_lambda': lower_m.get('lambda', self.lower_trainer.lambda_param),
            'lower_replay_size': len(self.replay_buffer),
            # upper policy
            'upper_delta_mean': round(ud_stat['mean'], 2),
            'upper_delta_std': round(ud_stat['std'], 2),
            'upper_delta_min': round(ud_stat['min'], 2),
            'upper_delta_max': round(ud_stat['max'], 2),
            'upper_reward_mean': round(ur_stat['mean'], 4),
            'upper_reward_std': round(ur_stat['std'], 4),
            # upper training
            'upper_q_mean': upper_m.get('upper_q_mean', 0.),
            'upper_q_std': upper_m.get('upper_q_std', 0.),
            'upper_q_loss': upper_m.get('upper_q_loss', 0.),
            'upper_q_mse': upper_m.get('upper_q_mse', 0.),
            'upper_ood_loss': upper_m.get('upper_ood_loss', 0.),
            'upper_policy_loss': upper_m.get('upper_policy_loss', 0.),
            'upper_pi_grad_norm': upper_m.get('upper_pi_grad_norm', 0.),
            'upper_q_grad_norm': upper_m.get('upper_q_grad_norm', 0.),
            'upper_alpha': upper_m.get('upper_alpha', 0.),
            'upper_replay_size': len(self.upper_trainer.replay_buffer),
            # coupling
            'hold_fb_mean': hold_summary.get('mean', 0.),
            'hold_fb_std': hold_summary.get('std', 0.),
            'hold_fb_n_trips': hold_summary.get('n_trips', 0),
            'hold_fb_dir0_mean': hold_dir0['rolling_mean'],
            'hold_fb_dir1_mean': hold_dir1['rolling_mean'],
            'hold_penalty_mean': hp_stat['mean'],
            'theta_wait': float(theta_w[0]),
            'theta_fleet': float(theta_w[1]),
            'theta_cv': float(theta_w[2]),
            # CS-BAPR belief
            'surprise': round(surprise, 4),
            'belief_window': round(self.belief_tracker.effective_window, 2),
            'belief_cp_prob': round(self.belief_tracker.changepoint_prob, 4),
            'belief_entropy': round(self.belief_tracker.entropy, 3),
            # v2j: belief-weighted MORL weights
            'w_wait': round(float(self._last_adj_weights[0]), 3) if hasattr(self, '_last_adj_weights') else 0.,
            'w_fleet': round(float(self._last_adj_weights[1]), 3) if hasattr(self, '_last_adj_weights') else 0.,
            'w_cv': round(float(self._last_adj_weights[2]), 3) if hasattr(self, '_last_adj_weights') else 0.,
            # v2k: elastic fleet
            'N_fleet': self._current_N_fleet,
            'fleet_overshoot': max(0, int(z[1]) - self._current_N_fleet),
        }
        self.diag.append(row)

        # Also keep lightweight history for quick plotting
        for k in ['avg_wait_min', 'peak_fleet', 'headway_cv',
                   'lower_lambda', 'lower_alpha', 'lower_q_mean', 'lower_q_std',
                   'upper_delta_mean', 'upper_q_mean',
                   'hold_fb_mean', 'hold_penalty_mean',
                   'theta_wait', 'theta_fleet',
                   'surprise', 'belief_window']:
            self.history[k].append(row[k])

        return row

    # ────────────────── Periodic deep dump ──────────────────

    def _print_diagnostic_block(self, row):
        """Print a detailed diagnostic block (every N episodes)."""
        ep = row['ep']
        print(f"\n{'─'*90}")
        print(f"  DIAGNOSTIC  ep={ep}  stage={row['stage']}  "
              f"wall={row['wall_env_s']}+{row['wall_train_s']}s")
        print(f"{'─'*90}")

        print(f"  ENV      wait={row['avg_wait_min']:.2f}m  fleet={row['peak_fleet']}  "
              f"cv={row['headway_cv']:.3f}  "
              f"R={row['ep_reward']:.2f}  C={row['ep_cost']:.2f}  "
              f"steps={row['ep_steps']}")

        print(f"  LOWER π  action μ={row['lower_action_mean']:.1f} "
              f"σ={row['lower_action_std']:.1f} "
              f"[{row['lower_action_min']:.1f}, {row['lower_action_max']:.1f}]  "
              f"reward μ={row['lower_reward_mean']:.3f} "
              f"σ={row['lower_reward_std']:.3f}")

        print(f"  LOWER Q  Q={row['lower_q_mean']:.2f}±{row['lower_q_std']:.2f}  "
              f"loss={row['lower_q_loss']:.4f} (mse={row['lower_q_mse']:.4f} "
              f"ood={row['lower_ood_loss']:.4f})  "
              f"CQ={row['lower_cost_q_mean']:.3f} "
              f"CQ_loss={row['lower_cost_q_loss']:.4f}")

        print(f"  LOWER ∇  π_grad={row['lower_pi_grad_norm']:.4f}  "
              f"Q_grad={row['lower_q_grad_norm']:.4f}  "
              f"α={row['lower_alpha']:.4f}  λ={row['lower_lambda']:.3f}  "
              f"buf={row['lower_replay_size']}")

        if row['stage'] == 'BiLevel':
            print(f"  UPPER δ  δ_t μ={row['upper_delta_mean']:.1f} "
                  f"σ={row['upper_delta_std']:.1f} "
                  f"[{row['upper_delta_min']:.1f}, {row['upper_delta_max']:.1f}]  "
                  f"reward μ={row['upper_reward_mean']:.3f} "
                  f"σ={row['upper_reward_std']:.3f}  "
                  f"n={row['n_dispatches']}")

            print(f"  UPPER Q  Q={row['upper_q_mean']:.3f}±{row['upper_q_std']:.3f}  "
                  f"loss={row['upper_q_loss']:.4f} (mse={row['upper_q_mse']:.4f} "
                  f"ood={row['upper_ood_loss']:.4f})  "
                  f"π_loss={row['upper_policy_loss']:.4f}")

            print(f"  UPPER ∇  π_grad={row['upper_pi_grad_norm']:.4f}  "
                  f"Q_grad={row['upper_q_grad_norm']:.4f}  "
                  f"α={row['upper_alpha']:.4f}  buf={row['upper_replay_size']}")

        print(f"  COUPLE   hold μ={row['hold_fb_mean']:.1f}s  "
              f"σ={row['hold_fb_std']:.1f}  "
              f"n_trips={row['hold_fb_n_trips']}  "
              f"dir0={row['hold_fb_dir0_mean']:.1f}  "
              f"dir1={row['hold_fb_dir1_mean']:.1f}  "
              f"penalty={row['hold_penalty_mean']:.3f}")

        print(f"  θ-OGD    w=[{row['theta_wait']:.3f}, "
              f"{row['theta_fleet']:.3f}, {row['theta_cv']:.3f}]")

        print(f"  BELIEF   surprise={row.get('surprise',0):.3f}  "
              f"window={row.get('belief_window',0):.1f}  "
              f"cp_prob={row.get('belief_cp_prob',0):.3f}  "
              f"entropy={row.get('belief_entropy',0):.2f}")
        print(f"{'─'*90}\n")

    # ────────────────── Per-trip dump ──────────────────

    def _dump_trip_breakdown(self, ep):
        """Write per-trip detail to CSV and print summary."""
        if not self._ep_trip_records:
            return

        trip_csv = os.path.join(self.log_dir, 'trip_details.csv')
        write_header = not os.path.exists(trip_csv)
        fields = ['ep', 'tid', 'dir', 'hour', 'period', 'delta_t',
                  'base_hw', 'eff_hw', 's_hold_mean', 's_hold_std',
                  'hold_mean', 'hold_std', 'hold_max', 'hold_n',
                  'gap_dev', 'penalty', 'reward']

        with open(trip_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            if write_header:
                w.writeheader()
            for rec in self._ep_trip_records:
                rec['ep'] = ep
                w.writerow(rec)

        # Print per-period summary
        from collections import defaultdict
        by_period = defaultdict(list)
        for r in self._ep_trip_records:
            by_period[r['period']].append(r)

        print(f"  TRIPS    ep={ep}  n={len(self._ep_trip_records)}")
        for period in ['peak', 'off', 'trans']:
            recs = by_period.get(period, [])
            if not recs:
                continue
            deltas = [r['delta_t'] for r in recs]
            holds = [r['hold_mean'] for r in recs]
            pens = [r['penalty'] for r in recs]
            print(f"    {period:5s} n={len(recs):3d}  "
                  f"δ={np.mean(deltas):+5.1f}±{np.std(deltas):4.1f}  "
                  f"hold={np.mean(holds):+5.1f}±{np.std(holds):4.1f}  "
                  f"pen={np.mean(pens):.3f}")

            # Flag worst trips
            worst = sorted(recs, key=lambda r: -abs(r['hold_mean']))[:3]
            for w in worst:
                if abs(w['hold_mean']) > 20:
                    print(f"      ⚠ tid={w['tid']} h={w['hour']}:00 "
                          f"δ={w['delta_t']:+.0f}s hold_μ={w['hold_mean']:+.0f}s "
                          f"pen={w['penalty']:.2f}")

    # ────────────────── Train loop ──────────────────

    def maybe_resume(self):
        """Scan checkpoints/ for latest ep; if found, load networks and set resume_from_ep.
        Returns the ep to start from (0 if no resume, last_ep+1 otherwise).
        """
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            return 0
        import re
        eps = []
        for fn in os.listdir(ckpt_dir):
            m = re.match(r'lower_ep(\d+)\.pt$', fn)
            if m and os.path.exists(os.path.join(ckpt_dir, f'upper_ep{m.group(1)}.pt')):
                eps.append(int(m.group(1)))
        if not eps:
            return 0
        last_ep = max(eps)
        try:
            self.lower_trainer.load(os.path.join(ckpt_dir, f'lower_ep{last_ep}.pt'))
            self.upper_trainer.load(os.path.join(ckpt_dir, f'upper_ep{last_ep}.pt'))
        except Exception as e:
            print(f"  [Resume] Failed to load ep{last_ep} checkpoint: {e}. Starting fresh.")
            return 0
        self.resume_from_ep = last_ep + 1
        print(f"  [Resume] Loaded checkpoint ep{last_ep}. Resuming from ep{self.resume_from_ep}.")
        return self.resume_from_ep

    def train(self, total_episodes=300):
        diag_freq = self.cfg.get('training', {}).get('diag_freq', 10)
        trip_dump_freq = self.cfg.get('training', {}).get('trip_dump_freq', 25)
        # Init diag now (after resume decision so CSV header handled correctly)
        if self.diag is None:
            self.diag = DiagnosticLog(self.log_dir, resume=(self.resume_from_ep > 0))

        print(f"TransitDuet v2 | eps={total_episodes} | warmup={self.upper_warmup} | "
              f"δ∈[-{self.delta_max},+{self.delta_max}] | α_hold={self.alpha_holding} | "
              f"dev={self.device}")
        print(f"  Lower: state={self.env.state_dim}  K={self.lower_trainer.ensemble_size}  "
              f"batch={self.batch_size}  updates/ep={self.updates_per_episode}")
        print(f"  Upper: state={self.upper_state_dim}  K={self.upper_trainer.ensemble_size}  "
              f"batch={self.upper_batch_size}  updates/ep={self.upper_updates}")
        print(f"  Diag CSV: {self.diag.csv_path}")
        print("=" * 90)

        for ep in range(self.resume_from_ep, total_episodes):
            row = self.run_episode(ep, training=True)

            # ── Compact per-episode line ──
            if ep % 5 == 0 or ep < 5:
                line = (f"[{ep:3d}] {row['stage']:7s} N={row.get('N_fleet',12):2d} | "
                        f"w={row['avg_wait_min']:4.1f} f={row['peak_fleet']:2d} "
                        f"cv={row['headway_cv']:.2f} | "
                        f"Lπ a={row['lower_action_mean']:+5.1f}±{row['lower_action_std']:4.1f} "
                        f"Q={row['lower_q_mean']:+6.1f} λ={row['lower_lambda']:.2f} | ")
                if row['stage'] == 'BiLevel':
                    line += (f"Uδ={row['upper_delta_mean']:+5.1f}±{row['upper_delta_std']:4.1f} "
                             f"Q={row['upper_q_mean']:+6.3f} | "
                             f"h_μ={row['hold_fb_mean']:+5.1f} "
                             f"pen={row['hold_penalty_mean']:.2f} | ")
                line += f"{row['wall_env_s']:.0f}+{row['wall_train_s']:.0f}s"
                print(line)

            # ── Detailed diagnostic block ──
            if ep % diag_freq == 0 or ep == total_episodes - 1:
                self._print_diagnostic_block(row)

            # ── Per-trip breakdown ──
            if ep % trip_dump_freq == 0 and row['stage'] == 'BiLevel':
                self._dump_trip_breakdown(ep)

            # ── Checkpoint ──
            if (ep + 1) % 50 == 0:
                self._save_checkpoint(ep)

        self._save_checkpoint(total_episodes - 1)
        self.diag.save_json()
        self._save_history()
        print(f"\nDone. Results in {self.log_dir}/")

    def _save_checkpoint(self, ep):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.lower_trainer.save(os.path.join(ckpt_dir, f'lower_ep{ep}.pt'))
        self.upper_trainer.save(os.path.join(ckpt_dir, f'upper_ep{ep}.pt'))
        print(f"  [Checkpoint ep {ep}]")

    def _save_history(self):
        results = {}
        for key, values in self.history.items():
            try:
                results[key] = [float(x) if not isinstance(x, list) else x
                                for x in values]
            except (TypeError, ValueError):
                results[key] = [str(x) for x in values]
        with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
            json.dump(results, f)


def eval_pareto_frontier(runner, n_eval=10, fleet_values=None):
    """v2k: Sweep N_fleet values and record (fleet, wait, cv) Pareto points."""
    if fleet_values is None:
        fleet_values = list(range(8, 17))
    results = []
    for N in fleet_values:
        waits, cvs, overshoots = [], [], []
        for i in range(n_eval):
            row = runner.run_episode(ep=9999, training=False, N_fleet_override=N)
            waits.append(row['avg_wait_min'])
            cvs.append(row['headway_cv'])
            overshoots.append(row.get('fleet_overshoot', 0))
        results.append({
            'N_fleet': N,
            'wait_mean': float(np.mean(waits)),
            'wait_std': float(np.std(waits)),
            'cv_mean': float(np.mean(cvs)),
            'cv_std': float(np.std(cvs)),
            'overshoot_mean': float(np.mean(overshoots)),
        })
        print(f"  N_fleet={N:2d}: wait={np.mean(waits):4.1f}±{np.std(waits):.1f}m  "
              f"cv={np.mean(cvs):.2f}  overshoot={np.mean(overshoots):.1f}")
    return results


def main():
    parser = argparse.ArgumentParser(description='TransitDuet v2')
    parser.add_argument('--config', type=str, default='config_v2.yaml')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--eval_pareto', action='store_true',
                        help='After training, evaluate Pareto frontier over N_fleet ∈ [8,16]')
    parser.add_argument('--n_eval', type=int, default=5, help='eps per N_fleet for eval')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                        help='Resume from latest checkpoint if found (default: on)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start from scratch even if checkpoints exist')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_path = os.path.join(str(SCRIPT_DIR), args.config)
    config = load_config(config_path)
    config['seed'] = args.seed

    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'

    runner = TransitDuetV2Runner(config, device=device)
    if args.resume:
        runner.maybe_resume()
    runner.train(total_episodes=args.episodes)

    if args.eval_pareto:
        print("\n" + "="*80)
        print("  PARETO FRONTIER EVALUATION")
        print("="*80)
        results = eval_pareto_frontier(runner, n_eval=args.n_eval)
        with open(os.path.join(runner.log_dir, 'pareto_frontier.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {runner.log_dir}/pareto_frontier.json")


if __name__ == '__main__':
    main()
