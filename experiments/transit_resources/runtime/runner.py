"""
runner.py
=========
TransitDuet main training loop — full 3-stage bi-level training.

Both levels use RE-SAC:
  - Upper: RE-SAC (no Lagrangian) with per-dispatch proxy reward
  - Lower: RE-SAC Lagrangian with headway cost constraint

Stage 1 (warmup):  β=0, only lower trains, fixed target_headway=360s
Stage 2 (ramp):    β ramps 0→0.5, upper activates, TAP couples layers
Stage 3 (full):    β=0.5, full coupling

Usage:
    cd transit_duet/
    python -u runner.py [--config config.yaml] [--episodes 300] [--gpu]
"""

import os
import sys
import argparse
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
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from env.sim import env_bus
from upper.resac_upper import RESACUpperTrainer
from upper.measurement_proj import MeasurementProjection
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer
from coupling.tap import TAPManager
from coupling.beta_schedule import BetaSchedule


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class TransitDuetRunner:
    """Main training loop for bi-level bus control."""

    def __init__(self, config, device='cpu'):
        self.cfg = config
        self.device = device

        # Environment
        env_path = os.path.join(str(SCRIPT_DIR), config['env']['path'])
        self.env = env_bus(env_path, route_sigma=config['env']['route_sigma'])
        self.env.enable_plot = False
        self.env._n_fleet_target = config['upper']['N_fleet']

        state_dim = self.env.state_dim

        # ── Upper policy: RE-SAC (no Lagrangian) ──
        self.upper_trainer = RESACUpperTrainer(
            state_dim=config['upper']['state_dim'],
            action_dim=config['upper']['action_dim'],
            hidden_dim=config['upper']['hidden_dim'],
            action_low=config['upper']['action_low'],
            action_high=config['upper']['action_high'],
            ensemble_size=config['upper'].get('ensemble_size', 10),
            beta=config['upper'].get('resac_beta', -2.0),
            lr=config['upper']['lr'],
            gamma=config['upper'].get('gamma', 0.95),  # shorter horizon for planning
            replay_capacity=config['upper'].get('replay_capacity', 50000),
            device=device,
        )

        # ── Lower policy: RE-SAC Lagrangian ──
        self.replay_buffer = CostReplayBuffer(config['training']['replay_buffer_size'])
        self.lower_trainer = RESACLagrangianTrainer(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=config['lower']['hidden_dim'],
            action_range=config['lower']['action_range'],
            cost_limit=config['lower']['cost_limit'],
            ensemble_size=config['lower'].get('ensemble_size', 10),
            beta=config['lower'].get('resac_beta', -2.0),
            beta_ood=config['lower'].get('beta_ood', 0.01),
            weight_reg=config['lower'].get('weight_reg', 0.01),
            lr=config['lower']['lr'],
            lambda_lr=config['lower']['lambda_lr'],
            gamma=config['lower']['gamma'],
            soft_tau=config['lower']['soft_tau'],
            auto_entropy=config['lower']['auto_entropy'],
            maximum_alpha=config['lower']['maximum_alpha'],
            device=device,
        )

        # ApproPO measurement projection
        self.measurement_proj = MeasurementProjection(
            N_fleet=config['upper']['N_fleet'],
            lr=config['coupling']['measurement_lr'],
        )

        # TAP coupling
        self.beta_schedule = BetaSchedule(
            warmup_eps=config['coupling']['beta_warmup_eps'],
            ramp_eps=config['coupling']['beta_ramp_eps'],
            beta_max=config['coupling']['beta_max'],
        )
        self.tap = TAPManager(self.beta_schedule)

        # Training params
        self.batch_size = config['lower']['batch_size']
        self.updates_per_episode = config['lower'].get('updates_per_episode', 30)
        self.upper_batch_size = config['upper'].get('batch_size', 64)
        self.upper_updates = config['upper'].get('updates_per_episode', 10)

        # TAP reverse signal from previous episode
        self._prev_upper_reward_per_trip = {}

        # Track upper transitions within episode for replay buffer
        self._episode_upper_transitions = []
        self._prev_upper_state = None

        # Logging
        self.log_dir = os.path.join(str(SCRIPT_DIR), 'logs', 'full_training')
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = defaultdict(list)

    def _upper_callback(self, s_upper, trip):
        """Called by env.step() at each dispatch event."""
        action = self.upper_trainer.policy_net.get_action(
            s_upper, deterministic=False)
        trip_id = trip.launch_turn

        # Record for TAP
        self.tap.record_upper_transition(s_upper, action, trip_id)

        # Store transition: when next dispatch happens, we get (s, a, r, s')
        dispatch_rewards = getattr(self.env, '_dispatch_rewards', {})
        if self._prev_upper_state is not None:
            prev_s, prev_a, prev_tid = self._prev_upper_state
            r = dispatch_rewards.get(prev_tid, 0.0)
            # TAP forward: augment with lower rewards from this trip
            beta = self.beta_schedule.get_beta(self._current_ep)
            if beta > 0:
                lower_rs = self.tap._lower_rewards_by_trip.get(prev_tid, [])
                if lower_rs:
                    r += beta * np.mean(lower_rs)
            self._episode_upper_transitions.append(
                (prev_s, prev_a, r, s_upper.copy(), False))

        self._prev_upper_state = (s_upper.copy(), action.copy(), trip_id)

        # Select headway based on time-of-day
        hour = 6 + trip.launch_time // 3600
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            target_hw = action[0]
        elif 9 < hour < 17:
            target_hw = action[1]
        else:
            target_hw = action[2]

        return float(target_hw)

    def run_episode(self, ep, training=True):
        """Run one complete episode."""
        t0 = time.time()
        self.tap.clear()
        self.env.reset()
        self._current_ep = ep
        self._episode_upper_transitions = []
        self._prev_upper_state = None

        beta = self.beta_schedule.get_beta(ep)

        # Register upper callback only after warmup
        if ep > self.beta_schedule.warmup and training:
            self.env._upper_policy_callback = self._upper_callback
        else:
            self.env._upper_policy_callback = None

        # Initialize
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
                        trip_id = int(state_dict[key][0][0])

                        if training:
                            self.replay_buffer.push(
                                state, action_dict[key], reward, cost,
                                next_state, False, trip_id)
                            self.tap.record_lower_reward(reward, trip_id)

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

        # ──── Episode-end: finalize last upper transition ────
        if self._prev_upper_state is not None:
            prev_s, prev_a, prev_tid = self._prev_upper_state
            dispatch_rewards = getattr(self.env, '_dispatch_rewards', {})
            r = dispatch_rewards.get(prev_tid, 0.0)
            # Terminal transition — use s as next_state (doesn't matter much)
            self._episode_upper_transitions.append(
                (prev_s, prev_a, r, prev_s, True))

        # ──── Episode-end training ────
        t1 = time.time()
        train_metrics = {}
        upper_metrics = {}

        # Lower policy training
        if training and len(self.replay_buffer) > self.batch_size:
            tap_signal = None
            if beta > 0 and self._prev_upper_reward_per_trip:
                tap_signal = {tid: beta * r for tid, r in
                              self._prev_upper_reward_per_trip.items()}

            for _ in range(self.updates_per_episode):
                train_metrics = self.lower_trainer.update(
                    self.replay_buffer, self.batch_size,
                    reward_scale=1.0, tap_signal=tap_signal)

        # Upper policy training (only after warmup)
        if ep > self.beta_schedule.warmup and training:
            # Push episode's upper transitions into upper replay buffer
            for (s, a, r, ns, d) in self._episode_upper_transitions:
                self.upper_trainer.replay_buffer.push(s, a, r, ns, d)

            # Train upper RE-SAC
            if len(self.upper_trainer.replay_buffer) > self.upper_batch_size:
                for _ in range(self.upper_updates):
                    upper_metrics = self.upper_trainer.update(self.upper_batch_size)

            # Store per-trip rewards for next episode's TAP reverse
            dispatch_rewards = getattr(self.env, '_dispatch_rewards', {})
            self._prev_upper_reward_per_trip = dict(dispatch_rewards)

        # Measurement projection update
        z = self.env.measurement_vector
        self.measurement_proj.update(z)
        r_upper = self.measurement_proj.compute_upper_reward(z)

        train_time = time.time() - t1

        # ──── Logging ────
        avg_r = episode_reward / max(episode_steps, 1)
        q_mean = train_metrics.get('q_mean', 0)
        q_std = train_metrics.get('q_std', 0)
        uq_mean = upper_metrics.get('upper_q_mean', 0)
        uq_std = upper_metrics.get('upper_q_std', 0)

        self.history['ep_reward'].append(episode_reward)
        self.history['avg_reward'].append(avg_r)
        self.history['ep_cost'].append(episode_cost / max(episode_steps, 1))
        self.history['r_upper'].append(r_upper)
        self.history['avg_wait'].append(float(z[0]))
        self.history['peak_fleet'].append(float(z[1]))
        self.history['headway_cv'].append(float(z[2]))
        self.history['lambda'].append(self.lower_trainer.lambda_param)
        self.history['alpha'].append(self.lower_trainer.alpha)
        self.history['beta'].append(beta)
        self.history['q_mean'].append(q_mean)
        self.history['q_std'].append(q_std)
        self.history['upper_q_mean'].append(uq_mean)
        self.history['upper_q_std'].append(uq_std)
        self.history['n_upper_trans'].append(len(self._episode_upper_transitions))
        self.history['theta_weights'].append(
            self.measurement_proj.get_reward_weights().tolist())

        return {
            'ep': ep, 'reward': episode_reward, 'avg_r': avg_r,
            'r_upper': r_upper, 'steps': episode_steps,
            'z': z.tolist(), 'lambda': self.lower_trainer.lambda_param,
            'beta': beta, 'stage': self.beta_schedule.stage_name(ep),
            'q_mean': q_mean, 'q_std': q_std,
            'uq_mean': uq_mean, 'uq_std': uq_std,
            'n_upper': len(self._episode_upper_transitions),
            'env_time': env_time, 'train_time': train_time,
        }

    def train(self, total_episodes=None):
        """Full 3-stage training loop."""
        if total_episodes is None:
            total_episodes = self.cfg['training']['total_episodes']

        print(f"TransitDuet Full Training: {total_episodes} episodes")
        print(f"  Stage 1 warmup: ep 0-{self.beta_schedule.warmup}")
        print(f"  Stage 2 ramp:   ep {self.beta_schedule.warmup}-"
              f"{self.beta_schedule.warmup + self.beta_schedule.ramp}")
        print(f"  Stage 3 full:   ep {self.beta_schedule.warmup + self.beta_schedule.ramp}+")
        print(f"  Device: {self.device}, state_dim: {self.env.state_dim}")
        print(f"  Upper: RE-SAC (K={self.upper_trainer.ensemble_size})")
        print(f"  Lower: RE-SAC Lagrangian (K={self.lower_trainer.ensemble_size})")
        print("=" * 90)

        for ep in range(total_episodes):
            info = self.run_episode(ep, training=True)

            if ep % 5 == 0 or ep < 5:
                print(f"[Ep {ep:3d}] {info['stage']:15s} | "
                      f"R={info['avg_r']:6.3f} R_up={info['r_upper']:6.1f} | "
                      f"w={info['z'][0]:3.1f}m f={info['z'][1]:2.0f} "
                      f"cv={info['z'][2]:.2f} | "
                      f"Ql={info['q_mean']:.1f}±{info['q_std']:.1f} "
                      f"Qu={info['uq_mean']:.2f}±{info['uq_std']:.2f} | "
                      f"λ={info['lambda']:.2f} β={info['beta']:.2f} | "
                      f"{info['env_time']:.0f}+{info['train_time']:.0f}s")

            if (ep + 1) % self.cfg['training']['save_freq'] == 0:
                self._save_checkpoint(ep)

        self._save_checkpoint(total_episodes - 1)
        self._save_history()
        print("Training complete.")

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


def main():
    parser = argparse.ArgumentParser(description='TransitDuet Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    config_path = os.path.join(str(SCRIPT_DIR), args.config)
    config = load_config(config_path)

    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'

    runner = TransitDuetRunner(config, device=device)
    episodes = args.episodes or config['training']['total_episodes']
    runner.train(total_episodes=episodes)


if __name__ == '__main__':
    main()
