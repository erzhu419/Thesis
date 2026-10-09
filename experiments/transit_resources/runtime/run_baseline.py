"""
run_baseline.py
===============
TransitDuet baseline experiment: lower-only (Stage 1) convergence test.

Uses RE-SAC (Robust Ensemble SAC) with Lagrangian cost constraint.
No upper policy — fixed timetable with target_headway=360s.

Usage:
    cd transit_duet/
    python -u run_baseline.py --episodes 100 [--seed 42] [--gpu]
"""

import sys, os, time, argparse, json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer


def run_baseline(args):
    device = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    # ---- Environment ----
    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    state_dim = env.state_dim  # 29

    # ---- Lower policy (RE-SAC Lagrangian) ----
    trainer = RESACLagrangianTrainer(
        state_dim=state_dim,
        action_dim=1,
        hidden_dim=64,
        action_range=60.0,
        cost_limit=0.15,
        ensemble_size=10,
        beta=-2.0,             # pessimistic (RE-SAC core)
        beta_ood=0.01,         # OOD regularization
        weight_reg=0.01,       # L1 critic regularization
        lr=3e-4,
        lambda_lr=1e-3,
        gamma=0.99,
        soft_tau=0.005,
        auto_entropy=True,
        maximum_alpha=0.3,
        device=device,
    )
    replay_buffer = CostReplayBuffer(500_000)

    batch_size = 512
    updates_per_episode = 30   # more updates per ep for ensemble
    warmup_episodes = 3        # collect data before training

    # ---- Logging ----
    log_dir = SCRIPT_DIR / 'logs' / f'resac_baseline_seed{args.seed}'
    os.makedirs(log_dir, exist_ok=True)
    history = defaultdict(list)

    print(f"TransitDuet RE-SAC Baseline: {args.episodes} episodes, seed={args.seed}")
    print(f"  state_dim={state_dim}, device={device}")
    print(f"  ensemble=10, beta=-2.0, hidden=64, lr=3e-4")
    print(f"  batch={batch_size}, updates/ep={updates_per_episode}")
    print("=" * 70)

    for ep in range(args.episodes):
        t0 = time.time()
        env.reset()
        state_dict, reward_dict, _ = env.initialize_state()
        action_dict = {k: None for k in range(env.max_agent_num)}

        ep_reward = 0.0
        ep_cost = 0.0
        ep_steps = 0

        while not env.done:
            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        obs = np.array(state_dict[key][0], dtype=np.float32)
                        a = trainer.policy_net.get_action(
                            torch.from_numpy(obs).float().to(device),
                            deterministic=False)
                        action_dict[key] = a

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        state = np.array(state_dict[key][0], dtype=np.float32)
                        next_state = np.array(state_dict[key][1], dtype=np.float32)
                        reward = reward_dict[key]
                        cost = env.cost.get(key, 0.0)
                        trip_id = int(state_dict[key][0][0])

                        replay_buffer.push(
                            state, action_dict[key], reward, cost,
                            next_state, False, trip_id)

                        ep_reward += reward
                        ep_cost += cost
                        ep_steps += 1

                    state_dict[key] = state_dict[key][1:]
                    obs = np.array(state_dict[key][0], dtype=np.float32)
                    action_dict[key] = trainer.policy_net.get_action(
                        torch.from_numpy(obs).float().to(device),
                        deterministic=False)

            state_dict, reward_dict, cost_dict, done = env.step(
                action_dict, render=False)

        env_time = time.time() - t0

        # ---- Episode-end batch training ----
        train_time = 0.0
        train_metrics = {}
        if ep >= warmup_episodes and len(replay_buffer) > batch_size:
            t1 = time.time()
            for _ in range(updates_per_episode):
                train_metrics = trainer.update(
                    replay_buffer, batch_size, reward_scale=1.0)
            train_time = time.time() - t1

        # ---- Collect measurements ----
        z = env.measurement_vector  # [avg_wait_min, peak_fleet, headway_cv]

        # ---- Logging ----
        avg_reward = ep_reward / max(ep_steps, 1)
        history['ep_reward'].append(ep_reward)
        history['avg_reward'].append(avg_reward)
        history['ep_cost'].append(ep_cost / max(ep_steps, 1))
        history['avg_wait'].append(float(z[0]))
        history['peak_fleet'].append(float(z[1]))
        history['headway_cv'].append(float(z[2]))
        history['lambda'].append(trainer.lambda_param)
        history['alpha'].append(trainer.alpha)
        history['q_mean'].append(train_metrics.get('q_mean', 0))
        history['q_std'].append(train_metrics.get('q_std', 0))
        history['ep_steps'].append(ep_steps)

        q_mean = train_metrics.get('q_mean', 0)
        q_std = train_metrics.get('q_std', 0)
        cost_q = train_metrics.get('cost_q_mean', 0)

        print(f"[Ep {ep:3d}] R={ep_reward:8.1f} | avg_r={avg_reward:6.2f} | "
              f"steps={ep_steps:4d} | "
              f"wait={z[0]:5.1f}m | fleet={z[1]:2.0f} | hw_cv={z[2]:.3f} | "
              f"λ={trainer.lambda_param:.3f} α={trainer.alpha:.3f} | "
              f"Q={q_mean:.1f}±{q_std:.1f} Qc={cost_q:.3f} | "
              f"env={env_time:.1f}s train={train_time:.1f}s")

    # ---- Save results ----
    results = {k: [float(x) for x in v] for k, v in history.items()}
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(results, f)

    trainer.save(str(log_dir / 'model_final.pt'))
    print(f"\nResults saved to {log_dir}")
    print(f"Final: avg_wait={history['avg_wait'][-1]:.1f}min, "
          f"hw_cv={history['headway_cv'][-1]:.3f}, "
          f"λ={history['lambda'][-1]:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    run_baseline(args)
