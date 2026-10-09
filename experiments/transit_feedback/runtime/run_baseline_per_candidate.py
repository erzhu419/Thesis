"""
run_baseline_per_candidate.py
=============================
Stronger GA/CMA-ES baseline that retrains a fresh lower SAC controller for
each candidate timetable in the search population, addressing the reviewer
concern that the shared-lower protocol used in the main paper may understate
search-based methods.

Per generation:
  for each candidate triple (H_peak, H_off, H_trans) in pop:
    1. Spawn a fresh RESAC lower (same hyperparams as A_full / Fixed lower)
    2. Train it for K=30 episodes against this static timetable
    3. Evaluate it for J=10 episodes (deterministic policy)
    4. Use mean composite cost as the candidate's fitness (lower = better)

We use a small population (8) and few generations (5), giving 40 total
candidates and ~3.5 h on a single 8 GB GPU. The output is the best-candidate's
trained lower + its triple, evaluated under the unified protocol.
"""

import sys, os, time, argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer
from upper.upper_ga import GAUpperPolicy
from upper.upper_cmaes import CMAESUpperPolicy


def hour_to_slot(hour):
    if (7 <= hour <= 9) or (17 <= hour <= 19):
        return 0
    if 9 < hour < 17:
        return 1
    return 2


def make_upper_callback(triple):
    peak, off_peak, trans = triple

    def upper_cb(s_upper, trip):
        trip._upper_queried = True
        hour = 6 + trip.launch_time // 3600
        return float([peak, off_peak, trans][hour_to_slot(hour)])

    return upper_cb


def run_episode(env, lower, replay_buffer, training, device):
    env.reset()
    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: None for k in range(env.max_agent_num)}
    ep_reward, ep_steps = 0.0, 0
    while not env.done:
        for key in state_dict:
            obs_list = state_dict[key]
            if len(obs_list) == 1:
                if action_dict[key] is None:
                    obs = np.array(obs_list[0], dtype=np.float32)
                    a = lower.policy_net.get_action(
                        torch.from_numpy(obs).float().to(device),
                        deterministic=not training)
                    action_dict[key] = a
            elif len(obs_list) == 2:
                if obs_list[0][1] != obs_list[1][1]:
                    s = np.array(obs_list[0], dtype=np.float32)
                    ns = np.array(obs_list[1], dtype=np.float32)
                    r = reward_dict[key]
                    c = env.cost.get(key, 0.0)
                    tid = int(obs_list[0][0])
                    if training and replay_buffer is not None:
                        replay_buffer.push(s, action_dict[key], r, c, ns, False, tid)
                    ep_reward += r
                    ep_steps += 1
                state_dict[key] = state_dict[key][1:]
                obs = np.array(state_dict[key][0], dtype=np.float32)
                action_dict[key] = lower.policy_net.get_action(
                    torch.from_numpy(obs).float().to(device),
                    deterministic=not training)
        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)
    return env.measurement_vector


def composite(z, n_fleet):
    over = max(0.0, z[1] - n_fleet)
    return z[0] / 10.0 + (over ** 2) / max(n_fleet, 1) + z[2]


def evaluate_candidate(triple, env, device,
                       train_eps=30, eval_eps=10, fleet_min=8, fleet_max=16,
                       warm_start_ckpt=None):
    state_dim = env.state_dim
    lower = RESACLagrangianTrainer(
        state_dim=state_dim, action_dim=1, hidden_dim=64,
        action_range=60.0, cost_limit=0.5,
        ensemble_size=10, beta=-2.0, lr=3e-4,
        lambda_lr=1e-3, gamma=0.99, soft_tau=0.005,
        auto_entropy=True, maximum_alpha=0.3, device=device)
    if warm_start_ckpt is not None:
        try:
            lower.load(str(warm_start_ckpt))
        except Exception as e:
            print(f"  [warm-start] failed to load {warm_start_ckpt}: {e}")
    replay_buffer = CostReplayBuffer(200_000)
    env._upper_policy_callback = make_upper_callback(triple)

    # Training (elastic fleet sampling, demand_noise already set on env)
    for ep in range(train_eps):
        n = int(np.random.randint(fleet_min, fleet_max + 1))
        env._n_fleet_target = n
        run_episode(env, lower, replay_buffer, training=True, device=device)
        if len(replay_buffer) > 512:
            for _ in range(10):
                lower.update(replay_buffer, batch_size=256, reward_scale=1.0)

    # Deterministic evaluation
    comp_vals = []
    for ep in range(eval_eps):
        n = int(np.random.randint(fleet_min, fleet_max + 1))
        env._n_fleet_target = n
        z = run_episode(env, lower, replay_buffer=None, training=False, device=device)
        comp_vals.append(composite(z, n))
    return float(np.mean(comp_vals)), float(np.std(comp_vals)), lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ga', 'cmaes'], default='ga')
    parser.add_argument('--pop_size', type=int, default=8)
    parser.add_argument('--gens', type=int, default=5)
    parser.add_argument('--train_eps', type=int, default=30)
    parser.add_argument('--eval_eps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--demand_noise', type=float, default=0.15)
    parser.add_argument('--fleet_min', type=int, default=8)
    parser.add_argument('--fleet_max', type=int, default=16)
    parser.add_argument('--warm_start',
                        default='logs/H_hiro_seed42/checkpoints/lower_ep299.pt',
                        help='Pretrained lower SAC ckpt to warm-start each candidate')
    args = parser.parse_args()

    device = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = args.demand_noise

    if args.method == 'ga':
        upper = GAUpperPolicy(action_low=[180., 300., 240.],
                              action_high=[600., 1200., 900.],
                              pop_size=args.pop_size, mutation_sigma=0.15)
    else:
        upper = CMAESUpperPolicy(action_low=[180., 300., 240.],
                                 action_high=[600., 1200., 900.],
                                 pop_size=args.pop_size, sigma0=0.3)

    log_dir = SCRIPT_DIR / 'logs' / f'baseline_per_cand_{args.method}_seed{args.seed}'
    log_dir.mkdir(parents=True, exist_ok=True)
    history = defaultdict(list)
    best = {'composite': float('inf'), 'triple': None, 'gen': -1, 'cand': -1}

    print(f"Per-candidate retrain {args.method.upper()}: pop={args.pop_size}, "
          f"gens={args.gens}, train_eps={args.train_eps}, eval_eps={args.eval_eps}, "
          f"device={device}")

    for gen in range(args.gens):
        for cand in range(args.pop_size):
            t0 = time.time()
            triple = upper.suggest()
            triple = tuple(float(x) for x in triple)
            warm_start = args.warm_start if args.warm_start else None
            if warm_start and not Path(warm_start).is_absolute():
                warm_start = SCRIPT_DIR / warm_start
            mean_comp, std_comp, lower = evaluate_candidate(
                triple, env, device,
                train_eps=args.train_eps, eval_eps=args.eval_eps,
                fleet_min=args.fleet_min, fleet_max=args.fleet_max,
                warm_start_ckpt=warm_start)
            upper.report(-mean_comp)  # higher = better
            wall_s = round(time.time() - t0, 1)
            history['gen'].append(int(gen))
            history['cand'].append(int(cand))
            history['triple'].append([float(x) for x in triple])
            history['mean_composite'].append(mean_comp)
            history['std_composite'].append(std_comp)
            history['wall_s'].append(wall_s)
            print(f"  gen{gen} cand{cand}  H={tuple(round(x,1) for x in triple)}  "
                  f"comp={mean_comp:.3f}±{std_comp:.3f}  ({wall_s}s)")
            if mean_comp < best['composite']:
                best = {'composite': mean_comp, 'std': std_comp,
                        'triple': list(triple), 'gen': int(gen), 'cand': int(cand)}
                lower.save(str(log_dir / 'best_lower.pt'))

    (log_dir / 'history.json').write_text(json.dumps(history, indent=1))
    (log_dir / 'best.json').write_text(json.dumps(best, indent=1))
    print(f"\nBest: gen{best['gen']} cand{best['cand']}  H={best['triple']}  "
          f"comp={best['composite']:.3f}±{best['std']:.3f}")


if __name__ == '__main__':
    main()
