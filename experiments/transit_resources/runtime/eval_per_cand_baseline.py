"""
eval_per_cand_baseline.py
=========================
Held-out evaluation of a per-candidate-retrained baseline result. Loads the
saved best_lower.pt and the best-candidate timetable triple from
logs/baseline_per_cand_ga_seed*/, then runs n_eval deterministic episodes
under the unified protocol so the result is comparable to other Table 1 rows.
"""

import sys, os, argparse, json
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from lower.resac_lagrangian import RESACLagrangianTrainer
from run_baseline_per_candidate import make_upper_callback, run_episode, composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_eval', type=int, default=20)
    ap.add_argument('--triple', nargs=3, type=float,
                    default=[485.8, 323.9, 288.5],
                    help='best candidate triple (H_peak, H_off, H_trans)')
    ap.add_argument('--ckpt',
                    default='logs/baseline_per_cand_ga_seed42/best_lower.pt')
    ap.add_argument('--fleet_min', type=int, default=8)
    ap.add_argument('--fleet_max', type=int, default=16)
    ap.add_argument('--demand_noise', type=float, default=0.15)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--eval_seed', type=int, default=12345)
    args = ap.parse_args()

    device = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'
    np.random.seed(args.eval_seed)

    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = args.demand_noise

    state_dim = env.state_dim
    lower = RESACLagrangianTrainer(
        state_dim=state_dim, action_dim=1, hidden_dim=64,
        action_range=60.0, cost_limit=0.5,
        ensemble_size=10, beta=-2.0, lr=3e-4,
        lambda_lr=1e-3, gamma=0.99, soft_tau=0.005,
        auto_entropy=True, maximum_alpha=0.3, device=device)
    ckpt = SCRIPT_DIR / args.ckpt
    lower.load(str(ckpt))
    print(f"Loaded {ckpt}")

    triple = tuple(args.triple)
    env._upper_policy_callback = make_upper_callback(triple)

    waits, cvs, overs, comps = [], [], [], []
    for ep in range(args.n_eval):
        n = int(np.random.randint(args.fleet_min, args.fleet_max + 1))
        env._n_fleet_target = n
        z = run_episode(env, lower, replay_buffer=None, training=False, device=device)
        waits.append(float(z[0])); cvs.append(float(z[2]))
        overs.append(max(0, float(z[1]) - n))
        comps.append(composite(z, n))
        print(f'  ep{ep:2d}  N={n:2d}  wait={z[0]:5.2f}  cv={z[2]:.3f}  over={overs[-1]:.1f}  comp={comps[-1]:.3f}')

    print(f"\nPer-candidate-retrain GA held-out (n={args.n_eval}, triple={triple}):")
    print(f"  wait      = {np.mean(waits):.2f} ± {np.std(waits):.2f}")
    print(f"  cv        = {np.mean(cvs):.3f} ± {np.std(cvs):.3f}")
    print(f"  overshoot = {np.mean(overs):.2f} ± {np.std(overs):.2f}")
    print(f"  composite = {np.mean(comps):.3f} ± {np.std(comps):.3f}")

    log_dir = SCRIPT_DIR / 'logs' / f'baseline_per_cand_ga_seed{args.seed}'
    out = {
        'triple': list(triple),
        'n_eval': args.n_eval,
        'wait_mean': float(np.mean(waits)), 'wait_std': float(np.std(waits)),
        'cv_mean': float(np.mean(cvs)), 'cv_std': float(np.std(cvs)),
        'overshoot_mean': float(np.mean(overs)), 'overshoot_std': float(np.std(overs)),
        'composite_mean': float(np.mean(comps)), 'composite_std': float(np.std(comps)),
    }
    (log_dir / 'eval_holdout.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
