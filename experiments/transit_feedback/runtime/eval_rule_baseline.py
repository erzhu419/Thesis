"""
eval_rule_baseline.py
=====================
Held-out evaluation of a rule-baseline result, matching the protocol used by
scripts/per_ckpt_eval.py (n_eval episodes per fleet bin in [8, 16],
deterministic) so the resulting numbers are directly comparable to
TransitDuet / Fixed / GA / CMA-ES rows in the main table.
"""

import sys, os, argparse, json, time
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from run_baseline_rule import (
    run_episode, composite, mpc_plan, hour_to_slot,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True,
                    choices=['rule_fixed', 'rule_daganzo', 'rule_xuan',
                             'rule_ga', 'rule_cmaes', 'rule_mpc'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_eval', type=int, default=20)
    ap.add_argument('--fleet_min', type=int, default=8)
    ap.add_argument('--fleet_max', type=int, default=16)
    ap.add_argument('--demand_noise', type=float, default=0.15)
    ap.add_argument('--eval_seed', type=int, default=12345)
    args = ap.parse_args()

    log_dir = SCRIPT_DIR / 'logs' / f'baseline_{args.variant}_seed{args.seed}'
    best = json.load(open(log_dir / 'best.json'))
    lower_rule = 'proportional'
    if args.variant == 'rule_fixed':
        params = (360.0, 360.0, 360.0)
    elif args.variant == 'rule_daganzo':
        params = (360.0, 360.0, 360.0)
        lower_rule = 'daganzo'
    elif args.variant == 'rule_xuan':
        params = (360.0, 360.0, 360.0)
        lower_rule = 'xuan'
    elif args.variant == 'rule_mpc':
        # MPC re-plans each dispatch: build a per-dispatch callable
        mpc_candidates = []
        for hp in [240, 300, 360, 420, 480]:
            for ho in [360, 480, 600, 720]:
                for ht in [300, 360, 420]:
                    mpc_candidates.append((hp, ho, ht))
    else:
        params = tuple(best['params'])

    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = args.demand_noise

    np.random.seed(args.eval_seed)
    waits, cvs, overs, comps = [], [], [], []
    metric_keys = [
        'avg_holding_sec', 'total_holding_sec',
        'avg_onboard_time_min', 'avg_total_passenger_time_min',
        'n_arrived_passengers',
        'planned_dispatch_headway_mean', 'planned_dispatch_headway_std',
        'planned_dispatch_headway_cv',
        'actual_dispatch_headway_mean', 'actual_dispatch_headway_std',
        'actual_dispatch_headway_cv',
        'planned_shift_mean', 'planned_shift_std',
        'dispatch_lateness_mean', 'dispatch_lateness_max',
    ]
    metric_series = {k: [] for k in metric_keys}
    for ep in range(args.n_eval):
        n = int(np.random.randint(args.fleet_min, args.fleet_max + 1))
        env._n_fleet_target = n
        if args.variant == 'rule_mpc':
            demand_proxy = float(np.clip(np.random.normal(1.0, args.demand_noise), 0.3, 2.0))
            chosen = mpc_plan(current_hour=12, last_dispatch_time_per_dir=None,
                              episode_budget_n=n, current_demand_proxy=demand_proxy,
                              candidates=mpc_candidates)
            z, metrics = run_episode(env, chosen)
        else:
            z, metrics = run_episode(env, params, lower_rule=lower_rule)
        waits.append(float(z[0])); cvs.append(float(z[2]))
        overs.append(max(0, float(z[1]) - n))
        comps.append(composite(z, n))
        for key in metric_keys:
            metric_series[key].append(float(metrics.get(key, 0.0)))

    print(f"\n{args.variant} held-out (n={args.n_eval}, params={params if args.variant != 'rule_mpc' else 'MPC re-plan'}):")
    print(f"  wait      = {np.mean(waits):.2f} ± {np.std(waits):.2f}")
    print(f"  cv        = {np.mean(cvs):.3f} ± {np.std(cvs):.3f}")
    print(f"  overshoot = {np.mean(overs):.2f} ± {np.std(overs):.2f}")
    print(f"  composite = {np.mean(comps):.3f} ± {np.std(comps):.3f}")

    out = {
        'variant': args.variant,
        'seed': args.seed,
        'params': params if args.variant != 'rule_mpc' else 'MPC',
        'n_eval': args.n_eval,
        'wait_mean': float(np.mean(waits)), 'wait_std': float(np.std(waits)),
        'cv_mean': float(np.mean(cvs)), 'cv_std': float(np.std(cvs)),
        'overshoot_mean': float(np.mean(overs)), 'overshoot_std': float(np.std(overs)),
        'composite_mean': float(np.mean(comps)), 'composite_std': float(np.std(comps)),
    }
    for key, vals in metric_series.items():
        out[f'{key}_mean'] = float(np.mean(vals))
        out[f'{key}_std'] = float(np.std(vals))
    (log_dir / 'eval_holdout.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
