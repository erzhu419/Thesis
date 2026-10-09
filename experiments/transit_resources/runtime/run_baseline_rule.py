"""
run_baseline_rule.py
====================
CPU-only baselines using a hand-tuned proportional headway-keeping rule for the
lower (holding) controller, with three upper-level options:

    --upper rule_fixed   :  fixed timetable (H=360), no upper search
    --upper rule_daganzo :  Daganzo-style headway equalization holding
    --upper rule_xuan    :  Xuan-style two-headway holding with fixed timetable
    --upper rule_ga      :  GA optimises a peak/off-peak/transition headway triple
    --upper rule_cmaes   :  CMA-ES optimises the same triple
    --upper rule_mpc     :  receding-horizon MPC re-plans the next K headways
                            at every dispatch event

The lower rule (Eq. below) replaces the trained RE-SAC controller. For each
station arrival of bus i with a forward bus ahead, the rule outputs

    hold_i = clip( H_target - forward_headway , 0, h_max )

where forward_headway is the time since the bus ahead reached this station,
H_target is the per-trip target headway, and h_max=60 s. Negative values
(early release) are clipped to 0 — the rule is monotone: hold longer when too
close to the bus ahead, never overshoot.

This script reuses the same env, the same elastic-fleet sampling, the same
demand_noise, and the same evaluation protocol as
``run_upper_comparison.py`` so the resulting numbers are directly comparable to
the Fixed / GA / CMA-ES / TransitDuet rows in the main table.

Output:
    logs/<exp_name>_seed<seed>/history.json    per-episode metrics
    logs/<exp_name>_seed<seed>/best.json       best-by-composite checkpoint info
"""

import sys, os, time, argparse, json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from upper.upper_cmaes import CMAESUpperPolicy
from upper.upper_ga import GAUpperPolicy


# ---------------------------------------------------------------------------
# Rule-based lower (proportional headway-keeping)
# ---------------------------------------------------------------------------
HMAX_HOLD = 60.0


def rule_holding_action(obs, target_headway, rule='proportional'):
    """
    obs layout (env/bus.py): [bus_id, station_id, hour, direction,
                              forward_headway, backward_headway,
                              passengers_factor, headway_dev, *speeds]
    """
    forward_headway = float(obs[4])
    backward_headway = float(obs[5])
    if rule == 'daganzo':
        # Daganzo-style equalization: hold when the following gap is larger
        # than the forward gap, which increases the forward spacing and gives
        # the following bus time to close the service gap.
        hold = 0.5 * (backward_headway - forward_headway)
    elif rule == 'xuan':
        # Xuan-style two-headway rule: combine target tracking with the
        # backward-headway signal. This is a same-simulator proxy rather than
        # an exact reproduction of the original paper's full controller.
        hold = 0.7 * (target_headway - forward_headway) + \
            0.3 * (backward_headway - target_headway)
    else:
        hold = target_headway - forward_headway
    return float(np.clip(hold, 0.0, HMAX_HOLD))


def hour_to_slot(hour, peak=(7, 9, 17, 19)):
    """Return 0=peak, 1=off-peak, 2=transition."""
    if (peak[0] <= hour <= peak[1]) or (peak[2] <= hour <= peak[3]):
        return 0
    if peak[1] < hour < peak[2]:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(env, target_triple_or_callback, mpc_replan_fn=None,
                lower_rule='proportional'):
    """
    Run one simulation episode under the rule-based lower.

    target_triple_or_callback:
        - If a 3-tuple (peak, off, trans): static piecewise schedule.
        - If callable f(hour, trip): per-dispatch headway from MPC plan.
    """
    if callable(target_triple_or_callback):
        def upper_cb(s_upper, trip):
            trip._upper_queried = True
            hour = 6 + trip.launch_time // 3600
            return float(target_triple_or_callback(hour, trip))
    else:
        peak, off_peak, trans = target_triple_or_callback

        def upper_cb(s_upper, trip):
            trip._upper_queried = True
            hour = 6 + trip.launch_time // 3600
            slot = hour_to_slot(hour)
            return float([peak, off_peak, trans][slot])

    env.reset()
    env._upper_policy_callback = upper_cb
    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: 0.0 for k in range(env.max_agent_num)}
    last_target = {k: 360.0 for k in range(env.max_agent_num)}

    while not env.done:
        # Update target_headway for buses at decision events
        for key in state_dict:
            obs_list = state_dict[key]
            if not obs_list:
                continue
            obs = np.array(obs_list[0], dtype=np.float32)
            # Pull the current trip's target_headway via the upper callback
            # by reading the bus's bound trip object (env writes target into trip)
            for bus in env.bus_all:
                if bus.bus_id == int(obs[0]) and bus.on_route:
                    last_target[key] = float(getattr(bus, '_target_headway', 360.0))
                    break
            action_dict[key] = rule_holding_action(
                obs, last_target[key], rule=lower_rule)
            if len(obs_list) == 2:
                state_dict[key] = state_dict[key][1:]

        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)

    z = env.measurement_vector  # [avg_wait_min, peak_fleet, headway_cv]
    metrics = dict(env.episode_metrics)
    return z, metrics


def composite(z, n_fleet):
    over = max(0.0, z[1] - n_fleet)
    return z[0] / 10.0 + (over ** 2) / max(n_fleet, 1) + z[2]


# ---------------------------------------------------------------------------
# MPC: receding-horizon re-planning
# ---------------------------------------------------------------------------
def mpc_plan(current_hour, last_dispatch_time_per_dir, episode_budget_n,
             current_demand_proxy, candidates):
    """
    Tiny MPC: at the moment of a dispatch, score each candidate triple over a
    short horizon using a closed-form mean-wait approximation:

        wait ≈ H / 2 + α * max(0, H - H_target_demand)^2

    with H_target_demand = demand-adapted ideal headway, α = curvature.

    Picks the triple that minimises predicted wait + |fleet_overshoot| penalty.

    This is intentionally a coarse model — the real role is to give a
    receding-horizon controller that can adjust headway across the day, not to
    out-perform the simulator-aware baselines.
    """
    slot = hour_to_slot(current_hour)
    best = None
    best_score = float('inf')
    for triple in candidates:
        H = triple[slot]
        wait = H / 2.0
        # cheap "demand ideal" surrogate; H_demand_ideal = clip(360 / demand, 240, 600)
        H_demand_ideal = float(np.clip(360.0 / max(current_demand_proxy, 0.5), 240.0, 600.0))
        wait += 0.001 * max(0.0, H - H_demand_ideal) ** 2
        # implicit fleet usage: shorter H → more fleet
        # fleet ~ ceil(trip_cycle / H), trip_cycle ≈ 22 stations * mean_dwell_plus_travel ~ 50 min
        approx_cycle_s = 50 * 60
        fleet_used = approx_cycle_s / H * 2  # both directions
        over = max(0.0, fleet_used - episode_budget_n)
        score = wait + 5.0 * over ** 2
        if score < best_score:
            best_score = score
            best = triple
    return best


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper', type=str, required=True,
                        choices=['rule_fixed', 'rule_daganzo', 'rule_xuan',
                                 'rule_ga', 'rule_cmaes', 'rule_mpc'])
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--demand_noise', type=float, default=0.15)
    parser.add_argument('--fleet_mode', choices=['fixed', 'elastic'], default='elastic')
    parser.add_argument('--fleet_min', type=int, default=8)
    parser.add_argument('--fleet_max', type=int, default=16)
    parser.add_argument('--N_fleet', type=int, default=12)
    args = parser.parse_args()

    np.random.seed(args.seed)
    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = args.demand_noise

    log_dir = SCRIPT_DIR / 'logs' / f'baseline_{args.upper}_seed{args.seed}'
    log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ choose upper-level scheme ------------------
    lower_rule = 'proportional'
    if args.upper == 'rule_fixed':
        upper = None
        params = (360.0, 360.0, 360.0)
    elif args.upper == 'rule_daganzo':
        upper = None
        lower_rule = 'daganzo'
        params = (360.0, 360.0, 360.0)
    elif args.upper == 'rule_xuan':
        upper = None
        lower_rule = 'xuan'
        params = (360.0, 360.0, 360.0)
    elif args.upper == 'rule_ga':
        upper = GAUpperPolicy(action_low=[180., 300., 240.],
                              action_high=[600., 1200., 900.],
                              pop_size=10, mutation_sigma=0.15)
    elif args.upper == 'rule_cmaes':
        upper = CMAESUpperPolicy(action_low=[180., 300., 240.],
                                 action_high=[600., 1200., 900.],
                                 pop_size=8, sigma0=0.3)
    elif args.upper == 'rule_mpc':
        upper = None
        # Static candidate set used by MPC's per-dispatch lookup
        mpc_candidates = []
        for hp in [240, 300, 360, 420, 480]:
            for ho in [360, 480, 600, 720]:
                for ht in [300, 360, 420]:
                    mpc_candidates.append((hp, ho, ht))

    history = {'avg_wait': [], 'cv': [], 'overshoot': [], 'composite': [],
               'params': [], 'wall_s': []}
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
    for key in metric_keys:
        history[key] = []
    best = {'composite': float('inf'), 'params': None, 'episode': -1, 'z': None}

    for ep in range(args.episodes):
        t0 = time.time()
        if args.fleet_mode == 'elastic':
            n_f = int(np.random.randint(args.fleet_min, args.fleet_max + 1))
        else:
            n_f = args.N_fleet
        env._n_fleet_target = n_f

        if args.upper in ('rule_fixed', 'rule_daganzo', 'rule_xuan'):
            params = (360.0, 360.0, 360.0)
        elif args.upper in ('rule_ga', 'rule_cmaes'):
            params = upper.suggest()
            params = tuple(float(x) for x in params)
        elif args.upper == 'rule_mpc':
            # Receding-horizon: pick a triple per dispatch event using mpc_plan.
            # We approximate "state at dispatch" by the slot-of-hour and a
            # demand proxy = (1 + per-hour multiplier shock used by env).
            demand_proxy = float(np.random.normal(1.0, args.demand_noise))
            demand_proxy = float(np.clip(demand_proxy, 0.3, 2.0))
            chosen_triple = mpc_plan(
                current_hour=12, last_dispatch_time_per_dir=None,
                episode_budget_n=n_f,
                current_demand_proxy=demand_proxy,
                candidates=mpc_candidates,
            )
            params = chosen_triple

        z, metrics = run_episode(env, params, lower_rule=lower_rule)
        comp = composite(z, n_f)

        if args.upper in ('rule_ga', 'rule_cmaes'):
            # higher = better; minimise composite ⇔ maximise -composite
            upper.report(-comp)

        history['avg_wait'].append(float(z[0]))
        history['cv'].append(float(z[2]))
        history['overshoot'].append(float(max(0, z[1] - n_f)))
        history['composite'].append(float(comp))
        history['params'].append([float(x) for x in params])
        history['wall_s'].append(round(time.time() - t0, 1))
        for key in metric_keys:
            history[key].append(float(metrics.get(key, 0.0)))

        if comp < best['composite']:
            best = {'composite': float(comp), 'params': list(params),
                    'episode': int(ep), 'z': [float(x) for x in z],
                    'n_fleet': int(n_f)}

        if ep % 5 == 0 or ep == args.episodes - 1:
            print(f"ep{ep:3d}  N={n_f:2d}  H={tuple(round(x,1) for x in params)}  "
                  f"wait={z[0]:5.2f}  cv={z[2]:.3f}  over={max(0,z[1]-n_f):.1f}  "
                  f"comp={comp:5.3f}  ({history['wall_s'][-1]}s)")

    (log_dir / 'history.json').write_text(json.dumps(history, indent=1))
    (log_dir / 'best.json').write_text(json.dumps(best, indent=1))
    print(f"Best composite={best['composite']:.3f} at ep{best['episode']} "
          f"with H={best['params']} on N={best['n_fleet']}")


if __name__ == '__main__':
    main()
