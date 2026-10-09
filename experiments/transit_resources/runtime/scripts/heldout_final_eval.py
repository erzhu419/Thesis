#!/usr/bin/env python3
"""Independent final-checkpoint evaluation for the TransitDuet paper.

This script deliberately separates checkpoint selection from test evaluation.
For each learned run, the pre-specified final checkpoint (episode 299) is
loaded and evaluated on new stochastic episodes. Test outcomes are never used
to choose a checkpoint. Daganzo-style and Xuan-style rules use the same test
episodes but have no fitted checkpoint.

The output is one compact JSON file per (method, seed), so the scheduler only
returns summary metrics rather than checkpoints or episode-level traces.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from run_baseline_rule import composite as rule_composite
from run_baseline_rule import run_episode as run_rule_episode
from runner_v3 import TransitDuetV2Runner, load_config


SEEDS = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
CHECKPOINT_EPISODE = 299
PROTOCOL_VERSION = "final-checkpoint-heldout-v1"

LEARNED_METHODS = {
    "H_timetable_v5_stable_300": "configs_ablation/H_timetable_v5_stable_300.yaml",
    "H_hiro_final": "configs_ablation/H_hiro_final.yaml",
    "H_haar_final": "configs_ablation/H_haar_final.yaml",
    "H_fixed_timetable_300_final": "configs_ablation/H_fixed_timetable_300_final.yaml",
    "H_fixed_timetable_360_final": "configs_ablation/H_fixed_timetable_360_final.yaml",
}
RULE_METHODS = {
    "baseline_rule_daganzo": "daganzo",
    "baseline_rule_xuan": "xuan",
}
MAIN_METHODS = tuple(LEARNED_METHODS) + tuple(RULE_METHODS)

METRIC_KEYS = (
    "wait",
    "cv",
    "overshoot",
    "composite",
    "avg_holding_sec",
    "avg_onboard_time_min",
    "avg_total_passenger_time_min",
    "planned_dispatch_headway_mean",
    "planned_dispatch_headway_std",
    "planned_dispatch_headway_cv",
    "actual_dispatch_headway_mean",
    "actual_dispatch_headway_std",
    "actual_dispatch_headway_cv",
    "planned_shift_mean",
    "planned_shift_std",
    "dispatch_lateness_mean",
    "dispatch_lateness_max",
)


class _NullDiag:
    def append(self, row):
        del row

    def save_json(self):
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_plan(seed: int, n_test: int, fleet_min: int, fleet_max: int):
    """Create a method-independent fleet schedule and episode seeds."""
    rng = np.random.RandomState(700_000 + seed)
    fleets = [int(rng.randint(fleet_min, fleet_max + 1)) for _ in range(n_test)]
    episode_seeds = [800_000 + 1_000 * seed + i for i in range(n_test)]
    return fleets, episode_seeds


def summarise(series: dict[str, list[float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, values in series.items():
        values_array = np.asarray(values, dtype=float)
        output[f"{key}_mean"] = float(values_array.mean())
        output[f"{key}_std"] = float(values_array.std(ddof=0))
    return output


def load_learned(method: str, seed: int, checkpoint_episode: int, device: str):
    config_path = SCRIPT_DIR / LEARNED_METHODS[method]
    config = load_config(str(config_path))
    config["seed"] = seed
    runner = TransitDuetV2Runner(config, device=device)
    exp_dir = SCRIPT_DIR / "logs" / f"{method}_seed{seed}"
    checkpoint_dir = exp_dir / "checkpoints"
    lower_path = checkpoint_dir / f"lower_ep{checkpoint_episode}.pt"
    upper_path = checkpoint_dir / f"upper_ep{checkpoint_episode}.pt"
    if not lower_path.exists() or not upper_path.exists():
        missing = [str(path) for path in (lower_path, upper_path) if not path.exists()]
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))
    runner.lower_trainer.load(str(lower_path))
    runner.upper_trainer.load(str(upper_path))
    runner.diag = _NullDiag()
    return runner


def evaluate_learned(method: str, seed: int, n_test: int, checkpoint_episode: int,
                     device: str) -> dict[str, float]:
    runner = load_learned(method, seed, checkpoint_episode, device)
    fleets, episode_seeds = test_plan(seed, n_test, runner.fleet_min, runner.fleet_max)
    series = {key: [] for key in METRIC_KEYS}
    for fleet, episode_seed in zip(fleets, episode_seeds):
        seed_everything(episode_seed)
        runner.env._n_fleet_target = fleet
        row = runner.run_episode(ep=10_000, training=False, N_fleet_override=fleet)
        wait = float(row["avg_wait_min"])
        cv = float(row["headway_cv"])
        overshoot = float(row["fleet_overshoot"])
        series["wait"].append(wait)
        series["cv"].append(cv)
        series["overshoot"].append(overshoot)
        series["composite"].append(wait / 10.0 + overshoot ** 2 / max(fleet, 1) + cv)
        for key in METRIC_KEYS[4:]:
            series[key].append(float(row.get(key, 0.0)))
    return summarise(series)


def evaluate_rule(method: str, seed: int, n_test: int) -> dict[str, float]:
    rule_name = RULE_METHODS[method]
    env = env_bus(str(SCRIPT_DIR / "env"), route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = 0.15
    fleets, episode_seeds = test_plan(seed, n_test, fleet_min=8, fleet_max=16)
    series = {key: [] for key in METRIC_KEYS}
    for fleet, episode_seed in zip(fleets, episode_seeds):
        seed_everything(episode_seed)
        env._n_fleet_target = fleet
        measurement, metrics = run_rule_episode(
            env, (360.0, 360.0, 360.0), lower_rule=rule_name)
        wait = float(measurement[0])
        cv = float(measurement[2])
        overshoot = max(0.0, float(measurement[1]) - fleet)
        series["wait"].append(wait)
        series["cv"].append(cv)
        series["overshoot"].append(overshoot)
        series["composite"].append(rule_composite(measurement, fleet))
        for key in METRIC_KEYS[4:]:
            series[key].append(float(metrics.get(key, 0.0)))
    return summarise(series)


def evaluate_item(method: str, seed: int, n_test: int, checkpoint_episode: int,
                  device: str, output_dir: Path) -> Path:
    if method in LEARNED_METHODS:
        metrics = evaluate_learned(method, seed, n_test, checkpoint_episode, device)
        selection = {
            "type": "pre-specified final checkpoint",
            "checkpoint_episode": checkpoint_episode,
            "test_used_for_selection": False,
        }
    elif method in RULE_METHODS:
        metrics = evaluate_rule(method, seed, n_test)
        selection = {
            "type": "fixed analytical-rule parameterization",
            "checkpoint_episode": None,
            "test_used_for_selection": False,
        }
    else:
        raise ValueError(f"unknown method: {method}")

    fleets, episode_seeds = test_plan(seed, n_test, fleet_min=8, fleet_max=16)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "training_seed": seed,
        "selection": selection,
        "n_test_episodes": n_test,
        "fleet_schedule": fleets,
        "test_episode_seeds": episode_seeds,
        "metrics": metrics,
    }
    path = output_dir / method / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=MAIN_METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-main", action="store_true")
    parser.add_argument("--all-rules", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--checkpoint-episode", type=int, default=CHECKPOINT_EPISODE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results_remote/revision3_heldout")
    args = parser.parse_args()

    if sum((args.all_main, args.all_rules, args.method is not None)) != 1:
        parser.error("provide exactly one of --all-main, --all-rules, or --method")
    if args.method is not None and args.seed is None:
        parser.error("--seed is required with --method")
    if args.n_test < 1:
        parser.error("--n-test must be positive")

    torch.set_num_threads(1)
    output_dir = SCRIPT_DIR / args.output_dir
    if args.all_main or args.all_rules:
        methods = MAIN_METHODS if args.all_main else tuple(RULE_METHODS)
        items = [(method, seed) for method in methods for seed in SEEDS]
        end = len(items) if args.end is None else args.end
        if args.start < 0 or end < args.start or end > len(items):
            parser.error(f"invalid item range [{args.start}, {end}) for {len(items)} items")
        for method, seed in items[args.start:end]:
            path = evaluate_item(method, seed, args.n_test, args.checkpoint_episode,
                                 args.device, output_dir)
            print(path)
    else:
        path = evaluate_item(args.method, args.seed, args.n_test,
                             args.checkpoint_episode, args.device, output_dir)
        print(path)
    print("Eval complete", flush=True)


if __name__ == "__main__":
    main()
