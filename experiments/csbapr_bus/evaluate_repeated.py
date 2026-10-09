"""Single and repeated pulses at equal integrated excess demand multipliers."""

import argparse
import json
from pathlib import Path
from evaluate_recovery import evaluate, METHODS

SCENES = {"normal": [(0, 1.)]}
for multiplier in (2., 5.):
    SCENES[f"single_{int(multiplier)}x"] = [(0, 1.), (3600, multiplier), (5400, 1.)]
    SCENES[f"repeated_{int(multiplier)}x"] = [
        (0, 1.), (3600, multiplier), (4200, 1.),
        (7200, multiplier), (7800, 1.), (10800, multiplier), (11400, 1.)]

PROTOCOL = dict(training_seeds=[0, 1, 2], evaluation_seeds=[17301, 17302, 17303],
                methods=list(METHODS), schedules=SCENES, horizon_seconds=18000,
                window_seconds=300, primary_metric="full_horizon_wait_passenger_seconds",
                secondary_metrics=["headway_imbalance", "holding", "completion", "queue_tail"],
                comparison="same training/evaluation seed paired across methods; single vs repeated has equal peak and 30min high-demand duration",
                demand_note="Equal integrated multiplier, not identical realized arrival counts; no extra shock selected after seeing outcomes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    for scene, schedule in SCENES.items():
        for evaluation_seed in PROTOCOL["evaluation_seeds"]:
            row = evaluate((args.method, args.seed, scene, args.out, evaluation_seed, schedule))
            print(json.dumps({k: v for k, v in row.items() if k != "windows"}), flush=True)
    (args.out / "completed.json").write_text(json.dumps(dict(evaluation_episodes=15)) + "\n")


if __name__ == "__main__":
    main()
