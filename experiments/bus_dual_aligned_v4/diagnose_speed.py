"""Frozen-policy speed interventions; never modify existing actors or results."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bus_control_repair_v3"))
from repair_common import module, write
mc = module("aligned_speed_attribution", HERE.parent / "bus_control_repair_v3/mc.py")


def evaluate(args):
    variant = args.branch
    folder = HERE / ("speed_smoke" if args.smoke else "speed_diagnosis") / variant / args.objective / args.scene / f"{args.arm}_seed{args.seed}" / f"speed{args.speed}"
    mc.mc_folder = lambda unused: folder
    original = mc.physical_action
    if variant == "new_executed":
        args.branch = "new"
        original_execute = mc.execute_hold

        def executed_history(*values):
            action = original_execute(*values)
            action[1] = (args.speed-1.)/.2
            return action

        mc.execute_hold = executed_history

    def intervention(action):
        command = original(action)
        command[1] = args.speed
        return command

    mc.physical_action = intervention
    mc.evaluate(args)
    target = folder / f"case{args.case:02d}.json"
    row = json.loads(target.read_text())
    row["branch"] = variant
    row["speed_intervention"] = args.speed
    row["previous_speed_history"] = "executed override" if variant=="new_executed" else "native policy proposal" if variant=="new" else "absent from original holding state"
    row["interpretation"] = "Frozen holding policy with prespecified speed override; no retraining or result replacement"
    write(target, row)
    print("DONE speed intervention", {k:row[k] for k in ("case","horizon_seconds","speed_intervention",
        "generalized_seconds_per_departed_passenger","headway_ridge_mean","wall_seconds")},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("old", "new", "new_executed"), required=True)
    parser.add_argument("--mode", default="dual")
    parser.add_argument("--objective", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--speed", type=float, choices=(1., 1.1, 1.2), required=True)
    parser.add_argument("--layer", default="guarded")
    parser.add_argument("--smoke", action="store_true")
    import torch
    torch.set_num_threads(1)
    evaluate(parser.parse_args())
