"""Replay the original frozen recipe with only the output location changed."""

import argparse
import sys

from settings import EXPERIMENTS, P, run_folder, module

sys.path.insert(0, str(EXPERIMENTS / "reward_cross_v1"))
old = module("symbolic_bus_legacy_reward", EXPERIMENTS / "reward_cross_v1/cs.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = run_folder(args.objective, args.arm, args.seed, args.smoke)
    args.requested_objective = args.objective
    args.evaluate = args.phase == "evaluate"
    args.arm = "csbapr" if args.arm == "legacy_nau" else "csbapr_no_nau"
    old.P = dict(old.P, episodes=P["episodes"], evaluation_seeds=P["evaluation_seeds"], cs_scenes=P["scenes"])
    old.run(args)
