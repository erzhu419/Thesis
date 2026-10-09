"""Optional reward selector; old defaults and old output folders stay intact."""

import argparse
from cross_common import P, folder, module, HERE
from rewards import OBJECTIVES, resolve_objective


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("legacy", "cs", "mc"), required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="original")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.requested_objective = args.objective
    args.objective = resolve_objective(args.kind, args.objective)
    if args.initialize and (args.kind != "mc" or args.evaluate):
        parser.error("Only MC offline-prefix jobs use --initialize")
    if not args.initialize and args.arm not in P[args.kind+"_arms"]:
        parser.error("Choose an arm from "+str(P[args.kind+"_arms"]))
    if args.kind == "mc" and not args.initialize and args.scenario not in P["mc_scenes"]:
        parser.error("Choose a source condition from "+str(P["mc_scenes"]))
    if args.evaluate and args.case not in range(10 if args.kind == "mc" else 9):
        parser.error("Evaluation case out of range")
    args.out = folder(args.kind, args.objective, args.arm or "initializer", args.seed, args.scenario, args.smoke)
    args.out.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = arguments()
    module("cross_runner_"+args.kind, HERE / (args.kind+".py")).run(args)
    print("DONE reward cross "+args.kind+" "+args.objective, flush=True)
