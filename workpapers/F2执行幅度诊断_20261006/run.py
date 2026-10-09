#!/usr/bin/env python3
"""Evaluate one original F-2 actor/scenario with only the extra hold scale changed."""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=os.environ.get("F2_EVAL_ROOT"))
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--scale", type=float, choices=(0.25, 1.0), default=1.0)
    args = parser.parse_args()
    if args.eval_root is None:
        parser.error("--eval-root or F2_EVAL_ROOT is required")
    root = args.eval_root.resolve(strict=True)
    original = (args.original_root or root / "runs/sumo125_v3").resolve()
    policies = (args.policy_root or root.parent / "mcwm_bus/runs/campaign_v3").resolve()
    manifest = json.loads((HERE / "cases.json").read_text())
    cases = manifest["cases"]
    if not 0 <= args.case < len(cases):
        parser.error("--case must be between 0 and 31")
    case = cases[args.case]
    protocol = json.loads((policies / case["protocol"]).read_text())
    if (protocol["deployment_hold_scale"] != 0.25
            or protocol["evaluation_blocks"] != [manifest["evaluation_block"]]
            or protocol["evaluation_demand_scales"] != [1.0, 1.25]
            or protocol["evaluation_horizon_seconds"] != 24000):
        raise ValueError("input protocol differs from the original F-2 design")
    protocol["deployment_hold_scale"] = args.scale
    actor = (policies / case["actor"]).resolve(strict=True)
    scenario = (original / case["scenario"]).resolve(strict=True)
    passengers = (original / case["passengers"]).resolve(strict=True)
    reference = json.loads((original / case["result"]).read_text())
    out = args.out_root.resolve() / ("scale_%g" % args.scale) / ("case_%02d" % args.case)
    if out == original / case["cell"]:
        raise ValueError("diagnostic output must be separate from the original campaign")
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    actor_link = out / (case["tag"] + ".pt")
    if actor_link.exists():
        if actor_link.resolve() != actor:
            raise ValueError("existing diagnostic actor link points to another policy")
    else:
        actor_link.symlink_to(actor)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "case.json").write_text(json.dumps(dict(case, extra_hold_scale=args.scale), indent=2) + "\n")

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["THESIS_H2O_ROOT"] = str(root / "runtime/H2Oplus")
    os.environ["THESIS_SUMO_ROOT"] = str(root / "assets/SUMO_ruiguang")
    os.environ["THESIS_DATASET"] = str(root / "contract.h5")
    os.environ["SUMO_HOME"] = str(root / "assets/SUMO_ruiguang")
    os.environ["LIBSUMO_AS_TRACI"] = "quiet"
    sys.path.insert(0, str(root))
    import traci  # Bind libsumo before libsumo itself imports traci submodules.
    import libsumo
    if libsumo.getVersion()[1] != "SUMO 1.25.0":
        raise RuntimeError("this diagnostic requires the original SUMO 1.25.0 runtime")
    from evaluate_control import evaluate
    evaluate((case["tag"], case["demand_scale"], manifest["evaluation_block"],
              str(scenario), str(passengers), str(out), protocol))
    result_path = out / "evaluation" / Path(case["result"]).name
    result = json.loads(result_path.read_text())
    if args.scale == 0.25:
        # Exact full-episode replay, excluding only elapsed wall-clock time.
        changed = [key for key in reference.keys() | result.keys()
                   if key != "elapsed_seconds" and reference.get(key) != result.get(key)]
        if changed:
            raise RuntimeError("original-scale replay differs in fields: " + ", ".join(sorted(changed)))
        (out / "replay_check.json").write_text(json.dumps({
            "passed": True,
            "case": args.case,
            "cost_seconds": result["generalized_seconds_per_departed_passenger"],
            "compared_fields": sorted((reference.keys() | result.keys()) - {"elapsed_seconds"}),
        }, indent=2) + "\n")
    print(json.dumps({"case": args.case, "scale": args.scale,
                      "cost_seconds": result["generalized_seconds_per_departed_passenger"],
                      "original_scale_replay_exact": args.scale == 0.25,
                      "result": str(result_path)}), flush=True)


if __name__ == "__main__":
    main()
