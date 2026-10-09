"""Resolve one common evaluation case using this server's resident inputs."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("mcwm_aligned", "h2oplus_final", "sac_guarded"), required=True)
parser.add_argument("--seed", type=int, choices=range(6201, 6206), required=True)
parser.add_argument("--case", type=int, choices=range(15), required=True)
args = parser.parse_args()
runtime = json.loads((HERE / "eval_runtime.json").read_text())
case = json.loads((HERE / "scenarios/scenario_index.json").read_text())["cases"][args.case]
name = "%s_seed%d_sumo%d_demand%g.json" % (args.method, args.seed, case["sumo_seed"], case["demand_scale"])
values = {"method": args.method, "training-seed": args.seed, "sumo-seed": case["sumo_seed"],
          "checkpoint": HERE / "policies" / ("%s_seed%d.pt" % (args.method, args.seed)),
          "scenario-manifest": HERE / "scenarios" / case["scenario_manifest"],
          "output": HERE / "results" / name}
for key in ("h2o-root", "dataset", "sumo-dir", "edge-xml", "sumo-runtime-lock",
            "verified-dataset-sha256", "verified-dataset-size-bytes", "verified-dataset-mtime-ns"):
    values[key] = runtime[key]
if args.method == "mcwm_aligned":
    values["training-record"] = HERE / "training_records" / ("seed%d.json" % args.seed)
command = [sys.executable, "-u", str(HERE / "evaluate.py")]
for key, value in values.items():
    command.extend(["--" + key, str(value)])
os.execv(sys.executable, command + ["--quiet"])
