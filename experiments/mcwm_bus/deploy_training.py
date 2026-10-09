"""Stage the bounded training inputs on the existing GPU servers."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

HERE = Path(__file__).resolve().parent
H2O = Path("/home/erzhu419/mine_code/sumo-rl/H2Oplus")
parser = argparse.ArgumentParser()
parser.add_argument("node", choices=["jtl110gpu", "jtl110gpu2", "jtl311linux", "node007"])
args = parser.parse_args()
remote = str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work") if args.node == "node007" else str(HERE)
python = ("/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
          if args.node == "node007" else "/home/erzhu419/miniconda3/envs/csbapr/bin/python"
          if args.node == "jtl110gpu2" else "/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python")
target = scheduler._ssh_target_for_node(args.node)
shell = scheduler._ssh_rsync_shell_for_node(args.node)
scheduler.run_on(args.node, f"mkdir -p {remote}/runtime/source_env {remote}/prefixes {remote}/runtime/H2Oplus/bus_h2o/network_data", timeout=45)


def sync(sources, destination, flags=()):
    subprocess.run(["rsync", "-a", "-e", shell, *flags, *map(str, sources),
                    f"{target}:{remote}/{destination}"], check=True)


sync([str(HERE)+"/"], "", ["--include=*.py", "--include=*.json", "--exclude=*"])
sync([str(HERE / "runtime")+"/"], "runtime/", ["--exclude=__pycache__"])
sync([H2O / "bus_h2o/network_data/a_sorted_busline_edge.xml"],
     "runtime/H2Oplus/bus_h2o/network_data/")
sync([str(HERE / "runs/budget_v2")+"/./blocks2/offline_initializer.pt",
      str(HERE / "runs/budget_v2")+"/./blocks6/offline_initializer.pt"], "prefixes/", ["-R"])
sync([str(H2O / "bus_h2o/calibrated_env")+"/"], "runtime/source_env/",
     ["--include=data/***", "--include=config.json", "--include=segment_runtime_calibration_v7_v4.json",
      "--include=observation_speed_profiles_v7_v1.npz", "--include=simulator_journeys_v1.json",
      "--include=simulator_demand_calibration.json", "--exclude=*"])
if args.node in ("jtl110gpu", "jtl110gpu2"):
    dataset = str(H2O / "reproducibility/deployments/v7_formal_v2_training_v7r1/workspace/bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5")
else:
    sync([H2O / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"], "runtime/dataset.h5")
    dataset = remote+"/runtime/dataset.h5"
env = dict(THESIS_H2O_ROOT=remote+"/runtime/H2Oplus", THESIS_MCWM_ROOT=remote+"/runtime/MC-WM",
           THESIS_SOURCE_ENV=remote+"/runtime/source_env", THESIS_DATASET=dataset,
           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
code = (f"import sys;sys.path.insert(0,{remote!r});import run_campaign_cell as c;"
        f"e=c.AllLinesSimEnv({env['THESIS_SOURCE_ENV']!r},observation_schema='v7',observation_speed_seed=14201);"
        "print('source-environment-ready',len(e.line_map))")
command = "env "+" ".join(shlex.quote(k+"="+v) for k,v in env.items())+" "+python+" -c "+shlex.quote(code)
result = scheduler.run_on(args.node, command, timeout=90)
print(result, flush=True)
(HERE / f"deployment_{args.node}.json").write_text(json.dumps(dict(node=args.node, python=python,
              cwd=remote, env=env, ready=True), indent=2)+"\n")
