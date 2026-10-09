"""Select already installed source/SUMO runtimes for independent CPU shards."""

import json
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
family = sys.argv[sys.argv.index("--family") + 1]
phase = sys.argv[sys.argv.index("--phase") + 1]
if family == "mc":
    manifest = json.loads((here.parent / "bus_control_v2/node_runtime.json").read_text())
    runtime = manifest["evaluate" if phase == "evaluate" else "train"]
else:
    runtime = json.loads((here.parent / "reward_cross_v1/node_runtime.json").read_text())["train"]
env = dict(os.environ, **runtime["env"], CUDA_VISIBLE_DEVICES="")
env["CS_BUS_ENV_ROOT"] = str(here.parent / "enhancement_v1/runtime/cs")
os.chdir(here)
entry = "mc_run.py" if family == "mc" else "cs_run.py"
os.execve(runtime["python"], [runtime["python"], "-u", str(here / entry), *sys.argv[1:]], env)
