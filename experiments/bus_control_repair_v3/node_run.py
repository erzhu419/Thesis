"""Enter an already resident runtime without loading training data in evaluation."""

import json
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
kind = sys.argv[1]
runtime = json.loads((here.parent / "bus_control_v2/node_runtime.json").read_text())
chosen = runtime["evaluate" if kind == "mc" else "train"]
env = dict(os.environ, **chosen["env"], CUDA_VISIBLE_DEVICES="")
env["CS_BUS_ENV_ROOT"] = str(here.parent / "enhancement_v1/runtime/cs")
os.chdir(here)
os.execve(chosen["python"], [chosen["python"], "-u", str(here / f"{kind}.py"), *sys.argv[2:]], env)
