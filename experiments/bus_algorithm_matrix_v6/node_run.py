"""Read the existing multi-line runtime; no static-data staging."""

import json
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
runtime = json.loads((here.parent / "reward_cross_v1/node_runtime.json").read_text())["train"]
env = dict(os.environ, **runtime["env"], CUDA_VISIBLE_DEVICES="",
           CS_BUS_ENV_ROOT=str(here.parent / "enhancement_v1/runtime/cs"))
os.chdir(here)
os.execve(runtime["python"], [runtime["python"], "-u", str(here / "run.py"), *sys.argv[1:]], env)
