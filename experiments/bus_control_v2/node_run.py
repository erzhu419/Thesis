"""Enter the resident source or SUMO runtime on any allowed shared-storage node."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    runtime = json.loads((here / "node_runtime.json").read_text())
    phase = sys.argv[sys.argv.index("--phase")+1]
    selected = runtime["evaluate" if phase in ("dual-data", "evaluate") else "train"]
    env = dict(os.environ, **selected["env"], CUDA_VISIBLE_DEVICES="")
    os.chdir(here)
    os.execve(selected["python"], [selected["python"], "-u", str(here / "run.py"), *sys.argv[1:]], env)
