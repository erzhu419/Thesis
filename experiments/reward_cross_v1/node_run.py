"""Select resident simulation/SUMO runtimes, without moving any dataset."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    runtimes = json.loads((here / "node_runtime.json").read_text())
    kind = sys.argv[sys.argv.index("--kind")+1]
    runtime = runtimes["evaluate" if kind == "mc" and "--evaluate" in sys.argv else "train"]
    environment = dict(os.environ, **runtime["env"], CUDA_VISIBLE_DEVICES="")
    os.chdir(here)
    os.execve(runtime["python"], [runtime["python"], "-u", str(here / "run.py"), *sys.argv[1:]], environment)
