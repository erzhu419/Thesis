"""Use the shared CPU training and SUMO evaluation runtimes."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    config = json.loads((here / "node_runtime.json").read_text())
    runtime = config["evaluate" if "--evaluate" in sys.argv else "train"]
    os.chdir(here)
    os.execve(runtime["python"], [runtime["python"], "-u", str(here / "run.py"), *sys.argv[1:]],
        dict(os.environ, **runtime["env"]))
