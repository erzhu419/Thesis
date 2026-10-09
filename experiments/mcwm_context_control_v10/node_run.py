"""Enter the resident node-local Python environment."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    settings = json.loads((here / "node_runtime.json").read_text())
    runtime = settings["evaluate" if "--evaluate" in sys.argv else "train"]
    os.chdir(here)
    os.execve(runtime["python"], [runtime["python"], "-u", str(here / "run.py"), *sys.argv[1:]],
        dict(os.environ, **runtime["env"]))
