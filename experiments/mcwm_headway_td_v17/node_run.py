"""Select shared source/SUMO runtimes without copying datasets."""

import json
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
runtime = json.loads((here / "node_runtime.json").read_text())
selected = runtime["evaluate" if "evaluate" in sys.argv else "train"]
os.chdir(here)
os.execve(selected["python"], [selected["python"], "-u", str(here / "run.py"), *sys.argv[1:]],
          dict(os.environ, **selected["env"], CUDA_VISIBLE_DEVICES=""))
