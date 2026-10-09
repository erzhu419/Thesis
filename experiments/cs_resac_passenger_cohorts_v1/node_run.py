"""Use each server's verified CPU runtime."""

import json
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
runtime = json.loads((here / "node_runtime.json").read_text())
environment = {
    **os.environ, **runtime["env"], "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "SDL_VIDEODRIVER": "dummy", "MPLBACKEND": "Agg",
    "OFFLINE_SUMO_ENV": str(here / "inputs/runtime/offline-sumo/env"),
    "THESIS_CSBAPR_ROOT": str(here / "inputs/runtime"),
}
os.execve(runtime["python"], [runtime["python"], "-u", str(here / "run.py"), *sys.argv[1:]], environment)
