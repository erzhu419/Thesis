"""Use existing shared source/SUMO runtimes, without transferring their data."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    runtime = json.loads((here.parent / "bus_control_v2/node_runtime.json").read_text())
    command, arguments = sys.argv[1], sys.argv[2:]
    evaluation = command == "diagnose_speed" or any(x in arguments for x in ("collect", "evaluate"))
    selected = runtime["evaluate" if evaluation else "train"]
    env = dict(os.environ, **selected["env"], CUDA_VISIBLE_DEVICES="")
    os.chdir(here)
    os.execve(selected["python"], [selected["python"], "-u", str(here / (command+".py")), *arguments], env)
