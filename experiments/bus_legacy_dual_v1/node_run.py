"""Use the existing server runtime; stage neither data nor checkpoints."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    runtime = json.loads((here.parent / "reward_cross_v1/node_runtime.json").read_text())["train"]
    env = dict(os.environ, **runtime["env"], CUDA_VISIBLE_DEVICES="")
    env["CS_BUS_ENV_ROOT"] = str(here.parent / "enhancement_v1/runtime/cs")
    os.chdir(here)
    entry = "contracts.py" if "--contracts" in sys.argv else "run.py"
    arguments = [x for x in sys.argv[1:] if x != "--contracts"]
    os.execve(runtime["python"], [runtime["python"], "-u", str(here / entry), *arguments], env)
