"""Use the resident bus runtime; sparse fitting has its own dependencies."""

import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    runtime = json.loads((here / "node_runtime.json").read_text())["train"]
    phase = sys.argv[sys.argv.index("--phase")+1]
    arm = sys.argv[sys.argv.index("--arm")+1]
    env = dict(os.environ, **runtime["env"], CUDA_VISIBLE_DEVICES="")
    env["CS_BUS_ENV_ROOT"] = str(here.parent / "enhancement_v1/runtime/cs")
    if phase == "fit":
        python = str(here / "fit_env/bin/python")
        arguments = ["--seed", sys.argv[sys.argv.index("--seed")+1]]
        if "--smoke" in sys.argv:
            arguments.append("--smoke")
        entry = "fit_reference.py"
    else:
        python, arguments = runtime["python"], sys.argv[1:]
        entry = "legacy_run.py" if arm.startswith("legacy_") else "run.py"
    os.chdir(here)
    os.execve(python, [python, "-u", str(here / entry), *arguments], env)
