"""Independent CS bus artifacts; no changes to an existing campaign."""

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())
FROZEN = HERE / "runtime"
ENVROOT = Path(os.environ.get("CS_BUS_ENV_ROOT", EXPERIMENTS / "enhancement_v1/runtime/cs"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def use_snapshot():
    sys.path.insert(0, str(FROZEN))


def model_folder(seed, smoke=False):
    return HERE / ("smoke_models" if smoke else "models") / f"seed{seed}"


def run_folder(objective, arm, seed, smoke=False):
    return HERE / ("smoke" if smoke else "runs") / objective / f"{arm}_seed{seed}"


def day_folder(seed, day):
    return HERE / "identification" / f"seed{seed}" / f"day{day}"
