"""Separate artifacts for the reward-objective crossover campaign."""

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())


def module(name, path):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    temporary.replace(path)


def folder(kind, objective, arm, seed, scene="normal", smoke=False):
    return HERE / ("smoke" if smoke else "runs") / kind / objective / scene / f"{arm}_seed{seed}"


def initializer_folder(objective, seed, smoke=False):
    return HERE / ("smoke_inputs" if smoke else "inputs") / objective / f"seed{seed}"
