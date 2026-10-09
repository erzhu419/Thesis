"""Separate output paths without shadowing the SUMO runtime's common package."""

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def cs_folder(objective, arm, seed, smoke=False):
    return HERE / ("cs_smoke" if smoke else "cs_runs") / objective / f"{arm}_seed{seed}"


def mc_folder(args):
    return HERE / ("mc_smoke" if args.smoke else "mc_runs") / args.branch / args.mode / args.objective / args.scene / f"{args.arm}_seed{args.seed}" / args.layer
