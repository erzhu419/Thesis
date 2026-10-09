"""Independent dual extensions of the two established bus learners."""

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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def folder(kind, mode, objective, arm, seed, smoke=False):
    return HERE / ("smoke" if smoke else "runs") / kind / mode / objective / f"{arm}_seed{seed}"


def holding_folder(kind, objective, arm, seed):
    if kind == "cs_symbolic":
        return EXPERIMENTS / "bus_control_repair_v3/cs_runs" / objective / f"{arm}_seed{seed}"
    inherited = "resac" if arm == "resac" else "ensemble_sac_control" if kind == "resac" else "csbapr" if arm == "legacy_nau" else "csbapr_no_nau"
    return EXPERIMENTS / "reward_cross_v1/runs" / ("legacy" if kind == "resac" else "cs") / objective / "normal" / f"{inherited}_seed{seed}"


def placements(smoke=False):
    for kind, arms in P["arms"].items():
        seeds = P["symbolic_seeds"] if kind == "cs_symbolic" else P["native_seeds"]
        for objective in P["objectives"]:
            for seed in seeds[:1] if smoke else seeds:
                for arm in arms:
                    yield dict(kind=kind, mode="dual", objective=objective, arm=arm, seed=seed, smoke=smoke)


def label(item):
    prefix = "smoke" if item.get("smoke") else "formal"
    return f"{prefix}/{item['kind']}/{item['mode']}/{item['objective']}/{item['arm']}/{item['seed']}"
