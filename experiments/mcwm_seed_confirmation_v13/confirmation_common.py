"""Fresh initialization seeds and disjoint development/test bus days."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_context_midweight_v12"))
from midweight_common import P as PREVIOUS, module, write, smoke_protocol as old_smoke
sys.path.insert(0, str(HERE))
P = dict(PREVIOUS, **json.loads((HERE / "protocol.json").read_text()))
FIXED = P["fixed_coefficients"]
CONTEXTS = [s["name"] for s in P["scenarios"]]


def smoke_protocol(p):
    return dict(old_smoke(p), offline_updates_in_initializer=64)


def initializer_folder(seed, smoke=False):
    return HERE.parent / P["initializer_experiment"] / ("smoke_inputs" if smoke else "inputs") / f"seed{seed}"


def initializer_path(seed, smoke=False):
    return initializer_folder(seed, smoke) / f"h2oplus_seed{seed}.pt"


def bundle_path(seed, smoke=False):
    return HERE.parent / P["initializer_experiment"] / "adaptation" / f"seed{seed}.json"


def policy_folder(scene, arm, seed, smoke=False):
    experiment = P.get("reference_experiment") if arm not in P["training_arms"] else HERE.name
    return HERE.parent / experiment / ("smoke" if smoke else "runs") / scene / f"{arm}_seed{seed}"


def evaluation_protocol(p, case):
    split, local = divmod(case, 10)
    return dict(p, evaluation_blocks=p["development_blocks" if split == 0 else "test_blocks"]), local
