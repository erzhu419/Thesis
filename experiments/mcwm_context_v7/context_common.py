"""Isolated context protocol and two disjoint evaluation streams."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_usage_v3"))
from usage_common import P as PREVIOUS, module, write
sys.path.insert(0, str(HERE))

P = dict(PREVIOUS, **json.loads((HERE / "protocol.json").read_text()))
FIXED = P["fixed_coefficients"]
CONTEXTS = [s["name"] for s in P["scenarios"]]


def cases(arm):
    return range(20) if arm in FIXED else range(10, 20)


def evaluation_protocol(p, case):
    group, local = divmod(case, 10)
    return dict(p, evaluation_blocks=p["development_blocks" if group == 0 else "test_blocks"]), local


def bundle_path(seed, smoke=False):
    return HERE / ("smoke_adaptation.json" if smoke else f"adaptation/seed{seed}.json")
