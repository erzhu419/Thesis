"""Intermediate-weight comparison using the frozen v11 CPU protocol."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_context_mapping_v11"))
from mapping_common import P as PREVIOUS, module, write, smoke_protocol
sys.path.insert(0, str(HERE))
P = dict(PREVIOUS, **json.loads((HERE / "protocol.json").read_text()))
FIXED = P["fixed_coefficients"]
CONTEXTS = [s["name"] for s in P["scenarios"]]


def bundle_path(seed, smoke=False):
    return HERE / "adaptation" / f"seed{seed}.json"
