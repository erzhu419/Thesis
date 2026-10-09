"""Isolated model-use experiment, reusing the completed bus study's protocol."""

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def module(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


P = dict(json.loads((EXPERIMENTS / "mcwm_alignment_v2/protocol.json").read_text()),
         **json.loads((HERE / "protocol.json").read_text()))
MODEL_ARMS = ("mcwm", "normalized", "fixed_half", "source_repair")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)
