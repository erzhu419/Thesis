"""Review settings, named separately from the simulator's common package."""

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def module(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def distribution(values):
    import numpy as np
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x):
        return dict(n=0, mean=None, p10=None, median=None, p90=None)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite diagnostic values")
    q = np.quantile(x, [.1, .5, .9])
    return dict(n=len(x), mean=float(x.mean()), p10=float(q[0]), median=float(q[1]), p90=float(q[2]))
