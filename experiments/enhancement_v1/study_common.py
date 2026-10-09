"""Study protocol and small result records; distinct from bus simulator packages."""

import argparse
import json
import os
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def arguments(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=P[kind + "_arms"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    if kind == "cs":
        parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke and "smoke" not in args.out.parts:
        parser.error("smoke outputs must be under a separate smoke directory")
    return args
