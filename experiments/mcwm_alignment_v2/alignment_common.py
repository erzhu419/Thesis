"""Paths and fixed settings for the bus adaptation, separate from completed work."""

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(EXPERIMENTS / "mcwm_bus"))
sys.path.insert(0, str(EXPERIMENTS / "enhancement_v1"))
from study_common import write, seed_all

P = json.loads((HERE / "protocol.json").read_text())


def label(arm, scenario, seed):
    return f"{scenario}/{arm}_seed{seed}"


def milestones(p):
    points = set(range(p["warmup_events"], p["source_event_budget"] + 1,
                       p["collection_interval"]))
    points.update(p["model_refresh_events"])
    points.update((p["actor_taper_event"], p["source_event_budget"]))
    return sorted(points)


def mix_model_loss(real_loss, model_loss, fraction):
    return real_loss if model_loss is None else (1-fraction)*real_loss + fraction*model_loss
