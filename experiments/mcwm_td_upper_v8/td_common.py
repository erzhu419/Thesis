"""Fixed-weight study settings, separate from simulator common packages."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_usage_v3"))
from usage_common import P as PREVIOUS, module, write
sys.path.insert(0, str(HERE))
P = dict(PREVIOUS)
P.update(json.loads((HERE.parent / "mcwm_context_v7/protocol.json").read_text()))
P.update(json.loads((HERE / "protocol.json").read_text()))
FIXED = P["fixed_coefficients"]
CONTEXTS = [s["name"] for s in P["scenarios"]]


def smoke_protocol(p):
    return dict(p, warmup_events=128, source_event_budget=192, actor_taper_event=160,
        model_refresh_events=[128, 160], model_updates_per_member=2, model_refit_updates=2,
        model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64,
        context_window_physical_rows=8, evaluation_horizon_seconds=1200)
