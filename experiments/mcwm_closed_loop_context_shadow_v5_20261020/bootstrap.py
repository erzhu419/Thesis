"""Load read-only resident inputs without shadowing the H2O common package."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("LIBSUMO_AS_TRACI", "quiet")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_formal_20261020"))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))
from experiments.mcwm_closed_loop_formal_20261020 import run as formal
import numpy as np
import torch

INPUTS = HERE.parent / "mcwm_closed_loop_formal_20261020/inputs"


def load_learner(seed, *, trainer_class=formal.ClosedLoopTrainer, **kwargs):
    torch.set_num_threads(1)
    metadata = json.loads((INPUTS/"history_metadata.json").read_text())
    with np.load(INPUTS/"history.npz") as archive:
        history = {key: archive[key] for key in
                   ("s", "a", "ns", "r", "dt", "terminal", "sample_weight")}
    source = formal.load_table(INPUTS/"source_9101.npz", 4096)
    contract = formal.TransitMDPConfigV7(**metadata["contract"])
    stations = np.asarray(metadata["station_max_by_line"], np.float32)

    def make_replay():
        return formal.FormalReplay(history, source, contract, stations, "cpu", 285)

    initializer = INPUTS/f"h2oplus_seed{seed}.pt"
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    if (payload["total_updates"] != 50000 or payload["online_critic_updates"] != 0
            or payload["extra"]["source_training_seed"] != seed):
        raise ValueError("Wrong read-only initializer")
    config = replace(formal.TransitTrainerConfig(**payload["source_config"]),
                     use_ratio_correction=False)
    del payload
    replay = make_replay()
    learner = trainer_class(replay, config=config, device="cpu", seed=seed, **kwargs)
    learner.load_offline_initializer(initializer)
    formal.set_cpu_optimizer_runtime(learner)
    return learner, replay, make_replay, contract, stations, initializer


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
