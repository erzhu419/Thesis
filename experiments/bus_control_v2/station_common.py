"""Station-level bus contract, shared by collection, learning and evaluation."""

import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())
ROOT = Path(os.environ.get("THESIS_H2O_ROOT", EXPERIMENTS / "mcwm_bus/runtime/H2Oplus"))
MCROOT = Path(os.environ.get("THESIS_MCWM_ROOT", EXPERIMENTS / "mcwm_bus/runtime/MC-WM"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bus_h2o"))
sys.path.insert(0, str(MCROOT))
sys.path.insert(0, str(EXPERIMENTS / "reward_cross_v1"))
from rewards import ridge_reward
from bus_h2o.mdp_contract_v5 import raw_event_to_canonical_v5, TransitMDPConfigV5
sys.path.insert(0, str(HERE))

CONFIG = TransitMDPConfigV5(reward_unit_passenger_seconds=1.)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def action_dim(mode):
    return 2 if mode == "dual" else 1


def physical_action(raw):
    raw = np.asarray(raw, np.float32).reshape(-1)
    return [30 * (float(np.clip(raw[0], -1, 1)) + 1),
            1 + .2 * float(np.clip(raw[1], -1, 1)) if len(raw) == 2 else 1.]


def neutral_action(mode):
    return np.array([-1., 0.] if mode == "dual" else [-1.], np.float32)


def observation(raw, previous_action, stations):
    return np.concatenate((raw_event_to_canonical_v5(raw, stations, CONFIG), previous_action)).astype(np.float32)


def headway_reward(raw):
    return float(ridge_reward(raw[3], raw[4], raw[5], raw[6], raw[11] > .5, raw[12] > .5))


def state_headway_reward(states):
    s = np.asarray(states)
    tf, tb = s[:, 5] * 720., s[:, 6] * 720.
    return ridge_reward(s[:, 3] * tf, s[:, 4] * tb, tf, tb, s[:, 11] > .5, s[:, 12] > .5)


def policy_folder(mode, objective, scene, arm, seed, smoke=False):
    return HERE / ("smoke" if smoke else "runs") / mode / objective / scene / f"{arm}_seed{seed}"


def data_folder(mode, smoke=False):
    return HERE / ("smoke_data" if smoke else "data") / mode
