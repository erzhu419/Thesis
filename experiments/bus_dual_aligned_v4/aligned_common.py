"""Extend the existing global bus contract rather than the station experiment."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, str(EXPERIMENTS / "bus_control_v2"))
from station_common import ROOT, MCROOT, write, physical_action, state_headway_reward
sys.path.insert(0, str(EXPERIMENTS / "daganzo_baseline_v1"))
from bus_h2o.mdp_contract_v7 import (
    TransitMDPConfigV7, pack_event_batch_v7, raw_state_to_canonical_v7,
    terminal_raw_state_v7, current_event_v7,
)

STATE_DIM = 245
CONFIG = TransitMDPConfigV7()


def state(raw, speeds, stations):
    import numpy as np
    return np.concatenate((raw_state_to_canonical_v7(raw, stations, CONFIG), speeds)).astype(np.float32)


def ridge_of_state(states):
    import numpy as np
    values = np.asarray(states)
    index = np.rint(values[:, 2] * 15).astype(int)
    slots = values[:, 5:229].reshape(-1, 16, 14)
    local = slots[np.arange(len(values)), index, 1:13]
    event = np.zeros((len(values), 15), np.float32)
    event[:, :2] = local[:, :2]
    event[:, 3:13] = local[:, 2:12]
    return state_headway_reward(event)


def policy_folder(objective, scene, arm, seed, smoke=False):
    return HERE / ("smoke" if smoke else "runs") / objective / scene / f"{arm}_seed{seed}"


def data_folder(smoke=False):
    return HERE / ("smoke_data" if smoke else "data")
