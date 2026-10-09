"""Check whether a constant-population formula can replace the terminal tail."""

import json

import h5py
import numpy as np

from data import HERE, load_protocol
from pathlib import Path


def main():
    p = load_protocol()
    with h5py.File(Path(p["h2o_root"]) / p["dataset"], "r") as h:
        rows = np.flatnonzero(h["terminals"][:].reshape(-1) > .5)
        s = h["raw_observations"][rows]
        ns = h["raw_next_observations"][rows]
        dt = h["delta_t_seconds"][rows].reshape(-1)
        cost = -h["rewards"][rows].reshape(-1)
    estimate = (2 * s[:, 3] + s[:, 4]) * dt
    error = np.abs(cost - estimate)
    result = {
        "terminal_rows": len(rows),
        "endpoint_count_changes": int(np.any(s[:, 3:5] != ns[:, 3:5], axis=1).sum()),
        "constant_population_cost_exact_rows": int((error < 1e-5).sum()),
        "mae_passenger_s": float(error.mean()),
        "max_relative_error": float((error / np.maximum(cost, 1)).max()),
        "onboard_at_last_decision_max": float(s[:, 4].max()),
        "conclusion": "Constant initial population is an approximation, not an exact terminal transition",
    }
    (HERE / "runs/pilot_v1/terminal_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
