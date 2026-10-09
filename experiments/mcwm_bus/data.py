"""Read existing bus logs and collect a bounded, separate source-sim sample."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
KEYS = ("s", "a", "ns", "r", "dt", "block")
H5_KEYS = dict(s="observations", a="actions", ns="next_observations",
               r="rewards", dt="delta_t_seconds", block="demand_realization_seed")


def load_protocol():
    protocol = json.loads((HERE / "protocol_v1.json").read_text())
    for key in ("h2o_root", "dataset", "source_env", "sumo_root", "mcwm_root"):
        if os.environ.get("THESIS_" + key.upper()):
            protocol[key] = os.environ["THESIS_" + key.upper()]
    return protocol


def within_batch(s):
    return np.rint(s[:, 2] * 15).astype(int) + 1 < np.rint(s[:, 1] * 16)


def analytic_next(s, a):
    if not within_batch(s).all():
        raise ValueError("Analytic transition only applies inside an event batch")
    ns = s.copy()
    current = np.rint(s[:, 2] * 15).astype(int)
    ns[:, 2] = (current + 1) / 15
    ns[np.arange(len(s)), 5 + current * 14 + 13] = a.reshape(-1)
    return ns


def target_data(p, out):
    rng = np.random.default_rng(p["data_seed"])
    records, audit = {}, {}
    with h5py.File(Path(p["h2o_root"]) / p["dataset"], "r") as h:
        blocks = h["demand_realization_seed"][:].reshape(-1)
        trajectories = h["trajectory_id"][:].reshape(-1)
        dt = h["delta_t_seconds"][:].reshape(-1)
        terminal = h["terminals"][:].reshape(-1) > .5
        rewards = h["rewards"][:].reshape(-1)
        audit["station_max_by_line"] = json.loads(h.attrs["station_max_by_line"])
        audit["blocks"] = {}
        for split, split_blocks in p["target_blocks"].items():
            chosen = []
            for block in split_blocks:
                mask = blocks == block
                ids = np.flatnonzero(mask & (dt > 0) & ~terminal)
                for trajectory in np.unique(trajectories[ids]):
                    candidates = ids[trajectories[ids] == trajectory]
                    chosen.extend(rng.choice(candidates, min(len(candidates),
                                  p["rows_per_target_trajectory"]), replace=False))
                zero = np.flatnonzero(mask & (dt == 0) & ~terminal)
                check = np.sort(rng.choice(zero, min(len(zero),
                                     p["analytic_checks_per_block"]), replace=False))
                s, a, ns = (h[key][check] for key in ("observations", "actions", "next_observations"))
                error = float(np.abs(analytic_next(s, a) - ns).max())
                if error > 1e-6 or np.any(rewards[check] != 0):
                    raise ValueError(f"Within-batch contract mismatch in block {block}: {error}")
                tail = mask & terminal
                audit["blocks"][str(block)] = {
                    "rows": int(mask.sum()), "trajectories": int(tail.sum()),
                    "within_batch_checks": len(check), "analytic_max_error": error,
                    "zero_duration_fraction": float((mask & (dt == 0)).sum() / mask.sum()),
                    "terminal_cost_fraction": float(rewards[tail].sum() / rewards[mask].sum()),
                    "terminal_duration_mean_s": float(dt[tail].mean()),
                }
            chosen = np.sort(chosen)
            records[split] = {key: h[value][chosen] for key, value in H5_KEYS.items()}
            if within_batch(records[split]["s"]).any():
                raise ValueError("A learned transition was actually within-batch")
            records[split]["row_id"] = chosen
            print(f"target {split}: {len(chosen)} boundary rows", flush=True)
    for split, data in records.items():
        np.savez_compressed(out / f"target_{split}.npz", **data)
    (out / "target_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.rows = []
        self.boundary_seen = self.total_seen = self.zero_seen = 0
        self.total_reward = 0.
        self.terminal = None
        self.max_analytic_error = 0.

    @property
    def online_size(self):
        return len(self.rows)

    def append(self, s, a, r, ns, discount, **info):
        self.total_seen += 1
        self.total_reward += r
        if info["terminal"]:
            self.terminal = {"dt": info["delta_t_seconds"], "reward": r}
            return
        if info["delta_t_seconds"] == 0:
            self.zero_seen += 1
            err = float(np.abs(analytic_next(s[None], a[None])[0] - ns).max())
            self.max_analytic_error = max(self.max_analytic_error, err)
            if err > 1e-6 or r != 0:
                raise ValueError("Source within-batch transition mismatch")
            return
        if within_batch(s[None])[0]:
            raise ValueError("Positive time inside source event batch")
        self.boundary_seen += 1
        row = (s.copy(), a.copy(), ns.copy(), r, info["delta_t_seconds"])
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            position = self.rng.integers(self.boundary_seen)
            if position < self.capacity:
                self.rows[position] = row


def source_episode(p, spec, out, stations):
    sys.path.insert(0, p["h2o_root"])
    from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
    from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
    from SimpleSAC.transit_v7.sampler import AllLinesTransitSamplerV7

    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    policy_rng = np.random.default_rng(seed + 100000)
    buffer = Reservoir(p["source_rows_per_episode"], seed + 200000)
    env = AllLinesSimEnv(str(Path(p["h2o_root"]) / "bus_h2o/calibrated_env"),
                        episode_seconds=p["horizon_seconds"],
                        passenger_arrival_end_seconds=p["passenger_arrival_end_seconds"],
                        demand_scale=spec["demand_scale"], observation_schema="v7",
                        observation_speed_seed=seed)
    sampler = AllLinesTransitSamplerV7(env, buffer, station_max_by_line=stations,
                                      contract_config=TransitMDPConfigV7())

    def policy(obs, deterministic):
        positive = policy_rng.beta(*p["source_policy"]["positive_beta"], size=(len(obs), 1))
        active = policy_rng.random((len(obs), 1)) >= p["source_policy"]["zero_probability"]
        return (2 * positive * active - 1).astype(np.float32)

    started = time.monotonic()
    while buffer.terminal is None:
        # One complete batch per call, so collection stops at the first full day.
        sampler.collect(policy, 1)
        if buffer.total_seen and buffer.total_seen % 2000 == 0:
            print(f"source {seed}: {buffer.total_seen} rows, {time.monotonic()-started:.1f}s", flush=True)
    data = {key: np.asarray(value) for key, value in zip(KEYS[:-1], zip(*buffer.rows))}
    data["block"] = np.full(len(buffer.rows), seed)
    np.savez_compressed(out / f"source_{seed}.npz", **data)
    summary = {**spec, "total_rows": buffer.total_seen, "boundary_rows": buffer.boundary_seen,
               "zero_rows": buffer.zero_seen, "sample_rows": len(buffer.rows),
               "analytic_max_error": buffer.max_analytic_error,
               "terminal": buffer.terminal, "total_reward": buffer.total_reward,
               "terminal_cost_fraction": buffer.terminal["reward"] / buffer.total_reward,
               "elapsed_seconds": time.monotonic() - started}
    (out / f"source_{seed}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs/pilot_v1")
    parser.add_argument("--stage", choices=("target", "source", "all"), default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = load_protocol()
    if args.stage in ("target", "all"):
        target_data(p, args.out)
    if args.stage in ("source", "all"):
        stations = json.loads((args.out / "target_audit.json").read_text())["station_max_by_line"]
        for spec in p["source_episodes"]:
            source_episode(p, spec, args.out, np.asarray(stations))
