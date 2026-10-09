"""Fit with an isolated scientific runtime; no simulator or Torch dependency."""

import argparse
import importlib.util
import time

import numpy as np
import pysindy as ps

from settings import FROZEN, P, day_folder, model_folder, write


def arrays(seed, days):
    parts = []
    for day in days:
        with np.load(day_folder(seed, day) / "transitions.npz") as data:
            parts.append({name:data[name] for name in data.files})
    return {name:np.concatenate([p[name] for p in parts]) for name in parts[0]}


def fit(seed, smoke=False):
    folder = model_folder(seed, smoke)
    if (folder / "ready.json").exists():
        return
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("bus_sparse_dynamics", FROZEN / "csbapr/sindy/world_model.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    train, valid = arrays(seed, [0] if smoke else [0, 1, 2]), arrays(seed, [3])
    feature_ids = P["sindy_features"]
    coefficients, means, stds, diagnostics = [], [], [], []
    for line in range(12):
        selected = np.rint(train["states"][:, 0]*12).astype(int) == line
        held_out = np.rint(valid["states"][:, 0]*12).astype(int) == line
        x, u = train["states"][selected][:, feature_ids], train["actions"][selected]
        delta = train["next_states"][selected][:, feature_ids]-x
        if len(x) < 20 or not held_out.any():
            raise ValueError(f"Insufficient identified/validation transitions for line {line}")
        dynamics = source.SymbolicWorldModel(len(feature_ids), 1, P["sindy_degree"], P["sindy_threshold"])
        dynamics.fit(x, U=u, X_dot=delta)
        costs = train["costs"][selected]
        mean, std = costs.mean(0), np.maximum(costs.std(0), 1.)
        library = dynamics.library.transform(np.concatenate((x, u), -1))
        regression = ps.STLSQ(threshold=P["sindy_threshold"])
        regression.fit(library, (costs-mean)/std)
        coefficients.append(np.concatenate((dynamics.coeffs, regression.coef_), 0))
        means.append(mean)
        stds.append(std)
        vx = valid["states"][held_out][:, feature_ids]
        vu = valid["actions"][held_out]
        dy = valid["next_states"][held_out][:, feature_ids]-vx
        vy = (valid["costs"][held_out]-mean)/std
        predicted_cost = dynamics.library.transform(np.concatenate((vx, vu), -1))@regression.coef_.T
        diagnostics.append(dict(line=line, fit_rows=len(x), validation_rows=len(vx),
            increment_mse=float(np.square(dynamics.predict(vx, vu)-dy).mean()),
            persistence_increment_mse=float(np.square(dy).mean()),
            cost_mse=np.square(predicted_cost-vy).mean(0).tolist(),
            constant_cost_mse=np.square(vy).mean(0).tolist(), sparsity=dynamics.sparsity))
    coefficients = np.stack(coefficients).astype(np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError("Sparse feedback coefficients are not finite")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "reference.npz", powers=dynamics.library.powers_,
        coefficients=coefficients, cost_mean=means, cost_std=stds)
    write(folder / "ready.json", dict(seed=seed, smoke=smoke, diagnostics=diagnostics,
        fitting_days=[0] if smoke else [0, 1, 2], validation_day=3,
        identification_rows=len(train["states"]), validation_rows=len(valid["states"]),
        symbolic_model="controlled per-line state increments and sparse stage costs",
        wall_seconds=time.monotonic()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    fit(args.seed, args.smoke)
