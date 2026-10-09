"""Matched candidate training, calibration selection, then untouched-block scoring."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import HERE, KEYS, load_protocol
from model import BusModel, MarginalModel, Normalizer, losses, metrics, targets


def read_data(path):
    with np.load(path) as data:
        return {key: data[key] for key in KEYS}


def join(rows):
    return {key: np.concatenate([row[key] for row in rows]) for key in KEYS}


def take(y, index):
    return {key: value[index] for key, value in y.items()}


@torch.no_grad()
def predict(models, y):
    return torch.cat([torch.stack([m(y["s"][start:start+256], y["a"][start:start+256])
                                  for m in models]).mean(0)
                      for start in range(0, len(y["s"]), 256)])


def train(model, fit, stop, norm, p, seed, name):
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                 lr=p["learning_rate"])
    rng = np.random.default_rng(seed)
    best, best_state, best_step, history = float("inf"), None, None, []
    for step in range(1, p["updates_per_member"] + 1):
        batch = take(fit, torch.as_tensor(rng.integers(len(fit["s"]), size=p["batch_size"]),
                                          device=fit["s"].device))
        objective = losses(model(batch["s"], batch["a"]), batch, norm).mean()
        if not torch.isfinite(objective):
            raise ValueError(f"Non-finite loss in {name} at {step}")
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step % p["validation_every"] == 0:
            with torch.no_grad():
                value = float(losses(predict([model], stop), stop, norm).mean())
            history.append({"step": step, "validation_loss": value})
            if value < best:
                best, best_step = value, step
                best_state = copy.deepcopy(model.state_dict())
            print(f"{name} step={step} stop_loss={value:.5f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return {"selected_step": best_step, "stop_loss": best, "history": history}


def main(out, device):
    p = load_protocol()
    if (out / "model_results.json").exists():
        raise FileExistsError("Use a separate run directory rather than overwrite a completed pilot")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    # CUDA GEMM determinism is configured by the launch command.
    started = time.monotonic()
    stations = torch.tensor(json.loads((out / "target_audit.json").read_text())["station_max_by_line"],
                            dtype=torch.float32, device=device)
    raw_target = {split: read_data(out / f"target_{split}.npz")
                  for split in p["target_blocks"] if split != "test"}
    target = {key: targets(value, stations, device) for key, value in raw_target.items()}
    source = {split: targets(join([read_data(out / f"source_{spec['seed']}.npz")
                                  for spec in p["source_episodes"] if spec["split"] == split]),
                             stations, device) for split in ("fit", "stop")}
    norm = Normalizer(source["fit"]).to(device)
    families = {name: [] for name in ("source", "direct", "residual")}
    history = {}
    for member in range(p["ensemble_members"]):
        seed = p["model_seed"] + member
        torch.manual_seed(seed)
        base = BusModel().to(device)
        history[f"source_{member}"] = train(base, source["fit"], source["stop"], norm, p, seed, f"source/{member}")
        families["source"].append(base)
        for family in ("direct", "residual"):
            torch.manual_seed(seed + 100)
            model = BusModel(base if family == "residual" else None).to(device)
            history[f"{family}_{member}"] = train(model, target["fit"], target["stop"], norm, p,
                                                  seed + 100, f"{family}/{member}")
            families[family].append(model)
    families["marginal"] = [MarginalModel(target["fit"], norm)]
    calibration = {name: metrics(predict(models, target["calibrate"]), target["calibrate"], norm)
                   for name, models in families.items()}
    selected = min(("residual", "direct"), key=lambda name: calibration[name]["selection_loss"])
    # Freeze choice before loading/scoring the two model-test blocks.
    decision = {"selected_family": selected, "calibration": calibration, "training": history}
    (out / "calibration_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    raw_target["test"] = read_data(out / "target_test.npz")
    test = targets(raw_target["test"], stations, device)
    by_block = {}
    for block in p["target_blocks"]["test"]:
        subset = take(test, torch.as_tensor(raw_target["test"]["block"].reshape(-1) == block, device=device))
        by_block[str(block)] = {name: metrics(predict(models, subset), subset, norm)
                               for name, models in families.items()}
    result = {"protocol": p, **decision, "test_by_block": by_block,
              "elapsed_seconds": time.monotonic() - started,
              "device": device, "torch_version": torch.__version__,
              "policy_training_steps": 0, "target_online_training_steps": 0,
              "normalization": "Source-fit rows only; identical frozen units for every family"}
    (out / "model_results.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"normalizer": norm.state_dict(), "models": {key: [m.state_dict() for m in models]
                for key, models in families.items()}, "protocol": p}, out / "pilot_models.pt")
    print(json.dumps({"selected": selected, "elapsed_seconds": result["elapsed_seconds"],
                      "test": {block: {name: {key: val for key, val in scores.items()
                                    if key in ("selection_loss", "cost_mae_passenger_s", "slot_identity_accuracy")}
                                       for name, scores in families.items()}
                               for block, families in by_block.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs/pilot_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args.out, args.device)
