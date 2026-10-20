"""Create one bounded shared history cache and verify the five resident IQL initializers."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from inputs import load_history

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("THESIS_DATASET", "dataset.h5")))
    args = parser.parse_args()
    p = json.loads((HERE / "protocol.json").read_text())
    out = HERE / "inputs"
    out.mkdir(exist_ok=True)
    if (out / "prepared.json").exists():
        print(json.dumps(dict(status="complete", already_prepared=True)), flush=True)
        return
    started = time.monotonic()
    print("SCHEDULEURM_PROGRESS "+json.dumps(dict(phase="history_loading",current=0,total=1,unit="history")), flush=True)
    table, metadata = load_history(args.dataset, max_rows=p["history_rows"],
        seed=p["history_seed"], blocks=p["history_blocks"])
    np.savez_compressed(out / "history.npz", **table)
    (out / "history_metadata.json").write_text(json.dumps(metadata)+"\n")
    configs, initializers = [], []
    for seed in p["training_seeds"]:
        path = out / f"h2oplus_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["total_updates"] != p["initializer_updates"] or payload["online_critic_updates"] != 0:
            raise ValueError(f"Initializer {seed} is not the formal offline-only prefix")
        extra = payload["extra"]
        identity = extra.get("source_training_seed", extra.get("training_seed", extra.get("seed")))
        if identity != seed:
            raise ValueError(f"Initializer seed differs: requested {seed}, recorded {identity}")
        configs.append(payload["source_config"])
        initializers.append(dict(seed=seed, path=str(path), updates=payload["total_updates"],
            seed_key_value=identity, metadata={key: extra.get(key) for key in (
                "source_run_name", "source_training_seed", "training_seed", "offline_updates_completed")}))
        del payload
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Formal initializers use different policy configurations")
    result = dict(status="complete", history={key: metadata[key] for key in (
        "selected_rows", "terminal_rows", "blocks", "strata", "retained_array_bytes")},
        initializers=initializers, source_config=configs[0], wall_seconds=time.monotonic()-started)
    (out / "prepared.json").write_text(json.dumps(result, indent=2)+"\n")
    print("SCHEDULEURM_PROGRESS "+json.dumps(dict(phase="prepared",current=1,total=1,unit="history",
                         elapsed_seconds=result["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
