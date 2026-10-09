"""Re-evaluate archived actors with the published operational collector."""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
VARIANTS = ["sac", "resac_K2", "resac_K5", "resac_K10", "resac_K20",
            "resac_K40", "aleatoric_only", "epistemic_only"]


def namespace(seed):
    source = RUNTIME / "sac_ensemble_original_logging.py"
    tree = ast.parse(source.read_text())
    names = {"EmbeddingLayer", "PolicyNetwork", "set_global_seed", "safe_mean",
             "safe_std", "safe_percentile", "scalar_action", "collect_operational_metrics"}
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = dict(torch=torch, nn=nn, F=F, Normal=Normal, np=np, random=random,
                 device=torch.device("cpu"),
                 args=SimpleNamespace(seed=seed, route_sigma=1.5, use_state_norm=False))
    exec(compile(tree, str(source), "exec"), scope)
    return scope


def load_actor(scope, name):
    state = torch.load(HERE / "weights" / name, map_location="cpu", weights_only=True)
    cols = ["bus_id", "station_id", "time_period", "direction"]
    codes = {key: dict.fromkeys(range(state[f"embedding_layer.embeddings.{key}.weight"].shape[0]))
             for key in cols}
    embedding = scope["EmbeddingLayer"](codes, cols)
    actor = scope["PolicyNetwork"](state["linear1.weight"].shape[1], 1,
                                    state["linear1.weight"].shape[0], embedding,
                                    action_range=60.0)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def prepare():
    archive = Path("/home/erzhu419/mine_code/LSTM-RL/RE-SAC")
    project = Path("/home/erzhu419/mine_code/RE-SAC")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (HERE / "weights").mkdir(exist_ok=True)
    shutil.copy2(project / "sac_ensemble_original_logging.py", RUNTIME)
    shutil.copytree(project / "env_original", RUNTIME / "env_original",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.png"))
    records = []
    for name in VARIANTS:
        if name == "sac":
            folder, curve = "sac_v2_bus", "rewards.npy"
            checkpoint = archive / folder / "sac_v2_bus_episode_final_policy"
            current = project / "results/baseline_sac/rewards.npy"
        else:
            folder = (name.replace("resac_K", "ensemble_") if name.startswith("resac")
                      else {"aleatoric_only": "Aleatoric_Only", "epistemic_only": "Epistemic_Only"}[name])
            curve = "logs/rewards.npy"
            checkpoint = archive / folder / "model/final_policy"
            current = project / "results" / folder.lower() / curve
        original = archive / folder / curve
        a, b = np.load(original), np.load(current)
        if not np.array_equal(a, b) or len(a) != 500:
            raise ValueError(f"Archive does not match the thesis training record: {name}")
        shutil.copy2(checkpoint, HERE / "weights" / name)
        actor = load_actor(namespace(8), name)
        records.append(dict(variant=name, archive_id=8, checkpoint=str(checkpoint),
                            archive_curve=str(original), thesis_curve=str(current),
                            equal_training_curve=True, training_episodes=len(a),
                            hidden_width=actor.linear1.out_features,
                            train_last100=float(a[-100:].mean())))
    reference = project / "results/bus_seed_extension"
    shutil.copy2(reference / "model/resac_K10_seed32/final_policy",
                 HERE / "weights/reference_K10_seed32")
    expected = json.loads((reference / "logs/resac_K10_seed32/operational_metrics.json").read_text())
    (HERE / "archive_manifest.json").write_text(json.dumps(dict(
        actors=records, reference_episode=expected["episodes"][0]), indent=2) + "\n")
    print(json.dumps(records, indent=2), flush=True)


def evaluate(item):
    index, output = item
    name = VARIANTS[index // 10] if index < 80 else "reference_K10_seed32"
    episode = index % 10 if index < 80 else 0
    training_id = 8 if index < 80 else 32
    destination = Path(output) / f"{name}_episode{episode:02d}.json"
    if destination.exists():
        return str(destination)
    torch.set_num_threads(1)
    sys.path.insert(0, str(RUNTIME))
    from env_original.sim import env_bus
    scope = namespace(training_id + episode)
    actor = load_actor(scope, name)
    environment = env_bus(str(RUNTIME / "env_original"), debug=True, route_sigma=1.5)
    started = time.monotonic()
    # The original collector reseeds each rollout before reset; offset the seed
    # so independent shards reproduce its original ten-episode sequence.
    with torch.inference_mode():
        result = scope["collect_operational_metrics"](
            SimpleNamespace(policy_net=actor), environment, 1, deterministic=True)
    row = result["episodes"][0]
    row["episode"] = episode
    row.update(variant=name, archive_id=training_id,
               evaluation_seed=100000 + training_id + episode,
               hidden_width=actor.linear1.out_features,
               elapsed_seconds=time.monotonic() - started)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return str(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=81)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    items = [(index, str(args.out)) for index in range(args.start, args.end)]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(evaluate, items))


if __name__ == "__main__":
    main()
