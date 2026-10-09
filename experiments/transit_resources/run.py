"""Extend the existing frozen-policy tests with full-service resource totals."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / "scripts"))

import torch
import runner_v3
import heldout_final_eval as original
from measurement import measured_environment

METHODS = ["H_timetable_v5_stable_300", "H_fixed_timetable_300_final", "H_hiro_final"]
SEEDS = original.SEEDS
BASE_ENV = runner_v3.env_bus
MEASURED_ENV = measured_environment(BASE_ENV)
CHECKPOINT_ROOT = Path(os.environ.get("TRANSIT_CHECKPOINT_ROOT",
    "/home/zhengliang01/scheduleurm_work/TransitDuet/transit_duet/logs"))
_runners = {}


def load_runner(method, seed, measured):
    config = runner_v3.load_config(str(RUNTIME / original.LEARNED_METHODS[method]))
    config["seed"] = seed
    config["_name"] = "resource_evaluation"
    runner_v3.env_bus = MEASURED_ENV if measured else BASE_ENV
    original.seed_everything(seed)
    runner = runner_v3.TransitDuetV2Runner(config, device="cpu")
    checkpoints = CHECKPOINT_ROOT / f"{method}_seed{seed}" / "checkpoints"
    runner.lower_trainer.load(str(checkpoints / "lower_ep299.pt"))
    runner.upper_trainer.load(str(checkpoints / "upper_ep299.pt"))
    runner.diag = original._NullDiag()
    return runner


def case(index):
    method = METHODS[index // 500]
    seed = SEEDS[(index % 500) // 50]
    episode = index % 50
    key = method, seed
    if key not in _runners:
        _runners.clear()
        _runners[key] = load_runner(method, seed, True)
    runner = _runners[key]
    fleets, seeds = original.test_plan(seed, 50, runner.fleet_min, runner.fleet_max)
    original.seed_everything(seeds[episode])
    start = time.monotonic()
    row = runner.run_episode(ep=10000, training=False, N_fleet_override=fleets[episode])
    resources = runner.env.resource_meter.results(runner.env)
    assert resources["completed_trips"] == 262, resources["completed_trips"]
    return {"index": index, "method": method, "training_seed": seed,
            "test_episode": episode, "episode_seed": seeds[episode],
            "fleet_budget": fleets[episode], "seconds": time.monotonic() - start,
            "wait_min": row["avg_wait_min"],
            "onboard_min": row["avg_onboard_time_min"],
            "total_passenger_min": row["avg_total_passenger_time_min"],
            "peak_fleet": row["peak_fleet"], **resources}


def smoke():
    observed = case(0)
    plain = load_runner(METHODS[0], SEEDS[0], False)
    original.seed_everything(observed["episode_seed"])
    reference = plain.run_episode(ep=10000, training=False,
                                 N_fleet_override=observed["fleet_budget"])
    for key, reference_key in (("wait_min", "avg_wait_min"),
                               ("onboard_min", "avg_onboard_time_min"),
                               ("total_passenger_min", "avg_total_passenger_time_min"),
                               ("peak_fleet", "peak_fleet")):
        assert observed[key] == reference[reference_key], (key, observed[key], reference[reference_key])
    assert plain._ep_lower_actions == _runners[(METHODS[0], SEEDS[0])]._ep_lower_actions
    print(json.dumps({"observer_invariance": True, "case": observed}), flush=True)


def worker_setup():
    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    worker_setup()
    if args.smoke:
        smoke()
        return
    indices = list(range(args.shard, 1500, 36))
    output = HERE / "runs" / f"shard{args.shard:02d}" / "results"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "episodes.jsonl"
    done = {json.loads(line)["index"] for line in destination.read_text().splitlines()} if destination.exists() else set()
    pending = [index for index in indices if index not in done]
    with destination.open("a") as stream:
        with mp.get_context("spawn").Pool(args.workers, initializer=worker_setup) as pool:
            for result in pool.imap_unordered(case, pending):
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                done.add(result["index"])
                print(f"progress={len(done)}/{len(indices)}", flush=True)
    assert done == set(indices)
    (output / "complete.json").write_text(json.dumps({"shard": args.shard, "episodes": len(done)}) + "\n")
    print("Resource evaluation complete", flush=True)


if __name__ == "__main__":
    main()
