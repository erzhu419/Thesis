"""Time existing bus RL operations without writing any scientific run artifacts."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from evaluate_pair import evaluate, SEEDS
from bus_experiments import frozen_policy_bank_core as core


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def restore(trainer, payload):
    trainer.load_controller_state(payload["controller"])
    for name in ("critic_optimizer", "policy_optimizer", "alpha_optimizer"):
        getattr(trainer, name).load_state_dict(deepcopy(payload[name]))
    core.random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if trainer.device.type == "cuda":
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])


def episode_profile(environment, trainer, replay, update_index):
    totals = defaultdict(float)
    counts = defaultdict(int)

    def measured(name, function):
        def call(*args, **kwargs):
            synchronize(trainer.device)
            start = time.perf_counter()
            result = function(*args, **kwargs)
            synchronize(trainer.device)
            totals[name] += time.perf_counter() - start
            counts[name] += 1
            return result
        return call

    step = measured("environment_step", environment.step)
    select = measured("policy_inference", trainer.select_action)
    push = measured("replay_push", replay.push)
    update = measured("learner_including_sample", trainer.update)
    replay.sample = measured("replay_sample", replay.sample)
    started = time.perf_counter()
    environment.reset()
    states, rewards, _ = environment.initialize_state(render=False)
    totals["reset_initialize"] = time.perf_counter() - started
    actions = dict.fromkeys(range(environment.max_agent_num))
    done, steps, last_update = False, 0, -1
    while not done:
        for key, observations in states.items():
            if len(observations) == 1:
                if actions[key] is None:
                    actions[key] = select(np.asarray(observations[0]), deterministic=False)
            elif len(observations) == 2:
                if observations[0][1] != observations[1][1]:
                    push(np.asarray(observations[0], np.float32), actions[key],
                         float(rewards[key]), np.asarray(observations[1], np.float32), False)
                    steps += 1
                states[key] = observations[1:]
                actions[key] = select(np.asarray(states[key][0]), deterministic=False)
        states, rewards, done = step(actions)
        if (len(replay) > trainer.config.batch_size
                and len(replay) % trainer.config.training_freq == 0
                and last_update != steps):
            last_update = steps
            update(replay, update_index)
            update_index += 1
    elapsed = time.perf_counter() - started
    totals["learner_excluding_sample"] = (
        totals["learner_including_sample"] - totals["replay_sample"])
    totals["loop_other"] = elapsed - sum(totals[k] for k in (
        "reset_initialize", "environment_step", "policy_inference",
        "replay_push", "learner_including_sample"))
    return dict(elapsed_s=elapsed, transitions=steps, counts=dict(counts), seconds=dict(totals))


def load_eval_policy(controller_path):
    config = core.TrainConfig(**json.loads(
        controller_path.with_name("train_summary.json").read_text())["config"])
    env = core.make_environment(config)
    env.reset()
    return core.load_controller(controller_path, env, torch.device("cpu")).policy


def evaluation_worker_init(controller_path):
    global evaluation_policy
    torch.set_num_threads(1)
    evaluation_policy = load_eval_policy(Path(controller_path))


def evaluation_cell(cell):
    scene, seed = cell
    row = evaluate(evaluation_policy, "exp60_hold", scene, seed)
    return {key: row[key] for key in (
        "scene", "evaluation_seed", "generalized_seconds_per_passenger",
        "arrived_passengers", "waiting_tail", "onboard_tail", "elapsed_seconds")}


def evaluation_profile(run, workers):
    controller = run / "bundle/controller.pt"
    cells = [(scene, seed) for scene in ("normal", "fast_switch") for seed in SEEDS]
    evaluation_worker_init(controller)
    start = time.perf_counter()
    serial = [evaluation_cell(cell) for cell in cells]
    serial_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=evaluation_worker_init,
                             initargs=(str(controller),)) as pool:
        parallel = list(pool.map(evaluation_cell, cells))
    parallel_seconds = time.perf_counter() - start
    differences = []
    for a, b in zip(serial, parallel):
        expected = json.loads((run / "results" / (
            a["scene"] + "_seed" + str(a["evaluation_seed"]) + ".json")).read_text())
        for key in ("generalized_seconds_per_passenger", "arrived_passengers",
                    "waiting_tail", "onboard_tail"):
            differences.extend([abs(a[key] - b[key]), abs(a[key] - expected[key])])
    return dict(cells=len(cells), workers=workers, serial_seconds=serial_seconds,
                parallel_seconds_including_startup=parallel_seconds,
                speedup=serial_seconds / parallel_seconds,
                max_outcome_difference=max(differences), serial=serial, parallel=parallel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episode-threads", type=int, default=1)
    parser.add_argument("--evaluation-workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    summary = json.loads((args.run / "bundle/train_summary.json").read_text())
    config = core.TrainConfig(**summary["config"])
    environment = core.make_environment(config)
    environment.reset()
    trainer = core.BusSAC(environment, config, device)
    replay = core.ReplayBuffer(config.replay_capacity)
    start = time.perf_counter()
    payload = core.load_training_checkpoint(args.run / "checkpoints", trainer, replay)
    result = dict(device=str(device), torch_version=torch.__version__,
                  affinity_cpus=len(os.sched_getaffinity(0)),
                  checkpoint_load_seconds=time.perf_counter() - start,
                  config=summary["config"], thread_benchmarks=[])
    for threads in args.threads:
        torch.set_num_threads(threads)
        restore(trainer, payload)
        index = int(payload["update_index"])
        for i in range(4):
            trainer.update(replay, index + i)
        durations = []
        for i in range(args.updates):
            synchronize(device)
            start = time.perf_counter()
            trainer.update(replay, index + 4 + i)
            synchronize(device)
            durations.append(time.perf_counter() - start)
        row = dict(threads=threads, updates=args.updates,
                   median_update_ms=statistics.median(durations) * 1000,
                   mean_update_ms=statistics.mean(durations) * 1000)
        result["thread_benchmarks"].append(row)
        print(json.dumps(row), flush=True)
    torch.set_num_threads(args.episode_threads)
    restore(trainer, payload)
    result["episode"] = episode_profile(environment, trainer, replay, int(payload["update_index"]))
    result["episode"]["intra_op_threads"] = args.episode_threads
    print(json.dumps(result["episode"]), flush=True)
    if device.type == "cpu" and args.evaluation_workers:
        result["evaluation"] = evaluation_profile(args.run, args.evaluation_workers)
        print(json.dumps({k: v for k, v in result["evaluation"].items()
                          if k not in ("serial", "parallel")}), flush=True)
    core.atomic_json(args.out / "profile.json", result)
    print("RUNTIME_PROFILE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
