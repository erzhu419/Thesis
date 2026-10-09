"""Matched retraining of the existing upper-state holding-statistics ablation."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts"), str(HERE)]
import numpy as np
import torch
import runner_v3
import heldout_final_eval as original
from measurement import measured_environment

CONFIG = RUNTIME / "configs_ablation/H_timetable_v5_stable_300.yaml"
ARMS = ("full", "masked")
SEEDS = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
TRAIN_EPISODES = 100
CHECKPOINT_EP = TRAIN_EPISODES - 1
MEASURED_ENV = measured_environment(runner_v3.env_bus)
_eval_runner = None
_eval_test_plan = "initial"


def config(arm, seed, training_episodes=TRAIN_EPISODES):
    value = runner_v3.load_config(str(CONFIG))
    value["seed"] = seed
    value["_name"] = "feedback_" + arm
    value["training"]["total_episodes"] = training_episodes
    value["coupling"]["ablate_holding_feedback"] = arm == "masked"
    return value


def make_runner(arm, seed, device="cpu", measured=False, training_episodes=TRAIN_EPISODES):
    original.seed_everything(seed)
    if measured:
        runner_v3.env_bus = MEASURED_ENV
    runner = runner_v3.TransitDuetV2Runner(config(arm, seed, training_episodes), device=device)
    runner.diag = original._NullDiag()
    return runner


def check_channel(runner):
    raw = np.arange(1, 12, dtype=np.float32) / 12
    saved_builder = runner.env._build_upper_state_v2
    saved_policy = runner.upper_trainer.policy_net.get_action
    saved_flag = runner.ablate_holding_feedback
    received = []
    runner.env._build_upper_state_v2 = lambda trip: raw.copy()

    def capture(state, **kwargs):
        received.append(state.copy())
        return np.array([0.0], dtype=np.float32)

    runner.upper_trainer.policy_net.get_action = capture
    for flag in (False, True):
        runner.ablate_holding_feedback = flag
        runner._upper_callback_v2(None, SimpleNamespace(direction=True, launch_time=0, launch_turn=0))
    expected = raw.copy()
    expected[5:8] = 0
    np.testing.assert_array_equal(received[0], raw)
    np.testing.assert_array_equal(received[1], expected)
    runner.env._build_upper_state_v2 = saved_builder
    runner.upper_trainer.policy_net.get_action = saved_policy
    runner.ablate_holding_feedback = saved_flag


def state_statistics(runner, arm):
    transitions = runner._episode_upper_transitions
    if not transitions:
        return 0
    values = np.stack([t[key][5:8] for t in transitions for key in ("s", "ns")])
    if arm == "masked":
        assert np.count_nonzero(values) == 0, "Mask absent from upper replay"
    return int(np.count_nonzero(values))


def benchmark(device, arm):
    torch.set_num_threads(1)
    if device == "cuda":
        assert torch.cuda.is_available()
    full, masked = config("full", 98765), config("masked", 98765)
    masked["_name"] = full["_name"]
    masked["coupling"]["ablate_holding_feedback"] = False
    assert masked == full
    runner = make_runner(arm, 98765, device)
    counterpart = make_runner("masked" if arm == "full" else "full", 98765, device)
    for layer in ("upper_trainer", "lower_trainer"):
        left = getattr(runner, layer).policy_net.state_dict()
        right = getattr(counterpart, layer).policy_net.state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)
    del counterpart
    check_channel(runner)
    original.seed_everything(98765)
    rows = []
    for episode in (15, 16):
        start = time.monotonic()
        row = runner.run_episode(episode, training=True)
        observed = state_statistics(runner, arm)
        rows.append(dict(seconds=time.monotonic() - start, env_seconds=row["wall_env_s"],
                         update_seconds=row["wall_train_s"], nonzero_feedback_values=observed))
    if arm == "full":
        assert sum(row["nonzero_feedback_values"] for row in rows) > 0
    result = dict(device=device, arm=arm, checks_passed=True, episodes=rows,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  peak_cuda_mb=torch.cuda.max_memory_allocated() / 1024 ** 2 if device == "cuda" else 0)
    output = HERE / "benchmarks"
    output.mkdir(exist_ok=True)
    (output / f"{device}_{arm}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def eval_setup(arm, seed, checkpoint_dir, checkpoint_ep=CHECKPOINT_EP, test_plan="initial"):
    global _eval_runner, _eval_test_plan
    torch.set_num_threads(1)
    _eval_test_plan = test_plan
    _eval_runner = make_runner(arm, seed, measured=True, training_episodes=checkpoint_ep + 1)
    _eval_runner.lower_trainer.load(str(Path(checkpoint_dir) / f"lower_ep{checkpoint_ep}.pt"))
    _eval_runner.upper_trainer.load(str(Path(checkpoint_dir) / f"upper_ep{checkpoint_ep}.pt"))


def eval_case(episode):
    runner = _eval_runner
    seed = runner.cfg["seed"]
    fleet_base, env_base = (9300000, 9400000) if _eval_test_plan == "initial" else (11300000, 12400000)
    fleet = int(np.random.RandomState(fleet_base + seed).randint(8, 17, size=50)[episode])
    env_seed = env_base + seed * 1000 + episode
    original.seed_everything(env_seed)
    row = runner.run_episode(10000, training=False, N_fleet_override=fleet)
    resources = runner.env.resource_meter.results(runner.env)
    assert resources["completed_trips"] == 262
    feedback_nonzero = state_statistics(runner, "masked" if runner.ablate_holding_feedback else "full")
    return dict(episode=episode, environment_seed=env_seed, fleet_budget=fleet,
                feedback_nonzero=feedback_nonzero,
                composite=row["avg_wait_min"] / 10 + row["fleet_overshoot"] ** 2 / fleet + row["headway_cv"],
                wait_min=row["avg_wait_min"], onboard_min=row["avg_onboard_time_min"],
                total_min=row["avg_total_passenger_time_min"], peak_fleet=row["peak_fleet"],
                headway_cv=row["headway_cv"], upper_delta_mean=row["upper_delta_mean"],
                dispatch_lateness=row["dispatch_lateness_mean"], **resources)


def train(arm, seed, device, workers, training_episodes=TRAIN_EPISODES,
          test_plan="initial", train_only=False):
    torch.set_num_threads(1)
    root = HERE / ("runs" if training_episodes == 100 else "runs300") / f"{arm}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "training.jsonl").exists(), "Use a fresh run directory for full retraining"
    runner = make_runner(arm, seed, device, training_episodes=training_episodes)
    runner.log_dir = str(root)
    (root / "config.json").write_text(json.dumps(runner.cfg, indent=2) + "\n")
    feedback_nonzero = 0
    with (root / "training.jsonl").open("w") as stream:
        for episode in range(training_episodes):
            row = runner.run_episode(episode, training=True)
            feedback_nonzero += state_statistics(runner, arm)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"phase=train progress={episode+1}/{training_episodes} env={row['wall_env_s']} update={row['wall_train_s']}", flush=True)
            if (episode + 1) % 50 == 0:
                runner._save_checkpoint(episode)
                if episode in (99, 299):
                    (root / "checkpoints" / f"ready_ep{episode}.json").write_text(json.dumps(dict(
                        arm=arm, seed=seed, checkpoint_episode=episode, run_budget=training_episodes)) + "\n")
    if arm == "full":
        assert feedback_nonzero > 0
    del runner
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Training complete: {training_episodes} episodes", flush=True)
    if not train_only:
        evaluate(arm, seed, workers, training_episodes, test_plan)


def evaluate(arm, seed, workers, training_episodes=TRAIN_EPISODES,
             test_plan="initial", checkpoint_ep=None):
    checkpoint_ep = training_episodes - 1 if checkpoint_ep is None else checkpoint_ep
    assert checkpoint_ep in (99, 299) and checkpoint_ep < training_episodes
    root = HERE / ("runs" if training_episodes == 100 else "runs300") / f"{arm}_seed{seed}"
    # Only read the saved prefix; the continuous run may still be appending later episodes.
    with (root / "training.jsonl").open() as stream:
        retained = [json.loads(next(stream)) for _ in range(checkpoint_ep + 1)]
    assert [row["ep"] for row in retained] == list(range(checkpoint_ep + 1))
    for layer in ("lower", "upper"):
        assert (root / "checkpoints" / f"{layer}_ep{checkpoint_ep}.pt").is_file()
    results = root / ("results" if test_plan == "initial" else f"results_ep{checkpoint_ep}_extended")
    results.mkdir(exist_ok=True)
    assert not (results / "episodes.jsonl").exists(), "Do not overwrite evaluation results"
    start = time.monotonic()
    rows = []
    with (results / "episodes.jsonl").open("w") as stream:
        with mp.get_context("spawn").Pool(workers, initializer=eval_setup,
                initargs=(arm, seed, str(root / "checkpoints"), checkpoint_ep, test_plan)) as pool:
            for row in pool.imap_unordered(eval_case, range(50)):
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"phase=evaluation progress={len(rows)}/50", flush=True)
    assert {r["episode"] for r in rows} == set(range(50))
    (results / "complete.json").write_text(json.dumps(dict(arm=arm, training_seed=seed,
        training_episodes=checkpoint_ep + 1, checkpoint_episode=checkpoint_ep,
        run_training_budget=training_episodes, test_plan=test_plan,
        evaluation_episodes=50, workers=workers,
        recorded_training_seconds=sum(r["wall_env_s"] + r["wall_train_s"] for r in retained),
        evaluation_seconds=time.monotonic() - start,
        evaluation_feedback_nonzero=sum(r["feedback_nonzero"] for r in rows))) + "\n")
    print("Feedback ablation complete", flush=True)


def evaluation_smoke():
    torch.set_num_threads(1)
    root = HERE / "benchmarks/smoke100_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    runner = make_runner("masked", 98765)
    runner.lower_trainer.save(str(root / f"lower_ep{CHECKPOINT_EP}.pt"))
    runner.upper_trainer.save(str(root / f"upper_ep{CHECKPOINT_EP}.pt"))
    del runner
    with mp.get_context("spawn").Pool(2, initializer=eval_setup,
            initargs=("masked", 98765, str(root))) as pool:
        rows = list(pool.imap_unordered(eval_case, (0, 1)))
    result = dict(checks_passed=True, test_only_random_policies=True,
                  evaluation_episodes=len(rows), completed_trips=[r["completed_trips"] for r in rows])
    (HERE / "benchmarks/evaluation_smoke_100.json").write_text(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--eval-smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--training-episodes", type=int, choices=(100, 300), default=100)
    parser.add_argument("--checkpoint-episode", type=int, choices=(99, 299))
    parser.add_argument("--test-plan", choices=("initial", "extended"), default="initial")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.eval_smoke:
        evaluation_smoke()
    elif args.benchmark:
        benchmark(args.device, args.arm)
    elif args.evaluate_only:
        evaluate(args.arm, args.seed, args.workers, args.training_episodes,
                 args.test_plan, args.checkpoint_episode)
    else:
        train(args.arm, args.seed, args.device, args.workers, args.training_episodes,
              args.test_plan, args.train_only)
