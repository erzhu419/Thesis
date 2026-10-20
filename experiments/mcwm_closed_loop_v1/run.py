#!/usr/bin/env python3
"""Run a bounded target-data MC-WM loop; keep the frozen thesis experiments intact."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import platform
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("LIBSUMO_AS_TRACI", "quiet")

import numpy as np
import torch

from learning import (BUS, H2O, ClosedLoopTrainer, DomainReplay, OnlineWorld,
                      TransitTrainerConfig, learned_rows, load_table, row_table)
from stream import EventStream, TransitMDPConfigV7, create_source_env, create_target_env


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("frozen", "replay", "online", "context"), default="online")
    parser.add_argument("--target", choices=("sumo", "source-probe"), default="sumo")
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--target-seed", type=int, default=7201)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--history", type=Path, default=BUS / "runs/pilot_v1/target_fit.npz")
    parser.add_argument("--source-history", type=Path, default=BUS / "runs/pilot_v1/source_9101.npz")
    parser.add_argument("--source-env", type=Path,
                        default=Path("/home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o/calibrated_env"))
    parser.add_argument("--sumo-root", type=Path,
                        default=Path("/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"))
    parser.add_argument("--edge-xml", type=Path, default=H2O / "bus_h2o/network_data/a_sorted_busline_edge.xml")
    parser.add_argument("--sumo-cfg", type=Path)
    parser.add_argument("--passenger-obj-xml", type=Path)
    parser.add_argument("--initializer", type=Path, help="Existing v7 offline IQL initializer, read only")
    parser.add_argument("--delay-seconds", type=float, default=30.)
    parser.add_argument("--record-every", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=2048)
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--offline-updates", type=int, default=20)
    parser.add_argument("--model-init-updates", type=int, default=20)
    parser.add_argument("--refit-every", type=int, default=16, help="Newly released target records per refit")
    parser.add_argument("--model-refit-updates", type=int, default=5)
    parser.add_argument("--policy-updates", type=int, default=4, help="Updates per target refit interval")
    parser.add_argument("--source-batches-per-update", type=int, default=1)
    parser.add_argument("--max-target-batches", type=int, default=200, help="0 means no batch cap")
    parser.add_argument("--max-target-seconds", type=float, default=900., help="0 means full 24000-s horizon")
    parser.add_argument("--smoke", action="store_true", help="Small neural/policy update budgets; preserve horizon")
    args = parser.parse_args()
    if args.smoke:
        args.offline_updates = args.model_init_updates = 2
        args.model_refit_updates = args.policy_updates = 1
        args.refit_every = 2
    if (args.history_limit < 2 or args.recent_window < 2 or args.refit_every < 1
            or min(args.offline_updates, args.model_init_updates, args.model_refit_updates,
                   args.policy_updates, args.source_batches_per_update, args.max_target_batches,
                   args.max_target_seconds) < 0):
        parser.error("Invalid negative budget or empty fitting/window budget")
    return args


def run(args):
    started = time.monotonic()
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing evidence will not be overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    contract = TransitMDPConfigV7()
    audit = json.loads((BUS / "runs/pilot_v1/target_audit.json").read_text())
    stations = np.asarray(audit["station_max_by_line"], np.float32)
    history, source = (load_table(path, args.history_limit) for path in (args.history, args.source_history))
    replay = DomainReplay(history, source, contract, stations, args.device, args.recent_window)
    if args.initializer:
        payload = torch.load(args.initializer, map_location="cpu", weights_only=False)
        config = TransitTrainerConfig(**dict(payload["source_config"], use_ratio_correction=False))
        del payload
    else:
        config = TransitTrainerConfig(offline_objective="iql", online_backup="h2oplus",
            mixed_backup_lambda=.1, use_ratio_correction=False,
            ensemble_size=2 if args.smoke else 5, hidden_dim=64 if args.smoke else 256,
            batch_size=16 if args.smoke else 128, critic_loss="huber", gradient_clip_norm=10.)
    learner = ClosedLoopTrainer(replay, config=config, device=args.device, seed=args.seed)
    if args.initializer:
        learner.load_offline_initializer(args.initializer)
    else:
        for _ in range(args.offline_updates):
            learner.update(online=False, collect_metrics=False)
    initial_updates = learner.total_updates
    initial_actor = {key: value.detach().clone() for key, value in learner.actor.state_dict().items()}
    if args.arm in ("online", "context"):
        devices = [torch.cuda.current_device()] if args.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed+20_000)
            world = OnlineWorld(replay, seed=args.seed, updates=args.model_init_updates, context=args.arm == "context")
        learner.world = world
    else:
        world = None

    source_env = target_env = None
    source_stream = stream = None
    last_fit_count = batches = update_rounds = after_update_decisions = 0
    last_metrics, release_bounds, sumo_version = {}, [], None
    def on_issue(issued):
        nonlocal after_update_decisions
        if update_rounds:
            after_update_decisions += 1
        if world is not None and issued["sequence"] % args.record_every == 0:
            world.register(issued)

    def policy(state):
        return learner.act(state[None], deterministic=True)[0]

    write(args.out / "protocol.json", dict(arguments={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()}, contract=asdict(contract), policy_config=asdict(config),
        historical_rows=len(history["s"]), initial_source_rows=len(source["s"]),
        initial_policy_updates=initial_updates, target_is_proxy=args.target == "sumo",
        observation_permission="v7 current state including planned eligible arrival rate",
        prediction_order="issue and save -> label completed+available -> score -> learn -> next action",
        source_model="frozen after initial fit", source_history=str(args.source_history.resolve()),
        h2o_runtime_root=str(H2O.resolve()),
        runtime=dict(python=platform.python_version(), torch=torch.__version__),
        old_experiment="F-FINAL-225 unchanged; frozen arm here is a same-initializer reference",
        generated_replay="on-demand one-step samples; no persistent synthetic cache"))
    try:
        source_env = create_source_env(args.source_env, seed=args.seed+1000)
        source_stream = EventStream(source_env, stations, contract, domain="source")
        if args.target == "sumo":
            target_env = create_target_env(args.sumo_root, args.edge_xml, seed=args.target_seed,
                sumo_cfg=args.sumo_cfg, passenger_obj_xml=args.passenger_obj_xml)
        else:
            target_env = create_source_env(args.source_env, seed=args.target_seed)
        stream = EventStream(target_env, stations, contract, domain="target",
            delay_seconds=args.delay_seconds, record_every=args.record_every, on_issue=on_issue)
        stream.reset()
        if args.target == "sumo":
            import libsumo
            sumo_version = libsumo.getVersion()[1]
        with (args.out / "prequential.jsonl").open("w") as predictions, \
                (args.out / "physical_intervals.jsonl").open("w") as physical:
            while not stream.done:
                if args.max_target_batches and batches >= args.max_target_batches:
                    break
                if args.max_target_seconds and stream.now >= args.max_target_seconds:
                    break
                rows = stream.step(policy)
                batches += 1
                for interval in stream.release_intervals():
                    # Small inspectable process labels, without full state/trajectory dumps.
                    physical.write(json.dumps({key: getattr(interval, key) for key in (
                        "vehicle_id", "line_id", "trip_id", "station_id", "next_station_id",
                        "started_at", "completed_at", "available_at", "dt", "sequence", "semantics")}
                        | {"executed_action": interval.a.tolist()})+"\n")
                physical.flush()
                if world is not None:
                    logs = world.observe(rows, stream.now)
                    for log in logs:
                        predictions.write(json.dumps(log)+"\n")
                    predictions.flush()
                if rows:
                    release_bounds.append(dict(now=stream.now, rows=len(rows),
                        max_available_at=max(row.available_at for row in rows)))
                if args.arm == "frozen":
                    continue
                replay.add_target(rows)
                if replay.target_received-last_fit_count < args.refit_every:
                    continue
                for _ in range(args.source_batches_per_update):
                    if source_stream.done:
                        source_stream.reset()
                    replay.add_source(source_stream.step(policy))
                if world is not None:
                    world.refit(args.model_refit_updates)
                for _ in range(args.policy_updates):
                    last_metrics = learner.continue_step()
                update_rounds += 1
                last_fit_count = replay.target_received
                progress = dict(target_time_seconds=stream.now, target_controls=stream.issued_count,
                    target_records=replay.target_received, policy_updates=learner.total_updates-initial_updates,
                    world=world.summary() if world else None, last_metrics=last_metrics)
                write(args.out / "progress.json", progress)
                print(json.dumps({key: progress[key] for key in (
                    "target_time_seconds", "target_controls", "target_records", "policy_updates")}), flush=True)
        actor_change = sum(float((value-initial_actor[key]).abs().sum())
                           for key, value in learner.actor.state_dict().items())
        snapshot = stream._snapshot()
        summary = dict(arm=args.arm, target=args.target, smoke=args.smoke,
            full_horizon_completed=stream.done, target_time_seconds=stream.now, batches=batches,
            target_control_events=stream.issued_count, target_completed_records=stream.completed_count,
            target_released_records=stream.released_count, target_thinned_records=stream.thinned_count,
            target_online_training_events=replay.target_received, target_pending_delayed_records=stream.pending_completed_count,
            target_recent_learned_event_records=len(learned_rows(row_table(replay.recent))["s"]) if replay.recent else 0,
            same_vehicle_completed_intervals=len(stream.physical_intervals),
            uncompleted_same_vehicle_intervals=len(stream._vehicle_pending),
            policy_updates=learner.total_updates-initial_updates, update_rounds=update_rounds,
            actor_parameter_l1_change=actor_change, decisions_after_policy_update=after_update_decisions,
            source_controls=source_stream.issued_count, world=world.summary() if world else None,
            sumo_version=sumo_version,
            observed_partial_generalized_passenger_seconds=(
                2*snapshot.accrued_wait_seconds + snapshot.accrued_in_vehicle_seconds),
            release_bounds=release_bounds, last_metrics=last_metrics,
            wall_seconds=time.monotonic()-started,
            scope="Wiring/protocol verification; partial runs are not full-day control results")
        learner.save_policy_artifact(args.out / "policy.pt", extra=dict(arm=args.arm, seed=args.seed,
            contract_config=asdict(contract), target_online_training_events=replay.target_received,
            policy_updates=summary["policy_updates"], smoke=args.smoke))
        if world is not None:
            torch.save(dict(models={key: model.state_dict() for key, model in world.models.items()},
                normalizer=world.norm.state_dict(), context=world.context, family=world.family,
                summary=world.summary()), args.out / "world.pt")
        write(args.out / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("arm", "target", "target_time_seconds",
            "target_online_training_events", "policy_updates", "decisions_after_policy_update", "wall_seconds")}), flush=True)
        return summary
    except Exception as error:
        write(args.out / "failure.json", dict(error=type(error).__name__, message=str(error),
            batches=batches, target_time_seconds=stream.now if stream is not None and stream.episode_id >= 0 else None))
        raise
    finally:
        for env in (target_env, source_env):
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()


if __name__ == "__main__":
    run(parse_args())
