#!/usr/bin/env python3
"""Isolated local-interval and model-TD mechanism diagnosis on the fixed bus day."""

import argparse
import copy
import json
import os
from pathlib import Path
import time

from bootstrap import HERE, formal, load_learner, write
import numpy as np
import torch
from stream import EventStream, create_target_env, policy_action_to_hold_seconds_v7
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario
from experiments.mcwm_closed_loop_formal_20261020.scenarios import (
    AFFECTED_LINES, UPSTREAM_STOP)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.interval import IntervalDiagnostic
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import DiagnosticTrainer
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.world_diagnostics import DiagnosticWorld


def period(at):
    if at < 9000:
        return "before"
    if at < 12000:
        return "early_after"
    if at < 15000:
        return "middle_after"
    if at < 18000:
        return "late_after"
    return "drain"


class Moments:
    def __init__(self):
        self.data = {}

    def add(self, values, group="all"):
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row = self.data.setdefault((group, key), [0, 0., float("inf"), -float("inf")])
                row[0] += 1
                row[1] += float(value)
                row[2] = min(row[2], float(value))
                row[3] = max(row[3], float(value))

    def summary(self):
        result = {}
        for (group, key), (n, total, low, high) in self.data.items():
            result.setdefault(group, {})[key] = dict(n=n, mean=total/n, min=low, max=high)
        return result


def action_probe(learner, packet, contract):
    proposed = learner.act(np.stack([row.s for row in packet]), deterministic=True)
    executed = [completion_safety_action(row.raw_s, action, "quarter-load-sixty", contract)
                for row, action in zip(packet, proposed)]
    hold = lambda actions: np.array([float(policy_action_to_hold_seconds_v7(a[0], contract))
                                    for a in actions])
    return hold(proposed), hold(executed)


def run(args):
    p = json.loads((HERE/"protocol.json").read_text())
    if args.preflight:
        p.update(model_init_updates=2, model_refit_updates=1, policy_updates_per_packet=1,
                 feedback_last_deadline=1800, target_record_budget=15,
                 policy_update_budget=3, shock_at_seconds=1350)
    if args.seed not in p["training_seeds"] or ((args.mode == "prediction") != (args.arm == "common")):
        raise ValueError("Invalid seed or mode/arm")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refuse to overwrite run {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(stage, now=0., **extra):
        print("SCHEDULEURM_PROGRESS "+json.dumps(dict(current=now, total=24000,
            unit="sim_seconds", phase=stage, elapsed_seconds=time.monotonic()-started,
            **extra)), flush=True)
    progress("load_initializer")
    learner, replay, make_replay, contract, stations, initializer = load_learner(
        args.seed, trainer_class=DiagnosticTrainer,
        arm=args.arm if args.mode == "control" else "real1")
    initial_updates = learner.total_updates
    worlds = {}
    physical = None
    if args.mode == "prediction" or args.arm in ("model", "context"):
        progress("initial_world_fit")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed+20000)
            base = DiagnosticWorld(replay, seed=args.seed, updates=p["model_init_updates"])
        if args.mode == "prediction":
            worlds = {"frozen_model_reference": copy.deepcopy(base), "online": base,
                      "context": copy.deepcopy(base)}
            for name, world in worlds.items():
                world.name = name
                world.replay = make_replay()
                world.context_enabled = name == "context"
                world.fixed_reference = name == "frozen_model_reference"
            source = json.loads((HERE/"inputs/source_intervals.json").read_text())
            physical = IntervalDiagnostic(source, contract, shock_at=p["shock_at_seconds"],
                delay_seconds=p["feedback_delay_seconds"],
                learning_end_seconds=p["physical_feedback_end_seconds"],
                min_source_per_line=p["source_intervals_per_line_minimum"],
                max_source_per_line=p["source_intervals_per_line_limit"])
        else:
            base.name = args.arm
            base.context_enabled = args.arm == "context"
            worlds[args.arm] = base
            learner.world = base
    write(out/"protocol.json", dict(arguments={**vars(args), "out": str(out)}, protocol=p,
        initializer=str(initializer), initial_updates=initial_updates,
        history_rows=len(replay.history["s"]), source_rows=len(replay.source["s"]),
        policy_config=formal.asdict(learner.config),
        physical_labels_enter_policy=False,
        policy_objectives={"real1": [1., 0.], "real075": [.75, 0.],
                           "model": [.75, .25], "context": [.75, .25]}))
    budget = formal.PacketBudget(p)
    action_stats, update_stats = Moments(), Moments()
    feedback_coverage = {}
    def issue(event):
        for world in worlds.values():
            world.register(event)
        if physical is not None:
            physical.register(event)
        proposed = float(policy_action_to_hold_seconds_v7(event["proposed_a"][0], contract))
        executed = float(policy_action_to_hold_seconds_v7(event["a"][0], contract))
        action_stats.add(dict(proposed_hold_seconds=proposed, executed_hold_seconds=executed,
            execution_changed=float(abs(executed-proposed)>1e-5)), period(event["started_at"]))
    env = hook = stream = None
    rounds = total_rows = after_update_decisions = 0
    last_progress = time.monotonic()
    try:
        env = create_target_env(os.environ["THESIS_SUMO_ROOT"],
            Path(os.environ["THESIS_H2O_ROOT"])/"bus_h2o/network_data/a_sorted_busline_edge.xml",
            episode_seconds=p["horizon_seconds"], seed=args.seed+p["target_seed_offset"])
        stream = EventStream(env, stations, contract, domain="target",
            delay_seconds=p["feedback_delay_seconds"], on_issue=issue)
        stream.reset()
        hook = install_scenario(env, args.scenario, shock_at=p["shock_at_seconds"],
            factor=p["speed_factor"], metric_every=p["metric_interval_seconds"])
        native_version = hook.api.getVersion()[1]
        with (out/"prequential.jsonl").open("w") as predictions, \
             (out/"interval_predictions.jsonl").open("w") as intervals, \
             (out/"td_diagnostics.jsonl").open("w") as td_file:
            while not stream.done:
                previous = stream.issued_count
                budget.ingest(stream.step(lambda s: learner.act(s[None], deterministic=True)[0]))
                if rounds and args.mode == "control":
                    after_update_decisions += stream.issued_count-previous
                actual_intervals = stream.release_intervals()
                if physical is not None:
                    for log in physical.observe(actual_intervals, stream.now):
                        intervals.write(json.dumps(log)+"\n")
                    intervals.flush()
                while budget.due(stream.now):
                    packet, discarded = budget.take(stream.now)
                    total_rows += len(packet)
                    for row in packet:
                        group = feedback_coverage.setdefault(period(row.started_at),
                            dict(records=0, by_line={}, directly_affected_upstream_records=0,
                                 directly_affected_upstream_by_line={line: 0 for line in AFFECTED_LINES}))
                        group["records"] += 1
                        group["by_line"][row.line_id] = group["by_line"].get(row.line_id, 0)+1
                        if row.line_id in AFFECTED_LINES and row.station_id == UPSTREAM_STOP:
                            group["directly_affected_upstream_records"] += 1
                            group["directly_affected_upstream_by_line"][row.line_id] += 1
                    for world in worlds.values():
                        for seq in discarded:
                            world.pending.pop(seq, None)
                        logs = world.observe(packet, stream.now)
                        if len(logs) != len(packet):
                            raise RuntimeError("Selected global label lacks issue-time prediction")
                        for log in logs:
                            predictions.write(json.dumps(log)+"\n")
                    predictions.flush()
                    if args.mode == "prediction":
                        for world in worlds.values():
                            if not world.fixed_reference:
                                world.replay.add_target(packet)
                                world.refit(p["model_refit_updates"])
                                world.recenter_context(packet)
                    else:
                        before = action_probe(learner, packet, contract)
                        replay.add_target(packet)
                        for world in worlds.values():
                            world.refit(p["model_refit_updates"])
                            world.recenter_context(packet)
                        diagnostic = None
                        for step in range(p["policy_updates_per_packet"]):
                            metrics = learner.continue_step(collect_diagnostics=step == 0)
                            update_stats.add(metrics, period(stream.now))
                            if step == 0:
                                diagnostic = metrics.get("td_diagnostic")
                        after = action_probe(learner, packet, contract)
                        td_file.write(json.dumps(dict(packet=rounds+1, deadline=budget.receipts[-1]["deadline"],
                            processed_at=stream.now, phase=period(stream.now), diagnostic=diagnostic,
                            proposed_action_change_seconds=float(np.abs(after[0]-before[0]).mean()),
                            executed_action_change_seconds=float(np.abs(after[1]-before[1]).mean())))+"\n")
                        td_file.flush()
                    rounds += 1
                    progress("target_loop", stream.now, packets=rounds, target_records=total_rows,
                             policy_updates=learner.total_updates-initial_updates)
                    write(out/"progress.json", dict(target_time_seconds=stream.now, packets=rounds,
                        target_records=total_rows, policy_updates=learner.total_updates-initial_updates))
                    last_progress = time.monotonic()
                if time.monotonic()-last_progress >= 45:
                    progress("target_loop", stream.now, packets=rounds, target_records=total_rows)
                    last_progress = time.monotonic()
                if args.preflight and stream.now >= 2100:
                    break
        hook.capture_final()
        snapshot = env._bridge.passenger_time_ledger.snapshot()
        expected = p["policy_update_budget"] if args.mode == "control" else 0
        if total_rows != p["target_record_budget"] or learner.total_updates-initial_updates != expected:
            raise RuntimeError("Diagnostic data/update budget differs")
        if not args.preflight and (not stream.done or snapshot.time_seconds != 24000):
            raise RuntimeError("Diagnostic target did not finish full horizon")
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        from SimpleSAC.transit_v3.evaluation import summarize_service
        passengers = summarize_passengers_v5(env._bridge, snapshot)
        if passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("Passenger ledger imbalance")
        cost = 2*snapshot.accrued_wait_seconds+snapshot.accrued_in_vehicle_seconds
        with (out/"metrics.jsonl").open("w") as f:
            for row in hook.metrics:
                f.write(json.dumps(row)+"\n")
        summary = dict(status="complete", seed=args.seed, scenario=args.scenario, arm=args.arm,
            mode=args.mode, preflight=args.preflight, full_horizon_completed=stream.done,
            target_time_seconds=stream.now, target_online_training_events=total_rows,
            policy_updates=learner.total_updates-initial_updates, packets=rounds,
            decisions_after_policy_update=after_update_decisions, sumo_version=native_version,
            generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            passengers=passengers, service=summarize_service(env._bridge),
            scenario_metadata=hook.metadata(), feedback_receipts=budget.receipts,
            worlds={name: world.summary() for name, world in worlds.items()},
            physical_intervals=physical.summary() if physical is not None else None,
            physical_learning_enters_policy=False, action_statistics=action_stats.summary(),
            global_feedback_coverage=feedback_coverage,
            update_statistics=update_stats.summary(), wall_seconds=time.monotonic()-started)
        write(out/"summary.json", summary)
        progress("complete", stream.now, target_records=total_rows)
        return summary
    except Exception as error:
        write(out/"failure.json", dict(status="failed", type=type(error).__name__,
            message=str(error), seed=args.seed, arm=args.arm, scenario=args.scenario,
            target_time_seconds=stream.now if stream is not None and stream.episode_id >= 0 else None))
        raise
    finally:
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "prediction"), required=True)
    parser.add_argument("--arm", choices=("real1", "real075", "model", "context", "common"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", choices=("stable", "local_speed_drop"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
