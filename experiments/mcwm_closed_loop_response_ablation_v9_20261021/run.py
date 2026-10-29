"""Replay one unchanged prefix; compare nested response fits and fixed-weight ablations."""

import argparse
import json
import os
import time
from pathlib import Path

from bootstrap import HERE, formal, load_learner, write
import numpy as np
import torch

from stream import create_target_env
from experiments.mcwm_closed_loop_formal_20261020.scenarios import AFFECTED_LINES, UPSTREAM_STOP, install_scenario
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalIntervalCalibration
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedTrainer
from experiments.mcwm_closed_loop_paired_world_v6_20261020.paired_world import PairedWorld
from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_action_fork_v7_20261020.run import counters, generalized, train_packet
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.predictions import AblationPrediction
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.ablation_world import AblationResponseWorld
from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5


class SelectionComplete(Exception):
    """Stop at the issue clock before applying a diagnostic action."""


class MemoryStream(ForkEventStream):
    def __init__(self, *args, response_memory, **kwargs):
        self.response_memory = response_memory
        super().__init__(*args, **kwargs)

    def _close_physical(self, event):
        # Includes real terminal events, which never reach on_issue.
        self.response_memory.observe_event(event)
        return super()._close_physical(event)


def run(args):
    p = json.loads((HERE / "protocol.json").read_text())
    if args.preflight:
        p.update(p["preflight"])
    cell = dict(seed=args.seed, scenario=args.scenario, target_line=args.target_line)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = hook = stream = None

    def progress(stage, now=0., **fields):
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=now,
            total=p["selection_window_seconds"][1], unit="sim_seconds", phase=stage,
            elapsed_seconds=time.monotonic()-started, **fields)), flush=True)

    try:
        reference_root = HERE.parent / p["reference_v7"] / ("preflight" if args.preflight else "runs")
        reference_root /= Path(args.scenario) / args.target_line / f"seed{args.seed}"
        references = {name: json.loads((reference_root / name / "summary.json").read_text())
                      for name in ("low", "high")}
        if any(r["status"] != "complete" for r in references.values()):
            raise ValueError("Both original v7 truth branches must be complete")
        v8root = HERE.parent / p["reference_v8"]
        v8path = v8root / "preflight/summary.json" if args.preflight else v8root / "runs" / args.scenario / args.target_line / f"seed{args.seed}" / "summary.json"
        v8reference = json.loads(v8path.read_text())
        if v8reference["status"] != "complete":
            raise ValueError("The original v8 response reference must be complete")
        progress("load_initializer")
        metadata = json.loads((HERE.parent / "mcwm_closed_loop_formal_20261020/inputs/history_metadata.json").read_text())
        contract = formal.TransitMDPConfigV7(**metadata["contract"])
        transform = ExecutionAction(contract)
        learner, replay, _, contract, stations, initializer = load_learner(args.seed,
            trainer_class=ProbedTrainer, arm="model", action_transform=transform)
        initial_updates = learner.total_updates
        source = json.loads((HERE / p["reference_source_intervals"]).resolve().read_text())
        physical = LocalIntervalCalibration(source, contract, shock_at=p["shock_at_seconds"],
            delay_seconds=p["feedback_delay_seconds"], learning_end_seconds=p["physical_feedback_end_seconds"],
            min_source_per_line=p["source_intervals_per_line_minimum"],
            max_source_per_line=p["source_intervals_per_line_limit"])
        progress("initial_world_fit")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 20000)
            world = PairedWorld(replay, seed=args.seed, updates=p["model_init_updates"],
                                action_transform=transform, physical=physical, enabled=False)
        world.name, world.context_enabled, world.paired_probe_deadlines = "action_model", False, []
        learner.world = world
        response = AblationResponseWorld(world, physical, ridge=p["response_ridge"])
        budget = formal.PacketBudget(p)
        last_proposal = None
        result = None
        external_seed = args.seed + p["target_seed_offset"]
        initializer_receipt = dict(path=str(initializer), initial_updates=initial_updates,
            history_rows=len(replay.history["s"]), source_rows=len(replay.source["s"]),
            model_init_updates=p["model_init_updates"], model_refit_updates=p["model_refit_updates"])

        def policy(state):
            nonlocal last_proposal
            last_proposal = learner.act(state[None], deterministic=True)[0]
            return transform.numpy(state, last_proposal)

        def issue(event):
            event["proposed_a"] = last_proposal.copy()
            context = response.context(event)
            physical.register(event)
            world.register(event)
            response.register(event, context)

        def selector(event):
            nonlocal result
            now = event["started_at"]
            lo, hi = p["selection_window_seconds"]
            if not (lo <= now <= hi and event["line_id"] == args.target_line
                    and event["station_id"] == UPSTREAM_STOP):
                return None
            state = event["s"]
            low = transform.numpy(state, np.array([-1.], np.float32))
            high = transform.numpy(state, np.array([1.], np.float32))
            if float(high[0]-low[0]) <= 1e-7:
                return None
            counts_before = counters(learner, world, physical)
            context = response.context(event)
            probe = AblationPrediction(learner, state, **context,
                numeric_lines=response.numeric_lines,
                readouts_by_arm={arm: dict(online=response.readouts_by_arm[arm],
                                          frozen_common=response.frozen_readouts_by_arm[arm])
                                 for arm in response.readouts_by_arm},
                seed=args.seed + 300000 + AFFECTED_LINES.index(args.target_line))
            prefix = dict(initializer_receipt=initializer_receipt, feedback_receipts=list(budget.receipts),
                policy_updates=learner.total_updates-initial_updates,
                world_updates=world.model_updates, world_refits=world.refits, trust_updates=world.trust_updates,
                physical_labels=sum(physical.learned_by_line.values()), external_seed=external_seed,
                target_labels=sum(row["records"] for row in budget.receipts), reference_feedback_prefix_exact=True)
            snapshot = stream._snapshot()
            passengers = summarize_passengers_v5(env._bridge, snapshot)
            selected = dict(time=now, sequence=event["sequence"], line=event["line_id"],
                vehicle_id=event["vehicle_id"], trip_id=event["trip_id"], station=event["station_id"],
                state=state.tolist(), raw_state=event["raw_s"].tolist(),
                low_action=float(low[0]), high_action=float(high[0]),
                prefix_cost=generalized(snapshot), prefix_departed=passengers["departed_passengers"])
            prediction_record = probe.summary()
            checks = dict(
                old_truth_pair_selected_identical=references["low"]["selected"] == references["high"]["selected"],
                selected_state_and_clock_exact=all(selected[k] == references["low"]["selected"][k] for k in selected),
                training_prefix_exact=prefix == references["low"]["prefix"] == references["high"]["prefix"],
                legacy_predictions_exact=probe.legacy.summary() == references["low"]["prediction"] == references["high"]["prediction"],
                v8_prefix_exact={k: v for k, v in prefix.items() if k != "feedback_receipts"} == v8reference["prefix"],
                full_v8_predictions_exact=prediction_record["arms"]["memory_interaction"]["branches"] == v8reference["prediction"]["branches"],
                all_arms_trust_exact=all(
                    arm["branches"][branch]["candidates"][candidate]["trust"] ==
                    prediction_record["legacy"]["branches"][branch]["candidates"][candidate]["trust"]
                    for arm in prediction_record["arms"].values()
                    for branch in ("online", "frozen_common") for candidate in ("low", "high")),
                no_candidate_training_or_deployment=counters(learner, world, physical) == counts_before,
                response_label_budget=response.labels == prefix["target_labels"],
                frozen_response_initialization_zero=all(not torch.count_nonzero(v).item() for v in response.frozen_readouts.values()),
                suffix_not_rerun=stream.fork is None,
                no_truth_labels_in_training=True)
            comparison = probe.compare(references["low"]["next_event"], references["high"]["next_event"],
                low_cost_300=references["low"]["suffix"]["cost"], high_cost_300=references["high"]["suffix"]["cost"])
            result = dict(status="complete" if all(checks.values()) else "integrity_failed",
                cell=cell, preflight=args.preflight, selected={k: v for k, v in selected.items() if k not in ("state", "raw_state")},
                integrity=dict(checks=checks, all_pass=all(checks.values())),
                prefix={k: v for k, v in prefix.items() if k != "feedback_receipts"},
                global_normalizer=dict(mean=world.norm.g_mean.cpu().tolist(), std=world.norm.g_std.cpu().tolist()),
                response=response.summary(now), prediction=prediction_record, comparison=comparison,
                truth={branch: {key: references[branch]["next_event"][key]
                                for key in ("global_decoded", "bellman", "zero_duration")}
                       for branch in ("low", "high")},
                native=dict(sumo_version=hook.api.getVersion()[1], stopped_before_diagnostic_command=True),
                budget=dict(prefix_sim_seconds=now, new_suffix_sim_seconds=0, reused_v7_branches=2,
                    selected_prefix_labels=prefix["target_labels"], response_fit_labels=response.labels,
                    truth_labels_enter_training=0, response_ridge_refits=response.refits, ridge_solutions=6*response.refits),
                wall_seconds=time.monotonic()-started)
            raise SelectionComplete()

        write(out / "protocol.json", dict(arguments={**vars(args), "out": str(out)}, protocol=p))
        env = create_target_env(os.environ["THESIS_SUMO_ROOT"],
            Path(os.environ["THESIS_H2O_ROOT"]) / "bus_h2o/network_data/a_sorted_busline_edge.xml",
            episode_seconds=p["horizon_seconds"], seed=external_seed)
        stream = MemoryStream(env, stations, contract, domain="target", response_memory=response.memory,
            delay_seconds=p["feedback_delay_seconds"], execute_transform=False,
            on_issue=issue, selector=selector, fork_duration_seconds=p["suffix_seconds"])
        stream.reset()
        hook = install_scenario(env, args.scenario, shock_at=p["shock_at_seconds"],
                                factor=p["speed_factor"], metric_every=p["metric_interval_seconds"])
        last_progress = time.monotonic()
        try:
            while not stream.done:
                released = stream.step(policy)
                if stream.now > p["selection_window_seconds"][1]:
                    break
                budget.ingest(released)
                physical.observe(stream.release_intervals(), stream.now)
                while budget.due(stream.now):
                    packet, discarded = budget.take(stream.now)
                    response.accept(packet, discarded, stream.now)
                    train_packet(learner, replay, world, packet, discarded,
                                 budget.receipts[-1], stream.now, p["model_refit_updates"], contract)
                    response.refit()
                    progress("prefix_training", stream.now, packets=len(budget.receipts))
                if time.monotonic()-last_progress >= 45:
                    progress("prefix", stream.now)
                    last_progress = time.monotonic()
        except SelectionComplete:
            pass
        if result is None:
            result = dict(status="selection_missing", cell=cell, preflight=args.preflight,
                          integrity=dict(all_pass=False), budget=dict(prefix_sim_seconds=stream.now,
                          new_suffix_sim_seconds=0, truth_labels_enter_training=0))
        write(out / "summary.json", result)
        if args.preflight:
            write(out / "gate.json", {k: result[k] for k in ("status", "cell", "integrity", "budget")})
        progress(result["status"], stream.now)
        return result
    except Exception as error:
        write(out / "failure.json", dict(status="failed", cell=cell, type=type(error).__name__, message=str(error),
              time=stream.now if stream is not None else None))
        raise
    finally:
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(6201, 6202), required=True)
    parser.add_argument("--scenario", choices=("stable", "local_speed_drop"), required=True)
    parser.add_argument("--target-line", choices=AFFECTED_LINES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())
