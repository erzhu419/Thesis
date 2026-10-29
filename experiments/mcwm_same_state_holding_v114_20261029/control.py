"""One fixed holding contrast with complete live-state inheritance."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT_PROCESS_PID = os.getpid()


class PrefixComplete(Exception):
    def __init__(self, receipt):
        self.receipt = receipt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def mechanism_checks(mechanism, arm, updates):
    fresh = arm == "fresh"
    expected = updates if fresh else 0
    return dict(
        declared_fresh_mechanism=mechanism["mode"] == (
            "critic_only_fresh_historical_coverage" if fresh else "original_no_fresh"),
        original_auxiliary_batches=all(mechanism[key] == expected for key in (
            "auxiliary_captures", "value_calls", "actor_calls", "temperature_calls"))
            and mechanism["original_batch_used"] == dict(
                value=expected, actor=expected, temperature=expected),
        critic_complete_replacements=mechanism["critic_replacement_unit"] == "rows"
            and mechanism["critic_complete_replacements"] == expected*64,
        historical_critic_coverage=mechanism["history_coverage_updates"] == expected
            and mechanism["history_critic_rows_per_update"] == (128 if fresh else 0),
        extra_critic_computation=mechanism["extra_critic_rows"] == expected*64,
        extra_bootstrap_computation=mechanism["extra_bootstrap_calls"] == expected
            and mechanism["extra_bootstrap_rows"] == expected*256,
        main_rng_preserved=mechanism["main_rng_preserved"] is True,
        common_targets_exact=mechanism["common_targets_exact"] is True)


class PacketBudget:
    """Original formal latest-five rule, without importing its other learner."""
    def __init__(self, p):
        self.p, self.rows, self.index, self.receipts = p, [], 0, []
        self.deadlines = list(range(p["feedback_first_deadline"], p["feedback_last_deadline"]+1,
                                    p["feedback_interval_seconds"]))

    def ingest(self, rows):
        self.rows.extend(rows)

    def due(self, now):
        return self.index < len(self.deadlines) and now >= self.deadlines[self.index]

    def take(self, now):
        deadline = self.deadlines[self.index]
        if now < deadline:
            raise ValueError("Feedback deadline has not arrived")
        lower = deadline-self.p["feedback_interval_seconds"]
        ready = [row for row in self.rows if row.available_at <= deadline]
        candidates = [row for row in ready if row.available_at > lower and row.dt > 0 and not row.terminal]
        n = self.p["feedback_records_per_packet"]
        if len(candidates) < n:
            raise RuntimeError(f"Insufficient feedback at {deadline}: {len(candidates)} eligible")
        selected = sorted(candidates, key=lambda row: (row.available_at, row.sequence))[-n:]
        self.rows = [row for row in self.rows if row.available_at > deadline]
        self.receipts.append(dict(deadline=deadline, processed_at=now, records=n,
            sequence_ids=[row.sequence for row in selected],
            max_available_at=max(row.available_at for row in selected)))
        self.index += 1
        return selected


class UpdateCounts:
    """Count completed calls to the unchanged optimizers and augmentation methods."""
    def __init__(self, learner):
        self.values = dict(actor=0, critic=0, value=0, alpha=0, ratio=0,
                           trust=0, trust_optimizer=0, model_batches=0, model_rows=0)
        self.model_batch_sizes = set()
        self.phase_start = self.phase_calls(learner)
        for name in ("actor", "critic", "value", "alpha"):
            self.wrap(getattr(learner, name+"_optimizer"), "step", name)
        self.wrap(learner.ratio.optimizer, "step", "ratio")
        self.wrap(learner.augmentation.trust.optimizer, "step", "trust_optimizer")
        self.wrap(learner.augmentation, "update_trust", "trust")
        generate = learner.augmentation.generate

        def counted_generate(actor, size):
            result = generate(actor, size)
            self.values["model_batches"] += 1
            self.values["model_rows"] += int(result[0]["observations"].shape[0])
            self.model_batch_sizes.add(int(size))
            return result
        learner.augmentation.generate = counted_generate

    def wrap(self, owner, name, counter):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            self.values[counter] += 1
            return result
        setattr(owner, name, call)

    @staticmethod
    def phase_calls(learner):
        return {key: value["calls"] for key, value in learner.costs.rows.items()}

    def phase_deltas(self, learner):
        current = self.phase_calls(learner)
        return {key: current.get(key, 0)-self.phase_start.get(key, 0)
                for key in current.keys() | self.phase_start.keys()}

    def snapshot(self):
        return dict(self.values, policy=self.values["actor"], world_refit=0)


def run(args):
    p = json.loads((HERE / "protocol.json").read_text())
    root, out = args.thesis_root.resolve(), args.out.resolve()
    if args.arm not in p["arms"] or args.scenario not in p["scenarios"]:
        raise ValueError("Case is outside the fixed pilot")
    out.mkdir(parents=True, exist_ok=False)
    cell = dict(training_seed=p["training_seed"], external_seed=p["external_seed"],
                scenario=args.scenario, arm=args.arm)
    result = dict(status="failed", cell=cell, periods={}, budget={}, counts={},
        integrity=dict(all_pass=False, checks={}, failed_checks=["runtime_error"]))
    env = hook = stream = learner = replay = counts = trace = services = updates = observer = None
    budget = PacketBudget(p)
    branch = None
    parent_out = out
    root_clock = None
    root_counts = root_feedback_slots = None
    branch_actions = root_receipt = None
    root_issue_count = 0
    binding = json.loads((HERE / "anchor_binding.json").read_text())
    anchor_spec = next(a for a in binding["anchors"] if a["scenario"] == args.scenario)
    anchor = dict(action=anchor_spec["records"]["fresh"]["actions"],
        command=anchor_spec["records"]["fresh"]["commands"], visit=anchor_spec["records"]["fresh"]["visits"])
    started, native_started = time.monotonic(), False
    native_started_wall = native_finished_wall = None
    initial_updates = initial_online = None
    update_attempts = completed_updates = 0
    prefix = None
    service_rows = service_events = 0
    frozen_counts = post18300_counts = None
    freeze_clock = post18300_clock = None
    last_update_clock = None

    def progress(phase):
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(phase=phase,
            current=stream.now if stream is not None and native_started else 0.,
            total=p["horizon_seconds"], unit="sim_seconds", packets=budget.index,
            updates=completed_updates, elapsed_seconds=time.monotonic()-started, **cell)), flush=True)

    def integrity(checks):
        return dict(all_pass=all(checks.values()), checks=checks,
                    failed_checks=[key for key, value in checks.items() if not value])

    try:
        import numpy as np
        import torch
        from experiments.mcwm_historical_coverage_v111_20261029.learning import load_learner, mechanism_summary
        progress("restore_full_mcwm")
        learner, replay, contract, stations, initialization = load_learner(root, out, args.arm, p)
        from state import global_rng, restore_global_rng
        learning_rng = global_rng(learner.device)
        from experiments.mcwm_closed_loop_v1.stream import EventStream, create_target_env
        from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.run import periods, scenario_checks
        from experiments.mcwm_closed_loop_formal_20261020.scenarios import install_scenario, LANE_IDS
        from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
        from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
        from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
        initial_updates, initial_online = learner.total_updates, learner.online_critic_updates
        initial_actor = {key: value.detach().clone() for key, value in learner.actor.state_dict().items()}
        counts = UpdateCounts(learner)
        transform = ExecutionAction(contract)
        trace = NativePrefixTrace(out / "prefix_trace.jsonl")
        services = (out / "prefix_services.jsonl").open("x")
        updates = (out / "updates.jsonl").open("x")
        clock_audit = dict(actor_real_clock=True, first_feedback_applied_at=None, policy_issues=0)
        action_audit = dict(raw_execution_exact=True, canonical_observation_exact=True,
            issues=0, canonical_different_events=0, max_canonical_hold_gap_seconds=0.,
            canonical_difference_threshold_seconds=1e-4)

        def policy(state):
            nonlocal branch, out, updates, root_clock, root_counts, root_feedback_slots
            nonlocal branch_actions, root_receipt, native_started_wall
            clock = float(env._current_events[0].sim_time)
            clock_audit["actor_real_clock"] &= bool(state[0] == np.float32(np.float32(clock)/24000))
            clock_audit["policy_issues"] += 1
            proposal = learner.act(state[None], deterministic=True)[0]
            if branch is None and stream.issued_count == anchor["action"]["sequence"]:
                from support import root_actions, flush_writers, child_detach, state_receipt, trace_adapter, compare_prefix_actions
                from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children
                if not np.array_equal(state, np.asarray(anchor["action"]["s"], np.float32)) or stream.now != anchor["action"]["started_at"]:
                    raise RuntimeError("Fixed root state or clock did not reproduce")
                branch_actions = root_actions(anchor, np.asarray(anchor["action"]["raw_s"], np.float32), proposal,
                    contract, anchor_spec["low"])
                if not branch_actions["all_pass"]:
                    raise RuntimeError("Original F cannot realize the frozen root actions")
                flush_writers(observer, updates)
                prefix_check = compare_prefix_actions(parent_out / "actions.jsonl",
                    root / p["v113_root"] / "runs" / args.scenario / "fresh" / "actions.jsonl",
                    anchor["action"]["sequence"]-1)
                write(parent_out / "root_prefix_check.json", prefix_check)
                if not prefix_check["all_pass"]:
                    raise RuntimeError("Complete action prefix differs before root")
                root_clock, root_counts = stream.now, counts.snapshot()
                root_feedback_slots = replay.summary()["target_received"]
                root_receipt = state_receipt(stream, learner, replay, budget, counts)
                write(parent_out / "root_state.json", root_receipt)
                live_trace = trace_adapter(observer)

                def prepare_child(key):
                    nonlocal branch, out, updates, native_started_wall
                    branch = key
                    out = parent_out / "branches" / key
                    updates, manifest = child_detach(observer, updates, out)
                    actual = state_receipt(stream, learner, replay, budget, counts)
                    write(out / "root_state.json", actual)
                    write(out / "prefix_sources.json", manifest)
                    if actual != root_receipt:
                        raise RuntimeError("Child root state differs from its live parent")
                    native_started_wall = time.time()

                selected = fork_children(parent_out / "branches", live_trace, lambda: stream.now,
                    counts.snapshot, prepare_child, timeout_seconds=p["native_limits"]["child_timeout_seconds"])
                if isinstance(selected, dict):
                    raise PrefixComplete(selected)
                return np.asarray(branch_actions[branch]["proposal"], np.float32)
            return proposal

        def issue(event):
            nonlocal root_issue_count
            raw = completion_safety_action(event["raw_s"], event["proposed_a"], "quarter-load-sixty", contract)
            canonical = transform.numpy(event["s"], event["proposed_a"])
            gap = abs(float(canonical[0])-float(event["a"][0]))*.5*contract.max_hold_seconds
            action_audit["issues"] += 1
            action_audit["raw_execution_exact"] &= bool(np.array_equal(raw, event["a"]))
            action_audit["canonical_observation_exact"] &= bool(np.array_equal(event["s"],
                raw_state_to_canonical_v7(event["raw_s"], stations, contract)))
            action_audit["canonical_different_events"] += gap > 1e-4
            action_audit["max_canonical_hold_gap_seconds"] = max(gap, action_audit["max_canonical_hold_gap_seconds"])
            if not trace.writer.closed:
                trace.observe(event, stream._snapshot())
            if event["sequence"] == anchor["action"]["sequence"]:
                from bus_h2o.mdp_contract_v7 import policy_action_to_hold_seconds_v7
                root_issue_count += 1
                selected = branch_actions[branch]
                state_exact = (np.array_equal(event["raw_s"], np.asarray(anchor["action"]["raw_s"], np.float32))
                    and np.array_equal(event["s"], np.asarray(anchor["action"]["s"], np.float32))
                    and event["started_at"] == anchor["action"]["started_at"])
                action_exact = np.array_equal(event["a"], np.asarray(selected["executed"], np.float32))
                write(out / "root_action.json", dict(branch=branch, sequence=event["sequence"],
                    started_at=event["started_at"], trip_id=event["trip_id"], station_id=event["station_id"],
                    expected_hold_seconds=anchor_spec[branch], proposed_a=event["proposed_a"].tolist(),
                    executed_a=event["a"].tolist(), actual_hold_seconds=float(policy_action_to_hold_seconds_v7(event["a"][0], contract)),
                    raw_F_exact=bool(action_exact), root_state_exact=bool(state_exact), only_root_override=True))
                if not state_exact or not action_exact or root_issue_count != 1:
                    raise RuntimeError("Root intervention state or F execution differs")
            observer.issue(event)

        edge_xml = root / p["edge_xml"]
        if not edge_xml.is_file():
            raise FileNotFoundError(edge_xml)
        env = create_target_env(os.environ["THESIS_SUMO_ROOT"], edge_xml,
            episode_seconds=p["horizon_seconds"], seed=p["external_seed"])
        env._ensure_bridge()
        from experiments.mcwm_behavior_replay_v113_20261029.observer import Observer
        observer = Observer(env._bridge, out)
        stream = EventStream(env, stations, contract, domain="target",
            delay_seconds=p["feedback_delay_seconds"], execute_transform=True, on_issue=issue)
        native_started, native_started_wall = True, time.time()
        stream.reset()
        # reset installs online_control before the bridge imports sim_obj.
        from bus_h2o.sumo_env import rl_bridge
        version = rl_bridge.traci.getVersion()
        if "1.25.0" not in str(version[1]):
            raise RuntimeError("Expected the v104 SUMO 1.25.0 runtime")
        hook = install_scenario(env, args.scenario, shock_at=p["shock_at_seconds"],
            factor=p["speed_factor"], metric_every=p["metric_interval_seconds"])
        restore_global_rng(learning_rng, learner.device)
        runtime = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
            sumo=version, h2o_root=os.environ["THESIS_H2O_ROOT"],
            sumo_root=os.environ["THESIS_SUMO_ROOT"], edge_xml=str(edge_xml),
            learner_module=type(learner).__module__, learner_class=type(learner).__name__)
        result.update(initialization=initialization, runtime=runtime)
        last_progress = time.monotonic()
        progress("native_day")
        while not stream.done:
            if not services.closed:
                batch = stream._batch()  # Pure event read; step uses the same unmodified batch.
                rows = [dict(**{key: str(event[key]) for key in (
                    "agent", "line_id", "vehicle_id", "trip_id", "station_id")},
                    event=np.asarray(event["event"]).tolist(), terminal=event["terminal"],
                    terminal_command=[0., 1.] if event["terminal"] else None) for event in batch]
                services.write(json.dumps(dict(clock=stream.now, events=rows), allow_nan=False)+"\n")
                service_rows += 1
                service_events += len(rows)
            budget.ingest(stream.step(policy))
            stream.release_intervals()
            while budget.due(stream.now):
                packet = budget.take(stream.now)
                if budget.index == 1:
                    trace.close(); services.close()
                    prefix = dict(trace_path=str(trace.path), trace_rows=trace.observed_issues,
                        services_path=str(out / "prefix_services.jsonl"), service_rows=service_rows,
                        service_events=service_events, clock=stream.now)
                    prefix_actor_exact = all(torch.equal(value, initial_actor[key])
                        for key, value in learner.actor.state_dict().items())
                    first_updates = learner.total_updates
                    clock_audit["first_feedback_applied_at"] = stream.now
                replay.add_target(packet)
                for _ in range(p["policy_updates_per_packet"]):
                    update_attempts += 1
                    learner.update(online=True, collect_metrics=False)
                    completed_updates += 1
                last_update_clock = stream.now
                updates.write(json.dumps(dict(packet=budget.index, clock=stream.now,
                    total_updates=learner.total_updates, counts=counts.snapshot()), allow_nan=False)+"\n")
                updates.flush()
                if budget.index == 1:
                    c = counts.snapshot()
                    replay_summary = replay.summary()
                    mechanism = mechanism_summary(learner, replay)
                    checks = dict(initialization=initialization["all_pass"],
                        original_endpoint=initial_updates == first_updates == 70000 and initial_online == 20000,
                        prefix_actor_exact=prefix_actor_exact,
                        updates8=completed_updates == 8 and all(c[k] == 8 for k in ("actor", "critic", "value", "alpha")),
                        trust2=c["trust"] == 2 and c["trust_optimizer"] == 2*learner.protocol["qdelta_updates_per_extra_step"],
                        model128=counts.model_batch_sizes == {128}
                            and c["model_batches"] == 8 and c["model_rows"] == 1024,
                        original_mixture=learner.protocol["real_td_coefficient"] == .75 and learner.coefficient == .25,
                        raw_F=action_audit["raw_execution_exact"], actor_clock=clock_audit["actor_real_clock"],
                        canonical_observation=action_audit["canonical_observation_exact"],
                        five_records=len(packet) == 5 and replay_summary["target_received"] == 5,
                        outcome_rights=replay_summary["fresh_outcomes"] == (5 if args.arm == "fresh" else 0),
                        feedback_available=budget.receipts[0]["max_available_at"] <= budget.receipts[0]["deadline"] <= stream.now,
                        source_unchanged=replay_summary["source_unchanged"], main_rng_unchanged=replay_summary["main_rng_unchanged"],
                        **mechanism_checks(mechanism, args.arm, 8))
                    preflight = dict(status="complete" if all(checks.values()) else "integrity_failed",
                        integrity=integrity(checks), cell=cell, initial_updates=initial_updates,
                        initial_online_critic_updates=initial_online, counts=c, prefix=prefix,
                        execution_wall=dict(native_started=native_started_wall, first_packet_finished=time.time()),
                        feedback=dict(records_received=5, training_outcomes=replay_summary["fresh_outcomes"],
                            first_applied_at=stream.now, receipt=budget.receipts[0]),
                        mix=dict(real=.75, model=.25), model_batch_sizes=sorted(counts.model_batch_sizes),
                        runtime=runtime, initialization=initialization,
                        mechanism=mechanism,
                        phase_count_deltas=counts.phase_deltas(learner))
                    write(out / "preflight.json", preflight)
                    if not preflight["integrity"]["all_pass"]:
                        raise RuntimeError("First packet technical gate failed")
                progress("packet_training")
            if budget.index == len(budget.deadlines) and frozen_counts is None:
                frozen_counts, freeze_clock = counts.snapshot(), stream.now
            if stream.now >= 18300 and post18300_counts is None:
                post18300_counts, post18300_clock = counts.snapshot(), stream.now
            if time.monotonic()-last_progress >= 45:
                progress("native_day"); last_progress = time.monotonic()
        if branch is None:
            raise RuntimeError("Fixed root was not reached")
        native_finished_wall = time.time()
        hook.capture_final()
        final = asdict(stream._snapshot())
        passengers = summarize_passengers_v5(env._bridge, stream._snapshot())
        costs = periods(hook.metrics, final, passengers["departed_passengers"], p["periods"])
        scene, c, replay_summary = hook.metadata(), counts.snapshot(), replay.summary()
        mechanism = mechanism_summary(learner, replay)
        ids = [seq for receipt in budget.receipts for seq in receipt["sequence_ids"]]
        phases = counts.phase_deltas(learner)
        checks = dict(initialization=initialization["all_pass"],
            initial70000=initial_updates == 70000 and initial_online == 20000,
            original_learner=type(learner).__name__ == "MappingTrainer",
            original_mixture=learner.protocol["real_td_coefficient"] == .75 and learner.coefficient == .25,
            policy456=completed_updates == update_attempts == 456 and all(c[k] == 456 for k in ("actor", "critic", "value", "alpha")),
            learner_counters=learner.total_updates == 70456 and learner.online_critic_updates == 20456,
            trust114=c["trust"] == p["trust_updates_per_arm"]
                and c["trust_optimizer"] == p["trust_updates_per_arm"]*learner.protocol["qdelta_updates_per_extra_step"],
            actor_lr_preserved=[group["lr"] for group in learner.actor_optimizer.param_groups]
                == initialization["actor_learning_rates"],
            model_batches456=c["model_batches"] == 456 and c["model_rows"] == 456*128 and counts.model_batch_sizes == {128},
            no_ratio_updates=c["ratio"] == 0, phase_counts=phases.get("bellman_trust_update", 0) == c["trust"]
                and phases.get("model_training_batch", 0) == c["model_batches"],
            full_packets=budget.index == 57 and all(r["records"] == 5 for r in budget.receipts),
            unique285=len(ids) == len(set(ids)) == replay_summary["target_received"] == 285,
            outcome_rights=replay_summary["fresh_outcomes"] == (285 if args.arm == "fresh" else 0),
            source_unchanged=replay_summary["source_unchanged"], main_rng_unchanged=replay_summary["main_rng_unchanged"],
            no_early_feedback=all(r["max_available_at"] <= r["deadline"] <= r["processed_at"] for r in budget.receipts),
            raw_F=action_audit["raw_execution_exact"], canonical_observation=action_audit["canonical_observation_exact"],
            actor_clock=clock_audit["actor_real_clock"], all_issues=action_audit["issues"] == stream.issued_count,
            full_day=stream.done and final["time_seconds"] == 24000,
            cohort=passengers["departed_passengers"] == 15079,
            ledger_balance=passengers["active_state_count_balance_error"] == 0,
            exact_windows=all(row["status"] == "complete" for row in costs.values()),
            frozen_tail=freeze_clock <= 18300 and last_update_clock <= 18300
                and frozen_counts == post18300_counts == c,
            **mechanism_checks(mechanism, args.arm, 456),
            **scenario_checks(scene, args.scenario, p, LANE_IDS))
        write(out / "packet_receipts.json", budget.receipts)
        write(out / "metrics.json", hook.metrics)
        result.update(status="complete" if all(checks.values()) else "integrity_failed",
            integrity=integrity(checks), periods=costs, passengers=passengers, ledger=final,
            passenger_denominator=passengers["departed_passengers"], replay=replay_summary,
            mechanism=mechanism,
            action_audit=action_audit, clock_audit=clock_audit, prefix=prefix,
            scenario_metadata=scene, model_batch_sizes=sorted(counts.model_batch_sizes),
            phase_count_deltas=phases, post_budget_counts=dict(frozen_at=freeze_clock,
                before=frozen_counts, after=c, post18300_observed_at=post18300_clock,
                post18300=post18300_counts, last_update_clock=last_update_clock),
            packet_receipts_path=str(out / "packet_receipts.json"), metrics_path=str(out / "metrics.json"))
        observer.finish()
        return result
    except PrefixComplete as complete:
        result.update(status="prefix_complete", fork=complete.receipt, integrity=dict(all_pass=complete.receipt["all_pass"], checks={"live_fork":complete.receipt["all_pass"]}, failed_checks=[]))
        return result
    except Exception as error:
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        from experiments.mcwm_closed_loop_historical_comparison_v104_20261028.run import consumed
        seconds, lower_bound = consumed(env, native_started)
        result["counts"] = counts.snapshot() if counts is not None else {}
        result["budget"] = dict(native_seconds=seconds, native_seconds_is_lower_bound=lower_bound,
            packets=budget.index, feedback_slots_selected=budget.index*5,
            feedback_slots=replay.summary()["target_received"] if replay is not None else 0,
            target_training_labels=replay.summary()["fresh_outcomes"] if replay is not None else 0,
            policy_updates=completed_updates, update_attempts=update_attempts,
            trust_updates=counts.snapshot()["trust"] if counts is not None else 0,
            model_generated_rows=counts.snapshot()["model_rows"] if counts is not None else 0,
            partial_update_possible=update_attempts != completed_updates,
            source_collection_events=0, model_refits=0, offline_training_updates=0,
            training_stopped_at=freeze_clock)
        if result["status"] in ("complete", "integrity_failed", "prefix_complete"):
            result["budget"]["native_seconds_is_lower_bound"] = False
        if replay is not None:
            result["replay"] = replay.summary()
            result["mechanism"] = mechanism_summary(learner, replay)
        result["execution_wall"] = dict(native_started=native_started_wall, native_finished=native_finished_wall,
            native_started_wall=native_started_wall, native_finished_wall=native_finished_wall)
        if branch is not None:
            result["branch"] = dict(name=branch, root_clock=root_clock, root_counts=root_counts,
                root_feedback_slots=root_feedback_slots,
                suffix_native_seconds=None if seconds is None else seconds-root_clock,
                suffix_updates=completed_updates-root_counts["policy"],
                suffix_feedback_slots=budget.index*5-root_feedback_slots, root_issue_count=root_issue_count,
                proposal_overrides=1 if branch == "low" else 0)
        result["wall_seconds"] = time.monotonic()-started
        for handle in (services, updates):
            if handle is not None and not handle.closed: handle.close()
        if trace is not None and not trace.writer.closed: trace.close()
        write(out / "summary.json", result)
        if stream is not None and native_started:
            progress(result["status"])
        if observer is not None: observer.close()
        if hook is not None: hook.close()
        if env is not None: env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=("stable", "local_speed_drop"), required=True)
    parser.add_argument("--arm", choices=("fresh", "no_fresh"), required=True)
    exit_code = 0
    try:
        result = run(parser.parse_args())
        exit_code = 0 if result["status"] in ("complete", "prefix_complete") else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    if os.getpid() != ROOT_PROCESS_PID:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
