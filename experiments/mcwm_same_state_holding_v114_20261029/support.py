"""Small live-fork utilities; the native fork itself is the existing v34 path."""
from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def _one(path, matches):
    found = [row for row in rows(path) if matches(row)]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one frozen anchor in {path}; found {len(found)}")
    return found[0]


def load_anchor(case_dir, sequence):
    """Read only the fixed sequence and its actual command/visit on the server."""
    case_dir = Path(case_dir)
    action = _one(case_dir / "actions.jsonl", lambda row: row["sequence"] == sequence)
    command = _one(case_dir / "commands.jsonl", lambda row: row["sequence"] == sequence)
    key = (action["trip_id"], action["station_id"], action["visit_index"])
    visit = _one(case_dir / "visits.jsonl", lambda row:
                 (row["trip_id"], row["stop_id"], row["visit_index"]) == key)
    if ((command["trip_id"], command["stop_id"], command["visit_index"]) != key
            or not command["accepted"]):
        raise ValueError("Frozen anchor requires the matching accepted native command")
    return dict(action=action, command=command, visit=visit)


def root_actions(anchor, raw_s, original_proposal, contract, target_hold_seconds):
    """One fixed low proposal, checked through the unmodified physical F.

    Both declared v114 anchors lie on F's direct reserve branch. This is not a
    search for another dose: an unreachable frozen target returns a failed gate.
    """
    from bus_h2o.mdp_contract_v7 import hold_seconds_to_policy_action_v7, policy_action_to_hold_seconds_v7
    from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action

    original = anchor["action"]
    high_proposal = np.asarray(original_proposal, np.float32).reshape(1)
    low_proposal = np.asarray([hold_seconds_to_policy_action_v7(target_hold_seconds, contract)], np.float32)

    def execute(proposal):
        action = completion_safety_action(raw_s, proposal, "quarter-load-sixty", contract)
        return dict(proposal=proposal.tolist(), executed=action.tolist(),
                    hold=float(policy_action_to_hold_seconds_v7(action[0], contract)))

    high, low = execute(high_proposal), execute(low_proposal)
    checks = dict(
        raw_state_exact=np.array_equal(np.asarray(raw_s, np.float32), np.asarray(original["raw_s"], np.float32)),
        original_proposal_exact=np.array_equal(high_proposal, np.asarray(original["proposed_a"], np.float32)),
        high_execution_exact=high["executed"] == original["a"],
        high_native_command_exact=high["hold"] == anchor["command"]["requested_additional_hold_seconds"],
        low_frozen_dose_reached=abs(low["hold"]-target_hold_seconds) <= 1e-4,
        lower_physical_dose=low["hold"] < high["hold"],
        original_command_accepted=anchor["command"]["accepted"] is True)
    return dict(high=high, low=low, target_hold_seconds=float(target_hold_seconds),
                checks=checks, all_pass=all(checks.values()))


def compare_prefix_actions(actual_path, reference_path, through_sequence):
    """Full parsed-JSON equality through the inclusive boundary; no hashes."""
    observed = 0
    with Path(actual_path).open() as actual, Path(reference_path).open() as reference:
        for line in reference:
            expected = json.loads(line)
            if expected["sequence"] > through_sequence:
                break
            candidate = actual.readline()
            if not candidate or json.loads(candidate) != expected:
                return dict(all_pass=False, rows=observed,
                            first_difference=expected["sequence"], unexpected_extra=False)
            observed += 1
        extra = bool(actual.readline())
    complete = observed == through_sequence+1
    return dict(all_pass=complete and not extra, rows=observed,
                first_difference=None if complete else observed, unexpected_extra=extra)


def flush_writers(observer, updates):
    for writer in (*observer.writers.values(), updates):
        writer.flush()


def trace_adapter(observer):
    return SimpleNamespace(writer=observer.writers["actions"], reference=None)


def child_detach(observer, updates, child_out):
    """Retain the copied Observer and its cumulative accounts; detach only IO.

    Call in a child after flush_writers in the parent and v34 route detachment.
    Full logs are parent prefix followed by child suffix, never copied wholesale.
    """
    child_out = Path(child_out)
    child_out.mkdir(parents=True, exist_ok=True)
    prefix = {name: str(Path(writer.name).resolve()) for name, writer in observer.writers.items()}
    prefix["updates"] = str(Path(updates.name).resolve())
    manifest = dict(parent_prefix=prefix,
        child_suffix={name: str((child_out / (name+".jsonl")).resolve()) for name in prefix},
        prefix_issues=observer.issues, prefix_ticks=observer.ticks,
        prefix_commands=observer.commands, prefix_clock=observer.end,
        cumulative_observer_preserved=True)
    for writer in observer.writers.values():
        writer.close()
    updates.close()
    observer.out = child_out
    observer.writers = {name: (child_out / (name+".jsonl")).open("x") for name in observer.writers}
    new_updates = (child_out / "updates.jsonl").open("x")
    with (child_out / "log_manifest.json").open("x") as handle:
        json.dump(manifest, handle, separators=(",", ":"), allow_nan=False)
    return new_updates, manifest


def state_receipt(stream, learner, replay, budget, counts):
    """Read small live state and complete RNG states, without saving weights.

    COW copies the full learner/replay; this receipt is a boundary check, not a
    serialized replacement for that live state or the exact high-arm replay.
    """
    if learner.device.type != "cpu":
        raise ValueError("v114 live-fork continuation is fixed to the original CPU runtime")
    ledger = asdict(stream._snapshot())
    base, aug = replay.replay, learner.augmentation
    return json_value(dict(
        native_ledger=ledger,
        stream=dict(episode_id=stream.episode_id, issued_count=stream.issued_count,
            completed_count=stream.completed_count, released_count=stream.released_count,
            thinned_count=stream.thinned_count,
            pending_sequence=None if stream._pending is None else stream._pending["sequence"],
            vehicle_pending={str(key): row["sequence"] for key, row in stream._vehicle_pending.items()},
            queued_sequences=[row.sequence for row in stream._queue],
            queued_interval_sequences=[row.sequence for row in stream._interval_queue]),
        packets=dict(index=budget.index, waiting_sequences=[row.sequence for row in budget.rows],
                     receipts=budget.receipts),
        learner=dict(total_updates=learner.total_updates, online_critic_updates=learner.online_critic_updates,
                     model_family=aug.family, trust_tau=aug.trust._tau, counts=counts.snapshot()),
        replay=dict(summary=replay.summary(), online_pointer=base._online_pointer,
                    online_total_appended=base._online_total_appended),
        private_rng=dict(learner=learner.rng.bit_generator.state,
            diagnostics=learner.diagnostic_rng.bit_generator.state,
            fresh=replay.private_rng.bit_generator.state, augmentation=aug.rng.bit_generator.state,
            augmentation_generator=aug.generator.get_state()),
        global_rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())))
