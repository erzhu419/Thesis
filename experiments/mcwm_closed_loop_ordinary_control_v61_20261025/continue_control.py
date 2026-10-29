"""Finish the existing paired campaign, then continue ordinary dripping control."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
V60 = HERE.parent / "mcwm_closed_loop_control_campaign_v60_20261025"
MAX_JSON_BYTES = 48000
TERMINAL = {"done", "failed", "cancelled"}


class ReadUnavailable(RuntimeError):
    """A read-only observation is temporarily unavailable."""


class Attention(RuntimeError):
    """Existing work needs repair or an unresolved submission needs review."""


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Continuation only reads JSON of at most48000 bytes: " + str(path))
    return json.loads(path.read_text())


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(raw.encode()) > MAX_JSON_BYTES:
        raise ValueError("Continuation JSON exceeds48000 bytes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


class Continuation:
    def __init__(self, root=HERE, source=V60, *, infra=None, advance=None,
                 observer=None, pid_alive=process_alive):
        self.root, self.source = Path(root), Path(source)
        self.root.mkdir(parents=True, exist_ok=True)
        # Keep the source exception class and its read wrapper together. Loading
        # another experiment's bare "infra" or supervisor would split that class.
        self.observer = observer or load("v61_observed_v60_supervisor", self.source / "supervisor.py")
        self.infra = infra or load("v61_observed_v60_infra", self.source / "infra.py")
        self.advance, self.pid_alive = advance, pid_alive
        self.state = read_json(self.root / "continue_control_state.json") or {}

    def read_only(self, function, *args, **kwargs):
        try:
            return self.observer.read_only(function, *args, **kwargs)
        except self.observer.ReadUnavailable as error:
            raise ReadUnavailable(str(error)) from error

    def save(self, value):
        self.state.update(value)
        self.state["updated_unix"] = time.time()
        write_json(self.root / "continue_control_state.json", self.state)

    def event(self, value):
        raw = json.dumps(dict(at_unix=time.time(), **value), ensure_ascii=False, allow_nan=False)
        with (self.root / "continue_control_events.jsonl").open("a") as handle:
            handle.write(raw + "\n")
        print(raw, flush=True)

    def receipts(self, stage):
        return {name: read_json(self.source / ("paired_readout_" + stage + suffix))
                for name, suffix in (("native", "_submission.json"),
                                     ("analysis", "_analysis_submission.json"))}

    def observe(self, stage, receipts):
        process = read_json(self.source / "supervisor_process.json") or {}
        pid = process.get("pid")
        source_state = read_json(self.source / "supervisor_state.json") or {}
        value = dict(supervisor_pid=pid, supervisor_alive=self.pid_alive(pid) if pid else False,
                     supervisor_stage=source_state.get("stage"), receipt_ids={}, task_counts={})
        for group, receipt in receipts.items():
            if receipt is None:
                continue
            ids = [row["id"] for row in receipt["submitted"]]
            if not ids:
                raise Attention("Empty existing paired receipt: " + stage + "/" + group)
            statuses = self.read_only(self.infra.status, ids)
            if {row["id"] for row in statuses} != set(ids):
                raise Attention("Scheduler omitted an existing paired task; no replacement submitted")
            value["receipt_ids"][group] = ids
            counts = {}
            for row in statuses:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            value["task_counts"][group] = counts
            value[group + "_terminal"] = all(row["status"] in TERMINAL for row in statuses)
        return value

    def result(self, stage):
        relative = "results/paired_readout/" + stage + ".json"
        local = read_json(self.source / relative)
        if local is not None:
            return local
        present = self.read_only(self.infra.presence, [relative])
        return self.read_only(self.infra.collect, relative) if present.get(relative) else None

    def paired_gate(self, result, stage, observation):
        if result.get("candidate", "paired_readout") != "paired_readout" or result.get("stage", stage) != stage:
            raise Attention("Existing paired summary has a different candidate or stage")
        integrity = result.get("integrity", {})
        complete = result.get("status") == "complete" and integrity.get("all_pass") is True
        if complete:
            key = "advance_to_confirmation" if stage == "development" else "confirmed_control_improvement"
            decision = result.get("decision", {}).get(key)
            if not isinstance(decision, bool):
                raise Attention("Completed paired summary has no declared decision: " + key)
            if stage == "development" and decision:
                return None
            return dict(paired_disposition=("closed_failed_development_gate" if stage == "development"
                else "confirmed_counterfactual_assisted_only" if decision else "closed_after_confirmation"),
                paired_performance_verdict="declared_control_gate", ordinary_admitted=True,
                paired_result="results/paired_readout/" + stage + ".json")
        if (result.get("status") in ("integrity_failed", "failed") or result.get("failures")
                or integrity.get("failed_checks")):
            raise Attention("v60 paired technical failure requires repair; ordinary dispatch held. "
                            "No performance failure inferred: " + stage)
        if result.get("missing") and observation.get("native_terminal") and observation.get("analysis_terminal"):
            return dict(paired_disposition="not_admitted_incomplete", ordinary_admitted=True,
                paired_performance_verdict="not_evaluated_incomplete",
                paired_result="results/paired_readout/" + stage + ".json")
        return None

    def ordinary(self):
        ready = read_json(self.root / "execution_ready.json")
        if ready is None or ready.get("ready") is not True:
            return dict(stage="preparing_ordinary_control_code", done=False, goal_achieved=False)
        if self.state.get("advance_uncertain"):
            raise Attention("Ordinary launch raised previously; inspect its receipts before resuming")
        if self.advance is None:
            with self.observer.binding("infra", load("v61_ordinary_infra", self.root / "infra.py")):
                self.advance = load("v61_ordinary_launch", self.root / "launch.py").advance
        try:
            current = self.advance()
        except Exception as error:
            # The launcher owns durable submission receipts. A timeout here may
            # follow a submission, so the watcher never retries the mutation.
            self.save(dict(advance_uncertain=True, advance_error=str(error)))
            raise Attention("Ordinary advance outcome needs review: " + str(error)) from error
        if not isinstance(current, dict) or not isinstance(current.get("done"), bool):
            raise ValueError("v61 launch.advance() must return a dictionary with boolean done")
        value = dict(current, ordinary_campaign_done=current["done"], done=False)
        if current.get("goal_achieved"):
            value.update(stage="ordinary_gain_requires_root_review", requires_agent_review=True)
        return value

    def step(self):
        ordinary_receipt = read_json(self.root / "recent30_development_submission.json")
        if not self.state.get("ordinary_admitted") and ordinary_receipt is not None:
            submitted = ordinary_receipt.get("submitted", [])
            if not submitted or any(not row.get("id") for row in submitted):
                raise Attention("Existing ordinary development receipt has no task handles; inspect submission")
            admission = dict(ordinary_admitted=True,
                ordinary_admission_reason="predeclared_parallel_actual_control",
                ordinary_existing_receipt="recent30_development_submission.json",
                paired_disposition="separate_counterfactual_assisted_campaign",
                paired_performance_verdict="not_used_for_ordinary_admission")
            self.save(admission)
            self.event(dict(event="ordinary_control_admitted", **admission))
        if not self.state.get("ordinary_admitted"):
            for stage in ("development", "confirmation"):
                receipts = self.receipts(stage)
                if receipts["native"] is None:
                    if stage == "development":
                        raise Attention("No v60 paired development receipt; never replace its prefix")
                    return dict(stage="waiting_paired_confirmation_submission", done=False,
                                goal_achieved=False, paired_observation=self.state.get("paired_observation"))
                observation = self.observe(stage, receipts)
                self.save(dict(paired_observation=observation))
                result = self.result(stage)
                if result is not None:
                    admission = self.paired_gate(result, stage, observation)
                    if admission:
                        self.save(admission)
                        self.event(dict(event="ordinary_control_admitted", **admission))
                        break
                    if (stage == "development" and result.get("status") == "complete"
                            and result.get("integrity", {}).get("all_pass") is True
                            and result.get("decision", {}).get("advance_to_confirmation") is True):
                        continue
                return dict(stage="waiting_paired_" + stage, done=False, goal_achieved=False,
                    paired_observation=observation,
                    attention=None if observation["supervisor_alive"] else
                        "v60 supervisor is not live; existing tasks are observed without rerun")
        current = self.ordinary()
        return dict(current, paired_disposition=self.state["paired_disposition"],
                    paired_performance_verdict=self.state["paired_performance_verdict"],
                    ordinary_admission_reason=self.state.get("ordinary_admission_reason", "paired_campaign_finished"))

    def run(self, *, watch=False, interval=30):
        failures, previous = 0, None
        while True:
            try:
                current = self.step()
                failures = 0
            except ReadUnavailable as error:
                failures += 1
                current = dict(stage="waiting_transient_read", done=False, goal_achieved=False,
                               message=str(error), consecutive_read_failures=failures)
            except Attention as error:
                current = dict(stage="attention", done=False, goal_achieved=False, message=str(error))
            except Exception as error:
                current = dict(stage="attention", done=False, goal_achieved=False,
                               message=str(error), unexpected_error=True)
                self.save(current)
                self.event(current)
                raise
            delay = min(60, interval * 2 ** min(failures, 2)) if failures else interval
            current["next_poll_seconds"] = delay
            self.save(current)
            if current != previous:
                self.event(current)
                previous = current
            if not watch:
                return current
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30, choices=range(1, 61))
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "continue_control.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("The existing v61 continuation owns this watcher; no duplicate started.", flush=True)
            return
        write_json(HERE / "continue_control_process.json", dict(pid=os.getpid(), started_unix=time.time()))
        Continuation().run(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
