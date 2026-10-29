"""Resume v59 receipts, then advance the v60 actual-control campaign."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V59 = HERE.parent / "mcwm_closed_loop_matched_response_v59_20261025"
TERMINAL = {"done", "failed", "cancelled"}
MAX_JSON_BYTES = 48000


class ReadUnavailable(RuntimeError):
    """An explicitly transient read-only scheduler/SSH operation failed."""


class Attention(RuntimeError):
    """Known uncertainty that must not trigger another submission."""


def transient_read(error):
    if isinstance(error, (subprocess.TimeoutExpired, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, OSError) and error.errno in (
            errno.ECONNRESET, errno.ECONNREFUSED, errno.ETIMEDOUT, errno.ENETUNREACH):
        return True
    message = " ".join(str(value) for value in (
        error, getattr(error, "stderr", ""), getattr(error, "stdout", ""))).lower()
    return any(text in message for text in (
        "waiting for scheduler state lock", "connection timed out", "connection reset",
        "connection refused", "connection closed by", "network is unreachable",
        "temporary failure in name resolution", "broken pipe", "ssh: connect to host"))


def read_only(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        if transient_read(error):
            raise ReadUnavailable(str(error)) from error
        raise


@contextmanager
def binding(name, module):
    absent = object()
    previous = sys.modules.get(name, absent)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v59_modules(directory=V59):
    infra = load("v60_receipted_v59_infra", directory / "infra.py")
    # v59's launch uses a bare import; never borrow another experiment's infra.
    with binding("infra", infra):
        launch = load("v60_receipted_v59_launch", directory / "launch.py")
    return infra, launch


def write_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(raw.encode()) > MAX_JSON_BYTES:
        raise ValueError("Supervisor JSON exceeds48000-byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


class Supervisor:
    def __init__(self, root=HERE, source=V59, *, infra=None, launch=None, advance=None):
        self.root, self.source = Path(root), Path(source)
        self.root.mkdir(parents=True, exist_ok=True)
        if infra is None or launch is None:
            infra, launch = v59_modules(self.source)
        self.infra, self.launch, self.advance = infra, launch, advance
        state = self.root / "supervisor_state.json"
        self.state = json.loads(state.read_text()) if state.exists() else {}

    def save(self, current):
        self.state.update(current)
        self.state["updated_unix"] = time.time()
        write_json(self.root / "supervisor_state.json", self.state)

    def event(self, value):
        record = dict(at_unix=time.time(), **value)
        raw = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (self.root / "supervisor_events.jsonl").open("a") as handle:
            handle.write(raw + "\n")
        print(raw, flush=True)

    def receipt(self, batch):
        path = self.source / (batch + "_submission.json")
        return json.loads(path.read_text()) if path.exists() else None

    def submit_once(self, batch, function):
        receipt = self.receipt(batch)
        if receipt is not None:
            return receipt
        attempt = self.source / (batch + "_submission_attempt.json")
        if attempt.exists() or self.state.get("submission_intent") == batch:
            raise Attention("Unresolved submission attempt for " + batch + "; no automatic retry")
        self.save(dict(submission_intent=batch))
        try:
            result = function()
        except Exception as error:
            raise Attention("Submission outcome unknown for " + batch + ": " + str(error)) from error
        if self.receipt(batch) is None:
            raise Attention("Submission returned without a durable receipt: " + batch)
        self.save(dict(submission_intent=None))
        self.event(dict(event="submitted", batch=batch,
                        task_ids=[row["id"] for row in result["submitted"]]))
        return result

    def outcomes(self, jobs):
        paths = [relative + "/" + name for relative in jobs.values()
                 for name in ("summary.json", "failure.json")]
        found = read_only(self.infra.presence, paths)
        results, pending = {}, {}

        def resident(task_id, relative):
            failure, summary = relative + "/failure.json", relative + "/summary.json"
            if found.get(failure):
                value = read_only(self.infra.collect, failure, max_bytes=MAX_JSON_BYTES)
                return dict(id=task_id, state="failed", output=failure,
                            message=value.get("message", str(value)),
                            failure_budget=value.get("failure_budget"))
            if found.get(summary):
                value = read_only(self.infra.collect, summary, max_bytes=MAX_JSON_BYTES)
                acceptable = value.get("status") == "complete" or (
                    relative == "analysis" and value.get("status") == "partial_complete")
                return dict(id=task_id, state="complete" if acceptable else "failed",
                            output=summary, result_status=value.get("status"))
            return None

        for task_id, relative in jobs.items():
            value = resident(task_id, relative)
            if value is None:
                pending[task_id] = relative
            else:
                results[task_id] = value
        if pending:
            statuses = {row["id"]: row for row in read_only(self.infra.status, list(pending))}
            if set(statuses) != set(pending):
                raise Attention("Scheduler omitted a receipted task; no task will be resubmitted")
            terminal = {key: pending[key] for key, row in statuses.items() if row["status"] in TERMINAL}
            if terminal:
                # Terminal status can precede the first existence snapshot.
                found.update(read_only(self.infra.presence,
                    [path + "/" + name for path in terminal.values()
                     for name in ("summary.json", "failure.json")]))
            for task_id, relative in pending.items():
                row = statuses[task_id]
                value = resident(task_id, relative) if task_id in terminal else None
                if value is None:
                    value = dict(id=task_id, state="failed" if task_id in terminal else "pending",
                                 scheduler_status=row["status"], node=row.get("node"))
                    if task_id in terminal:
                        value["message"] = "Terminal task without declared resident output"
                results[task_id] = value
        return results

    def control(self, summary):
        if summary.get("status") not in ("complete", "partial_complete"):
            raise Attention("v59 analysis is not a completed or declared partial summary")
        gate = summary.get("next_control_gate")
        complete = summary["status"] == "complete" and bool(summary.get("integrity", {}).get("all_pass"))
        if complete and (not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool)):
            raise Attention("v59 summary has no declared control gate")
        disposition = ("candidate_admitted" if gate["passed"] else "candidate_closed") if complete else "candidate_not_admitted_incomplete"
        if self.state.get("v59_disposition") != disposition:
            self.save(dict(v59_disposition=disposition))
            self.event(dict(event=disposition, gate=gate,
                performance_verdict="not_evaluated_incomplete" if not complete else "declared_gate"))
        if self.advance is None:
            module = load("v60_control_campaign_launch", self.root / "control_launch.py")
            self.advance = module.advance
        current = self.advance(summary)
        if not isinstance(current, dict) or "done" not in current:
            raise ValueError("control_launch.advance must return a dictionary with done")
        return dict(current, v59_disposition=disposition)

    def step(self):
        protocol = self.launch.protocol()
        native = self.receipt("native")
        if native is None:
            raise Attention("No v59 native receipt; supervisor never submits a replacement prefix")
        cases, jobs, learning = {}, {}, {}
        prefix = protocol["name"] + "/"
        native_ids = {row["signature"].removeprefix(prefix + "collection/"): row["id"]
                      for row in native["submitted"]}
        if set(native_ids) != set(protocol["scenarios"]):
            raise Attention("Native receipt does not match the declared scenarios")
        for case, task_id in native_ids.items():
            jobs[task_id] = "runs/collection/" + case
            learning[case] = self.receipt("learning_" + case)
            if learning[case]:
                for row in learning[case]["submitted"]:
                    arm = row["signature"].removeprefix(prefix + "learning/" + case + "/")
                    if arm not in self.launch.ARMS:
                        raise Attention("Unexpected learning arm in receipt: " + arm)
                    jobs[row["id"]] = "runs/learning/" + case + "/" + arm
        final = self.receipt("finalize")
        if final:
            jobs[final["submitted"][0]["id"]] = "analysis"
        outcomes = self.outcomes(jobs)
        for case, task_id in native_ids.items():
            collection, workers = outcomes[task_id], {}
            receipt = learning[case]
            if collection["state"] == "complete" and receipt is None:
                receipt = self.submit_once("learning_" + case, lambda case=case: self.launch.learning(case))
                for row in receipt["submitted"]:
                    arm = row["signature"].rsplit("/", 1)[1]
                    workers[arm] = dict(id=row["id"], state="pending", scheduler_status="submitted")
            elif receipt:
                workers = {row["signature"].rsplit("/", 1)[1]: outcomes[row["id"]]
                           for row in receipt["submitted"]}
            elif collection["state"] == "failed":
                workers = {arm: dict(state="skipped", reason="Native collection failed; no rerun")
                           for arm in self.launch.ARMS}
            cases[case] = dict(collection=collection, learning=workers)
        all_terminal = all(case["collection"]["state"] in ("complete", "failed") and
            set(case["learning"]) == set(self.launch.ARMS) and all(
                row["state"] in ("complete", "failed", "skipped") for row in case["learning"].values())
            for case in cases.values())
        if final:
            analysis = outcomes[final["submitted"][0]["id"]]
            if analysis["state"] == "complete":
                summary = read_only(self.infra.collect, "analysis/summary.json", max_bytes=MAX_JSON_BYTES)
                return self.control(summary)
            if analysis["state"] == "failed":
                raise Attention("v59 summary task failed: " + str(analysis))
        elif all_terminal:
            lifecycle = dict(protocol=protocol["name"], cases=cases, all_terminal=True,
                             feedback_slots_per_arm=285, training_updates_per_arm=456,
                             preflight_reused=True, automatic_retries=0)
            read_only(self.infra.publish_state, lifecycle)
            self.submit_once("finalize", self.launch.finalize)
        return dict(stage="waiting_v59_summary" if all_terminal else "waiting_v59_collection_or_learning",
                    done=False, all_terminal=all_terminal, cases=cases)

    def run(self, *, watch=False, interval=30):
        failures, previous = 0, None
        while True:
            try:
                current = self.step()
                failures = 0
            except ReadUnavailable as error:
                failures += 1
                current = dict(stage="waiting_transient_read", done=False, message=str(error),
                               consecutive_read_failures=failures)
            except Attention as error:
                current = dict(stage="attention", done=False, message=str(error))
            except Exception as error:
                current = dict(stage="attention", done=False, error_type=type(error).__name__,
                               message=str(error), unexpected_error=True)
                self.save(current)
                self.event(current)
                raise
            delay = min(120, interval * 2 ** min(failures, 3)) if failures else interval
            current["next_poll_seconds"] = delay
            self.save(current)
            if current != previous:
                self.event(current)
                previous = current
            if current["done"] or not watch:
                return current
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30, choices=range(1, 61))
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("An existing v60 supervisor owns the receipts; no duplicate watcher started.", flush=True)
            return
        write_json(HERE / "supervisor_process.json", dict(pid=os.getpid(), started_unix=time.time()))
        Supervisor().run(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
