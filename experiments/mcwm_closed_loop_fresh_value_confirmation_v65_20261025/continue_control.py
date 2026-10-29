"""Watch one v65 batch and exit once its result is ready for root review."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
MAX_JSON_BYTES = 48000


class Attention(RuntimeError):
    """A submission outcome is uncertain; review existing receipts."""


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Continuation only reads JSON of at most48000 bytes")
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


class Continuation:
    def __init__(self, root=HERE, *, advance=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.advance = advance
        self.state = read_json(self.root / "continue_control_state.json") or {}

    def save(self, value):
        self.state.update(value)
        self.state["updated_unix"] = time.time()
        write_json(self.root / "continue_control_state.json", self.state)

    def event(self, value):
        raw = json.dumps(dict(at_unix=time.time(), **value), ensure_ascii=False, allow_nan=False)
        with (self.root / "continue_control_events.jsonl").open("a") as handle:
            handle.write(raw + "\n")
        print(raw, flush=True)

    def step(self):
        ready = read_json(self.root / "execution_ready.json")
        if ready is None or ready.get("ready") is not True:
            return dict(stage="preparing_fresh_value_confirmation", done=False, goal_achieved=False,
                        completed=0, pending=[], failures=[], all_terminal=False)
        if self.state.get("advance_uncertain"):
            raise Attention("Prior confirmation advance raised; review existing receipts before resuming")
        if self.advance is None:
            spec = importlib.util.spec_from_file_location("v65_confirmation_launch", self.root / "launch.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.advance = module.advance
        try:
            current = self.advance()
        except Exception as error:
            self.save(dict(advance_uncertain=True, advance_error=str(error)))
            raise Attention("Confirmation advance outcome requires receipt review: " + str(error)) from error
        if not isinstance(current, dict) or not isinstance(current.get("done"), bool):
            raise ValueError("launch.advance() must return a dictionary with boolean done")
        return dict(current, campaign_done=current["done"], goal_achieved=False)

    def run(self, *, watch=False, interval=30):
        failures, previous = 0, None
        while True:
            try:
                current = self.step()
            except Attention as error:
                current = dict(stage="attention", done=False, goal_achieved=False, message=str(error))
            except Exception as error:
                current = dict(stage="attention", done=False, goal_achieved=False,
                               message=str(error), unexpected_error=True)
                self.save(current)
                self.event(current)
                raise
            failures = failures + 1 if current["stage"].startswith("waiting_transient_") else 0
            current["next_poll_seconds"] = (0 if current["done"] else
                min(60, interval * 2 ** min(failures, 2)) if failures else interval)
            self.save(current)
            if current != previous:
                self.event(current)
                previous = current
            if current["done"] or not watch:
                return current
            time.sleep(current["next_poll_seconds"])


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
            print("An existing v65 watcher owns these receipts; no duplicate started.", flush=True)
            return
        write_json(HERE / "continue_control_process.json", dict(pid=os.getpid(), started_unix=time.time()))
        Continuation().run(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
