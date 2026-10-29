"""Run the two outcome-supervision arms concurrently on one two-CPU task."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ARMS = ("fresh", "no_fresh")


def last_progress(path):
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell()-8192))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if line.startswith("SCHEDULEURM_PROGRESS "):
            try:
                return json.loads(line.split(" ", 1)[1])
            except json.JSONDecodeError:
                continue  # The child may currently be writing its final line.
    return None


def parallel_timing(summaries):
    clocks = {arm: value.get("execution_wall", {}) for arm, value in summaries.items()}
    valid = len(clocks) == 2 and all(
        isinstance(value.get(key), (int, float))
        for value in clocks.values()
        for key in ("native_started_wall", "native_finished_wall"))
    if not valid:
        return dict(positive_overlap=False, native_overlap_seconds=None,
                    native_pair_span_seconds=None, arm_occupancy_factor=None, arms=clocks)
    durations = {arm: value["native_finished_wall"]-value["native_started_wall"]
                 for arm, value in clocks.items()}
    overlap = max(0., min(value["native_finished_wall"] for value in clocks.values())
                  - max(value["native_started_wall"] for value in clocks.values()))
    span = max(value["native_finished_wall"] for value in clocks.values()) \
        - min(value["native_started_wall"] for value in clocks.values())
    return dict(positive_overlap=overlap > 0 and min(durations.values()) > 0,
        native_overlap_seconds=overlap, native_pair_span_seconds=span,
        arm_occupancy_factor=sum(durations.values())/span if span > 0 else None,
        arm_native_wall_seconds=durations, arms=clocks,
        interpretation="Measured concurrent occupancy, not a serial-counterfactual speedup.")


def run(seed, scenario, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh paired output directory")
    out.mkdir(parents=True, exist_ok=True)
    processes, handles, exitcodes, summaries = {}, {}, {}, {}
    started = time.monotonic()
    error = None
    try:
        # Both children are launched before either is waited on.
        for arm in ARMS:
            handles[arm] = (out/f"{arm}.log").open("w")
            env = dict(os.environ)
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            processes[arm] = subprocess.Popen([sys.executable, "-u", str(HERE/"run.py"),
                "--seed", str(seed), "--scenario", scenario, "--arm", arm,
                "--out", str(out/arm)], stdout=handles[arm], stderr=subprocess.STDOUT, env=env)
    except OSError as exc:
        error = str(exc)
    finally:
        print(json.dumps(dict(phase="paired_children_started", child_pids={
            arm: process.pid for arm, process in processes.items()})), flush=True)
        next_progress = time.monotonic()
        while any(process.poll() is None for process in processes.values()):
            if time.monotonic() >= next_progress:
                child_progress = {arm: last_progress(out/f"{arm}.log") for arm in processes}
                values = dict(phase="paired_native_days", alive_arms=[arm for arm, process
                    in processes.items() if process.poll() is None],
                    elapsed_seconds=time.monotonic()-started, child_progress=child_progress)
                if len(child_progress) == 2 and all(
                        row is not None and isinstance(row.get("current"), (int, float))
                        for row in child_progress.values()):
                    values.update(current=min(row["current"] for row in child_progress.values()),
                                  total=24000, unit="sim_seconds_slowest_arm")
                print("SCHEDULEURM_PROGRESS "+json.dumps(values), flush=True)
                next_progress = time.monotonic()+45
            time.sleep(5)
        for arm, process in processes.items():
            exitcodes[arm] = process.wait()
        for handle in handles.values():
            handle.close()
    for arm in ARMS:
        path = out/arm/"summary.json"
        if path.exists():
            summaries[arm] = json.loads(path.read_text())
    timing = parallel_timing(summaries)
    checks = dict(both_children_exited_zero=set(exitcodes) == set(ARMS)
        and all(value == 0 for value in exitcodes.values()),
        both_native_days_complete=set(summaries) == set(ARMS) and all(
            value.get("status") == "complete" and value.get("integrity", {}).get("all_pass") is True
            for value in summaries.values()), actual_native_parallel=timing["positive_overlap"])
    result = dict(status="complete" if all(checks.values()) else "integrity_failed",
        cell=dict(seed=seed, scenario=scenario), integrity=dict(all_pass=all(checks.values()), checks=checks),
        exitcodes=exitcodes, child_pids={arm: process.pid for arm, process in processes.items()},
        parallel=timing, wall_seconds=time.monotonic()-started)
    if error:
        result["launch_error"] = error
    (out/"pair_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", choices=("stable", "local_speed_drop"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.scenario, args.out.resolve())
    print(json.dumps(result))
    return 0 if result["integrity"]["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
