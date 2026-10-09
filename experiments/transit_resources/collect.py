"""Fetch small resource summaries and aggregate at the training-seed level."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[2] / "TransitDuet/transit_duet/results_remote/revision3_heldout"
METHODS = ["H_timetable_v5_stable_300", "H_fixed_timetable_300_final", "H_hiro_final"]
SEEDS = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
METRICS = ["vehicle_hours", "service_km", "time_weighted_load", "vehicles_used",
           "peak_fleet", "full_day_holding_mean_sec", "full_day_holding_vehicle_hours",
           "wait_min", "onboard_min", "total_passenger_min"]


def fetch():
    sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
    import scheduler
    queue = json.loads(Path.home().joinpath(".claude/scheduler/queue.json").read_text())
    tasks = [t for t in queue["tasks"]
             if t.get("signature", "").startswith("thesis-transit-resources-v1/")]
    deployment = [{k: t.get(k) for k in ("id", "signature", "status", "node")}
                  for t in tasks]
    (HERE / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    print(json.dumps(dict(Counter(t["status"] for t in tasks))), flush=True)
    remote = str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work")
    (HERE / "runs").mkdir(exist_ok=True)
    for node in sorted({t["node"] for t in tasks if t.get("node")}):
        subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                        "--include=*/", "--include=*.json", "--include=*.jsonl", "--exclude=*",
                        f"{scheduler._ssh_target_for_node(node)}:{remote}/runs/",
                        str(HERE / "runs") + "/"], check=True)


def aggregate():
    rows = []
    completed = 0
    for shard in range(36):
        root = HERE / "runs" / f"shard{shard:02d}" / "results"
        path = root / "episodes.jsonl"
        if path.exists():
            # A running writer may leave the last line incomplete during transfer.
            rows.extend(json.loads(line) for line in path.read_text().splitlines(keepends=True)
                        if line.endswith("\n"))
        if (root / "complete.json").exists():
            finished = json.loads((root / "complete.json").read_text())
            assert finished == {"shard": shard, "episodes": len(range(shard, 1500, 36))}
            completed += 1
    print(json.dumps({"completed_shards": completed, "episodes": len(rows)}), flush=True)
    if completed != 36:
        return
    assert len(rows) == 1500 and {r["index"] for r in rows} == set(range(1500))
    seed_means = {}
    history_errors = []
    for method in METHODS:
        seed_means[method] = {}
        for seed in SEEDS:
            group = sorted([r for r in rows if r["method"] == method and r["training_seed"] == seed],
                           key=lambda r: r["test_episode"])
            assert [r["test_episode"] for r in group] == list(range(50))
            old = json.loads((HISTORY / method / f"seed{seed}.json").read_text())
            assert [r["episode_seed"] for r in group] == old["test_episode_seeds"]
            assert [r["fleet_budget"] for r in group] == old["fleet_schedule"]
            for r in group:
                assert r["completed_trips"] == 262
                assert abs(r["vehicle_hours"] - r["vehicle_hours_integral"]) < 1e-8
            values = {metric: mean(r[metric] for r in group) for metric in METRICS}
            for new, prior in (("wait_min", "wait_mean"),
                               ("onboard_min", "avg_onboard_time_min_mean"),
                               ("total_passenger_min", "avg_total_passenger_time_min_mean")):
                error = values[new] - old["metrics"][prior]
                history_errors.append(abs(error))
                assert abs(error) < 1e-8, (method, seed, new, error)
            seed_means[method][seed] = values
    summary = {method: {metric: {
        "mean": mean(seed_means[method][s][metric] for s in SEEDS),
        "sd_across_training_seeds": stdev(seed_means[method][s][metric] for s in SEEDS)}
        for metric in METRICS} for method in METHODS}
    paired = {}
    for reference in METHODS[1:]:
        paired[reference] = {}
        for metric in METRICS:
            differences = [seed_means[METHODS[0]][s][metric] - seed_means[reference][s][metric]
                           for s in SEEDS]
            average = mean(differences)
            half_width = 2.262157163 * stdev(differences) / len(SEEDS) ** 0.5
            paired[reference][metric] = dict(mean=average,
                ci95=[average - half_width, average + half_width],
                lower_count=sum(d < 0 for d in differences), seed_differences=differences)
    result = dict(episodes=len(rows), training_seeds=SEEDS, completed_trips_per_episode=262,
                  historical_passenger_metrics_max_error=max(history_errors),
                  summary=summary, paired=paired, seed_means=seed_means)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("summary", "paired")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    if parser.parse_args().fetch:
        fetch()
    aggregate()
