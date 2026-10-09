"""Collect numeric outputs only; analyse ten paired training-seed means."""

import argparse
import json
from pathlib import Path
import shlex
from statistics import mean, stdev
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REMOTE = str(HERE).replace("/home/erzhu419/mine_code", "/home/zhengliang01/scheduleurm_work")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler


def fetch(benchmarks, continuous300=False):
    folder = "runs300" if continuous300 else ("benchmarks" if benchmarks else "runs")
    (HERE / folder).mkdir(exist_ok=True)
    filename = ("continuous300_submission.json" if continuous300 else
                ("benchmark_submission.json" if benchmarks else "production_submission.json"))
    submission = json.loads((HERE / filename).read_text())
    jobs = submission["submitted"]
    if not benchmarks and not continuous300:
        for path in sorted(HERE.glob("cap100_eval_*_submission.json")):
            jobs += json.loads(path.read_text())["submitted"]
    if benchmarks and (HERE / "evaluation_smoke_submission.json").exists():
        jobs += json.loads((HERE / "evaluation_smoke_submission.json").read_text())["submitted"]
    for node in sorted({t["require_node"] for t in jobs}):
        includes = ["--include=*.json"] if benchmarks else ["--include=episodes.jsonl", "--include=complete.json"]
        subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                        "--include=*/", *includes, "--exclude=*",
                        f"{scheduler._ssh_target_for_node(node)}:{REMOTE}/{folder}/",
                        str(HERE / folder) + "/"], check=True)
        ids = [t["id"] for t in jobs if t["require_node"] == node]
        code = ("import json; from pathlib import Path; "
                f"ids={ids!r}; "
                "print(json.dumps({i: Path('/tmp/sched_'+i+'.log').read_text(errors='replace')[-240:] "
                "if Path('/tmp/sched_'+i+'.log').exists() else 'no log yet' for i in ids}))")
        status, output, errors = scheduler.run_on(node, f"{PYTHON} -c {shlex.quote(code)}", timeout=45, check=False)
        print(node, output.strip() if status == 0 else errors[-500:], flush=True)


def summarise(benchmarks):
    if benchmarks:
        files = list((HERE / "benchmarks").glob("*.json"))
        for path in files:
            value = json.loads(path.read_text())
            assert value["checks_passed"]
            print(path.name, json.dumps(value), flush=True)
        print(f"benchmark_and_smoke_outputs={len(files)}", flush=True)
        return
    seeds = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
    arms = ("full", "masked")
    complete = list((HERE / "runs").glob("*/results/complete.json"))
    print(f"completed_training_and_test={len(complete)}/20", flush=True)
    if len(complete) != 20:
        return
    metric_names = ("composite", "wait_min", "onboard_min", "total_min", "vehicle_hours",
                    "full_day_holding_vehicle_hours", "service_km", "vehicles_used",
                    "peak_fleet", "time_weighted_load", "dispatch_lateness", "headway_cv")
    all_rows, seed_means = {}, {}
    for arm in arms:
        seed_means[arm] = {}
        for seed in seeds:
            root = HERE / "runs" / f"{arm}_seed{seed}" / "results"
            done = json.loads((root / "complete.json").read_text())
            assert done["arm"] == arm and done["training_seed"] == seed
            assert done["training_episodes"] == 100 and done["checkpoint_episode"] == 99
            assert done["evaluation_episodes"] == 50
            if arm == "masked":
                assert done["evaluation_feedback_nonzero"] == 0
            rows = sorted([json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()],
                          key=lambda r: r["episode"])
            assert [r["episode"] for r in rows] == list(range(50))
            assert all(r["completed_trips"] == 262 for r in rows)
            all_rows[arm, seed] = rows
            seed_means[arm][seed] = {key: mean(r[key] for r in rows) for key in metric_names}
    for seed in seeds:
        for left, right in zip(all_rows["full", seed], all_rows["masked", seed]):
            assert (left["environment_seed"], left["fleet_budget"]) == (
                right["environment_seed"], right["fleet_budget"])
    paired = {}
    for key in metric_names:
        differences = [seed_means["full"][seed][key] - seed_means["masked"][seed][key] for seed in seeds]
        average = mean(differences)
        half_width = 2.262157163 * stdev(differences) / len(seeds) ** 0.5
        paired[key] = dict(mean=average, ci95=[average-half_width, average+half_width],
                           full_lower=sum(d < 0 for d in differences), seed_differences=differences)
    result = dict(training_seeds=seeds, training_episodes=100, checkpoint_episode=99,
                  episodes_per_seed=50, seed_means=seed_means, paired=paired)
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(paired, indent=2))


def compare_continuous_budgets():
    seeds = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
    arms, endpoints = ("full", "masked"), (99, 299)
    root = HERE / "runs300"
    missing = {endpoint: [f"{arm}_seed{seed}" for arm in arms for seed in seeds
                         if not (root / f"{arm}_seed{seed}" /
                                 f"results_ep{endpoint}_extended/complete.json").exists()]
               for endpoint in endpoints}
    complete_endpoints = [endpoint for endpoint in endpoints if not missing[endpoint]]
    for endpoint in endpoints:
        print(f"endpoint={endpoint+1} completed={20-len(missing[endpoint])}/20")
    metrics = ("composite", "wait_min", "total_min", "onboard_min", "vehicle_hours",
               "full_day_holding_vehicle_hours", "service_km", "peak_fleet", "headway_cv")
    means, schedules = {}, {}
    for endpoint in endpoints:
        means[endpoint] = {arm: {} for arm in arms}
        for arm in arms:
            for seed in seeds:
                if f"{arm}_seed{seed}" in missing[endpoint]:
                    continue
                result = root / f"{arm}_seed{seed}" / f"results_ep{endpoint}_extended"
                done = json.loads((result / "complete.json").read_text())
                assert done["training_seed"] == seed and done["arm"] == arm
                assert done["training_episodes"] == endpoint+1 and done["checkpoint_episode"] == endpoint
                assert done["run_training_budget"] == 300 and done["test_plan"] == "extended"
                assert done["evaluation_episodes"] == 50
                if arm == "masked":
                    assert done["evaluation_feedback_nonzero"] == 0
                rows = sorted((json.loads(line) for line in (result / "episodes.jsonl").read_text().splitlines()),
                              key=lambda row: row["episode"])
                assert [row["episode"] for row in rows] == list(range(50))
                assert all(row["completed_trips"] == 262 for row in rows)
                plan = [(row["environment_seed"], row["fleet_budget"]) for row in rows]
                if seed in schedules:
                    assert plan == schedules[seed]
                schedules[seed] = plan
                means[endpoint][arm][seed] = {key: mean(row[key] for row in rows) for key in metrics}

    def paired(differences):
        center = mean(differences)
        half = 2.262157163 * stdev(differences) / len(differences) ** 0.5
        return dict(mean=center, ci95=[center-half, center+half],
                    negative_seeds=sum(x < 0 for x in differences), seed_differences=differences)

    between = {endpoint: {key: paired([means[endpoint]["full"][s][key] -
                means[endpoint]["masked"][s][key] for s in seeds]) for key in metrics}
               for endpoint in complete_endpoints}
    # Missing intermediate weights do not invalidate a complete final endpoint.
    changes, interaction = {}, {}
    if len(complete_endpoints) == 2:
        changes = {arm: {key: paired([means[299][arm][s][key] - means[99][arm][s][key]
                    for s in seeds]) for key in metrics} for arm in arms}
        interaction = {key: paired([means[299]["full"][s][key] - means[299]["masked"][s][key] -
                                   means[99]["full"][s][key] + means[99]["masked"][s][key]
                                   for s in seeds]) for key in metrics}
    output = dict(training_seeds=seeds, seed_means=means, full_minus_masked=between,
                  primary_complete=299 in complete_endpoints, missing_evaluations=missing,
                  budget_comparison_complete=len(complete_endpoints) == 2,
                  budget300_minus100=changes, difference_change=interaction)
    (HERE / "continuous300_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(between.get(299, {"primary_complete": False}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--continuous300", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.benchmarks, args.continuous300)
    if args.continuous300:
        compare_continuous_budgets()
    else:
        summarise(args.benchmarks)
