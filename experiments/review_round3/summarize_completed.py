"""Summarize complete operational comparisons, preserving training-level repeats."""

from review_common import HERE, P, write
from collections import defaultdict
import json
from statistics import mean


def reward_results():
    files = [HERE / f"runs/reward/{arm}_seed{seed}/case{case:02d}/results/eval{case:02d}.json"
             for arm in P["reward"]["arms"] for seed in P["reward"]["training_seeds"] for case in range(6)]
    if not all(path.exists() for path in files):
        return dict(ready=sum(path.exists() for path in files), expected=len(files))
    groups = defaultdict(list)
    for path in files:
        row = json.loads(path.read_text())
        assert row["horizon_seconds"] == 24000
        ops = row["operational"]
        metrics = dict(generalized_seconds=row["generalized_seconds_per_passenger"],
                       completion_fraction=row["arrived_passengers"]/row["generated_passengers"],
                       front_plan_absolute_seconds=ops["front_plan_absolute_error_seconds"]["mean"],
                       back_plan_absolute_seconds=ops["back_plan_absolute_error_seconds"]["mean"],
                       departure_lateness_seconds=ops["departure_lateness_seconds"]["mean"],
                       trip_duration_seconds=ops["completed_trip_duration_seconds"]["mean"],
                       terminal_turnaround_seconds=ops["terminal_turnaround_seconds"]["mean"],
                       inference_ms=ops["policy_inference_ms"]["median"],
                       launched_trips=ops["launched_trips"], completed_trips=ops["completed_trips"])
        groups[(row["scene"], row["method"], row["training_seed"])].append(metrics)
    per_seed = {key: {metric: mean(r[metric] for r in rows) for metric in rows[0]}
                for key, rows in groups.items()}
    assert all(len(rows) == 3 for rows in groups.values())
    result = []
    for scene in P["reward"]["scenes"]:
        for arm in P["reward"]["arms"]:
            rows = [per_seed[(scene, arm, seed)] for seed in P["reward"]["training_seeds"]]
            result.append(dict(scene=scene, arm=arm, training_seeds=3,
                               metrics={key: mean(r[key] for r in rows) for key in rows[0]}, per_seed=rows))
    return dict(ready=len(files), expected=len(files), rows=result)


def response_results():
    files = [HERE / f"runs/response/case{case:03d}/results/response.json" for case in range(54)]
    if not all(path.exists() for path in files):
        return dict(ready=sum(path.exists() for path in files), expected=len(files))
    grouped = defaultdict(list)
    for path in files:
        row = json.loads(path.read_text())
        assert row["horizon"] == 24000
        case = row["case"]
        for group, metrics in row["groups"].items():
            grouped[(case["domain"], case["hold"], case["demand"], group)].append(metrics)
    result = []
    for (domain, hold, demand, group), rows in sorted(grouped.items()):
        assert len(rows) == 3
        result.append(dict(domain=domain, hold=hold, demand=demand, group=group, days=3,
            transitions=sum(r["completed_same_vehicle_transitions"] for r in rows),
            decisions=sum(r["decisions"] for r in rows),
            metrics={metric: mean(r[metric]["mean"] for r in rows if r[metric]["n"])
                     for metric in ("next_event_seconds", "onboard_change", "next_waiting",
                                    "next_front_absolute_plan_error")},
            day_transition_counts=[r["completed_same_vehicle_transitions"] for r in rows]))
    return dict(ready=len(files), expected=len(files), rows=result,
                scope="Equal mean across three days within each condition. Source and SUMO conditional populations differ.")


def guard_results():
    path = HERE / "runs/guard/paired_summary.json"
    if not path.exists():
        return dict(ready=0, expected=36)
    data = json.loads(path.read_text())
    if data["completed_pairs"] != 36:
        return dict(ready=data["completed_pairs"], expected=36)
    rows = []
    for method in P["guard"]["methods"]:
        selected = [r for r in data["rows"] if r["method"] == method]
        groups = {}
        for cohort in selected[0]["groups"]:
            metrics = [r["groups"][cohort] for r in selected]
            groups[cohort] = dict(mean_passengers_per_day=mean(r["passengers"] for r in metrics),
                wait_seconds=mean(r["wait_difference"]["mean"] for r in metrics),
                ride_seconds=mean(r["ride_difference"]["mean"] for r in metrics),
                generalized_seconds=mean(r["generalized_difference"]["mean"] for r in metrics),
                completed_only_with_guard=sum(r["completed_only_with_guard"] for r in metrics),
                completed_only_without_guard=sum(r["completed_only_without_guard"] for r in metrics))
        rows.append(dict(method=method, paired_days=9, groups=groups))
    return dict(ready=36, expected=36, rows=rows,
                scope="Equal mean across demands and days. Trigger-exposed subgroups are descriptive; all_departed is the paired policy effect.")


def main():
    results = dict(reward=reward_results(), response=response_results(), guard=guard_results())
    write(HERE / "completed_operational_summary.json", results)
    print(json.dumps({key: {k: value[k] for k in ("ready", "expected")} for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
