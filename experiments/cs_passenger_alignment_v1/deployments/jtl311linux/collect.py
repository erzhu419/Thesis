"""Aggregate on the shared server; export summaries, not passenger records."""

from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def collect():
    rows = [json.loads(p.read_text()) for p in (HERE / "results").glob("*/seed*/day*.json")]
    expected = 234
    keys = {(r["arm"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows}
    assert len(keys) == len(rows)
    counts = defaultdict(set)
    baseline_counts = defaultdict(set)
    for row in rows:
        assert row["policy_phase"] == "formal" and row["horizon_seconds"] == P["horizon_seconds"]
        assert row["timestamp_cost_identity_passed"]
        counts[row["scene"], row["evaluation_seed"]].add(row["groups"]["all"]["passengers"])
        baseline_counts[row["evaluation_seed"]].add(row["groups"]["baseline"]["passengers"])
    assert all(len(v) == 1 for v in counts.values()), "Unpaired generated passenger counts"
    assert all(len(v) == 1 for v in baseline_counts.values()), "Baseline arrivals changed across scenes"
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"], row["scene"]].append(row)
    summary = {}
    for (arm, scene), cases in sorted(grouped.items()):
        value = dict(cases=len(cases),
            cost_seconds=statistics.mean(r["cost"] for r in cases),
            cost_min_seconds=min(r["cost"] for r in cases),
            cost_max_seconds=max(r["cost"] for r in cases),
            holding_seconds=statistics.mean(r["mean_holding_seconds"] for r in cases),
            plan_mae_seconds=statistics.mean(r["plan_mae_seconds"] for r in cases),
            native_headway_penalty=statistics.mean(r["native_headway_penalty"] for r in cases),
            completion_rate=statistics.mean(r["groups"]["all"]["completion_rate"] for r in cases),
            baseline_cost_seconds=statistics.mean(r["groups"]["baseline"]["mean_generalized_seconds"] for r in cases),
            original_window_cost_seconds=statistics.mean(r["groups"]["baseline_during"]["mean_generalized_seconds"] for r in cases),
            peak_rss_mb=max(r["peak_rss_mb"] for r in cases))
        summary[arm + "/" + scene] = value
    by_case = {(r["arm"], r["seed"], r["evaluation_seed"], r["scene"]): r for r in rows}
    contrasts = {}
    for candidate, reference in [("observed_overlap", "local_overlap"),
                                  ("global_event", "observed_overlap"),
                                  ("global_shaped", "global_event"),
                                  ("global_shaped", "frozen_passenger")]:
        for scene in P["scenes"]:
            pairs = []
            for seed in P["training_seeds"]:
                for day in P["evaluation_seeds"]:
                    a, b = by_case.get((candidate, seed, day, scene)), by_case.get((reference, seed, day, scene))
                    if a and b:
                        pairs.append(b["cost"] - a["cost"])
            if pairs:
                contrasts[f"{candidate}-vs-{reference}/{scene}"] = dict(
                    paired_cases=len(pairs), mean_cost_reduction_seconds=statistics.mean(pairs),
                    improved_cases=sum(p > 0 for p in pairs), reductions_seconds=pairs)
    trains = {}
    for path in (HERE / "runs").glob("*/seed*/train_summary.json"):
        row = json.loads(path.read_text())
        last = row["history"][-1]
        assert len(row["history"]) == 100 and row["training_steps"] == 40000
        trains[str(path.parent.relative_to(HERE / "runs"))] = dict(
            episodes=len(row["history"]), updates=row["training_steps"],
            wall_seconds=row["training_wall_seconds"], peak_rss_mb=row["peak_rss_mb"],
            final_training_cost=last["cost"], final_holding_seconds=last["mean_holding_seconds"],
            learning_metrics=last["update_last"], reward_ema_std=last["reward_ema_std"],
            credit=last["credit"])
    result = dict(complete=len(rows) == expected and len(trains) == 8,
        analysis_location="server", evaluations=len(rows), expected_evaluations=expected,
        training_runs=len(trains), paired_passenger_counts_passed=True,
        groups=summary, contrasts=contrasts, training=trains)
    (HERE / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["complete", "evaluations", "training_runs"]}))


if __name__ == "__main__":
    collect()
