#!/usr/bin/env python3
"""Pair frozen SAC under the candidate executor with the original formal runs."""

import argparse
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD_ROOT = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
    "h2oplus_v20_formal_confirmation_results/evaluation_v4/single_jtl110gpu"
)
SEEDS = tuple(range(6201, 6206))
BLOCKS = tuple(range(8101, 8111))
DEMANDS = (0.75, 1.0, 1.25)
EXPECTED = {(block, demand, seed) for block in BLOCKS for demand in DEMANDS for seed in SEEDS}
CANDIDATE = "h2oplus-v19-completion-safety-quarter-load-sixty"
METRICS = (
    "cost_seconds", "completion_percent", "wait_seconds", "in_vehicle_seconds",
    "unfinished_passengers", "bus_completion_percent",
    "hold_seconds_per_decision", "nonzero_hold_percent",
)
SCOPE = (
    "Post-hoc frozen-policy execution diagnosis on the ten previously used formal "
    "test days. No retraining or new independent confirmation. All five fixed "
    "training seeds, three demand scales and ten environment blocks are retained. "
    "The primary difference is candidate minus guarded SAC. Negative cost and "
    "positive passenger completion differences favor the candidate. Completion "
    "differences are percentage points. Block intervals describe these reused "
    "evaluation days conditional on the five frozen policies; they are descriptive."
)


def read_record(path):
    data = json.loads(path.read_text())
    seed = int(data["policy"]["training_metadata"]["training_seed"])
    block = int(data["sumo_seed"])
    demand = float(data["scenario"]["demand_scale"])
    key = (block, demand, seed)
    if key not in EXPECTED:
        raise ValueError(f"Unexpected evaluation cell {key}: {path}")
    if int(data["scenario"]["selection_seed"]) != block + 100:
        raise ValueError(f"Demand realization does not match the formal block: {path}")
    episode = data["episode"]
    if (float(episode["observed_end_time_seconds"]) != 24000.0
            or episode["stopped_by_event_limit"] or episode["stopped_by_time_limit"]):
        raise ValueError(f"Evaluation did not finish the shared 24000-second horizon: {path}")
    passengers = data["passengers"]
    metrics = {
        "cost_seconds": float(data["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"]),
        "completion_percent": 100.0 * float(passengers["completion_rate"]),
        "wait_seconds": float(passengers["accrued_wait_seconds_per_departed_passenger"]),
        "in_vehicle_seconds": float(passengers["accrued_in_vehicle_seconds_per_departed_passenger"]),
        "unfinished_passengers": int(passengers["unfinished_passengers"]),
        "bus_completion_percent": 100.0 * float(data["service"]["completion_rate"]),
        "hold_seconds_per_decision": float(data["operations"]["hold_seconds"]["mean"]),
        "nonzero_hold_percent": 100.0 * float(data["operations"]["nonzero_hold_rate"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Nonfinite service metric: {path}")
    if not math.isclose(metrics["cost_seconds"], 2 * metrics["wait_seconds"] + metrics["in_vehicle_seconds"], abs_tol=1e-6):
        raise ValueError(f"Passenger cost does not follow the original accounting: {path}")
    return key, {
        "path": str(path.resolve()), "method": data["method"],
        "departed_passengers": int(passengers["departed_passengers"]),
        "metrics": metrics,
    }


def collect(paths):
    rows = {}
    for path in sorted(paths):
        key, row = read_record(path)
        if key in rows:
            raise ValueError(f"Duplicate evaluation cell {key}: {path}")
        rows[key] = row
    return rows


def means(rows):
    return {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}


def grouped_means(rows, position, labels):
    return {
        str(label): means([row for key, row in rows.items() if key[position] == label])
        for label in labels
    }


def directions(values, lower_is_better):
    return {
        "improved": sum(value < 0 if lower_is_better else value > 0 for value in values),
        "equal": sum(value == 0 for value in values),
        "worsened": sum(value > 0 if lower_is_better else value < 0 for value in values),
    }


def compare(left, right):
    deltas = {
        key: {metric: left[key]["metrics"][metric] - right[key]["metrics"][metric] for metric in METRICS}
        for key in sorted(EXPECTED)
    }
    by_block = grouped_means(deltas, 0, BLOCKS)
    by_seed = grouped_means(deltas, 2, SEEDS)
    intervals = {}
    direction_counts = {}
    # Two-sided t(.975, 9); ten block means, each averaging all 15 paired cases.
    for metric in ("cost_seconds", "completion_percent"):
        block_values = [row[metric] for row in by_block.values()]
        mean = statistics.mean(block_values)
        half_width = 2.2621571628540993 * statistics.stdev(block_values) / math.sqrt(10)
        intervals[metric] = [mean - half_width, mean + half_width]
        direction_counts[metric] = {
            "blocks": directions(block_values, metric == "cost_seconds"),
            "training_seeds": directions([row[metric] for row in by_seed.values()], metric == "cost_seconds"),
            "paired_cells": directions([row[metric] for row in deltas.values()], metric == "cost_seconds"),
        }
    return {
        "paired_cells": len(deltas), "mean_difference": means(list(deltas.values())),
        "descriptive_block_t95_interval": intervals,
        "direction_counts": direction_counts,
        "by_environment_block": by_block, "by_training_seed": by_seed,
        "by_demand_scale": grouped_means(deltas, 1, DEMANDS),
    }


def analyze(candidate, guarded, raw):
    for label, rows in (("candidate", candidate), ("guarded_sac", guarded), ("raw_sac", raw)):
        if set(rows) != EXPECTED:
            raise ValueError(f"Incomplete {label} comparison: {len(rows)}/150")
    for key in EXPECTED:
        counts = {rows[key]["departed_passengers"] for rows in (candidate, guarded, raw)}
        if len(counts) != 1:
            raise ValueError(f"Passenger denominator differs within paired cell {key}")
    methods = {"candidate": candidate, "guarded_sac": guarded, "raw_sac": raw}
    report = {
        "status": "complete", "scope": SCOPE,
        "training_seeds": list(SEEDS), "sumo_seeds": list(BLOCKS),
        "demand_scales": list(DEMANDS), "new_evaluations": 150, "reused_evaluations": 300,
        "aggregate_method_means": {
            label: means([row["metrics"] for row in rows.values()]) for label, rows in methods.items()
        },
        "comparisons": {
            "candidate_minus_guarded_sac": compare(candidate, guarded),
            "guarded_sac_minus_raw_sac": compare(guarded, raw),
            "candidate_minus_raw_sac_original_comparison": compare(candidate, raw),
        },
        "paired_results": [],
    }
    for block, demand, seed in sorted(EXPECTED):
        key = (block, demand, seed)
        report["paired_results"].append({
            "sumo_seed": block, "demand_seed": block + 100,
            "demand_scale": demand, "training_seed": seed,
            **{label: rows[key] for label, rows in methods.items()},
            "candidate_minus_guarded_sac": {
                metric: candidate[key]["metrics"][metric] - guarded[key]["metrics"][metric]
                for metric in METRICS
            },
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-results", type=Path, default=HERE / "results")
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--output", type=Path, default=HERE / "analysis.json")
    args = parser.parse_args()
    guarded = collect(args.new_results.glob("sac_guarded_seed*_sumo*_demand*.json"))
    if set(guarded) != EXPECTED:
        report = {
            "status": "incomplete", "complete_new_results": len(guarded), "expected_new_results": 150,
            "scope": SCOPE,
            "missing_cells": [
                {"sumo_seed": b, "demand_scale": d, "training_seed": s}
                for b, d, s in sorted(EXPECTED - set(guarded))
            ],
        }
    else:
        candidate = collect(args.old_root.glob(f"v20-formal-confirmation-*/{CANDIDATE}-seed*-events000025000/*/sumo_seed*.json"))
        raw = collect(args.old_root.glob("v20-formal-confirmation-*/sac-online-seed*-events000025000/*/sumo_seed*.json"))
        report = analyze(candidate, guarded, raw)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "complete_new_results", "expected_new_results", "aggregate_method_means", "comparisons") if key in report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
