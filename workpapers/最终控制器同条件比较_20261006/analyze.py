#!/usr/bin/env python3
"""Compare three frozen controllers on the complete paired 225-episode grid."""

import argparse
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
METHODS = ("mcwm_aligned", "h2oplus_final", "sac_guarded")
LABELS = {"mcwm_aligned": "对齐续训 MC-WM", "h2oplus_final": "正式 H2O+", "sac_guarded": "同执行规则纯源 SAC"}
SEEDS = tuple(range(6201, 6206))
BLOCKS = tuple(range(108101, 108106))
DEMANDS = (0.75, 1.0, 1.25)
EXPECTED = {(m, s, b, d) for m in METHODS for s in SEEDS for b in BLOCKS for d in DEMANDS}
PAIRS = {(s, b, d) for s in SEEDS for b in BLOCKS for d in DEMANDS}
METRICS = (
    "cost_seconds", "completion_percent", "wait_seconds", "in_vehicle_seconds",
    "hold_seconds_per_decision", "nonzero_hold_percent", "unfinished_passengers",
    "bus_completion_percent",
)
T975_DF4 = 2.7764451051977987
SCOPE = (
    "Five training seeds, five new environment blocks and three demand scales. "
    "All controllers use the v19 quarter-load-sixty execution transform with "
    "deployment scale 1.0 and a 24000-second horizon. MC-WM uses newly aligned "
    "continuation policies; H2O+ and SAC reuse frozen policies. Each primary "
    "interval uses five paired training-seed means, each first averaging all "
    "15 evaluation conditions. The 225 episodes are not independent training "
    "replicates. Environment-block intervals are descriptive conditional on "
    "these fixed policies. Intervals are unadjusted paired t intervals; no "
    "practical-superiority margin is imposed. SAC has no 50000-update offline "
    "initialization, so its comparison is not a compute-matched ablation."
)


def cell_dict(key):
    method, seed, block, demand = key
    return {"method": method, "training_seed": seed, "sumo_seed": block,
            "demand_seed": block + 100, "demand_scale": demand}


def read_record(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data["comparison_case"]
    key = (case["method"], int(case["training_seed"]), int(case["sumo_seed"]), float(case["demand_scale"]))
    if key not in EXPECTED:
        raise ValueError(f"Unexpected evaluation cell {key}: {path}")
    _, seed, block, demand = key
    if (int(case["demand_seed"]) != block + 100
            or int(data["sumo_seed"]) != block
            or int(data["scenario"]["selection_seed"]) != block + 100
            or float(data["scenario"]["demand_scale"]) != demand
            or int(data["policy"]["training_metadata"]["training_seed"]) != seed):
        raise ValueError(f"Evaluation and comparison metadata disagree: {path}")
    if (case["execution_transform"] != "v19:quarter-load-sixty"
            or float(case["deployment_hold_scale"]) != 1.0):
        raise ValueError(f"Controller does not use the common execution rule: {path}")
    episode = data["episode"]
    if (float(episode["observed_end_time_seconds"]) != 24000.0
            or float(episode["evaluation_horizon_seconds"]) != 24000.0
            or episode["stopped_by_event_limit"] or episode["stopped_by_time_limit"]):
        raise ValueError(f"Evaluation did not complete the shared 24000-second horizon: {path}")
    passengers = data["passengers"]
    metrics = {
        "cost_seconds": float(data["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"]),
        "completion_percent": 100.0 * float(passengers["completion_rate"]),
        "wait_seconds": float(passengers["accrued_wait_seconds_per_departed_passenger"]),
        "in_vehicle_seconds": float(passengers["accrued_in_vehicle_seconds_per_departed_passenger"]),
        "hold_seconds_per_decision": float(data["operations"]["hold_seconds"]["mean"]),
        "nonzero_hold_percent": 100.0 * float(data["operations"]["nonzero_hold_rate"]),
        "unfinished_passengers": int(passengers["unfinished_passengers"]),
        "bus_completion_percent": 100.0 * float(data["service"]["completion_rate"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Nonfinite service metric: {path}")
    if not math.isclose(metrics["cost_seconds"], 2 * metrics["wait_seconds"] + metrics["in_vehicle_seconds"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"Passenger cost differs from 2 * wait + ride: {path}")
    return key, {"path": str(path.resolve()), "departed_passengers": int(passengers["departed_passengers"]), "metrics": metrics}


def collect(directory):
    rows = {}
    for path in sorted(directory.glob("*.json")):
        key, row = read_record(path)
        if key in rows:
            raise ValueError(f"Duplicate evaluation cell {key}: {rows[key]['path']} and {path}")
        rows[key] = row
    # Each block and demand uses the same passenger realization across methods
    # and training seeds, not merely within one method pair.
    denominators = {}
    for (_, _, block, demand), row in rows.items():
        denominator = row["departed_passengers"]
        if denominator <= 0:
            raise ValueError(f"No departed passengers: {row['path']}")
        environment = (block, demand)
        if environment in denominators and denominators[environment] != denominator:
            raise ValueError(f"Passenger denominator differs within environment {environment}")
        denominators[environment] = denominator
    return rows


def means(rows):
    return {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}


def grouped(rows, position, labels):
    return {str(label): means([row for key, row in rows.items() if key[position] == label]) for label in labels}


def t95(values):
    center = statistics.mean(values)
    half = T975_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return [center - half, center + half]


def directions(values, lower_is_better):
    signed = [value if lower_is_better else -value for value in values]
    return {"improved": sum(value < 0 for value in signed),
            "equal": sum(value == 0 for value in signed),
            "worsened": sum(value > 0 for value in signed)}


def compare(left, right):
    deltas = {key: {metric: left[key][metric] - right[key][metric] for metric in METRICS} for key in sorted(PAIRS)}
    by_seed = grouped(deltas, 0, SEEDS)
    by_block = grouped(deltas, 1, BLOCKS)
    mean_delta = means(list(deltas.values()))
    baseline_cost = means(list(right.values()))["cost_seconds"]
    return {
        "paired_episodes": len(deltas), "independent_training_seed_pairs": len(SEEDS),
        "mean_difference": mean_delta,
        "cost_reduction_percent_of_right_mean": -100.0 * mean_delta["cost_seconds"] / baseline_cost,
        "training_seed_paired_t95_interval": {metric: t95([row[metric] for row in by_seed.values()]) for metric in METRICS},
        "descriptive_environment_block_t95_interval": {metric: t95([row[metric] for row in by_block.values()]) for metric in METRICS},
        "direction_counts": {
            metric: {
                "training_seeds": directions([row[metric] for row in by_seed.values()], metric == "cost_seconds"),
                "environment_blocks": directions([row[metric] for row in by_block.values()], metric == "cost_seconds"),
                "paired_episodes_descriptive_only": directions([row[metric] for row in deltas.values()], metric == "cost_seconds"),
            } for metric in ("cost_seconds", "completion_percent")
        },
        "by_training_seed": by_seed,
        "by_environment_block": by_block,
        "by_demand_scale": grouped(deltas, 2, DEMANDS),
    }


def conclusion(left, right, comparison):
    mean = comparison["mean_difference"]["cost_seconds"]
    interval = comparison["training_seed_paired_t95_interval"]["cost_seconds"]
    counts = comparison["direction_counts"]["cost_seconds"]["training_seeds"]
    if interval[1] < 0:
        evidence = "五个训练种子的配对 t95 区间整体低于零"
    elif interval[0] > 0:
        evidence = "五个训练种子的配对 t95 区间整体高于零"
    else:
        evidence = "五个训练种子的配对 t95 区间包含零，尚不能据此确定稳定的时间优势"
    return (
        f"{LABELS[left]}减{LABELS[right]}：广义乘客时间差{mean:+.2f}秒，"
        f"配对 t95 区间[{interval[0]:+.2f}, {interval[1]:+.2f}]秒；{evidence}。"
        f"5个训练种子中{counts['improved']}个改善、{counts['equal']}个持平、{counts['worsened']}个变差。"
        f"乘客完成率差{comparison['mean_difference']['completion_percent']:+.4f}个百分点。"
    )


def analyze(rows):
    missing = sorted(EXPECTED - set(rows))
    if missing:
        return {"status": "incomplete", "completed_evaluations": len(rows), "expected_evaluations": len(EXPECTED),
                "completed_by_method": {method: sum(key[0] == method for key in rows) for method in METHODS},
                "missing_cells": [cell_dict(key) for key in missing]}
    methods = {method: {key[1:]: row["metrics"] for key, row in rows.items() if key[0] == method} for method in METHODS}
    pairs = (("mcwm_aligned", "h2oplus_final", "primary"),
             ("mcwm_aligned", "sac_guarded", "primary"),
             ("h2oplus_final", "sac_guarded", "supporting"))
    comparisons = {}
    conclusions = []
    for left, right, role in pairs:
        comparison = compare(methods[left], methods[right])
        comparison.update({"role": role, "left": left, "right": right})
        comparisons[f"{left}_minus_{right}"] = comparison
        conclusions.append(conclusion(left, right, comparison))
    return {
        "status": "complete", "completed_evaluations": len(rows), "expected_evaluations": len(EXPECTED),
        "scope": SCOPE, "training_seeds": list(SEEDS), "sumo_seeds": list(BLOCKS),
        "demand_seeds": [block + 100 for block in BLOCKS], "demand_scales": list(DEMANDS),
        "aggregation": "Equal weight per evaluation cell; each training-seed mean averages 15 conditions. No passenger pooling across conditions.",
        "difference_units": {"completion_percent": "percentage points", "bus_completion_percent": "percentage points", "nonzero_hold_percent": "percentage points"},
        "aggregate_method_means": {method: means(list(values.values())) for method, values in methods.items()},
        "method_means_by_training_seed": {method: grouped(values, 0, SEEDS) for method, values in methods.items()},
        "method_means_by_environment_block": {method: grouped(values, 1, BLOCKS) for method, values in methods.items()},
        "method_means_by_demand_scale": {method: grouped(values, 2, DEMANDS) for method, values in methods.items()},
        "comparisons": comparisons, "conclusions": conclusions,
    }


def markdown(report):
    lines = [
        "225回合全部完成：3种方法、5个训练种子、5个新环境块和3档需求。广义乘客时间为2倍等待时间加乘车时间，越低越好。",
        "", "| 方法 | 广义时间/秒 | 完成率/% | 等待/秒 | 乘车/秒 | 每次决策停站控制/秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        m = report["aggregate_method_means"][method]
        lines.append(f"| {LABELS[method]} | {m['cost_seconds']:.2f} | {m['completion_percent']:.4f} | {m['wait_seconds']:.2f} | {m['in_vehicle_seconds']:.2f} | {m['hold_seconds_per_decision']:.2f} |")
    lines.extend(["", *[f"- {text}" for text in report["conclusions"]], "",
                  "区间按每个训练种子先平均15个测试条件后，以5个配对差计算；225回合不作为225个独立训练样本。"
                  "按5个环境块计算的区间、逐种子和逐需求结果见 analysis.json。", "",
                  "局限：配对 t 区间仅有5个训练种子且未做多重比较调整；环境块区间仅描述这组固定策略。"
                  "MC-WM为对齐客流后的新续训策略，H2O+和SAC为复用策略；纯源SAC不含5万次离线初始化，因此不是计算量相同的消融。"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "analysis.json")
    args = parser.parse_args()
    try:
        report = analyze(collect(args.results))
    except (ValueError, KeyError) as error:
        report = {"status": "invalid", "reason": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] == "complete":
        args.output.with_name("结果.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "completed_evaluations", "expected_evaluations", "completed_by_method", "reason", "conclusions") if key in report}, ensure_ascii=False))
    return 1 if report["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
