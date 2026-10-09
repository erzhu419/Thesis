"""Summarize complete paired blocks, never count service days as training seeds."""

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("cost_seconds", "completion_percentage_points", "waiting_seconds",
           "riding_seconds", "hold_seconds", "completed_bus_trips")


def values(row):
    p = row["passengers"]
    return dict(cost_seconds=row["generalized_seconds_per_departed_passenger"],
                completion_percentage_points=100 * p["completion_rate"],
                waiting_seconds=p["accrued_wait_seconds_per_departed_passenger"],
                riding_seconds=p["accrued_in_vehicle_seconds_per_departed_passenger"],
                hold_seconds=row["mean_hold_seconds"],
                completed_bus_trips=row["service"]["completed_trips"])


def contrasts(p):
    rows = []
    def add(name, terms, primary=False):
        rows.append(dict(name=name, terms=terms, primary=primary))
    for layer in p["execution_layers"]:
        for method in ("offline", "source", "zero", "headway", "welfare"):
            add(f"hybrid-minus-{method}/{layer}",
                [("hybrid", layer, 1), (method, layer, -1)], method in ("offline", "source"))
    for method in p["methods"]:
        for reference in ("no_floor", "late20"):
            add(f"{method}/early60-minus-{reference}",
                [(method, "early60", 1), (method, reference, -1)])
    for method in p["native_rule_references"]:
        add(f"hybrid-early60-minus-{method}-native",
            [("hybrid", "early60", 1), (method, "native", -1)])
    for method in ("zero", "headway", "welfare"):
        add(f"guard-effect/hybrid-minus-{method}",
            [("hybrid", "early60", 1), ("hybrid", "no_floor", -1),
             (method, "early60", -1), (method, "no_floor", 1)])
    return rows


def summarize(blocks):
    n = len(blocks)
    mean = statistics.mean(blocks.values())
    sd = statistics.stdev(blocks.values())
    # The frozen design has six independent environment blocks, hence five df.
    if n != 6:
        raise ValueError("This analysis requires all six frozen blocks")
    half_width = 2.570581835636314 * sd / math.sqrt(n)
    return dict(mean_difference=mean, block_sd=sd, environment_blocks=n,
                approximate_95pct_t_interval=[mean - half_width, mean + half_width],
                paired_block_differences=blocks)


def main():
    p = json.loads((HERE / "protocol.json").read_text())
    treatments = [(m, layer) for m in p["methods"] for layer in p["execution_layers"]]
    treatments += [(m, "native") for m in p["native_rule_references"]]
    expected = {(m, layer, scale, b["sumo_seed"], b["demand_seed"])
                for m, layer in treatments for scale in p["evaluation_demand_scales"]
                for b in p["evaluation_blocks"]}
    rows = {}
    for path in sorted((HERE / "runs").glob("shard*/results/case*.json")):
        row = json.loads(path.read_text())
        key = tuple(row[k] for k in ("method", "execution_layer", "demand_scale", "sumo_seed", "demand_seed"))
        if key not in expected or key in rows:
            raise ValueError(f"Unexpected or repeated evaluation: {path}")
        if row["horizon_seconds"] != p["evaluation_horizon_seconds"]:
            raise ValueError(f"Incomplete service horizon: {path}")
        rows[key] = row
    if set(rows) != expected:
        raise SystemExit(f"Pending: {len(rows)}/{len(expected)} complete local results; no partial ranking produced.")
    comparisons = []
    for comparison in contrasts(p):
        for scale in [*p["evaluation_demand_scales"], "all_equal_weight"]:
            scales = p["evaluation_demand_scales"] if isinstance(scale, str) else [scale]
            metrics = {metric: {} for metric in METRICS}
            for block in p["evaluation_blocks"]:
                for metric in METRICS:
                    differences = [sum(weight * values(rows[(method, layer, d, block["sumo_seed"],
                                                              block["demand_seed"])])[metric]
                                       for method, layer, weight in comparison["terms"]) for d in scales]
                    metrics[metric][str(block["sumo_seed"])] = statistics.mean(differences)
            comparisons.append(dict(**comparison, demand_scale=scale,
                                    metrics={name: summarize(v) for name, v in metrics.items()}))
    summary = dict(cases=len(rows), training_seeds=1, independent_environment_blocks=6,
                   interpretation="Frozen-policy diagnostic; marginal approximate intervals, not multiplicity-adjusted or training-seed inference.",
                   comparisons=comparisons)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(cases=len(rows), comparisons=len(comparisons), output=str(HERE / "summary.json"))))


if __name__ == "__main__":
    main()
