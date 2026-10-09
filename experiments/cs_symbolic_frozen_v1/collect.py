"""Paired server-side attribution without moving checkpoints or trajectories."""

from itertools import product
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / P["reference_campaign"]
spec = importlib.util.spec_from_file_location("frozen_metrics", HERE.parent / "cs_source_envelope_v1/collect.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def path_for(domain, variant, arm, seed, day, scene, folder="results"):
    return HERE / folder / domain / variant / arm / f"seed{seed}" / f"day{day}_{scene}.json"


def smoke_report():
    native = json.loads(path_for("source", "native", "symbolic0", 64001, 69421, "normal", "smoke").read_text())
    prior = json.loads((PRIOR / "results/source/symbolic0/seed64001/day69421_normal.json").read_text())
    keys = ["cost", "exact_cost", "decisions", "transitions", "mean_holding_seconds", "groups", "input_stats"]
    assert all(native[key] == prior[key] for key in keys), "Frozen native execution differs"
    probes = []
    for variant, arm, seed in [("bias", "symbolic10", 64002), ("sindy", "reference", 64001)]:
        row = json.loads(path_for("target", variant, arm, seed, 78101, "static10", "smoke").read_text())
        assert row["horizon_seconds"] == 7200 and row["updates"] == 0 and row["timestamp_cost_identity_passed"]
        assert row["input_stats"]["source_cap_exceeded_fraction"] > 0
        if variant == "bias":
            contracts = json.loads((HERE / "contracts.json").read_text())["records"]
            expected = next(r["bias_holding_seconds"] for r in contracts if (r["arm"], r["seed"]) == (arm, seed))
            assert abs(row["mean_holding_seconds"] - expected) < 1e-5
        probes.append({k: row[k] for k in ["variant", "cost", "mean_holding_seconds", "wall_seconds", "peak_rss_mb"]})
    result = dict(passed=True, native_full_horizon_execution_identical=True, high_demand_probes=probes)
    metrics.write(HERE / "smoke_passed.json", result)
    print(json.dumps(result), flush=True)


def baseline(domain, label, seed, day, scene):
    if label in P["arms"]:
        root = PRIOR / "results" / domain / label
    elif label == "old_raw":
        root = HERE.parent / ("cs_source_validation_v1/results/old_raw" if domain == "source"
                              else "cs_terminal_coverage_v1/results/global_coverage")
    else:
        seed = 0
        root = HERE.parent / ("cs_source_validation_v1/results/fixed0" if domain == "source"
                              else "cs_passenger_alignment_v1/results/fixed0")
    return json.loads((root / f"seed{seed}/day{day}_{scene}.json").read_text())


def summarize(rows):
    result = metrics.metrics(rows)
    result.update({key: mean(row[key] for row in rows) for key in
                   ["plan_mae_seconds", "imbalance_seconds", "native_headway_penalty"]})
    for label in P["arms"] + ["old_raw", "fixed0"]:
        pairs = [(row, baseline(row["domain"], label, row["seed"], row["evaluation_seed"], row["scene"])) for row in rows]
        assert all(a["groups"][name]["passengers"] == b["groups"][name]["passengers"]
                   for a, b in pairs for name in ["all", "baseline", "extra", "baseline_during"])
        differences = [b["cost"] - a["cost"] for a, b in pairs]
        result["reduction_vs_" + label] = dict(mean_seconds=mean(differences),
            improved_cases=sum(x > 1e-5 for x in differences), worse_cases=sum(x < -1e-5 for x in differences),
            mean_wait_seconds=mean(b["groups"]["all"]["mean_wait_seconds"] - a["groups"]["all"]["mean_wait_seconds"] for a, b in pairs),
            mean_onboard_seconds=mean(b["groups"]["all"]["mean_onboard_seconds"] - a["groups"]["all"]["mean_onboard_seconds"] for a, b in pairs),
            plan_mae_seconds=mean(b["plan_mae_seconds"] - a["plan_mae_seconds"] for a, b in pairs),
            imbalance_seconds=mean(b["imbalance_seconds"] - a["imbalance_seconds"] for a, b in pairs),
            original_window_mean_seconds=mean(b["groups"]["baseline_during"]["mean_generalized_seconds"] - a["groups"]["baseline_during"]["mean_generalized_seconds"] for a, b in pairs),
            completion_percentage_point_change=100 * mean(a["groups"]["all"]["completion_rate"] - b["groups"]["all"]["completion_rate"] for a, b in pairs))
    return result


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/*/*/seed*/day*.json")]
    expected = {(domain, variant, arm, seed, day, scene) for domain in ["source", "target"]
                for variant, arms in P["controllers"].items() for arm, seed, day, scene in
                product(arms, P["training_seeds"], P[domain + "_days"], P[domain + "_scenes"])}
    assert len(rows) == len(expected) == 204
    assert {(r["domain"], r["variant"], r["arm"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected
    assert all(r["updates"] == 0 and r["horizon_seconds"] == P["horizon_seconds"] and r["timestamp_cost_identity_passed"] for r in rows)
    output = dict(complete=True, evaluations=len(rows), reused_full_evaluations=136, training_runs=0,
        analysis_location="server", groups={}, per_seed={}, source_weighted_cost_seconds={},
        maximum_wall_seconds=max(r["wall_seconds"] for r in rows), maximum_rss_mb=max(r["peak_rss_mb"] for r in rows))
    prior = json.loads((PRIOR / "analysis_summary.json").read_text())
    output["full_policy_controls"] = prior["groups"]
    for domain in ["source", "target"]:
        for variant, arms in P["controllers"].items():
            for arm, scene in product(arms, P[domain + "_scenes"]):
                cases = [r for r in rows if (r["domain"], r["variant"], r["arm"], r["scene"]) == (domain, variant, arm, scene)]
                output["groups"][f"{domain}/{variant}/{arm}/{scene}"] = summarize(cases)
                for seed in P["training_seeds"]:
                    output["per_seed"][f"{domain}/{variant}/{arm}/seed{seed}/{scene}"] = summarize([r for r in cases if r["seed"] == seed])
    for variant, arms in P["controllers"].items():
        for arm in arms:
            output["source_weighted_cost_seconds"][f"{variant}/{arm}"] = sum(P["source_weights"][scene] * output["groups"][f"source/{variant}/{arm}/{scene}"]["cost_seconds"] for scene in P["source_scenes"])
    output["source_weighted_cost_seconds"].update({"full/" + k: v for k, v in prior["source_weighted_cost_seconds"].items()})
    metrics.write(HERE / "analysis_summary.json", output)
    print(json.dumps({k: output[k] for k in ["complete", "evaluations", "training_runs", "source_weighted_cost_seconds"]}), flush=True)


if __name__ == "__main__":
    (smoke_report if len(sys.argv) > 1 and sys.argv[1] == "smoke-report" else collect)()
