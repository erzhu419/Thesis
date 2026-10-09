"""Compare frozen feedback variants against existing matched controls on-server."""

from itertools import product
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / P["reference_campaign"]
spec = importlib.util.spec_from_file_location("temperature_previous_metrics", PRIOR / "collect.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
write = previous.metrics.write


def smoke_report():
    native = json.loads((HERE / "smoke/target/identity/seed64001/day78101_normal.json").read_text())
    original = json.loads((PRIOR / "results/target/sindy/reference/seed64001/day78101_normal.json").read_text())
    keys = ["cost", "exact_cost", "decisions", "transitions", "mean_holding_seconds", "groups", "input_stats"]
    assert all(native[key] == original[key] for key in keys), "Original feedback execution differs"
    probes = []
    for variant in P["variants"]:
        row = json.loads((HERE / "smoke/target" / variant / "seed64002/day78101_static10.json").read_text())
        assert row["updates"] == 0 and row["timestamp_cost_identity_passed"] and row["horizon_seconds"] == 7200
        assert row["input_stats"]["source_cap_exceeded_fraction"] > 0
        assert row["temperature"] == P["variants"][variant]
        probes.append({k: row[k] for k in ["variant", "cost", "mean_holding_seconds", "wall_seconds", "peak_rss_mb"]})
    result = dict(passed=True, original_full_horizon_execution_identical=True, high_demand_probes=probes)
    write(HERE / "smoke_passed.json", result)
    print(json.dumps(result), flush=True)


def summarize(rows):
    result = previous.summarize(rows)
    for label, folder in [("original_sindy", "sindy/reference"), ("bias", "bias/symbolic0")]:
        pairs = [(r, json.loads((PRIOR / "results" / r["domain"] / folder / f"seed{r['seed']}" /
                                  f"day{r['evaluation_seed']}_{r['scene']}.json").read_text())) for r in rows]
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
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/*/seed*/day*.json")]
    expected = {(domain, variant, seed, day, scene) for domain in ["source", "target"] for variant, seed, day, scene in
                product(P["variants"], P["training_seeds"], P[domain + "_days"], P[domain + "_scenes"])}
    assert len(rows) == len(expected) == 204
    assert {(r["domain"], r["variant"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected
    assert all(r["updates"] == 0 and r["horizon_seconds"] == P["horizon_seconds"] and r["timestamp_cost_identity_passed"]
               and r["temperature"] == P["variants"][r["variant"]] for r in rows)
    output = dict(complete=True, evaluations=len(rows), training_runs=0, analysis_location="server",
        groups={}, per_seed={}, source_weighted_cost_seconds={},
        maximum_wall_seconds=max(r["wall_seconds"] for r in rows), maximum_rss_mb=max(r["peak_rss_mb"] for r in rows))
    for domain in ["source", "target"]:
        for variant, scene in product(P["variants"], P[domain + "_scenes"]):
            cases = [r for r in rows if (r["domain"], r["variant"], r["scene"]) == (domain, variant, scene)]
            output["groups"][f"{domain}/{variant}/{scene}"] = summarize(cases)
            for seed in P["training_seeds"]:
                output["per_seed"][f"{domain}/{variant}/seed{seed}/{scene}"] = summarize([r for r in cases if r["seed"] == seed])
    for variant in P["variants"]:
        output["source_weighted_cost_seconds"][variant] = sum(P["source_weights"][scene] * output["groups"][f"source/{variant}/{scene}"]["cost_seconds"] for scene in P["source_scenes"])
    output["source_selected_variant"] = min(output["source_weighted_cost_seconds"], key=output["source_weighted_cost_seconds"].get)
    write(HERE / "analysis_summary.json", output)
    print(json.dumps({k: output[k] for k in ["complete", "evaluations", "source_weighted_cost_seconds", "source_selected_variant"]}), flush=True)


if __name__ == "__main__":
    (smoke_report if len(sys.argv) > 1 and sys.argv[1] == "smoke-report" else collect)()
