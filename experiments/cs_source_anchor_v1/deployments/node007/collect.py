"""Server-side source comparison, followed by separately reported target tests."""

import importlib.util
from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("anchor_metrics", HERE.parent / "cs_source_envelope_v1/collect.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/*/seed*/day*.json")]
    expected = {(domain, arm, seed, day, scene) for domain in ["source", "target"]
                for arm, seed, day, scene in product(P["arms"], P["training_seeds"], P[domain + "_days"], P[domain + "_scenes"])}
    assert len(rows) == len(expected) and {(r["domain"], r["arm"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected
    assert all(r["updates"] == 0 and r["timestamp_cost_identity_passed"] for r in rows)
    result = dict(complete=True, evaluations=len(rows), groups={}, per_seed={}, source_weighted_cost_seconds={},
                  training={}, controls=json.loads((HERE / "inputs/controls.json").read_text()), analysis_location="server")
    for arm, seed in product(P["arms"], P["training_seeds"]):
        summary = json.loads((HERE / "runs" / arm / f"seed{seed}/train_summary.json").read_text())
        assert summary["training_steps"] == 40000 and len(summary["history"]) == 100
        assert all(r["timestamp_cost_identity_passed"] and r["updates"] == 400 for r in summary["history"])
        assert [r["source_regime"] for r in summary["history"]].count("normal") == 40
        result["training"][f"{arm}/seed{seed}"] = {k:summary[k] for k in ["training_wall_seconds", "peak_rss_mb", "anchor"]}
    for domain in ["source", "target"]:
        for arm, scene in product(P["arms"], P[domain + "_scenes"]):
            cases = [r for r in rows if (r["domain"], r["arm"], r["scene"]) == (domain, arm, scene)]
            for key, selected in [(f"{domain}/{arm}/{scene}", cases)] + [
                    (f"{domain}/{arm}/seed{seed}/{scene}", [r for r in cases if r["seed"] == seed]) for seed in P["training_seeds"]]:
                value = shared.metrics(selected)
                value.update(reduction_vs_unanchored_seconds=mean(r["cost_reduction_vs_unanchored"] for r in selected),
                             improved_vs_unanchored_cases=sum(r["cost_reduction_vs_unanchored"] > 1e-5 for r in selected),
                             worse_vs_unanchored_cases=sum(r["cost_reduction_vs_unanchored"] < -1e-5 for r in selected))
                (result["groups"] if key == f"{domain}/{arm}/{scene}" else result["per_seed"])[key] = value
    for arm in P["arms"]:
        result["source_weighted_cost_seconds"][arm] = sum(P["source_weights"][scene] * result["groups"][f"source/{arm}/{scene}"]["cost_seconds"] for scene in P["source_scenes"])
    candidates = dict(result["source_weighted_cost_seconds"])
    candidates["unanchored"] = result["controls"]["source_weighted_cost_seconds"]["source_clip"]
    result["source_selected_arm"] = min(candidates, key=candidates.get)
    shared.write(HERE / "analysis_summary.json", result)
    print(json.dumps(dict(complete=True, evaluations=len(rows), source_costs=result["source_weighted_cost_seconds"])))


if __name__ == "__main__":
    collect()
