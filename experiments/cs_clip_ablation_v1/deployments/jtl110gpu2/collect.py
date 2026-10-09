"""Paired server-side analysis of the upper-bound ablation."""

import importlib.util
from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PRIOR = HERE.parent / P["policy_campaign"]
spec = importlib.util.spec_from_file_location("clip_ablation_metrics", HERE.parent / "cs_source_envelope_v1/collect.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def collect():
    rows = [json.loads(p.read_text()) for p in (HERE / "results/linear_total").glob("seed*/day*.json")]
    expected = set(product(P["training_seeds"], P["evaluation_days"], P["scenes"]))
    assert len(rows) == len(expected) and {(r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected
    assert all(r["updates"] == 0 and r["timestamp_cost_identity_passed"] and r["frozen_actor"] for r in rows)
    previous = json.loads((PRIOR / "analysis_summary.json").read_text())
    references = json.loads((HERE / "inputs/reference.json").read_text())
    output = dict(complete=True, evaluations=len(rows), new_training_runs=0, analysis_location="server",
        groups={}, per_seed={}, controls=previous["controls"], source_fit=previous["source_fit"],
        maximum_wall_seconds=max(r["wall_seconds"] for r in rows), maximum_rss_mb=max(r["peak_rss_mb"] for r in rows))
    for scene in P["scenes"]:
        cases = sorted([r for r in rows if r["scene"] == scene], key=lambda r: (r["seed"], r["evaluation_seed"]))
        clips = [references[f'{r["seed"]}/{r["evaluation_seed"]}/{scene}'] for r in cases]
        output["controls"]["trained_clip/" + scene] = shared.metrics(clips)
        for key, selected in [(scene, cases)] + [(f"seed{seed}/{scene}", [r for r in cases if r["seed"] == seed]) for seed in P["training_seeds"]]:
            value = shared.metrics(selected)
            n = sum(r["input_stats"]["feature_calls"] for r in selected)
            value.update(reference_matched_cases=sum(r["reference_rollout_matched"] for r in selected),
                reduction_vs_clip_seconds=mean(r["reduction_vs_clip"] for r in selected),
                improved_vs_clip_cases=sum(r["reduction_vs_clip"] > 1e-5 for r in selected),
                worse_vs_clip_cases=sum(r["reduction_vs_clip"] < -1e-5 for r in selected),
                source_cap_exceeded_fraction=sum(r["input_stats"]["source_cap_exceeded_fraction"] * r["input_stats"]["feature_calls"] for r in selected) / n,
                transformed_total_maximum=max(r["input_stats"]["transformed_total_maximum"] for r in selected))
            if key == scene:
                value["pairs"] = [dict(seed=r["seed"], day=r["evaluation_seed"], cost_seconds=r["cost"],
                    holding_seconds=r["mean_holding_seconds"], reduction_vs_clip_seconds=r["reduction_vs_clip"],
                    input_stats=r["input_stats"]) for r in selected]
                output["groups"][key] = value
            else:
                output["per_seed"][key] = value
    shared.write(HERE / "analysis_summary.json", output)
    print(json.dumps({k:output[k] for k in ["complete", "evaluations", "new_training_runs", "maximum_wall_seconds", "maximum_rss_mb"]}))


if __name__ == "__main__":
    collect()
