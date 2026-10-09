"""Source-only paired evidence, aggregated on the server."""

import importlib.util
from itertools import product
import json
from pathlib import Path
from statistics import mean

from response import describe, merge

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
spec = importlib.util.spec_from_file_location("source_validation_metrics", HERE.parent / "cs_source_envelope_v1/collect.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/seed*/day*.json")]
    expected = {(arm, seed, day, scene) for arm, seed in
                [(arm, seed) for arm, seed in product(["old_raw", "source_clip"], P["training_seeds"])] + [("fixed0", 0)]
                for day, scene in product(P["validation_days"], P["scenes"])}
    actual = {(r["arm"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows}
    assert len(rows) == len(expected) and actual == expected
    indexed = {(r["arm"], r["seed"], r["evaluation_seed"], r["scene"]): r for r in rows}
    assert all(r["updates"] == 0 and r["timestamp_cost_identity_passed"] and r["frozen_actor"] for r in rows)
    for row in rows:
        ref = indexed["fixed0", 0, row["evaluation_seed"], row["scene"]]
        for group in ["all", "baseline", "extra", "baseline_during"]:
            assert row["groups"][group]["passengers"] == ref["groups"][group]["passengers"]
    result = dict(complete=True, evaluations=len(rows), new_training_runs=0, analysis_location="server",
                  groups={}, per_seed={}, paired={}, source_weights=P["source_weights"],
                  validation_days=P["validation_days"], native_equivalence_cases=2,
                  maximum_wall_seconds=max(r["wall_seconds"] for r in rows),
                  maximum_rss_mb=max(r["peak_rss_mb"] for r in rows))
    for arm in ["old_raw", "source_clip", "fixed0"]:
        for scene in P["scenes"]:
            cases = [r for r in rows if r["arm"] == arm and r["scene"] == scene]
            result["groups"][arm + "/" + scene] = shared.metrics(cases)
            if arm == "fixed0":
                continue
            for key, selected in [(arm + "/" + scene, cases)] + [
                    (f"{arm}/seed{seed}/{scene}", [r for r in cases if r["seed"] == seed]) for seed in P["training_seeds"]]:
                value = shared.metrics(selected)
                value["response"] = describe(merge([r["response"] for r in selected]))
                n = sum(r["input_stats"]["feature_calls"] for r in selected)
                value["source_cap_exceeded_fraction"] = sum(r["input_stats"]["source_cap_exceeded_fraction"] * r["input_stats"]["feature_calls"] for r in selected) / n
                value["maximum_queue_ratio"] = max(r["input_stats"]["raw_total_maximum"] / r["source_cap"] for r in selected)
                if key == arm + "/" + scene:
                    result["groups"][key] = value
                else:
                    result["per_seed"][key] = value
    for scene in P["scenes"]:
        pairs = []
        for seed, day in product(P["training_seeds"], P["validation_days"]):
            old = indexed["old_raw", seed, day, scene]
            new = indexed["source_clip", seed, day, scene]
            pairs.append(dict(seed=seed, day=day, old_cost_seconds=old["cost"], new_cost_seconds=new["cost"],
                              reduction_seconds=old["cost"] - new["cost"],
                              holding_change_seconds=new["mean_holding_seconds"] - old["mean_holding_seconds"]))
        result["paired"][scene] = dict(cases=len(pairs), reduction_seconds=mean(r["reduction_seconds"] for r in pairs),
                                      improved_cases=sum(r["reduction_seconds"] > 1e-5 for r in pairs),
                                      worse_cases=sum(r["reduction_seconds"] < -1e-5 for r in pairs), pairs=pairs)
    result["source_weighted_cost_seconds"] = {
        arm:sum(P["source_weights"][scene] * result["groups"][arm + "/" + scene]["cost_seconds"] for scene in P["scenes"])
        for arm in ["old_raw", "source_clip", "fixed0"]}
    shared.write(HERE / "analysis_summary.json", result)
    print(json.dumps({k:result[k] for k in ["complete", "evaluations", "new_training_runs", "maximum_wall_seconds", "maximum_rss_mb"]}))


if __name__ == "__main__":
    collect()
