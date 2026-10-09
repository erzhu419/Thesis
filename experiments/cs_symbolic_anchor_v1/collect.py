"""Paired symbolic controls, analyzed on the server without weight downloads."""

from itertools import product
import importlib.util
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
path = HERE.parent / "cs_source_envelope_v1/collect.py"
spec = importlib.util.spec_from_file_location("symbolic_metrics", path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/*/seed*/day*.json")]
    expected = {(domain, arm, seed, day, scene) for domain in ["source", "target"]
                for arm, seed, day, scene in product(P["arms"], P["training_seeds"], P[domain + "_days"], P[domain + "_scenes"])}
    assert len(rows) == len(expected) and {(r["domain"], r["arm"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected
    assert all(r["updates"] == 0 and r["timestamp_cost_identity_passed"] for r in rows)
    controls = json.loads((HERE / "inputs/controls.json").read_text())
    output = dict(complete=True, evaluations=len(rows), groups={}, per_seed={}, training={},
                  controls=controls, source_weighted_cost_seconds={}, analysis_location="server")
    configs = {}
    for arm, seed in product(P["arms"], P["training_seeds"]):
        folder = HERE / "runs" / arm / f"seed{seed}"
        summary = json.loads((folder / "train_summary.json").read_text())
        configs[arm, seed] = json.loads((folder / "settings.json").read_text())["config"]
        history = summary["history"]
        assert len(history) == 100 and summary["training_steps"] == 40000
        assert all(r["updates"] == 400 and r["timestamp_cost_identity_passed"] for r in history)
        assert sum(r["source_regime"] == "normal" for r in history) == 40
        assert configs[arm, seed]["jac_weight"] == P["jac_weight"] and configs[arm, seed]["weight_sym"] == P["weight_sym"]
        for key in ["jac_loss", "sym_penalty"]:
            values = [r["update_mean"][key] for r in history]
            assert min(values) >= 0 and max(values) > 0
        output["training"][f"{arm}/seed{seed}"] = {k:summary[k] for k in ["training_wall_seconds", "peak_rss_mb", "anchor"]}
        output["training"][f"{arm}/seed{seed}"].update(mean_jac_loss=mean(r["update_mean"]["jac_loss"] for r in history),
                                                      mean_sym_penalty=mean(r["update_mean"]["sym_penalty"] for r in history))
    for seed in P["training_seeds"]:
        assert configs["symbolic0", seed] == configs["symbolic10", seed]
    for domain in ["source", "target"]:
        for arm, scene in product(P["arms"], P[domain + "_scenes"]):
            cases = [r for r in rows if (r["domain"], r["arm"], r["scene"]) == (domain, arm, scene)]
            for key, selected in [(f"{domain}/{arm}/{scene}", cases)] + [
                    (f"{domain}/{arm}/seed{seed}/{scene}", [r for r in cases if r["seed"] == seed]) for seed in P["training_seeds"]]:
                value = metrics.metrics(selected)
                for label in ["symbolic0", "old_raw", "fixed0"]:
                    comparisons = []
                    for r in selected:
                        if label == "symbolic0":
                            baseline = HERE / "results" / domain / label / f"seed{r['seed']}" / f"day{r['evaluation_seed']}_{scene}.json"
                        else:
                            folder = ("cs_source_validation_v1/results/" + label if domain == "source" else
                                      "cs_terminal_coverage_v1/results/global_coverage" if label == "old_raw" else
                                      "cs_passenger_alignment_v1/results/fixed0")
                            baseline = HERE.parent / folder / f"seed{0 if label == 'fixed0' else r['seed']}" / f"day{r['evaluation_seed']}_{scene}.json"
                        b = json.loads(baseline.read_text())
                        comparisons.append(dict(cost=b["cost"] - r["cost"], window=b["groups"]["baseline_during"]["mean_generalized_seconds"] - r["groups"]["baseline_during"]["mean_generalized_seconds"]))
                    value["reduction_vs_" + label] = dict(mean_seconds=mean(c["cost"] for c in comparisons),
                        improved_cases=sum(c["cost"] > 1e-5 for c in comparisons),
                        worse_cases=sum(c["cost"] < -1e-5 for c in comparisons),
                        original_window_mean_seconds=mean(c["window"] for c in comparisons))
                (output["groups"] if key == f"{domain}/{arm}/{scene}" else output["per_seed"])[key] = value
    for arm in P["arms"]:
        output["source_weighted_cost_seconds"][arm] = sum(P["source_weights"][scene] * output["groups"][f"source/{arm}/{scene}"]["cost_seconds"] for scene in P["source_scenes"])
    output["source_selected_arm"] = min(output["source_weighted_cost_seconds"], key=output["source_weighted_cost_seconds"].get)
    metrics.write(HERE / "analysis_summary.json", output)
    print(json.dumps(dict(complete=True, evaluations=len(rows), source_costs=output["source_weighted_cost_seconds"])))


if __name__ == "__main__":
    collect()
