"""Server-only paired analysis, including prior negative diagnostic controls."""

from collections import Counter
import importlib.util
from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
THRESHOLDS = HERE.parent / P["threshold_campaign"]
REFERENCE = HERE.parent / P["reference_campaign"]
spec = importlib.util.spec_from_file_location("frozen_queue_metrics", THRESHOLDS / "collect.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results").glob("*/seed*/day*.json")]
    expected = set(product(P["arms"], P["training_seeds"], P["evaluation_days"], P["scenes"]))
    assert {(r["arm"], r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected and len(rows) == len(expected)
    assert all(r["updates"] == 0 and r["timestamp_cost_identity_passed"] and r["policy_phase"] == "formal" for r in rows)
    originals = json.loads((THRESHOLDS / "inputs/reference.json").read_text())
    output = dict(complete=True, analysis_location="server", training_runs=4, evaluations=len(rows),
        training={}, groups={}, per_seed={}, controls={}, source_fit=json.loads((THRESHOLDS / "fit_summary.json").read_text()))
    for arm, seed in product(P["arms"], P["training_seeds"]):
        root = HERE / "runs" / arm / f"seed{seed}"
        settings = json.loads((root / "settings.json").read_text())
        record = json.loads((root / "train_summary.json").read_text())
        history = record["history"]
        old = json.loads((REFERENCE / "runs/global_coverage" / f"seed{seed}/settings.json").read_text())
        assert settings["config"] == old["config"] and settings["learner_change"] == "none"
        assert len(history) == P["episodes"] and record["training_steps"] == P["episodes"] * P["updates_per_episode"]
        for episode, row in enumerate(history):
            assert row["source_regime"] == P["source_training_cycle"][episode % len(P["source_training_cycle"])]
            assert row["updates"] == P["updates_per_episode"] and row["timestamp_cost_identity_passed"]
            assert row["credit"]["cost_identity_passed"] and row["credit"]["terminal_transitions"] == 1
            assert row["input_stats"]["feature_calls"] == row["decisions"]
        output["training"][f"{arm}/seed{seed}"] = dict(episodes=len(history), updates=record["training_steps"],
            wall_seconds=record["training_wall_seconds"], peak_rss_mb=record["peak_rss_mb"],
            source_regime_counts=dict(Counter(r["source_regime"] for r in history)),
            configuration_difference=[], mean_source_cap_exceeded_fraction=mean(r["input_stats"]["source_cap_exceeded_fraction"] for r in history),
            last10_cost_seconds=mean(r["cost"] for r in history[-10:]),
            last10_holding_seconds=mean(r["mean_holding_seconds"] for r in history[-10:]),
            final_update=history[-1]["update_last"])
    for arm, scene in product(P["arms"], P["scenes"]):
        cases = sorted([r for r in rows if r["arm"] == arm and r["scene"] == scene], key=lambda r: (r["seed"], r["evaluation_seed"]))
        pairs, controls = [], {k: [] for k in ["unchanged", "frozen_daily", "frozen_phase_total", "observed_overlap", "fixed0"]}
        for row in cases:
            seed, day = row["seed"], row["evaluation_seed"]
            control = {"unchanged": originals[f"{seed}/{day}/{scene}"],
                "frozen_daily": json.loads((THRESHOLDS / "results/daily_total" / f"seed{seed}" / f"day{day}_{scene}.json").read_text())["rollout"],
                "frozen_phase_total": json.loads((HERE.parent / "cs_phase_total_v1/results" / f"seed{seed}" / f"day{day}_{scene}.json").read_text())["rollout"],
                "observed_overlap": json.loads((HERE.parent / "cs_passenger_alignment_v1/results/observed_overlap" / f"seed{seed}" / f"day{day}_{scene}.json").read_text()),
                "fixed0": json.loads((HERE.parent / "cs_passenger_alignment_v1/results/fixed0/seed0" / f"day{day}_{scene}.json").read_text())}
            pair = dict(seed=seed, day=day, cost_seconds=row["cost"], holding_seconds=row["mean_holding_seconds"], input_stats=row["input_stats"])
            for label, value in control.items():
                for group in ["all", "baseline", "extra", "baseline_during"]:
                    assert value["groups"][group]["passengers"] == row["groups"][group]["passengers"]
                controls[label].append(value)
                pair["reduction_vs_" + label] = value["cost"] - row["cost"]
            pairs.append(pair)
        for label, values in controls.items():
            output["controls"][label + "/" + scene] = shared.metrics(values)
        for label, selected in [(f"{arm}/{scene}", cases)] + [(f"{arm}/seed{seed}/{scene}", [r for r in cases if r["seed"] == seed]) for seed in P["training_seeds"]]:
            selected_pairs = [pair for pair in pairs if any(pair["seed"] == r["seed"] and pair["day"] == r["evaluation_seed"] for r in selected)]
            value = shared.metrics(selected)
            for control in controls:
                differences = [pair["reduction_vs_" + control] for pair in selected_pairs]
                value["reduction_vs_" + control] = dict(mean_seconds=mean(differences), improved_cases=sum(v > 1e-5 for v in differences))
            if label == f"{arm}/{scene}":
                value["pairs"] = pairs
                output["groups"][label] = value
            else:
                output["per_seed"][label] = value
    shared.write(HERE / "analysis_summary.json", output)
    print(json.dumps({k: output[k] for k in ["complete", "training_runs", "evaluations"]}))


if __name__ == "__main__":
    collect()
