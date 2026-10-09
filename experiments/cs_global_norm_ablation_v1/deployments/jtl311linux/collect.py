"""Compare the isolated ablation with completed, matched server-side controls."""

from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
REFERENCE = HERE.parent / P["reference_campaign"]


def metrics(cases):
    return dict(cases=len(cases),
        cost_seconds=statistics.mean(r["cost"] for r in cases),
        holding_seconds=statistics.mean(r["mean_holding_seconds"] for r in cases),
        plan_mae_seconds=statistics.mean(r["plan_mae_seconds"] for r in cases),
        native_headway_penalty=statistics.mean(r["native_headway_penalty"] for r in cases),
        completion_rate=statistics.mean(r["groups"]["all"]["completion_rate"] for r in cases),
        baseline_cost_seconds=statistics.mean(r["groups"]["baseline"]["mean_generalized_seconds"] for r in cases),
        original_window_cost_seconds=statistics.mean(r["groups"]["baseline_during"]["mean_generalized_seconds"] for r in cases))


def training_metrics(root, seed):
    directory = root / "runs/global_event" / f"seed{seed}"
    row = json.loads((directory / "train_summary.json").read_text())
    settings = json.loads((directory / "settings.json").read_text())
    history = row["history"]
    assert len(history) == P["episodes"] and row["training_steps"] == P["episodes"] * P["updates_per_episode"]
    assert all(r["updates"] == P["updates_per_episode"] and r["timestamp_cost_identity_passed"]
               and r["credit"]["cost_identity_passed"] for r in history)
    blocks = {}
    for first, last in [(1, 10), (41, 50), (91, 100)]:
        cases = history[first - 1:last]
        blocks[f"episodes{first}-{last}"] = dict(
            cost_seconds=statistics.mean(r["cost"] for r in cases),
            holding_seconds=statistics.mean(r["mean_holding_seconds"] for r in cases),
            update_mean={key: statistics.mean(r["update_mean"][key] for r in cases)
                         for key in cases[0]["update_mean"]},
            reward_std=statistics.mean(r["replay_reward_std"] for r in cases))
    return settings["config"], dict(episodes=len(history), updates=row["training_steps"],
        wall_seconds=row["training_wall_seconds"], peak_rss_mb=row["peak_rss_mb"],
        blocks=blocks, final_update=history[-1]["update_last"])


def collect():
    rows = [json.loads(path.read_text()) for path in (HERE / "results/global_event").glob("seed*/day*.json")]
    expected = {(seed, day, scene) for seed in P["training_seeds"]
                for day in P["evaluation_seeds"] for scene in P["scenes"]}
    by_case = {(r["seed"], r["evaluation_seed"], r["scene"]): r for r in rows}
    assert len(by_case) == len(rows) and set(by_case) == expected
    references = ["global_event", "observed_overlap", "frozen_passenger", "fixed0"]
    grouped, per_seed, contrasts, control_groups = defaultdict(list), defaultdict(list), {}, {}
    for row in rows:
        assert row["arm"] == "global_event" and row["policy_phase"] == "formal"
        assert row["horizon_seconds"] == P["horizon_seconds"] and row["timestamp_cost_identity_passed"]
        grouped[row["scene"]].append(row)
        per_seed[row["seed"], row["scene"]].append(row)
    for reference in references:
        for scene in P["scenes"]:
            pairs, controls = [], {}
            for seed in P["training_seeds"]:
                for day in P["evaluation_seeds"]:
                    candidate = by_case[seed, day, scene]
                    control_seed = 0 if reference == "fixed0" else seed
                    path = REFERENCE / "results" / reference / f"seed{control_seed}" / f"day{day}_{scene}.json"
                    control = json.loads(path.read_text())
                    assert (control["arm"], control["seed"], control["evaluation_seed"], control["scene"]) == (reference, control_seed, day, scene)
                    assert control["policy_phase"] == "formal" and control["horizon_seconds"] == P["horizon_seconds"]
                    assert control["timestamp_cost_identity_passed"]
                    for group in ["all", "baseline", "extra", "baseline_during"]:
                        assert candidate["groups"][group]["passengers"] == control["groups"][group]["passengers"]
                    controls[control_seed, day] = control
                    pairs.append(dict(seed=seed, day=day,
                        cost_reduction_seconds=control["cost"] - candidate["cost"],
                        cost_reduction_percent=100 * (control["cost"] - candidate["cost"]) / control["cost"],
                        original_window_reduction_seconds=control["groups"]["baseline_during"]["mean_generalized_seconds"] - candidate["groups"]["baseline_during"]["mean_generalized_seconds"]))
            contrasts[f"no_norm-vs-{reference}/{scene}"] = dict(paired_cases=len(pairs),
                improved_cases=sum(r["cost_reduction_seconds"] > 0 for r in pairs),
                mean_cost_reduction_seconds=statistics.mean(r["cost_reduction_seconds"] for r in pairs),
                mean_cost_reduction_percent=statistics.mean(r["cost_reduction_percent"] for r in pairs), pairs=pairs)
            control_groups[reference + "/" + scene] = metrics(list(controls.values()))
    trains = {}
    for seed in P["training_seeds"]:
        config, current = training_metrics(HERE, seed)
        previous_config, previous = training_metrics(REFERENCE, seed)
        difference = {key: dict(previous=previous_config[key], current=config[key])
                      for key in config if config[key] != previous_config[key]}
        assert difference == {"weight_reg": dict(previous=.01, current=0.)}, difference
        trains[f"seed{seed}"] = dict(configuration_difference=difference, no_norm=current, norm_001=previous)
    result = dict(complete=True, analysis_location="server", evaluations=len(rows),
        expected_evaluations=len(expected), reused_control_evaluations=len(references) * len(expected) - 18,
        training_runs=len(trains), paired_passenger_counts_passed=True,
        groups={scene: metrics(cases) for scene, cases in grouped.items()},
        per_seed={f"seed{seed}/{scene}": metrics(cases) for (seed, scene), cases in per_seed.items()},
        control_groups=control_groups, contrasts=contrasts, training=trains)
    (HERE / "analysis_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ["complete", "evaluations", "training_runs"]}))


if __name__ == "__main__":
    collect()
