"""Server-side matched comparisons for two separate diagnostic branches."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from regimes import source_schedule, training_regime

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
REFERENCE = HERE.parent / P["reference_campaign"]
ALIGNMENT = HERE.parent / "cs_passenger_alignment_v1"


def metrics(cases):
    return dict(cases=len(cases),
        cost_seconds=mean(r["cost"] for r in cases),
        holding_seconds=mean(r["mean_holding_seconds"] for r in cases),
        mean_wait_seconds=mean(r["groups"]["all"]["mean_wait_seconds"] for r in cases),
        mean_onboard_seconds=mean(r["groups"]["all"]["mean_onboard_seconds"] for r in cases),
        plan_mae_seconds=mean(r["plan_mae_seconds"] for r in cases),
        native_headway_penalty=mean(r["native_headway_penalty"] for r in cases),
        completion_rate=mean(r["groups"]["all"]["completion_rate"] for r in cases),
        baseline_cost_seconds=mean(r["groups"]["baseline"]["mean_generalized_seconds"] for r in cases),
        original_window_cost_seconds=mean(r["groups"]["baseline_during"]["mean_generalized_seconds"] for r in cases))


def training(arm, seed):
    folder = HERE / "runs" / arm / f"seed{seed}"
    record = json.loads((folder / "train_summary.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())
    reference_config = json.loads((REFERENCE / "runs/global_event" / f"seed{seed}/settings.json").read_text())["config"]
    assert settings["config"] == reference_config
    history = record["history"]
    assert len(history) == P["episodes"] and record["training_steps"] == P["episodes"] * P["updates_per_episode"]
    for episode, row in enumerate(history):
        assert row["updates"] == P["updates_per_episode"] and row["timestamp_cost_identity_passed"]
        assert row["credit"]["cost_identity_passed"] and row["credit"]["terminal_transitions"] == 1
        regime = training_regime(arm, episode, P["source_training_cycle"])
        assert row["source_regime"] == regime
        assert row["source_schedule"] == source_schedule(regime, P["horizon_seconds"])
    if arm == "global_terminal":
        assert settings["boundary_weight"] == 1. and settings["learner_change"] == "terminal_boundary_loss"
        assert all(r["terminal_boundary_updates"] == P["updates_per_episode"] for r in history[1:])
    else:
        assert settings["boundary_weight"] == 0. and settings["learner_change"] == "none"
        assert all(r["terminal_boundary_updates"] == 0 for r in history)
    blocks = {}
    for first, last in [(1, 10), (41, 50), (91, 100)]:
        cases = history[first-1:last]
        updates = {key: mean(r["update_mean"][key] for r in cases) for key in cases[0]["update_mean"]}
        updates["q_loss_base"] = mean(r["update_mean"].get("q_loss_base", r["update_mean"]["q_loss"]) for r in cases)
        blocks[f"episodes{first}-{last}"] = dict(
            cost_seconds=mean(r["cost"] for r in cases),
            holding_seconds=mean(r["mean_holding_seconds"] for r in cases),
            transitions=mean(r["transitions"] for r in cases),
            zero_time_fraction=mean(r["credit"]["zero_time_transitions"] / r["transitions"] for r in cases),
            update_mean=updates)
    return dict(episodes=len(history), updates=record["training_steps"],
        configuration_difference=[], learner_change=settings["learner_change"],
        boundary_weight=settings["boundary_weight"],
        source_regime_counts=dict(Counter(r["source_regime"] for r in history)),
        boundary_updates=sum(r["terminal_boundary_updates"] for r in history),
        final_terminal_examples=history[-1]["terminal_examples_retained"],
        wall_seconds=record["training_wall_seconds"], peak_rss_mb=record["peak_rss_mb"], blocks=blocks)


def collect():
    rows = [json.loads(p.read_text()) for p in (HERE / "results").glob("*/seed*/day*.json")]
    expected = {(arm, seed, day, scene) for arm in P["arms"] for seed in P["training_seeds"]
                for day in P["evaluation_seeds"] for scene in P["scenes"]}
    by_case = {(r["arm"], r["seed"], r["evaluation_seed"], r["scene"]): r for r in rows}
    assert len(rows) == len(by_case) and set(by_case) == expected
    grouped, per_seed = defaultdict(list), defaultdict(list)
    for row in rows:
        assert row["policy_phase"] == "formal" and row["horizon_seconds"] == P["horizon_seconds"]
        assert row["timestamp_cost_identity_passed"]
        grouped[row["arm"], row["scene"]].append(row)
        per_seed[row["arm"], row["seed"], row["scene"]].append(row)
    controls = [("no_norm", REFERENCE, "global_event"),
                ("observed_overlap", ALIGNMENT, "observed_overlap"),
                ("frozen_passenger", ALIGNMENT, "frozen_passenger"),
                ("fixed0", ALIGNMENT, "fixed0")]
    contrasts, control_groups = {}, {}
    reused = set()
    for arm in P["arms"]:
        for label, root, control_arm in controls:
            for scene in P["scenes"]:
                pairs, control_rows = [], {}
                for seed in P["training_seeds"]:
                    for day in P["evaluation_seeds"]:
                        row = by_case[arm, seed, day, scene]
                        control_seed = 0 if label == "fixed0" else seed
                        path = root / "results" / control_arm / f"seed{control_seed}" / f"day{day}_{scene}.json"
                        control = json.loads(path.read_text())
                        assert (control["arm"], control["seed"], control["evaluation_seed"], control["scene"]) == (control_arm, control_seed, day, scene)
                        assert control["policy_phase"] == "formal" and control["horizon_seconds"] == P["horizon_seconds"]
                        for group in ["all", "baseline", "extra", "baseline_during"]:
                            assert row["groups"][group]["passengers"] == control["groups"][group]["passengers"]
                        control_rows[control_seed, day] = control
                        reused.add(str(path))
                        pairs.append(dict(seed=seed, day=day,
                            cost_reduction_seconds=control["cost"]-row["cost"],
                            cost_reduction_percent=100*(control["cost"]-row["cost"])/control["cost"],
                            original_window_reduction_seconds=control["groups"]["baseline_during"]["mean_generalized_seconds"]-row["groups"]["baseline_during"]["mean_generalized_seconds"]))
                contrasts[f"{arm}-vs-{label}/{scene}"] = dict(paired_cases=len(pairs),
                    improved_cases=sum(r["cost_reduction_seconds"] > 0 for r in pairs),
                    mean_cost_reduction_seconds=mean(r["cost_reduction_seconds"] for r in pairs),
                    mean_cost_reduction_percent=mean(r["cost_reduction_percent"] for r in pairs), pairs=pairs)
                control_groups[label + "/" + scene] = metrics(list(control_rows.values()))
    trains = {f"{arm}/seed{seed}": training(arm, seed)
              for arm in P["arms"] for seed in P["training_seeds"]}
    output = dict(complete=True, analysis_location="server", evaluations=len(rows),
        expected_evaluations=len(expected), training_runs=len(trains),
        reused_control_evaluations=len(reused), paired_passenger_counts_passed=True,
        groups={arm+"/"+scene: metrics(cases) for (arm, scene), cases in grouped.items()},
        per_seed={f"{arm}/seed{seed}/{scene}": metrics(cases) for (arm, seed, scene), cases in per_seed.items()},
        control_groups=control_groups, contrasts=contrasts, training=trains)
    (HERE / "analysis_summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in ["complete", "evaluations", "training_runs"]}))


if __name__ == "__main__":
    collect()
