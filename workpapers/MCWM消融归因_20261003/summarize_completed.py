"""Aggregate existing server JSONs; leave rollouts and model weights resident."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def metrics(row):
    p = row["passengers"]
    result = dict(cost=row["generalized_seconds_per_departed_passenger"],
                  wait=p["accrued_wait_seconds_per_departed_passenger"],
                  ride=p["accrued_in_vehicle_seconds_per_departed_passenger"],
                  completion=100*p["completion_rate"], hold=row["mean_hold_seconds"])
    for key in ("headway_ridge_mean", "headway_historical_mean", "headway_plan_mae_seconds",
                "headway_imbalance_seconds", "mean_speed_ratio"):
        if key in row:
            result[key] = row[key]
    if "guard_changed_decisions" in row:
        result["guard_percent"] = 100*row["guard_changed_decisions"]/row["decision_events"]
    if "headway" in row:
        result["headway_normalized_error"] = row["headway"]["normalized_absolute_error"]["mean"]
    return result


def average(rows):
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def paired(first, second):
    differences = []
    for a, b in zip(first, second):
        assert (a["case"], a["demand_scale"], a["passengers"]["departed_passengers"]) == (
            b["case"], b["demand_scale"], b["passengers"]["departed_passengers"])
        differences.append({key: value-metrics(b)[key] for key, value in metrics(a).items()
                            if key in metrics(b)})
    return average(differences)


def training(folder, passenger):
    row = load(folder / ("results/ready.json" if passenger else "ready.json"))
    history = row.get("model_history", [])
    path = folder / "models/history.json"
    if path.exists():
        history = load(path)
    result = dict(wall_seconds=row["training_wall_seconds"], peak_rss_mb=row["peak_host_rss_mb"],
                  families=dict(Counter(h["selected_family"] for h in history)))
    probes = sorted((folder / "models").glob("reward_probe_*.json"))
    if probes:
        result["reward_probe_first"] = load(probes[0])
        result["reward_probe_last"] = load(probes[-1])
    return result


def passenger(root):
    here = root / "mcwm_attribution_v15"
    protocol = load(here / "protocol.json")
    previous = root / "mcwm_seed_confirmation_v13"
    rules = [load(previous / f"rules/daganzo_duet/case{case:02d}/results/case{case:02d}.json")
             for case in protocol["cases"]]
    output = []
    for scene in protocol["scenes"]:
        for arm in protocol["arms"] + protocol["reference_arms"]:
            rows, timing, differences = [], [], []
            origin = here if arm in protocol["arms"] else previous
            for seed in protocol["training_seeds"]:
                folder = origin / "runs" / scene / f"{arm}_seed{seed}"
                cases = [load(folder / f"results/eval_{case:02d}.json") for case in protocol["cases"]]
                assert len(cases) == len(rules) == 10
                rows.append(dict(seed=seed, **average([metrics(c) for c in cases])))
                differences.append(dict(seed=seed, **paired(cases, rules)))
                timing.append(training(folder, True))
            output.append(dict(scene=scene, arm=arm, means=average([
                {k:v for k,v in row.items() if k != "seed"} for row in rows]),
                seeds=rows, vs_daganzo=differences, training=timing))
    return dict(rule=average([metrics(r) for r in rules]), arms=output)


def headway(root):
    here = root / "mcwm_headway_alignment_v16"
    protocol = load(here / "protocol.json")
    output, comparisons, rule_metrics, train = [], [], [], []
    rules = {}
    for arm in ("daganzo_duet", "zero"):
        for layer in protocol["executors"]:
            rows = [load(here / f"rules/{arm}/{layer}/case{c:02d}.json") for c in protocol["cases"]]
            rules[arm, layer] = rows
            rule_metrics.append(dict(arm=arm, layer=layer, means=average([metrics(r) for r in rows])))
    for scene in protocol["scenes"]:
        all_rows = {}
        for arm in protocol["arms"]:
            for seed in protocol["training_seeds"]:
                folder = here / "runs" / scene / f"{arm}_seed{seed}"
                train.append(dict(scene=scene, arm=arm, seed=seed, **training(folder, False)))
                for layer in protocol["executors"]:
                    rows = [load(folder / f"{layer}/case{c:02d}.json") for c in protocol["cases"]]
                    all_rows[arm, seed, layer] = rows
            for layer in protocol["executors"]:
                means = [dict(seed=seed, **average([metrics(r) for r in all_rows[arm, seed, layer]]))
                         for seed in protocol["training_seeds"]]
                output.append(dict(scene=scene, arm=arm, layer=layer, seeds=means,
                    means=average([{k:v for k,v in row.items() if k != "seed"} for row in means])))
        for arm in protocol["arms"]:
            for layer in protocol["executors"]:
                for reference in ("h2oplus", "h2o_matched", "daganzo_duet", "zero"):
                    if reference == arm:
                        continue
                    differences = [dict(seed=seed, **paired(all_rows[arm, seed, layer],
                        rules[reference, layer] if reference in ("daganzo_duet", "zero") else
                        all_rows[reference, seed, layer])) for seed in protocol["training_seeds"]]
                    comparisons.append(dict(scene=scene, arm=arm, layer=layer, reference=reference,
                        seeds=differences, means=average([
                            {k:v for k,v in row.items() if k != "seed"} for row in differences])))
    previous = root / "bus_dual_aligned_v4"
    before = []
    for scene in protocol["scenes"]:
        for arm in ("h2oplus", "h2o_matched", "mcwm"):
            for layer in protocol["executors"]:
                differences, values = [], []
                for seed in protocol["training_seeds"]:
                    folder = previous / "runs/headway" / scene / f"{arm}_seed{seed}"
                    rows = [load(folder / f"{layer}/case{c:02d}.json") for c in protocol["cases"]]
                    values.append(average([metrics(r) for r in rows]))
                    new_arm = "mc_state" if arm == "mcwm" else arm
                    new = here / "runs" / scene / f"{new_arm}_seed{seed}"
                    after = [load(new / f"{layer}/case{c:02d}.json") for c in protocol["cases"]]
                    differences.append(dict(seed=seed, **paired(after, rows)))
                before.append(dict(scene=scene, arm=arm, layer=layer, old_means=average(values),
                    new_minus_old=average([{k:v for k,v in row.items() if k != "seed"}
                                         for row in differences]), seed_differences=differences))
    return dict(rules=rule_metrics, arms=output, comparisons=comparisons, training=train,
                previous_v4=before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(analysis_location="server", passenger=passenger(args.experiments),
                  headway=headway(args.experiments))
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output), bytes=args.output.stat().st_size)))
