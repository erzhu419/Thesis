"""Paired analysis of the two independent bus campaigns on shared storage."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def average(rows, fields):
    return {field: mean(row[field] for row in rows) for field in fields}


def compare(treated, control, fields, cost, higher=()):
    changes = {field: treated[field]-control[field] for field in fields}
    return dict(differences=changes,
                cost_change_pct=100*(treated[cost]/control[cost]-1),
                metric_change_pct={field: 100*changes[field]/abs(control[field])
                                   for field in fields if control[field] != 0},
                improved={field: changes[field] > 0 if field in higher else changes[field] < 0
                          for field in fields})


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def transit():
    folder = HERE / "bus_control_v2"
    protocol = read(folder / "protocol.json")
    cost = "generalized_seconds_per_departed_passenger"
    fields = (cost, "headway_ridge_mean", "headway_plan_mae_seconds",
              "headway_imbalance_seconds", "completion_rate", "mean_hold_seconds", "mean_speed_ratio")
    rows = []
    for path in list(folder.glob("runs/*/*/*/*/evaluation/case*.json"))+list(folder.glob("rules/*/evaluation/case*.json")):
        row = read(path)
        row["completion_rate"] = row["passengers"]["completion_rate"]
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["objective"], row["source_scene"], row["arm"], row["training_seed"])].append(row)
    assert len(rows) == 560 and len(grouped) == 56 and all(len(v) == 10 for v in grouped.values())
    assert all(row["horizon_seconds"] == protocol["horizon_seconds"] for row in rows)
    index = {key: average(values, fields) for key, values in grouped.items()}
    case_index = {key: {r["case"]: r for r in values} for key, values in grouped.items()}
    groups, contrasts = [], []
    higher = ("headway_ridge_mean", "completion_rate")
    for mode, objective in protocol["cells"]:
        for scene in (s["name"] for s in protocol["scenarios"]):
            for arm in protocol["arms"]:
                keys = [(mode, objective, scene, arm, seed) for seed in protocol["training_seeds"]]
                groups.append(dict(mode=mode, objective=objective, scene=scene, arm=arm,
                                   **average([index[key] for key in keys], fields)))
                controls = [((mode, objective, scene, "h2oplus", seed), "h2oplus")
                            for seed in protocol["training_seeds"]] if arm == "mcwm" else []
                comparisons = defaultdict(list)
                if mode == "dual" and objective == "headway":
                    controls += [(("holding", "headway", scene, arm, seed), "holding_same_algorithm")
                                 for seed in protocol["training_seeds"]]
                for rule in ("zero", "daganzo_duet"):
                    controls += [(("holding", "headway", None, rule, None), rule)]*len(keys)
                for control_key, label in controls:
                    seed = control_key[-1]
                    if seed is None:
                        seed = protocol["training_seeds"][len(comparisons[label])]
                    key = (mode, objective, scene, arm, seed)
                    pair = compare(index[key], index[control_key], fields, cost, higher)
                    pair["seed"] = seed
                    pair["case_wins"] = {field: sum((case_index[key][c][field] > r[field]) if field in higher
                                                    else (case_index[key][c][field] < r[field])
                                                    for c, r in case_index[control_key].items()) for field in fields[:5]}
                    comparisons[label].append(pair)
                for label, pairs in comparisons.items():
                    contrasts.append(dict(mode=mode, objective=objective, scene=scene, arm=arm, control=label,
                        cost_change_pct=mean(p["cost_change_pct"] for p in pairs),
                        differences={field: mean(p["differences"][field] for p in pairs) for field in fields},
                        metric_change_pct={field: mean(p["metric_change_pct"][field] for p in pairs)
                                           for field in pairs[0]["metric_change_pct"]},
                        seed_wins={field: sum(p["improved"][field] for p in pairs) for field in fields[:5]},
                        case_wins={field: sum(p["case_wins"][field] for p in pairs) for field in fields[:5]},
                        pairs=pairs))
    write(folder / "result_analysis.json", dict(complete=True, evaluations=len(rows),
        groups=groups, contrasts=contrasts, rules=[dict(arm=k[3], **v) for k,v in index.items() if k[-1] is None],
        comparison="Within a training seed, average the same ten test cases; paired percentage changes are then averaged across all three seeds."))


def symbolic():
    folder = HERE / "csbapr_symbolic_bus_v2"
    cost = "generalized_seconds_per_passenger"
    fields = (cost, "native_reward_per_decision", "headway_plan_mae_seconds",
              "headway_imbalance_seconds", "completion_rate", "mean_hold_seconds")
    rows = []
    for path in folder.glob("runs/*/*/results/eval_*.json"):
        row = read(path)
        row["arm"] = path.parents[1].name.rsplit("_seed", 1)[0]
        row["training_seed"] = int(path.parents[1].name.rsplit("_seed", 1)[1])
        row["native_reward_per_decision"] = row.get("native_reward", row.get("reward"))/row["decisions"]
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["scene"], row["arm"], row["training_seed"])].append(row)
    assert len(rows) == 234 and len(grouped) == 78 and all(len(v) == 3 for v in grouped.values())
    assert all(row["horizon_seconds"] == 18000 for row in rows)
    index = {key: average(values, fields) for key, values in grouped.items()}
    families = defaultdict(list)
    for key, value in index.items():
        families[key[:3]].append(value)
    groups = [dict(zip(("objective", "scene", "arm"), key), **average(values, fields))
              for key, values in families.items()]
    contrasts = []
    for key, value in index.items():
        objective, scene, arm, seed = key
        if arm != "centered_symbolic":
            continue
        for control in ("uncentered_symbolic", "centered_no_symbolic", "legacy_nau", "legacy_mlp", "fitted_reference", "daganzo", "zero"):
            control_key = (("headway", scene, control, 64001) if control in ("daganzo", "zero")
                           else (objective, scene, control, seed))
            contrasts.append(dict(objective=objective, scene=scene, seed=seed, control=control,
                **compare(value, index[control_key], fields, cost, ("native_reward_per_decision", "completion_rate"))))
    diagnostics, learners = [], []
    for path in folder.glob("models/seed*/ready.json"):
        model = read(path)
        ratios = [{"line": r["line"],
                   "dynamics_mse_ratio": r["increment_mse"]/r["persistence_increment_mse"],
                   "headway_cost_mse_ratio": r["cost_mse"][0]/r["constant_cost_mse"][0],
                   "passenger_cost_mse_ratio": r["cost_mse"][1]/r["constant_cost_mse"][1]} for r in model["diagnostics"]]
        diagnostics.append(dict(seed=model["seed"], rows=model["identification_rows"],
            worse_lines={field: sum(r[field] > 1 for r in ratios) for field in ratios[0] if field != "line"},
            ratios=ratios))
    for path in folder.glob("runs/*/*/results/progress.json"):
        row = read(path)
        learners.append(dict(objective=path.parents[2].name, arm=path.parents[1].name,
            episode=row["episode"], metrics=row.get("metrics")))
    write(folder / "result_analysis.json", dict(complete=True, evaluations=len(rows), groups=groups,
        contrasts=contrasts, model_diagnostics=diagnostics, final_updates=learners,
        reward_note="Per-decision native reward is comparable within an objective, not between objectives; passenger intervals overlap."))


if __name__ == "__main__":
    transit()
    symbolic()
    print("Paired analyses complete: 560 SUMO cases and 234 independent CS bus cases.")
