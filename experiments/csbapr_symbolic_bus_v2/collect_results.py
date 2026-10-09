"""Server-side analysis, compact readiness markers and paired seed contrasts."""

from collections import defaultdict
import json
from statistics import mean

from settings import HERE, P, day_folder, model_folder, run_folder, write


def collect():
    ready, rows, training = [], [], []
    for seed in P["training_seeds"]:
        for day in range(P["identification_days"]):
            if (day_folder(seed, day) / "ready.json").exists():
                ready.append(f"day/{seed}/{day}")
    smoke_count = formal_count = 0
    for smoke in (True, False):
        prefix = "smoke/" if smoke else ""
        seeds = P["training_seeds"][:1] if smoke else P["training_seeds"]
        for seed in seeds:
            if (model_folder(seed, smoke) / "ready.json").exists():
                ready.append(prefix+f"model/{seed}")
            for objective in P["objectives"]:
                for arm in P["arms"]:
                    out = run_folder(objective, arm, seed, smoke)
                    path = out / "results/ready.json"
                    if path.exists():
                        ready.append(prefix+f"policy/{objective}/{arm}/{seed}")
                        training.append({**json.loads(path.read_text()), "smoke":smoke,
                                         "objective":objective, "arm":arm, "seed":seed})
                for arm in P["arms"]+["fitted_reference", "daganzo", "zero"]:
                    # Exact rule baselines need only one objective/seed copy.
                    if arm in ("daganzo", "zero") and (objective != "headway" or seed != seeds[0]):
                        continue
                    out = run_folder(objective, arm, seed, smoke)
                    for case in ([0] if smoke else range(9)):
                        path = out / "results" / f"eval_{case:02d}.json"
                        if path.exists():
                            row = json.loads(path.read_text())
                            if smoke:
                                smoke_count += 1
                            else:
                                formal_count += 1
                                rows.append(dict(row, arm=arm, training_seed=seed))
    # 10 learners, 2 fitted controls, 2 exact rule controls.
    if smoke_count == 14:
        ready.append("smoke-pass")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["arm"], row["training_seed"], row["scene"])].append(row)
    seed_means = []
    fields = ("generalized_seconds_per_passenger", "headway_plan_mae_seconds", "headway_imbalance_seconds", "completion_rate", "mean_hold_seconds")
    for key, values in grouped.items():
        seed_means.append(dict(zip(("objective", "arm", "seed", "scene"), key), cases=len(values),
                               **{k:mean(r[k] for r in values) for k in fields}))
    index = {(r["objective"], r["arm"], r["seed"], r["scene"]):r for r in seed_means}
    contrasts = []
    for row in seed_means:
        if row["arm"] != "centered_symbolic" or row["cases"] != 3:
            continue
        for arm in P["arms"]+["fitted_reference", "daganzo", "zero"]:
            control = index.get((row["objective"], arm, row["seed"], row["scene"]))
            if arm in ("daganzo", "zero"):
                control = index.get(("headway", arm, P["training_seeds"][0], row["scene"]))
            if control is not None and control["cases"] == 3 and arm != row["arm"]:
                contrasts.append(dict(objective=row["objective"], seed=row["seed"], scene=row["scene"],
                    control=arm, **{k:row[k]-control[k] for k in fields},
                    passenger_cost_change_pct=100*(row[fields[0]]/control[fields[0]]-1)))
    result = dict(experiment=P["name"], analysis_location="server", ready=ready,
        smoke_evaluations=smoke_count, formal_evaluations=formal_count,
        expected_formal_evaluations=234, training=training, seed_means=seed_means,
        paired_contrasts=contrasts, complete=formal_count == 234)
    write(HERE / "summary.json", result)
    print(json.dumps({k:result[k] for k in ("smoke_evaluations", "formal_evaluations", "complete")}))


if __name__ == "__main__":
    collect()
