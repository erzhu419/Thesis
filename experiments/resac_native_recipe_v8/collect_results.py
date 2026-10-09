"""Only compact endpoint and matched-case summaries leave server storage."""

from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
OLD = HERE.parent / "bus_algorithm_matrix_v6"


def collect():
    ready, progress, rows, contrasts, cs_comparisons = [], [], [], [], []
    training = evaluations = 0
    for mode, objective, arm, seed in product(P["modes"], P["objectives"], P["arms"], P["training_seeds"]):
        label = f"runs/{mode}/{objective}/{arm}_seed{seed}"
        root = HERE / label
        if (root / "ready.json").exists():
            r = json.loads((root / "ready.json").read_text())
            assert r["episodes"] == P["episodes"]
            training += 1
            ready.append(label)
        elif (root / "progress.json").exists():
            progress.append(dict(label=label, **json.loads((root / "progress.json").read_text())))
        for budget in P["checkpoints"]:
            endpoint = root / f"budget{budget}"
            if (endpoint / "ready.json").exists():
                assert (endpoint / "policy.pt").exists()
                ready.append(label+f"/budget{budget}")
            for case in range(18):
                path = endpoint / f"eval_{case:02d}.json"
                if not path.exists():
                    continue
                r = json.loads(path.read_text())
                assert (r["budget"], r["case"], r["horizon_seconds"]) == (budget, case, 18000)
                assert r["item"] == dict(mode=mode, objective=objective, arm=arm, seed=seed)
                evaluations += 1
                rows.append(dict(mode=mode, objective=objective, arm=arm, seed=seed, case=case,
                    budget=budget, scene=r["scene"], people=r["generated_passengers"], cost=r["cost"],
                    reward=r["native_reward"]/r["decisions"], plan=r["headway_plan_mae_seconds"],
                    imbalance=r["headway_imbalance_seconds"]))
    index = {(r["mode"], r["objective"], r["arm"], r["seed"], r["budget"], r["case"]):r for r in rows}
    for r in rows:
        refs = [(r["mode"], "sac"), (r["mode"], "resac")]
        if r["mode"] == "dual":
            refs.append(("holding", r["arm"]))
        for mode, arm in refs:
            if (mode, arm) == (r["mode"], r["arm"]):
                continue
            b = index.get((mode, r["objective"], arm, r["seed"], r["budget"], r["case"]))
            if b is not None:
                contrasts.append(dict(**r, reference_mode=mode, reference_arm=arm,
                    people_difference=r["people"]-b["people"],
                    **{k+"_delta":r[k]-b[k] for k in ("cost", "reward", "plan", "imbalance")}))
        if r["arm"] == "resac" and r["budget"] == 100:
            for arm in ("stabilized_mlp", "constant_symbolic", "constant_control"):
                if arm != "stabilized_mlp" and r["mode"] == "holding":
                    parent = OLD / "runs/holding" / r["objective"] / f"{arm}_seed{r['seed']}"
                else:
                    family = "cs_recipe" if arm == "stabilized_mlp" else "cs_context"
                    parent = HERE.parent / "bus_alignment_followup_v5/formal" / family / r["mode"] / r["objective"] / arm / str(r["seed"])
                path = parent / f"eval_{r['case']:02d}.json"
                if not path.exists():
                    continue
                b = json.loads(path.read_text())
                assert b["horizon_seconds"] == 18000 and b["case"] == r["case"]
                metrics = dict(cost=b["cost"], reward=b["native_reward"]/b["decisions"],
                    plan=b["headway_plan_mae_seconds"], imbalance=b["headway_imbalance_seconds"])
                cs_comparisons.append(dict(mode=r["mode"], objective=r["objective"], seed=r["seed"],
                    case=r["case"], scene=r["scene"], arm=arm, reference="resac", budget=100,
                    people_difference=b["generated_passengers"]-r["people"],
                    **{k+"_delta":v-r[k] for k,v in metrics.items()}))
    groups = defaultdict(list)
    for r in rows:
        groups[r["mode"], r["objective"], r["arm"], r["budget"], r["scene"]].append(r)
    aggregates = [dict(mode=k[0], objective=k[1], arm=k[2], budget=k[3], scene=k[4],
        **{m:mean(r[m] for r in rs) for m in ("cost", "reward", "plan", "imbalance")})
        for k,rs in groups.items() if len(rs) == 6]
    expected = 32*18*len(P["checkpoints"])
    report = dict(experiment=P["name"], ready=ready, progress=progress, analysis_location="server",
        training_ready=training, training_expected=32, evaluation_ready=evaluations, evaluation_expected=expected,
        complete=training == 32 and evaluations == expected, aggregates=aggregates,
        contrasts=contrasts, cs_comparisons=cs_comparisons, interpretation=P["interpretation"])
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("training_ready", "evaluation_ready", "complete")}))


if __name__ == "__main__":
    collect()
