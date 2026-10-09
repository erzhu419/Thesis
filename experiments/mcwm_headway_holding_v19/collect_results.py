"""Server-side matched headway comparisons for the fixed 0.125 weight."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def read(path):
    return json.loads(path.read_text())


def metrics(rows):
    return dict(cost=mean(r["generalized_seconds_per_departed_passenger"] for r in rows),
        reward=mean(r["headway_historical_mean"] for r in rows),
        plan=mean(r["headway_plan_mae_seconds"] for r in rows),
        imbalance=mean(r["headway_imbalance_seconds"] for r in rows))


def collect():
    ready, progress, aggregate, paired = [], [], [], []
    training = evaluations = smoke_training = smoke_evaluations = 0
    for smoke in (True, False):
        for scene, seed in product(P["scenes"][:1] if smoke else P["scenes"],
                                   P["training_seeds"][:1] if smoke else P["training_seeds"]):
            label = f"{'smoke' if smoke else 'runs'}/headway/{scene}/mcwm_seed{seed}"
            root = HERE / label
            if (root / "ready.json").exists():
                r = read(root / "ready.json")
                assert r["updates"] == (256 if smoke else 20000)
                assert r["model_td_coefficient"] == .125
                ready.append(label)
                if smoke:
                    smoke_training += 1
                else:
                    training += 1
            elif (root / "progress.json").exists():
                progress.append(dict(label=label, **read(root / "progress.json")))
            for layer in P["executors"]:
                cases = [0] if smoke else P["cases"]
                paths = [root / layer / f"case{case:02d}.json" for case in cases]
                rows = [read(path) for path in paths if path.exists()]
                for r in rows:
                    assert r["mean_speed_ratio"] == 1. and r["control_mode"] == "holding"
                if smoke:
                    smoke_evaluations += len(rows)
                    continue
                evaluations += len(rows)
                if len(rows) != 10:
                    continue
                a = metrics(rows)
                aggregate.append(dict(scene=scene, seed=seed, layer=layer, **a))
                refs = {
                    "dual_mc0125": HERE.parent / "mcwm_headway_td_v17/runs" / scene / f"mc_td0125_seed{seed}",
                    "holding_mc025": HERE.parent / "mcwm_action_matrix_v18/runs/headway" / scene / f"mcwm_seed{seed}",
                    "holding_h2oplus": HERE.parent / "mcwm_action_matrix_v18/runs/headway" / scene / f"h2oplus_seed{seed}",
                    "daganzo": HERE.parent / "mcwm_headway_alignment_v16/rules/daganzo_duet"}
                for name, parent in refs.items():
                    rs = [read(parent / layer / f"case{case:02d}.json") for case in cases]
                    assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in rs]
                    b = metrics(rs)
                    paired.append(dict(scene=scene, seed=seed, layer=layer, reference=name,
                        **{k+"_delta":a[k]-b[k] for k in a}))
    report = dict(experiment=P["name"], ready=ready, progress=progress, analysis_location="server",
        training_ready=training, training_expected=4, evaluation_ready=evaluations, evaluation_expected=80,
        smoke_complete=smoke_training == 1 and smoke_evaluations == 2,
        complete=training == 4 and evaluations == 80, aggregates=aggregate, paired=paired)
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))


if __name__ == "__main__":
    collect()
