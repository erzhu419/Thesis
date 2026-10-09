"""Resident paired summaries; only this small JSON returns to local storage."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
PREVIOUS = HERE.parent / "mcwm_headway_alignment_v16"


def read(path):
    return json.loads(path.read_text())


def collect():
    ready, progress, aggregates, paired = [], [], [], []
    training = evaluations = smoke_training = smoke_evaluations = 0
    for smoke in (True, False):
        prefix = "smoke" if smoke else "runs"
        for scene, arm, seed in product(P["scenes"][:1] if smoke else P["scenes"], P["arms"],
                                        P["training_seeds"][:1] if smoke else P["training_seeds"]):
            label = f"{prefix}/{scene}/{arm}_seed{seed}"
            root = HERE / label
            if (root / "ready.json").exists():
                r = read(root / "ready.json")
                assert r["updates"] == (256 if smoke else 20000)
                assert r["model_td_coefficient"] == P["coefficients"][arm]
                assert (root / "policy.pt").exists()
                ready.append(label)
                if smoke:
                    smoke_training += 1
                else:
                    training += 1
            elif (root / "progress.json").exists():
                r = read(root / "progress.json")
                progress.append(dict(label=label, updates=r["updates"], wall_seconds=r["wall_seconds"]))
            for layer in P["executors"]:
                cases = [0] if smoke else P["cases"]
                rows = []
                for case in cases:
                    path = root / layer / f"case{case:02d}.json"
                    if path.exists():
                        r = read(path)
                        assert (r["arm"], r["training_seed"], r["case"], r["source_scene"]) == (arm, seed, case, scene)
                        rows.append(r)
                if smoke:
                    smoke_evaluations += len(rows)
                    continue
                evaluations += len(rows)
                if len(rows) != len(cases):
                    continue
                def metrics(rs):
                    return dict(cost=mean(r["generalized_seconds_per_departed_passenger"] for r in rs),
                                reward=mean(r["headway_historical_mean"] for r in rs),
                                ridge=mean(r["headway_ridge_mean"] for r in rs))
                a = metrics(rows)
                aggregates.append(dict(scene=scene, arm=arm, seed=seed, layer=layer, **a))
                for reference in ("mc_state", "h2oplus", "h2o_matched", "daganzo_duet"):
                    parent = PREVIOUS / "rules" / reference if reference == "daganzo_duet" else PREVIOUS / "runs" / scene / f"{reference}_seed{seed}"
                    rs = [read(parent / layer / f"case{case:02d}.json") for case in cases]
                    assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in rs]
                    b = metrics(rs)
                    paired.append(dict(scene=scene, arm=arm, seed=seed, layer=layer, reference=reference,
                                       **{k+"_delta": a[k]-b[k] for k in a}))
    report = dict(experiment=P["name"], ready=ready, progress=progress, analysis_location="server",
                  training_ready=training, training_expected=8, evaluation_ready=evaluations, evaluation_expected=160,
                  smoke_complete=smoke_training == 2 and smoke_evaluations == 4,
                  complete=training == 8 and evaluations == 160, aggregates=aggregates, paired=paired)
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))


if __name__ == "__main__":
    collect()
