"""Summarize physical action restrictions versus resident dual endpoints."""

from itertools import product
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def read(path):
    return json.loads(path.read_text())


def reference(objective, scene, arm, seed):
    if objective == "headway":
        return HERE.parent / "mcwm_headway_alignment_v16/runs" / scene / f"{'mc_state' if arm == 'mcwm' else arm}_seed{seed}"
    return HERE.parent / "bus_dual_aligned_v4/runs/passenger" / scene / f"{arm}_seed{seed}"


def metrics(rows, objective):
    result = dict(cost=mean(r["generalized_seconds_per_departed_passenger"] for r in rows),
                  plan=mean(r["headway_plan_mae_seconds"] for r in rows),
                  imbalance=mean(r["headway_imbalance_seconds"] for r in rows),
                  ridge=mean(r["headway_ridge_mean"] for r in rows))
    if objective == "headway":
        result["reward"] = mean(r["headway_historical_mean"] for r in rows)
    return result


def collect():
    ready, progress, aggregates, paired = [], [], [], []
    training = evaluations = smoke_training = smoke_evaluations = 0
    for smoke in (True, False):
        for objective, scene, arm, seed in product(P["objectives"], P["scenes"][:1] if smoke else P["scenes"],
                P["arms"], P["training_seeds"][:1] if smoke else P["training_seeds"]):
            label = f"{'smoke' if smoke else 'runs'}/{objective}/{scene}/{arm}_seed{seed}"
            root = HERE / label
            if (root / "ready.json").exists():
                r = read(root / "ready.json")
                assert r["updates"] == (256 if smoke else 20000)
                assert r["policy_degrees_of_freedom"] == 1 and (root / "policy.pt").exists()
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
                        assert (r["arm"], r["training_seed"], r["source_scene"], r["case"]) == (arm, seed, scene, case)
                        assert r["mean_speed_ratio"] == 1.
                        rows.append(r)
                if smoke:
                    smoke_evaluations += len(rows)
                    continue
                evaluations += len(rows)
                if len(rows) != len(cases):
                    continue
                a = metrics(rows, objective)
                aggregates.append(dict(objective=objective, scene=scene, arm=arm, seed=seed, layer=layer, **a))
                references = [("dual", arm), ("dual", "h2oplus")]
                if ("dual", arm) == ("dual", "h2oplus"):
                    references = references[:1]
                for mode, ref in references+[("holding", "daganzo_duet")]:
                    if ref == "daganzo_duet":
                        parent = HERE.parent / ("mcwm_headway_alignment_v16" if objective == "headway" else "bus_dual_aligned_v4") / "rules/daganzo_duet"
                    else:
                        parent = reference(objective, scene, ref, seed)
                    rs = [read(parent / layer / f"case{case:02d}.json") for case in cases]
                    assert [r["passengers"]["departed_passengers"] for r in rows] == [r["passengers"]["departed_passengers"] for r in rs]
                    b = metrics(rs, objective)
                    paired.append(dict(objective=objective, scene=scene, arm=arm, seed=seed, layer=layer,
                        reference_mode=mode, reference_arm=ref, **{k+"_delta": a[k]-b[k] for k in a}))
    report = dict(experiment=P["name"], ready=ready, progress=progress, analysis_location="server",
        training_ready=training, training_expected=16, evaluation_ready=evaluations, evaluation_expected=320,
        smoke_complete=smoke_training == 4 and smoke_evaluations == 8,
        complete=training == 16 and evaluations == 320, aggregates=aggregates, paired=paired,
        interpretation=P["interpretation"])
    (HERE / "analysis_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("training_ready", "evaluation_ready", "smoke_complete", "complete")}))


if __name__ == "__main__":
    collect()
