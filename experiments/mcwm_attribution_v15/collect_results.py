"""Summarize resident paired policies without retrieving raw data or weights."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def collect():
    ready, results = [], {}
    progress = []
    for scene in P["scenes"]:
        for seed in P["training_seeds"]:
            for arm in P["arms"]+P["reference_arms"]:
                root = HERE if arm in P["arms"] else HERE.parent / "mcwm_seed_confirmation_v13"
                folder = root / "runs" / scene / f"{arm}_seed{seed}" / "results"
                label = f"runs/{scene}/{arm}_seed{seed}"
                path = folder / "ready.json"
                if path.exists() and arm in P["arms"]:
                    r = json.loads(path.read_text())
                    assert (r["arm"],r["seed"],r["scenario"],r["continuation_updates"],r["target_online_training_events"]) == (arm,seed,scene,20000,0)
                    assert (folder.parent / "policy.pt").exists()
                    ready.append(label)
                elif arm in P["arms"] and (folder / "progress.json").exists():
                    r = json.loads((folder / "progress.json").read_text())
                    progress.append(dict(label=label, updates=r["continuation_updates"], elapsed_seconds=r["elapsed_seconds"]))
                rows = []
                for case in P["cases"]:
                    path = folder / f"eval_{case:02d}.json"
                    if path.exists():
                        r = json.loads(path.read_text())
                        assert (r["arm"],r["training_seed"],r["scenario"],r["case"],r["horizon_seconds"]) == (arm,seed,scene,case,24000)
                        rows.append(r)
                results[scene,seed,arm] = rows
    paired = []
    for scene in P["scenes"]:
        for arm in P["arms"]+["fixed0250"]:
            for reference in ("scaled_zero", "target_replay", "h2o_matched"):
                if arm == reference:
                    continue
                deltas = []
                for seed in P["training_seeds"]:
                    a,b = results[scene,seed,arm],results[scene,seed,reference]
                    if len(a)==len(b)==10:
                        assert [r["passengers"]["departed_passengers"] for r in a] == [r["passengers"]["departed_passengers"] for r in b]
                        deltas.append(dict(seed=seed,delta_seconds=mean(x["generalized_seconds_per_departed_passenger"]-y["generalized_seconds_per_departed_passenger"] for x,y in zip(a,b))))
                if deltas:
                    paired.append(dict(scene=scene,arm=arm,reference=reference,seeds=len(deltas),
                        mean_delta_seconds=mean(r["delta_seconds"] for r in deltas),better_seeds=sum(r["delta_seconds"]<0 for r in deltas),seed_deltas=deltas))
    evaluations = sum(len(rows) for (scene,seed,arm),rows in results.items() if arm in P["arms"])
    report = dict(experiment=P["name"],analysis_location="server",ready=ready,progress=progress,
        training_ready=len(ready),training_expected=40,evaluation_ready=evaluations,evaluation_expected=400,
        complete=len(ready)==40 and evaluations==400,paired=paired,interpretation=P["interpretation"])
    (HERE / "analysis_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("training_ready","evaluation_ready","complete")}))


if __name__ == "__main__":
    collect()
