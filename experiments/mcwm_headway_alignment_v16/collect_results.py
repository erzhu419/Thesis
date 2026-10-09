"""Small dependency metadata and paired outcomes, computed on shared storage."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def collect():
    ready,progress,results = [],[],{}
    training,evaluations = 0,0
    smoke_evaluations,smoke_training,rule_count,smoke_rules = 0,0,0,0
    for smoke in (True,False):
        prefix = "smoke" if smoke else "runs"
        data = HERE / ("smoke_data" if smoke else "data")
        for arm in ("daganzo_duet","zero"):
            for layer in P["executors"]:
                for case in ([0] if smoke else P["cases"]):
                    path = HERE / ("rule_smoke" if smoke else "rules") / arm / layer / f"case{case:02d}.json"
                    if path.exists():
                        r = json.loads(path.read_text())
                        assert (r["arm"],r["case"],r["execution_layer"],r["horizon_seconds"])==(arm,case,layer,1800 if smoke else 24000)
                        assert "headway_historical_mean" in r
                        if smoke:
                            smoke_rules += 1
                        else:
                            rule_count += 1
        for shard in range(2 if smoke else P["offline_shards"]):
            if (data / f"shard{shard:02d}/ready.json").exists():
                ready.append(f"{prefix}/data/shard{shard}")
        if (data / "ready.json").exists():
            ready.append(f"{prefix}/data/merge")
        for seed in (P["training_seeds"][:1] if smoke else P["training_seeds"]):
            init = HERE / ("smoke_inputs" if smoke else "inputs") / f"seed{seed}"
            if (init / "ready.json").exists():
                r = json.loads((init / "ready.json").read_text())
                assert r["offline_updates"] == (32 if smoke else 50000)
                ready.append(f"{prefix}/init/{seed}")
            for scene in (P["scenes"][:1] if smoke else P["scenes"]):
                for arm in P["arms"]:
                    root = HERE / prefix / scene / f"{arm}_seed{seed}"
                    label = f"{prefix}/{scene}/{arm}_seed{seed}"
                    if (root / "ready.json").exists():
                        r = json.loads((root / "ready.json").read_text())
                        assert r["updates"] == (256 if smoke else 20000) and r["target_online_training_events"]==0
                        assert (root / "policy.pt").exists()
                        ready.append(label)
                        if smoke:
                            smoke_training += 1
                        else:
                            training += 1
                    elif (root / "progress.json").exists():
                        r = json.loads((root / "progress.json").read_text())
                        progress.append(dict(label=label,updates=r["updates"],wall_seconds=r["wall_seconds"]))
                    if not smoke and (init / "progress.json").exists() and not (init / "ready.json").exists():
                        r = json.loads((init / "progress.json").read_text())
                        progress.append(dict(label=f"{prefix}/init/{seed}",offline_updates=r["offline_updates"],wall_seconds=r["wall_seconds"]))
                    for layer in P["executors"]:
                        rows = []
                        for case in ([0] if smoke else P["cases"]):
                            path = root / layer / f"case{case:02d}.json"
                            if path.exists():
                                r = json.loads(path.read_text())
                                assert (r["arm"],r["training_seed"],r["source_scene"],r["case"],r["horizon_seconds"])==(arm,seed,scene,case,1800 if smoke else 24000)
                                rows.append(r)
                        results[smoke,scene,seed,arm,layer] = rows
                        if smoke:
                            smoke_evaluations += len(rows)
                        else:
                            evaluations += len(rows)
    paired = []
    for scene in P["scenes"]:
        for arm in P["arms"]:
            for reference in ("h2oplus","h2o_matched","mc_state","daganzo_duet"):
                if arm==reference:
                    continue
                for layer in P["executors"]:
                    deltas = []
                    for seed in P["training_seeds"]:
                        a = results[False,scene,seed,arm,layer]
                        if reference=="daganzo_duet":
                            paths = [HERE / "rules" / reference / layer / f"case{c:02d}.json" for c in P["cases"]]
                            b = [json.loads(p.read_text()) for p in paths if p.exists()]
                        else:
                            b = results[False,scene,seed,reference,layer]
                        if len(a)==len(b)==10:
                            assert [r["passengers"]["departed_passengers"] for r in a]==[r["passengers"]["departed_passengers"] for r in b]
                            deltas.append(dict(seed=seed,cost_delta_seconds=mean(x["generalized_seconds_per_departed_passenger"]-y["generalized_seconds_per_departed_passenger"] for x,y in zip(a,b)),
                                headway_ridge_delta=mean(x["headway_ridge_mean"]-y["headway_ridge_mean"] for x,y in zip(a,b)),
                                historical_reward_delta=mean(x["headway_historical_mean"]-y["headway_historical_mean"] for x,y in zip(a,b))))
                    if deltas:
                        paired.append(dict(scene=scene,arm=arm,reference=reference,layer=layer,seeds=len(deltas),seed_deltas=deltas,
                            mean_headway_ridge_delta=mean(r["headway_ridge_delta"] for r in deltas),
                            mean_cost_delta_seconds=mean(r["cost_delta_seconds"] for r in deltas)))
    report = dict(experiment=P["name"],analysis_location="server",ready=ready,progress=progress,training_ready=training,
        training_expected=16,evaluation_ready=evaluations,evaluation_expected=320,
        smoke_training_ready=smoke_training,smoke_evaluation_ready=smoke_evaluations,rule_ready=rule_count,rule_expected=40,
        smoke_complete=smoke_training==4 and smoke_evaluations==8 and smoke_rules==4,
        complete=training==16 and evaluations==320 and rule_count==40,paired=paired,interpretation=P["interpretation"])
    (HERE / "analysis_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("training_ready","evaluation_ready","smoke_complete","complete")}))


if __name__=="__main__":
    collect()
