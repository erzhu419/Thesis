"""Summarize the frozen learning comparisons without pooling evaluation days as seeds."""

from pathlib import Path
import json
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
RUNS = HERE / "runs"


def read(path):
    return json.loads(path.read_text())


def stats(values):
    x = np.asarray(values, dtype=float)
    assert len(x) > 1 and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(t.ppf(.975, len(x)-1) * sd / np.sqrt(len(x)))
    return dict(n=len(x), mean=mean, sd=sd, ci95=[mean-half, mean+half], values=x.tolist())


def mcwm():
    cells, diagnostics, training = [], [], []
    for scenario in P["mcwm"]["scenarios"]:
        for arm in P["mcwm"]["arms"]:
            for seed in P["mcwm"]["seeds"]:
                folder = RUNS / "mcwm" / scenario / f"{arm}_seed{seed}" / "results"
                run = read(folder / "ready.json")
                assert (run["arm"], run["seed"], run["scenario"]) == (arm, seed, scenario)
                assert run["continuation_updates"] == 20000 and run["total_policy_updates"] == 70000
                assert 25000 <= run["source_events"] <= 25002 and run["target_online_training_events"] == 0
                rows = [read(folder / f"eval_{case:02d}.json") for case in range(10)]
                for case, row in enumerate(rows):
                    block = P["mcwm"]["evaluation_blocks"][case % 5]
                    assert (row["arm"], row["scenario"], row["training_seed"], row["case"]) == (arm, scenario, seed, case)
                    assert row["demand_seed"] == block["demand_seed"] and row["sumo_seed"] == block["sumo_seed"]
                    assert row["horizon_seconds"] == 24000 and row["deployment_hold_scale"] == 1
                    assert row["demand_scale"] == (1. if case < 5 else 1.25)
                for demand in (1., 1.25):
                    days = [r for r in rows if r["demand_scale"] == demand]
                    fields = dict(time=lambda r:r["generalized_seconds_per_departed_passenger"],
                                  wait=lambda r:r["passengers"]["accrued_wait_seconds_per_departed_passenger"],
                                  ride=lambda r:r["passengers"]["accrued_in_vehicle_seconds_per_departed_passenger"],
                                  completion=lambda r:100*r["passengers"]["completion_rate"],
                                  hold=lambda r:r["mean_hold_seconds"])
                    cells.append(dict(scenario=scenario, arm=arm, seed=seed, demand=demand,
                                      **{k:float(np.mean([f(r) for r in days])) for k,f in fields.items()}))
                phases = {k:v["wall_seconds"] for k,v in run["phases"].items()}
                # Model fitting is nested in refresh; batch/trust diagnostics are nested in policy_update.
                extra = sum(phases.get(k, 0) for k in ("model_training_batch", "bellman_trust_update", "mechanism_diagnostics"))
                exclusive = dict(source=phases["source_collection"], refresh=phases.get("model_refresh", 0),
                                 model_batch=phases.get("model_training_batch", 0),
                                 trust=phases.get("bellman_trust_update", 0),
                                 diagnostic=phases.get("mechanism_diagnostics", 0),
                                 policy=phases["policy_update"]-extra)
                exclusive["other"] = run["training_wall_seconds"]-sum(exclusive.values())
                assert min(exclusive.values()) >= 0
                training.append(dict(scenario=scenario, arm=arm, seed=seed,
                    minutes=run["training_wall_seconds"]/60, gpu_mb=run["peak_gpu_allocated_mb"],
                    cpu_actor_ms=float(np.mean([r["policy_inference_ms"]["mean"] for r in rows])),
                    exclusive_seconds=exclusive, model_history=run["model_history"]))
                traces = [read(p) for p in sorted((folder / "diagnostics").glob("update*.json"))]
                assert len(traces) == 20 and [r["total_updates"] for r in traces] == list(range(51000, 70001, 1000))
                if arm == "mcwm":
                    for row in traces:
                        model = row["model"]["calibrate"]
                        diagnostics.append(dict(scenario=scenario, seed=seed, update=row["total_updates"],
                            error=model["unweighted_prediction_loss"]["mean"],
                            confidence=model["local_confidence"]["mean"], tau=row["model"]["tau"],
                            weight=model["weights"]["mean"], ess_fraction=model["effective_sample_size"]/model["weights"]["n"],
                            gradient_ratio=row["scaled_model_gradient_norm"]/row["scaled_real_gradient_norm"],
                            cosine=row["gradient_cosine"]))
    groups, contrasts = [], []
    for scenario in P["mcwm"]["scenarios"]:
        for demand in (1., 1.25):
            methods = {}
            for arm in P["mcwm"]["arms"]:
                selected = [r for r in cells if (r["scenario"],r["demand"],r["arm"]) == (scenario,demand,arm)]
                methods[arm] = selected
                groups.append(dict(scenario=scenario, demand=demand, arm=arm,
                    **{key:stats([r[key] for r in selected]) for key in ("time","wait","ride","completion","hold")}))
            for arm, reference in (("mcwm","h2o_matched"),("scaled_nomodel","h2o_matched"),("mcwm","scaled_nomodel")):
                a, b = methods[arm], methods[reference]
                assert [r["seed"] for r in a] == [r["seed"] for r in b]
                contrasts.append(dict(scenario=scenario, demand=demand, arm=arm, reference=reference,
                    **{key:stats([x[key]-y[key] for x,y in zip(a,b)]) for key in ("time","wait","ride","completion")},
                    time_percent=100*(np.mean([r["time"] for r in a])/np.mean([r["time"] for r in b])-1)))
    return dict(cells=cells, groups=groups, contrasts=contrasts, training=training, diagnostics=diagnostics)


def ensemble():
    cells, training, groups, contrasts = [], [], [], []
    for size in P["ensemble"]["sizes"]:
        for seed in P["ensemble"]["seeds"]:
            folder = RUNS / "ensemble" / f"k{size}_seed{seed}" / "results"
            ready = read(folder / "ready.json")
            assert (ready["episodes"],ready["seed"],ready["ensemble_size"]) == (100,seed,size)
            training.append(dict(size=size,seed=seed,minutes=ready["training_wall_seconds"]/60,
                                 gpu_mb=ready["peak_gpu_allocated_mb"]))
            rows = [read(folder / f"eval{case:02d}.json") for case in range(6)]
            for case,row in enumerate(rows):
                assert (row["ensemble_size"],row["training_seed"],row["scene"],row["evaluation_seed"],row["horizon_seconds"]) == (
                    size, seed, P["ensemble"]["scenes"][case//3], P["ensemble"]["evaluation_seeds"][case%3], 24000)
            for scene in P["ensemble"]["scenes"]:
                days = [r for r in rows if r["scene"] == scene]
                getters = dict(time=lambda r:r["generalized_seconds_per_passenger"],
                    reward=lambda r:r["reward"], front=lambda r:r["operational"]["front_plan_absolute_error_seconds"]["mean"],
                    back=lambda r:r["operational"]["back_plan_absolute_error_seconds"]["mean"],
                    completion=lambda r:100*r["arrived_passengers"]/r["generated_passengers"],
                    cpu_actor_ms=lambda r:r["operational"]["policy_inference_ms"]["mean"])
                cells.append(dict(size=size, seed=seed, scene=scene,
                    **{k:float(np.mean([f(r) for r in days])) for k,f in getters.items()}))
    for scene in P["ensemble"]["scenes"]:
        for size in P["ensemble"]["sizes"]:
            a = [r for r in cells if (r["scene"],r["size"]) == (scene,size)]
            groups.append(dict(scene=scene,size=size,**{k:stats([r[k] for r in a]) for k in getters}))
            if size != 2:
                b = [r for r in cells if (r["scene"],r["size"]) == (scene,2)]
                contrasts.append(dict(scene=scene,size=size,reference=2,
                    **{k:stats([x[k]-y[k] for x,y in zip(a,b)]) for k in getters}))
    return dict(cells=cells,training=training,groups=groups,contrasts=contrasts)


def ratio():
    cells = []
    for seed in P["ratio"]["enhancement_seeds"]:
        run = read(RUNS / "ratio" / f"enhancement_seed{seed}" / "results/train_summary.json")
        assert (run["policy_updates"], run["total_critic_updates"]) == (16144,17680)
        rows = []
        for case in range(3):
            folder = RUNS / "ratio_heldout" / f"seed{seed}" / f"case{case}" / "results"
            assert read(folder / "ready.json")["validation_available"]
            row = read(folder / "ratio_final.json")
            assert row["validation_available"] and row["new_gradient_updates"] == 0
            assert row["source"]["seed"] == P["ratio"]["validation_seeds"][case]
            assert row["source_holdout"] == "fresh frozen-policy episode after training"
            assert row["offline_holdout_blocks"] == [1202]
            rows.append(row)
        names = ("joint_auc","state_action_auc","joint_brier","state_action_brier", "weight_mean",
                 "ess_fraction","raw_ess_fraction","high_clip_fraction","low_clip_fraction")
        cells.append(dict(seed=seed, **{k:float(np.mean([r["metrics"]["ratio/validation_"+k] for r in rows])) for k in names}))
    return dict(cells=cells, summary={k:stats([r[k] for r in cells]) for k in names})


if __name__ == "__main__":
    result = dict(mcwm=mcwm(), ensemble=ensemble(), ratio=ratio(),
                  counts=dict(training=52, mcwm_evaluations=270, ensemble_evaluations=120, ratio_holdouts=15),
                  statistics="Evaluation days averaged within each training seed; two-sided descriptive t intervals, no multiplicity adjustment.")
    (HERE / "learning_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["counts"]))
    for row in result["mcwm"]["contrasts"]:
        print("MC",row["scenario"],row["demand"],row["arm"],row["reference"],row["time"],"completion",row["completion"]["mean"])
    for row in result["ensemble"]["groups"]:
        print("ENSEMBLE",row["scene"],row["size"],{k:round(row[k]["mean"],4) for k in ("time","front","back","reward","cpu_actor_ms")})
    print("RATIO",{k:round(v["mean"],6) for k,v in result["ratio"]["summary"].items()})
