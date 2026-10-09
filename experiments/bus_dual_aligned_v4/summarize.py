"""Analyze resident scalar endpoints; return one small gate/result summary."""

from itertools import product
import json
import math
from pathlib import Path

from aligned_common import HERE,P,policy_folder,data_folder,write
EXPERIMENTS = HERE.parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def mean(values):
    return sum(values)/len(values) if values else None


def metrics(rows):
    return dict(cases=len(rows),cost=mean([r["generalized_seconds_per_departed_passenger"] for r in rows]),
        ridge=mean([r["headway_ridge_mean"] for r in rows]),plan_mae=mean([r["headway_plan_mae_seconds"] for r in rows]),
        holding=mean([r["mean_hold_seconds"] for r in rows]),speed=mean([r["mean_speed_ratio"] for r in rows]))


def summarize():
    report = dict(ready=[],completed_labels=[],smoke_complete=False,formal_initializers=0,formal_training=0,formal_evaluations=0,
                  expected_initializers=4,expected_training=24,expected_evaluations=520,complete=False,analysis_location="server")
    for smoke in (True,False):
        prefix = "smoke/" if smoke else "formal/"
        count = 2 if smoke else P["offline_shards"]
        for shard in range(count):
            if read(data_folder(smoke) / f"shard{shard:02d}/ready.json") is not None:
                report["ready"].append(prefix+f"data/shard{shard}")
        if read(data_folder(smoke) / "ready.json") is not None:
            report["ready"].append(prefix+"data/merge")
        seeds,scenes,cases = (P["training_seeds"][:1],P["scenes"][:1],[0]) if smoke else (P["training_seeds"],P["scenes"],range(10))
        smoke_endpoints = 0
        for objective,seed in product(P["objectives"],seeds):
            init = HERE / ("smoke_inputs" if smoke else "inputs") / objective / f"seed{seed}/ready.json"
            if read(init) is not None:
                report["ready"].append(prefix+f"init/{objective}/{seed}")
                if not smoke:
                    report["formal_initializers"] += 1
            for scene,arm in product(scenes,P["arms"]):
                out = policy_folder(objective,scene,arm,seed,smoke)
                if read(out / "ready.json") is not None:
                    report["ready"].append(prefix+f"policy/{objective}/{scene}/{arm}/{seed}")
                    if not smoke:
                        report["formal_training"] += 1
                for layer,case in product(P["executors"],cases):
                    row = read(out / layer / f"case{case:02d}.json")
                    if row is not None:
                        if row["horizon_seconds"] != (1800 if smoke else 24000):
                            raise ValueError("Smoke/formal endpoint collision")
                        if not all(math.isfinite(row[k]) for k in ("generalized_seconds_per_departed_passenger","headway_ridge_mean","headway_plan_mae_seconds")):
                            raise ValueError("Non-finite learned endpoint")
                        if smoke:
                            smoke_endpoints += 1
                        else:
                            report["formal_evaluations"] += 1
                        report["completed_labels"].append(prefix+f"policy/{objective}/{scene}/{arm}/{seed}/eval/{layer}/{case}")
        for arm,layer,case in product(["daganzo_duet","zero"],P["executors"],cases):
            row = read(HERE / ("rule_smoke" if smoke else "rules") / arm / layer / f"case{case:02d}.json")
            if row is not None:
                if smoke:
                    smoke_endpoints += 1
                else:
                    report["formal_evaluations"] += 1
                report["completed_labels"].append(prefix+f"rule/{arm}/{layer}/{case}")
        if smoke and smoke_endpoints==16:
            report["smoke_complete"] = True
            report["ready"].append("smoke-pass")
    report["complete"] = report["formal_initializers"]==4 and report["formal_training"]==24 and report["formal_evaluations"]==520
    report["speed_diagnosis_evaluations"] = len(list((HERE / "speed_diagnosis").glob("**/case*.json")))
    report["speed_diagnosis_expected"] = 480
    report["speed_diagnosis"] = []
    for folder in sorted((HERE / "speed_diagnosis").glob("*/*/*/*/speed*")):
        rows = [read(p) for p in sorted(folder.glob("case*.json"))]
        if rows:
            pairs = []
            for r in rows:
                if r["horizon_seconds"]!=24000 or not math.isfinite(r["generalized_seconds_per_departed_passenger"]):
                    raise ValueError("Incomplete speed intervention")
                if r["branch"].startswith("new"):
                    label = f"diagnosis/{r['branch']}/{r['objective']}/{r['source_scene']}/{r['arm']}/{r['training_seed']}/{r['case']}"
                    native = read(EXPERIMENTS / "bus_control_repair_v3/mc_runs/new/dual" / r["objective"] / r["source_scene"] / f"{r['arm']}_seed{r['training_seed']}/guarded/case{r['case']:02d}.json")
                else:
                    label = f"diagnosis/old/{r['source_scene']}/{r['arm']}/{r['training_seed']}/{r['speed_intervention']}/{r['case']}"
                    native = read(EXPERIMENTS / "mcwm_seed_confirmation_v13/runs" / r["source_scene"] / f"{r['arm']}_seed{r['training_seed']}/results/eval_{r['case']+10:02d}.json")
                if native is not None:
                    block = native["block"] if r["branch"].startswith("new") else {k:native[k] for k in ("demand_seed","sumo_seed")}
                    if (r["block"],r["demand_scale"],r["horizon_seconds"]) != (block,native["demand_scale"],native["horizon_seconds"]):
                        raise ValueError("Unmatched native-speed counterfactual")
                    pairs.append((r,native))
                report["completed_labels"].append(label)
            result = dict(group=str(folder.relative_to(HERE)),**metrics(rows),paired_cases=len(pairs),
                          complete=len(rows)==10, cost_delta_vs_native_speed=mean([r["generalized_seconds_per_departed_passenger"]-n["generalized_seconds_per_departed_passenger"] for r,n in pairs]))
            if rows[0]["branch"].startswith("new") and pairs:
                result["ridge_delta_vs_native_speed"] = mean([r["headway_ridge_mean"]-n["headway_ridge_mean"] for r,n in pairs])
            report["speed_diagnosis"].append(result)
    report["results"] = []
    for objective,scene,arm,layer in product(P["objectives"],P["scenes"],P["arms"],P["executors"]):
        rows = []
        paired = []
        for seed,case in product(P["training_seeds"],range(10)):
            row = read(policy_folder(objective,scene,arm,seed) / layer / f"case{case:02d}.json")
            rule = read(HERE / "rules/daganzo_duet" / layer / f"case{case:02d}.json")
            if row is not None:
                rows.append(row)
                if rule is not None:
                    if (row["block"],row["demand_scale"],row["horizon_seconds"]) != (rule["block"],rule["demand_scale"],rule["horizon_seconds"]):
                        raise ValueError("Unmatched SUMO baseline endpoint")
                    paired.append((row,rule))
        if rows:
            result = dict(objective=objective,scene=scene,arm=arm,layer=layer,**metrics(rows),paired_cases=len(paired))
            if paired:
                result.update(cost_delta_vs_daganzo=mean([r["generalized_seconds_per_departed_passenger"]-b["generalized_seconds_per_departed_passenger"] for r,b in paired]),
                    ridge_delta_vs_daganzo=mean([r["headway_ridge_mean"]-b["headway_ridge_mean"] for r,b in paired]))
            report["results"].append(result)
    report["completed_labels"] += [x for x in report["ready"] if x!="smoke-pass"]
    write(HERE / "summary.json",report)
    print({k:report[k] for k in ("smoke_complete","formal_initializers","formal_training","formal_evaluations","speed_diagnosis_evaluations","complete")})


if __name__ == "__main__":
    summarize()
