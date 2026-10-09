"""Read resident scalar results and compare matched cases on the server."""

from collections import defaultdict
import json
import math

from followup_common import HERE, EXPERIMENTS, P, label, folder, parent, policies, needs_training, evaluation_cases, speed_controls, write


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def ready(item):
    out = parent(item)
    row = read(out / "ready.json") if needs_training(item) else read(out / "results/ready.json")
    if row is not None:
        if item["family"] == "mc":
            expected = 512 if item.get("smoke") else P["mc_source_events"]
            # The preserved sampler executes a complete batch of up to 16 buses.
            assert expected <= row["source_events"] < expected+16
            original = json.loads((EXPERIMENTS / "bus_dual_aligned_v4/protocol.json").read_text())
            assert row["updates"] == expected-(256 if item.get("smoke") else original["warmup_events"])
            assert row["target_online_training_events"] == 0
        else:
            assert row["episodes"] == (1 if item.get("smoke") else P["cs_episodes"])
        assert (out / "policy.pt").exists()
    return row


def endpoint(item, layer, case):
    out = folder(item)
    return read(out / layer / f"case{case:02d}.json") if item["family"] == "mc" else read(out / f"eval_{case:02d}.json")


def values(item, row):
    if item["family"] == "mc":
        return dict(cost=row["generalized_seconds_per_departed_passenger"], ridge=row["headway_ridge_mean"],
            plan_mae=row["headway_plan_mae_seconds"], speed=row["mean_speed_ratio"], holding=row["mean_hold_seconds"])
    return dict(cost=row["cost"], recorded_return=row["native_reward"], decisions=row["decisions"],
        recorded_reward_per_decision=row["native_reward"]/row["decisions"],
        plan_mae=row["headway_plan_mae_seconds"], imbalance=row["headway_imbalance_seconds"],
        completion=row["completion_rate"], speed=row["mean_speed_ratio"], holding=row["mean_hold_seconds"])


def validate(item, row, case):
    horizon = (1800 if item.get("smoke") else 24000) if item["family"] == "mc" else (3600 if item.get("smoke") else P["cs_horizon_seconds"])
    assert row["horizon_seconds"] == horizon
    assert all(math.isfinite(x) for x in values(item, row).values())
    if item["family"] != "mc":
        assert row["evaluation_seed"] == P["cs_evaluation_seeds"][case % 3]
        assert row["scene"] == list(P["cs_scenes"])[case // 3]


def mean(rows):
    return {k:sum(row[k] for row in rows)/len(rows) for k in rows[0]} if rows else {}


def grouped_results(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [dict(zip(fields, key), seeds=[row["seed"] for row in group], cases=sum(row["cases"] for row in group),
        complete=all(row["complete"] for row in group),
        **mean([{metric:row[metric] for metric in metrics} for row in group]))
        for key, group in sorted(groups.items())]


def aggregate_comparisons(rows, fields, comparisons, reward=None):
    index = {tuple(row[field] for field in fields)+(row["arm"],):row for row in rows}
    output = []
    for row in rows:
        for control in comparisons.get(row["arm"], []):
            baseline = index.get(tuple(row[field] for field in fields)+(control,))
            if baseline is None:
                continue
            result = dict(row, baseline=control,
                cost_change_pct=100*(row["cost"]/baseline["cost"]-1),
                plan_mae_change_pct=100*(row["plan_mae"]/baseline["plan_mae"]-1))
            if reward:
                result.update(reward_delta=row[reward]-baseline[reward],
                    reward_penalty_reduction_pct=100*(row[reward]-baseline[reward])/abs(baseline[reward]))
            output.append(result)
    return output


def pair(item, left, right):
    keys = ("block", "demand_scale", "horizon_seconds") if item["family"] == "mc" else ("evaluation_seed", "scene", "horizon_seconds", "generated_passengers")
    assert all(left[k] == right[k] for k in keys), "Unmatched control day"


def summarize():
    report = dict(ready=[], completed_labels=[], smoke_pass={}, formal_training=0, formal_evaluations=0,
        expected_training=40, expected_evaluations=1236, reused_policies=0, complete=False, analysis_location="server",
        training_progress=[], training_endpoints=[], cs_results=[], cs_contrasts=[], mc_results=[], mc_contrasts=[])
    families = ("cs_recipe", "cs_context", "mc")
    contracts = {}
    for family in families:
        contracts[family] = bool((read(HERE / "contracts" / (family+".json")) or {}).get("passed"))
        if contracts[family]:
            report["ready"].append("contracts/"+family)
    smoke_done, smoke_expected = defaultdict(int), defaultdict(int)
    for smoke in (True, False):
        for item in policies(smoke):
            name, train = label(item), ready(item)
            if smoke:
                smoke_expected[item["family"]] += 2
            if train is not None:
                report["ready"].append(name)
                if needs_training(item):
                    report["completed_labels"].append(name+"/train")
                    if smoke:
                        smoke_done[item["family"]] += 1
                    else:
                        report["formal_training"] += 1
                        progress = read(folder(item) / "progress.json") or {}
                        selected = ("alpha", "q_loss", "policy_loss", "jac_loss", "sym_penalty", "train/alpha",
                            "train/critic_grad_norm", "train/actor_grad_norm", "train/speed_prior_kl", "model_td_loss", "model_weight_mean")
                        report["training_endpoints"].append(dict(item=item, ready=train,
                            final_metrics={k:v for k,v in (progress.get("metrics") or {}).items() if k in selected}))
                elif not smoke:
                    report["reused_policies"] += 1
            elif not smoke and needs_training(item):
                progress = read(folder(item) / "progress.json")
                if progress is not None:
                    report["training_progress"].append(dict(policy=name,
                        **{k:progress[k] for k in ("episode", "source_events", "updates", "wall_seconds") if k in progress}))
            for layer, case in evaluation_cases(item):
                row = endpoint(item, layer, case)
                if row is not None:
                    validate(item, row, case)
                    report["completed_labels"].append(name+f"/eval/{layer}/{case}")
                    if smoke:
                        smoke_done[item["family"]] += 1
                    else:
                        report["formal_evaluations"] += 1
    for item in speed_controls():
        for layer, case in evaluation_cases(item):
            row = endpoint(item, layer, case)
            if row is not None:
                validate(item, row, case)
                report["formal_evaluations"] += 1
                report["completed_labels"].append(label(item)+f"/eval/{layer}/{case}")
    for family in families:
        passed = contracts[family] and smoke_done[family] == smoke_expected[family]
        report["smoke_pass"][family] = passed
        if passed:
            report["ready"].append("smoke-pass/"+family)
    cs_items = [item for item in policies() if item["family"] != "mc"] + list(speed_controls())
    groups = defaultdict(list)
    for item in cs_items:
        for case in range(18):
            row = endpoint(item, "native", case)
            if row is not None:
                key = (item["family"], item["mode"], item["objective"], item["arm"], item["seed"], row["scene"])
                groups[key].append(values(item, row))
    for key, rows in sorted(groups.items()):
        family, mode, objective, arm, seed, scene = key
        report["cs_results"].append(dict(family=family, mode=mode, objective=objective, arm=arm,
            seed=seed, scene=scene, cases=len(rows), complete=len(rows)==3, **mean(rows)))
    controls = dict(stabilized_mlp=["pristine_mlp"], stabilized_nau=["pristine_mlp", "stabilized_mlp"],
        context_symbolic=["context_control", "constant_symbolic"], context_control=["constant_control"])
    for item in cs_items:
        for baseline in controls.get(item["arm"], []):
            for scene_index, scene in enumerate(P["cs_scenes"]):
                pairs = []
                for case in range(scene_index*3, scene_index*3+3):
                    left = endpoint(item, "native", case)
                    right = endpoint(dict(item, arm=baseline), "native", case)
                    if left is not None and right is not None:
                        pair(item, left, right)
                        a, b = values(item, left), values(item, right)
                        pairs.append({k+"_delta":a[k]-b[k] for k in a} | {"cost_pct":100*(a["cost"]/b["cost"]-1)})
                if pairs:
                    report["cs_contrasts"].append(dict(family=item["family"], mode=item["mode"], objective=item["objective"],
                        arm=item["arm"], baseline=baseline, seed=item["seed"], scene=scene, cases=len(pairs), complete=len(pairs)==3, **mean(pairs)))
    old_mc = EXPERIMENTS / "bus_dual_aligned_v4"
    for item in (i for i in policies() if i["family"] == "mc"):
        for layer in ("direct", "guarded"):
            rows, pairs = [], defaultdict(list)
            for case in range(10):
                left = endpoint(item, layer, case)
                if left is None:
                    continue
                a = values(item, left)
                rows.append(a)
                comparison = {
                    "daganzo":read(old_mc / "rules/daganzo_duet" / layer / f"case{case:02d}.json"),
                    "unchanged_same_arm":read(old_mc / "runs/headway" / item["scene"] / f"{item['arm']}_seed{item['seed']}" / layer / f"case{case:02d}.json")}
                if item["arm"] == "mcwm":
                    comparison.update({arm:endpoint(dict(item, arm=arm), layer, case) for arm in ("h2oplus", "h2o_matched")})
                    comparison.update({"unchanged_"+arm:read(old_mc / "runs/headway" / item["scene"] /
                        f"{arm}_seed{item['seed']}" / layer / f"case{case:02d}.json") for arm in ("h2oplus", "h2o_matched")})
                for baseline, right in comparison.items():
                    if right is not None:
                        pair(item, left, right)
                        b = values(item, right)
                        pairs[baseline].append({k+"_delta":a[k]-b[k] for k in a} | {"cost_pct":100*(a["cost"]/b["cost"]-1)})
            identity = dict(variant=item["variant"], scene=item["scene"], arm=item["arm"], seed=item["seed"], layer=layer)
            if rows:
                report["mc_results"].append(dict(identity, cases=len(rows), complete=len(rows)==10, **mean(rows)))
            for baseline, deltas in pairs.items():
                report["mc_contrasts"].append(dict(identity, baseline=baseline, cases=len(deltas), complete=len(deltas)==10, **mean(deltas)))
    report["cs_aggregates"] = grouped_results(report["cs_results"], ("family", "mode", "objective", "arm", "scene"),
        ("cost", "recorded_return", "decisions", "recorded_reward_per_decision", "plan_mae", "imbalance", "completion", "holding", "speed"))
    report["cs_aggregate_comparisons"] = aggregate_comparisons(report["cs_aggregates"],
        ("family", "mode", "objective", "scene"), controls, "recorded_reward_per_decision")
    report["mc_aggregates"] = grouped_results(report["mc_results"], ("variant", "scene", "arm", "layer"),
        ("cost", "ridge", "plan_mae", "holding", "speed"))
    report["mc_aggregate_comparisons"] = aggregate_comparisons(report["mc_aggregates"],
        ("variant", "scene", "layer"), dict(mcwm=["h2oplus", "h2o_matched"]), "ridge")
    report["complete"] = report["formal_training"] == 40 and report["formal_evaluations"] == 1236
    write(HERE / "summary.json", report)
    print({k:report[k] for k in ("smoke_pass", "formal_training", "formal_evaluations", "complete")})


if __name__ == "__main__":
    summarize()
