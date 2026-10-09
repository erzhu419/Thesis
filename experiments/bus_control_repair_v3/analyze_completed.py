"""Keep the two repair branches and the earlier twenty-seed study separate."""

from itertools import product
from statistics import mean

from collect_results import read
from repair_common import HERE, EXPERIMENTS, P, write


def reward_loss(row):
    return -row.get("native_reward", row.get("reward"))/row["decisions"]


def cs_values(rows):
    return dict(cost_seconds=mean(r["generalized_seconds_per_passenger"] for r in rows),
        plan_error_seconds=mean(r["headway_plan_mae_seconds"] for r in rows),
        imbalance_seconds=mean(r["headway_imbalance_seconds"] for r in rows),
        reward_loss_per_decision=mean(reward_loss(r) for r in rows),
        completion_percent=100*mean(r["completion_rate"] for r in rows),
        holding_seconds=mean(r["mean_hold_seconds"] for r in rows))


def paired_cs(first, second):
    keys = ("scene", "evaluation_seed", "horizon_seconds", "generated_passengers")
    if any(tuple(x[k] for k in keys) != tuple(y[k] for k in keys) for x,y in zip(first,second)):
        raise ValueError("CS comparison uses different evaluation days or service horizons")
    a, b = cs_values(first), cs_values(second)
    return {key: 100*(a[key]/b[key]-1) for key in
            ("cost_seconds", "plan_error_seconds", "imbalance_seconds", "reward_loss_per_decision")} | dict(
        completion_delta_pp=a["completion_percent"]-b["completion_percent"],
        better_plan_cases=sum(x["headway_plan_mae_seconds"] < y["headway_plan_mae_seconds"] for x,y in zip(first,second)),
        better_cost_cases=sum(x["generalized_seconds_per_passenger"] < y["generalized_seconds_per_passenger"] for x,y in zip(first,second)),
        paired_cases=len(first))


def mc_values(rows, baseline):
    cost = lambda r: r["generalized_seconds_per_departed_passenger"]
    field = lambda key: mean(r[key] for r in rows)
    return dict(cost_seconds=mean(cost(r) for r in rows),
        daganzo_cost_change_pct=100*(mean(cost(r) for r in rows)/mean(cost(r) for r in baseline)-1),
        plan_error_seconds=field("headway_plan_mae_seconds"),
        plan_error_vs_daganzo_pct=100*(field("headway_plan_mae_seconds")/mean(r["headway_plan_mae_seconds"] for r in baseline)-1),
        imbalance_seconds=field("headway_imbalance_seconds"), ridge_reward=field("headway_ridge_mean"),
        ridge_loss_vs_daganzo_pct=100*(field("headway_ridge_mean")/mean(r["headway_ridge_mean"] for r in baseline)-1),
        completion_percent=100*mean(r["passengers"]["completion_rate"] for r in rows),
        holding_seconds=field("mean_hold_seconds"), speed_ratio=field("mean_speed_ratio"),
        better_cost_cases=sum(cost(x)<cost(y) for x,y in zip(rows,baseline)), paired_cases=len(rows))


def main():
    summary = read(HERE / "summary.json")
    if not summary["complete"]:
        raise ValueError("The formal repair campaign is not complete")
    report = dict(complete=True, analysis_location="server", cs=[], mc_new=[], mc_old=[], historical=[])
    legacy = ["centered_symbolic", "centered_no_symbolic", "legacy_nau", "legacy_mlp"]
    for objective in P["cs_objectives"]:
        groups = {}
        for arm in P["cs_arms"] + legacy + ["fitted_reference"]:
            root = EXPERIMENTS / "csbapr_symbolic_bus_v2/runs" if arm in legacy else HERE / "cs_runs"
            rows = [read(root/objective/f"{arm}_seed{seed}"/"results"/f"eval_{case:02d}.json")
                    for seed,case in product(P["cs_seeds"],range(9))]
            groups[arm] = rows
        for arm, rows in groups.items():
            report["cs"].append(dict(objective=objective, arm=arm, means=cs_values(rows),
                seeds=[dict(seed=seed, **cs_values(rows[i*9:(i+1)*9])) for i,seed in enumerate(P["cs_seeds"])],
                scenes=[dict(scene=scene, **cs_values([r for r in rows if r["scene"]==scene]))
                        for scene in ("normal","pulse","repeated")],
                comparisons={control: paired_cs(rows,groups[control]) for control in
                             ("centered_symbolic","legacy_nau","legacy_mlp","soft_no_symbolic")}))
    rules = {
        "guarded": [read(HERE/"mc_runs/rule/holding/passenger/calibrated/daganzo_duet_seed0/guarded"/f"case{c:02d}.json") for c in range(10)],
        "direct": [read(EXPERIMENTS/"bus_control_v2/rules/daganzo_duet/evaluation"/f"case{c:02d}.json") for c in range(10)]}
    for scene, arm, (mode,objective), layer in product(P["mc_scenes"],P["mc_new_arms"],P["mc_cells"],("direct","guarded")):
        rows = []
        for seed in P["mc_new_seeds"]:
            root = (EXPERIMENTS/"bus_control_v2/runs"/mode/objective/scene/f"{arm}_seed{seed}"/"evaluation"
                    if layer=="direct" else HERE/"mc_runs/new"/mode/objective/scene/f"{arm}_seed{seed}"/layer)
            rows.extend(read(root/f"case{c:02d}.json") for c in range(10))
        row = dict(scene=scene, arm=arm, mode=mode, objective=objective, layer=layer,
                   **mc_values(rows,rules[layer]*len(P["mc_new_seeds"])))
        if layer=="guarded":
            row["adjusted_raw_action_fraction"] = sum(r["guard_changed_decisions"] for r in rows)/sum(r["decision_events"] for r in rows)
            row["proposed_hold_seconds"] = mean(r["mean_proposed_hold_seconds"] for r in rows)
        report["mc_new"].append(row)
    for scene,arm in product(P["mc_scenes"],P["mc_old_arms"]):
        rows=[r for r in summary["mc_comparisons"] if r["branch"]=="old" and r["scene"]==scene and r["arm"]==arm]
        direct=mean(r["cost_seconds"] for r in rows)
        guarded=mean(r["previous_layer_cost_seconds"] for r in rows)
        report["mc_old"].append(dict(scene=scene,arm=arm,training_seeds=len(rows),direct_cost=direct,
            original_guarded_cost=guarded,removing_execution_layer_change_pct=100*(direct/guarded-1)))
    previous=read(EXPERIMENTS/"mcwm_seed_confirmation_v13/completed_analysis.json")
    report["historical_complete"]=previous["complete"]
    for row in previous["comparisons"]:
        if row["arm"]=="fixed0250" and row["reference"] in ("h2o_matched","daganzo_duet"):
            report["historical"].append(dict(scene=row["scenario"],reference=row["reference"],
                seeds=row["cost_delta_seconds"]["n"],better_seeds=row["better_seeds"],
                cost_delta_seconds=row["cost_delta_seconds"]["mean"],ci95=row["cost_delta_seconds"]["ci95"],
                completion_delta_pp=row["completion_delta_pp"]["mean"]))
    write(HERE/"completed_analysis.json",report)
    print(dict(complete=True,cs_groups=len(report["cs"]),mc_groups=len(report["mc_new"]),
               historical_comparisons=len(report["historical"])),flush=True)


if __name__=="__main__":
    main()
