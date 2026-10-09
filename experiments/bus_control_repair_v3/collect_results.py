"""Read case files on the server; transfer only small paired summaries."""

from itertools import product
import json
from statistics import mean

from repair_common import HERE, EXPERIMENTS, P, cs_folder, write

COST = "generalized_seconds_per_departed_passenger"


def read(path):
    return json.loads(path.read_text())


def average_cost(rows):
    return mean(row[COST] for row in rows)


def collect():
    ready, counts, cs_comparisons, mc_comparisons, mc_rules = [], {}, [], [], []
    for smoke in (True, False):
        prefix = "smoke/" if smoke else ""
        trains = evaluations = 0
        seeds = P["cs_seeds"][:1] if smoke else P["cs_seeds"]
        for objective, seed, arm in product(P["cs_objectives"], seeds, P["cs_arms"]):
            folder = cs_folder(objective, arm, seed, smoke) / "results"
            if (folder / "ready.json").exists():
                trains += 1
                ready.append(prefix + f"cs/{objective}/{arm}/{seed}/policy")
            cases = [folder / f"eval_{case:02d}.json" for case in ([0] if smoke else range(9))]
            evaluations += sum(path.exists() for path in cases)
            if not smoke and all(path.exists() for path in cases):
                rows = [read(path) for path in cases]
                for control in ("centered_symbolic", "centered_no_symbolic", "legacy_nau", "legacy_mlp"):
                    source = EXPERIMENTS / "csbapr_symbolic_bus_v2/runs" / objective / f"{control}_seed{seed}" / "results"
                    for scene in ("normal", "pulse", "repeated"):
                        treated = [r for r in rows if r["scene"] == scene]
                        baseline = [read(source / f"eval_{r['case']:02d}.json") for r in treated]
                        cost_field = "generalized_seconds_per_passenger"
                        native = lambda r: r.get("native_reward", r.get("reward")) / r["decisions"]
                        cs_comparisons.append(dict(objective=objective, arm=arm, seed=seed,
                            scene=scene, control=control,
                            cost_change_pct=100*(mean(r[cost_field] for r in treated)/mean(r[cost_field] for r in baseline)-1),
                            reward_per_decision_delta=mean(native(r) for r in treated)-mean(native(r) for r in baseline),
                            headway_plan_error_change_pct=100*(mean(r["headway_plan_mae_seconds"] for r in treated)/mean(r["headway_plan_mae_seconds"] for r in baseline)-1)))
        for objective, seed in product(P["cs_objectives"], seeds):
            source = cs_folder(objective, "fitted_reference", seed, smoke) / "results"
            evaluations += sum((source / f"eval_{c:02d}.json").exists() for c in ([0] if smoke else range(9)))
        counts["cs_smoke_training" if smoke else "cs_training"] = trains
        counts["cs_smoke_evaluations" if smoke else "cs_evaluations"] = evaluations
    mc_files = list((HERE / "mc_runs").glob("*/*/*/*/*/*/case*.json"))
    smoke_files = list((HERE / "mc_smoke").glob("*/*/*/*/*/*/case*.json"))
    counts.update(mc_evaluations=len(mc_files), mc_smoke_evaluations=len(smoke_files))
    if counts["cs_smoke_training"] == 8 and counts["cs_smoke_evaluations"] == 10:
        ready.append("cs-smoke-pass")
    if len(smoke_files) == 4:
        ready.append("mc-smoke-pass")
    for branch in ("old", "new"):
        seeds = P["mc_old_seeds"] if branch == "old" else P["mc_new_seeds"]
        arms = P["mc_old_arms"] if branch == "old" else P["mc_new_arms"]
        cells = [("holding", "passenger")] if branch == "old" else P["mc_cells"]
        layer = "direct" if branch == "old" else "guarded"
        rule_paths = [HERE / "mc_runs/rule/holding/passenger/calibrated/daganzo_duet_seed0/guarded" / f"case{c:02d}.json" for c in range(10)]
        for scene, seed, arm, (mode, objective) in product(P["mc_scenes"], seeds, arms, cells):
            folder = HERE / "mc_runs" / branch / mode / objective / scene / f"{arm}_seed{seed}" / layer
            paths = [folder / f"case{c:02d}.json" for c in range(10)]
            if not all(path.exists() for path in paths):
                continue
            new = [read(path) for path in paths]
            if branch == "new":
                source = EXPERIMENTS / "bus_control_v2/runs" / mode / objective / scene / f"{arm}_seed{seed}" / "evaluation"
                original = [read(source / f"case{c:02d}.json") for c in range(10)]
            else:
                source = EXPERIMENTS / "mcwm_seed_confirmation_v13/runs" / scene / f"{arm}_seed{seed}" / "results"
                original = [read(source / f"eval_{c:02d}.json") for c in range(10, 20)]
            row = dict(branch=branch, mode=mode, objective=objective, scene=scene, seed=seed, arm=arm,
                new_layer=layer, cost_seconds=average_cost(new),
                previous_layer_cost_seconds=average_cost(original),
                layer_cost_change_pct=100*(average_cost(new)/average_cost(original)-1),
                guard_changed_fraction=sum(r["guard_changed_decisions"] for r in new)/sum(r["decision_events"] for r in new))
            if branch == "new" and all(path.exists() for path in rule_paths):
                row["same_layer_daganzo_cost_change_pct"] = 100*(average_cost(new)/average_cost([read(path) for path in rule_paths])-1)
            if branch == "old":
                direct_rule = [read(EXPERIMENTS / "bus_control_v2/rules/daganzo_duet/evaluation" / f"case{c:02d}.json") for c in range(10)]
                row["same_layer_daganzo_cost_change_pct"] = 100*(average_cost(new)/average_cost(direct_rule)-1)
            mc_comparisons.append(row)
    zero = [read(EXPERIMENTS / "bus_control_v2/rules/zero/evaluation" / f"case{c:02d}.json") for c in range(10)]
    for arm, layer in [("daganzo_duet", "guarded"), ("speed1.1", "direct"), ("speed1.2", "direct")]:
        source = HERE / "mc_runs/rule/holding/passenger/calibrated" / f"{arm}_seed0" / layer
        paths = [source / f"case{c:02d}.json" for c in range(10)]
        if all(path.exists() for path in paths):
            rows = [read(path) for path in paths]
            mc_rules.append(dict(arm=arm, layer=layer, cost_seconds=average_cost(rows),
                zero_cost_change_pct=100*(average_cost(rows)/average_cost(zero)-1),
                headway_ridge_mean=mean(r["headway_ridge_mean"] for r in rows),
                headway_plan_mae_seconds=mean(r["headway_plan_mae_seconds"] for r in rows)))
    final_metrics = []
    for path in (HERE / "cs_runs").glob("*/*/results/progress.json"):
        row = read(path)
        final_metrics.append(dict(objective=path.parents[2].name, arm=path.parents[1].name,
                                  episode=row["episode"], metrics=row["metrics"]))
    summary = dict(ready=ready, **counts, expected_cs_training=16, expected_cs_evaluations=180,
        expected_mc_evaluations=350,
        complete=counts["cs_training"] == 16 and counts["cs_evaluations"] == 180 and len(mc_files) == 350,
        cs_comparisons=cs_comparisons, mc_comparisons=mc_comparisons, mc_rule_metrics=mc_rules, cs_latest_metrics=final_metrics,
        analysis_location="server", raw_results_downloaded=False)
    write(HERE / "summary.json", summary)
    print({key: summary[key] for key in (*counts, "complete")}, flush=True)


if __name__ == "__main__":
    collect()
