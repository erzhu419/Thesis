"""Separate completed cohorts and summarize resident model logs without weights."""

import argparse
import inspect
import json
from pathlib import Path
import shlex
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def collect_server(root):
    from collections import Counter
    import json
    from pathlib import Path
    from statistics import mean

    rows, groups = [], []
    root = Path(root)
    for scene in ("calibrated", "joint50", "joint75"):
        choices, finals, records = Counter(), Counter(), []
        for seed in range(63001, 63021):
            folder = root / "runs" / scene / f"fixed0250_seed{seed}" / "results"
            run = json.loads((folder / "train_summary.json").read_text())
            traces = [json.loads(p.read_text()) for p in sorted((folder / "diagnostics").glob("update*.json"))]
            assert run["continuation_updates"] == 20000 and len(traces) == 20
            assert run["real_td_coefficient"] == .75 and run["target_online_training_events"] == 0
            history = run["model_history"]
            choices.update(h["selected_family"] for h in history)
            finals.update([history[-1]["selected_family"]])
            last = traces[-1]
            calibration = last["model"]["calibrate"]
            records.append(dict(scene=scene, seed=seed, final_family=last["model"]["selected_family"],
                final_weight=calibration["weights"]["mean"],
                final_ess_fraction=calibration["effective_sample_size"]/calibration["weights"]["n"],
                final_prediction_loss=calibration["unweighted_prediction_loss"]["mean"],
                final_slot_identity_accuracy=calibration["components"]["slot_identity_accuracy"],
                mean_scaled_gradient_ratio=mean(r["scaled_model_gradient_norm"]/r["scaled_real_gradient_norm"] for r in traces),
                mean_gradient_cosine=mean(r["gradient_cosine"] for r in traces),
                negative_cosine_fraction=mean(r["gradient_cosine"] < 0 for r in traces),
                model_refreshes=len(history),
                direct_refreshes=sum(h["selected_family"] == "direct" for h in history)))
        fields = ("final_weight", "final_ess_fraction", "final_prediction_loss", "final_slot_identity_accuracy",
                  "mean_scaled_gradient_ratio", "mean_gradient_cosine", "negative_cosine_fraction")
        groups.append(dict(scene=scene, policies=len(records), refresh_choices=dict(choices),
            final_choices=dict(finals), diagnostics={k:dict(mean=mean(r[k] for r in records),
            minimum=min(r[k] for r in records), maximum=max(r[k] for r in records)) for k in fields}))
        rows.extend(records)
    return dict(cohort="mcwm_seed_confirmation_v13", arm="fixed0250", source=str(root),
        weights_or_datasets_downloaded=False, groups=groups, policies=rows,
        scope="Training/calibration diagnostics, not independent model generalization or a causal family intervention.")


def read(experiment, name="summary.json"):
    return json.loads((ROOT / "experiments" / experiment / name).read_text())


def analyze():
    result = {}
    early = read("ratio_split_v2")["mc"]
    result["old_single_fit"] = [dict(demand=demand, arm=arm,
        cost_delta_seconds=detail["cost"]["mean"], ci95=detail["cost"]["ci95"])
        for demand, group in early["results"].items()
        for arm, detail in group["paired_differences"].items()
        if arm.startswith(("fixed_direct", "fixed_residual", "calibrated_selection"))]
    usage = read("mcwm_usage_v3", "analysis.json")
    result["normalized_same_scaling_control"] = [dict(scene=r["scenario"], demand=r["demand_scale"],
        cost_delta_seconds=-r["cost_minus_normalized"], paired_seed_deltas=[-x for x in r["paired_cost_minus_normalized"]])
        for r in usage["cells"] if r["arm"] == "scaled_nomodel"]
    context = read("mcwm_context_v7")
    zeros = [r for r in context["outcomes"] if r["arm"] == "fixed_zero"
             and r["reference"] == "fixed_original" and r["reference_role"] == "fixed_original"]
    assert len(zeros) == 12
    result["fixed_coefficient_intervention"] = [dict(scene=scene, demand=demand,
        model_minus_zero_seconds=-mean(r["delta_seconds"] for r in zeros if r["scenario"] == scene and r["demand_scale"] == demand))
        for scene in ("calibrated", "joint50", "joint75") for demand in (1., 1.25)]
    dual = read("bus_dual_aligned_v4")
    guarded = {(r["objective"], r["scene"], r["arm"]):r for r in dual["results"] if r["layer"] == "guarded"}
    result["dual_passenger"] = []
    for scene in ("calibrated", "joint75"):
        mc, nomodel, h2o = [guarded["passenger", scene, arm] for arm in ("mcwm", "h2o_matched", "h2oplus")]
        total = h2o["cost"]-mc["cost"]
        result["dual_passenger"].append(dict(scene=scene, mcwm_cost=mc["cost"], h2o_cost=h2o["cost"],
            no_ratio_no_model_cost=nomodel["cost"], total_reduction=total,
            no_ratio_contrast=h2o["cost"]-nomodel["cost"],
            model_package_contrast=nomodel["cost"]-mc["cost"],
            scope="Ordered contrasts of observed means, not additive causal mediation; model package changes TD scaling too."))
    follow = read("bus_alignment_followup_v5")
    result["dual_clipped_headway"] = [dict(scene=r["scene"], baseline=r["baseline"],
        cost_change_percent=r["cost_change_pct"], headway_penalty_reduction_percent=r["reward_penalty_reduction_pct"])
        for r in follow["mc_aggregate_comparisons"] if r["layer"] == "guarded" and r["variant"] == "clip10"]
    confirm = read("mcwm_seed_confirmation_v13", "completed_analysis.json")
    result["confirmation_metrics"] = [dict(scene=r["scenario"], arm=r["arm"],
        cost=r["metrics"]["cost_seconds"]["mean"], wait=r["metrics"]["wait_seconds"]["mean"],
        ride=r["metrics"]["in_vehicle_seconds"]["mean"], hold=r["metrics"]["hold_seconds"]["mean"],
        reserve=r["metrics"]["reserve_percent"]["mean"])
        for r in confirm["all_seed_metrics"] if r["arm"] in ("fixed0250", "h2o_matched")]
    result["confirmation_contrasts"] = [dict(scene=r["scenario"], cost_delta=r["cost_delta_seconds"]["mean"],
        ci95=r["cost_delta_seconds"]["ci95"], better_seeds=r["better_seeds"],
        wait_delta=r["wait_delta_seconds"]["mean"], ride_delta=r["in_vehicle_delta_seconds"]["mean"])
        for r in confirm["comparisons"] if r["arm"] == "fixed0250" and r["reference"] == "h2o_matched"]
    (HERE / "existing_evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def remote():
    sys.path.insert(0, str(ROOT.parent / "scheduleurm/skill"))
    import scheduler
    root = "/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_seed_confirmation_v13"
    program = inspect.getsource(collect_server)+f"\nimport json\nprint(json.dumps(collect_server({root!r})))"
    python = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
    _, output, _ = scheduler.run_on("node007", shlex.quote(python)+" -c "+shlex.quote(program), timeout=90, check=True)
    result = json.loads(output)
    (HERE / "resident_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("local", "remote"))
    args = parser.parse_args()
    remote() if args.phase == "remote" else analyze()
