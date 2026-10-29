"""Compare completed frozen-world controls using existing small local summaries."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
V5 = EXPERIMENTS / "mcwm_closed_loop_context_shadow_v5_20261020"
V4 = EXPERIMENTS / "mcwm_closed_loop_v4_review_20261020"
HEADS = ("global", "local", "count", "line", "station", "neighbor")
TD_FIELDS = ("trust_mean", "effective_model_scale", "real_td_loss",
             "model_td_contribution", "model_real_gradient_norm_ratio")


def predictions(value):
    result = dict(value)
    n = sum(value[name]["n"] for name in ("before", "after"))
    result["all_selected"] = dict(n=n, selected_head_loss={head: sum(
        value[name]["n"] * value[name]["selected_head_loss"][head]
        for name in ("before", "after")) / n for head in HEADS})
    for row in result.values():
        row["six_head_mean"] = mean(row["selected_head_loss"].values())
    return result


def compare():
    summary = json.loads((V5 / "frozen_world_summary.json").read_text())
    prior = json.loads((V4 / "v4_summary.json").read_text())
    cells = lambda values: {(row["seed"], row["scenario"]): row for row in values}
    costs, contrasts = cells(summary["control_cells"]), cells(summary["frozen_world_contrasts"])
    current_td = cells(summary["td_diagnostics"])
    reference_td = cells([row for row in prior["td_diagnostics"]
                          if row["arm"] == "action_model" and row["seed"] in (6201, 6202)])
    runs = []
    for diagnostic in summary["frozen_world_diagnostics"]:
        key = diagnostic["seed"], diagnostic["scenario"]
        cost, contrast = costs[key], contrasts[key]
        full = cost["seconds_per_departed_passenger"]
        frozen_cost = dict(full_day=full, **{name: row["seconds_per_full_day_departed_passenger"]
                                          for name, row in cost["windows"].items()})
        delta = dict(full_day=contrast["full_day_difference"], **contrast["window_differences"])
        reference_cost = {name: frozen_cost[name] - delta[name] for name in delta}
        trust = {}
        for arm, td in (("frozen_world", current_td[key]), ("action_model", reference_td[key])):
            trust[arm] = {period: {field: row[field] for field in TD_FIELDS}
                          for period, row in td["periods"].items()}
            total = sum(td["periods"][period]["packets"] for period in ("before", "after"))
            trust[arm]["all_first_steps"] = {field: sum(td["periods"][period]["packets"] *
                td["periods"][period][field] for period in ("before", "after")) / total
                for field in TD_FIELDS}
        runs.append(dict(seed=key[0], scenario=key[1], departed=cost["departed"],
            last_active_update=cost["last_active_update_at"], frozen_parameters=diagnostic["frozen_parameters"],
            cost=dict(frozen_world=frozen_cost, action_model=reference_cost, frozen_minus_action_model=delta),
            prediction=dict(frozen_world=predictions(diagnostic["prequential_heads"]),
                            action_model=predictions(diagnostic["reference_prequential_heads"])), trust=trust))
    scenarios = {}
    for scenario in ("stable", "local_speed_drop"):
        selected = [row for row in runs if row["scenario"] == scenario]
        cost = {arm: {period: mean(row["cost"][arm][period] for row in selected)
                      for period in ("full_day", "adaptation", "post_training_frozen")}
                for arm in ("frozen_world", "action_model", "frozen_minus_action_model")}
        prediction = {arm: {period: {
            **{head: mean(row["prediction"][arm][period]["selected_head_loss"][head]
                          for row in selected) for head in HEADS},
            "six_head_mean": mean(row["prediction"][arm][period]["six_head_mean"] for row in selected)}
            for period in ("before", "after", "all_selected")}
            for arm in ("frozen_world", "action_model")}
        trust = {arm: {period: {field: mean(row["trust"][arm][period][field] for row in selected)
                                for field in TD_FIELDS}
                       for period in ("before", "after", "late_after", "all_first_steps")}
                 for arm in ("frozen_world", "action_model")}
        scenarios[scenario] = dict(cost=cost, prediction=prediction, trust=trust)
    return dict(status=summary["status"], integrity=summary["integrity"], by_seed=runs,
        scenario_means=scenarios,
        interpretation=dict(cost="Frozen minus action_model; negative favors frozen. Generalized passenger seconds divided by the full-day15079 departed cohort; windows are ledger increments, not separate passenger cohorts.",
            prediction="Causal issue-time predictions on each arm's own realized trajectory.135 before and150 after9000s labels per day;285 total. Global is standardized four-output MSE; six-head mean includes five other structured heads.",
            trust="Mean of the first optimizer step in each packet:26 before,31 after; late_after is a subset of11 from15000s, including the final18000s packet. This is not a mean over all456 optimizer steps.",
            post_stop="18300–24000 has actual ledger costs only. Selected prediction labels and trust updates stop around18000; no post-stop prediction/trust measurement exists.",
            inference="Two original development seeds. Freezing begins after initial source/history fit, before online adaptation, so shock-period effects include earlier policy and trajectory differences."))


if __name__ == "__main__":
    result = compare()
    (HERE / "comparison_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                           allow_nan=False) + "\n")
    print(json.dumps(result["scenario_means"], ensure_ascii=False, indent=2))
