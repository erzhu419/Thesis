"""Resident-only paired analysis for the two-seed action/local development pilot."""

import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
REFERENCE_ROOT = HERE.parent / "mcwm_closed_loop_diagnostic_v2_20261020/runs"
_spec = importlib.util.spec_from_file_location("mcwm_formal_window_analysis",
        HERE.parent / "mcwm_closed_loop_formal_20261020/analysis.py")
_windows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_windows)
CONTRASTS = (("action_model", "old_model"), ("action_model", "action_real"),
             ("action_local", "action_model"), ("action_local", "old_model"))
PERIODS = {"before": (0., 9000.), "after": (9000., 18000.), "late_after": (15000., 18000.)}
LINES = ("7X", "102X", "705X")
PHYSICAL_MODELS = ("frozen_source_prior", "online_linear", "local_context")
PHYSICAL_PERIODS = {"before": ("before",), "early_after": ("early",),
                    "late_after": ("late",), "all_after": ("early", "middle", "late")}


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def jsonl(path):
    path = Path(path)
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def at(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return statistics.mean(values) if values else None


def difference(candidate, reference):
    return candidate - reference if finite(candidate) and finite(reference) else None


def paired(values):
    observed = [float(value) for value in values if finite(value)]
    return dict(values=values, n_pairs=len(observed), mean=mean(observed),
                minimum=min(observed) if observed else None, maximum=max(observed) if observed else None,
                negative=sum(value < 0 for value in observed), positive=sum(value > 0 for value in observed))


def weighted(records, field="mean"):
    count = sum(row.get("n", 0) for row in records)
    if not count or any(row.get("n", 0) and not finite(row.get(field)) for row in records):
        return None
    return sum(row.get("n", 0) * (row.get(field) or 0.) for row in records) / count


TD_FIELDS = {
    "trust_mean": ("absolute_trust", "mean"),
    "effective_model_scale": ("effective_model_mean_weight_scale",),
    "real_td_loss": ("real_td_loss",), "model_td_contribution": ("model_td_contribution",),
    "model_real_gradient_norm_ratio": ("gradients", "model_to_real_scaled_norm_ratio"),
    "real_model_gradient_cosine": ("gradients", "cosine_real_model"),
    "combined_real_gradient_cosine": ("gradients", "cosine_combined_real"),
    "actor_parameter_change_l2": ("actor_parameter_change", "l2"),
    "critic_parameter_change_l2": ("critic_parameter_change", "l2"),
}


def _td_summary(logs, summary, arm):
    result = {}
    for period, (start, stop) in PERIODS.items():
        # The final 18,000-second receipt can be processed a few seconds later;
        # all completed post-shock optimizer packets belong in TD summaries.
        td_stop = math.inf if period != "before" else stop
        selected = [row for row in logs if start <= row.get("processed_at", row["deadline"]) < td_stop]
        observed = [{field: at(row.get("diagnostic") or {}, *path) for field, path in TD_FIELDS.items()}
                    for row in selected]
        if arm == "action_real":
            for row in observed:
                for field in ("model_real_gradient_norm_ratio", "real_model_gradient_cosine",
                              "combined_real_gradient_cosine"):
                    row[field] = None
        values = {field: mean([row[field] for row in observed]) for field in TD_FIELDS}
        values.update({field: mean([row.get(field) for row in selected]) for field in
                      ("proposed_action_change_seconds", "executed_action_change_seconds")})
        ratios = [row["model_real_gradient_norm_ratio"] for row in observed
                  if finite(row["model_real_gradient_norm_ratio"])]
        cosines = [row["real_model_gradient_cosine"] for row in observed
                   if finite(row["real_model_gradient_cosine"])]
        values["model_real_gradient_norm_ratio_max"] = max(ratios) if ratios else None
        values["negative_gradient_cosine_fraction"] = sum(value < 0 for value in cosines) / len(cosines) if cosines else None
        updates = [at(row, "train/actor_updated") or {} for phase, row in summary.get("update_statistics", {}).items()
                   if (phase == "before") == (period == "before")]
        # Update moments are only before/after aggregates; do not relabel the after aggregate as late.
        values["actor_updated_fraction"] = weighted(updates) if period != "late_after" else None
        probes = [row.get("local_counterfactual") or at(row, "diagnostic", "local_counterfactual")
                  for row in selected]
        probe_fields = sorted({key for probe in probes if isinstance(probe, dict)
                               for key, value in probe.items() if finite(value)})
        values["local_counterfactual"] = {field: mean([at(probe, field) for probe in probes])
                                          for field in probe_fields}
        probe_rows = sum(at(probe, "rows") or 0 for probe in probes)
        if probe_rows:
            values["local_counterfactual"]["applied_row_fraction"] = sum(
                at(probe, "applied_rows") or 0 for probe in probes) / probe_rows
        result[period] = dict(packets=len(selected), **values)
    return result


def _action_summary(summary):
    result = {}
    for phase in ("before", "after"):
        selected = [value for name, value in summary.get("action_statistics", {}).items()
                    if (name == "before") == (phase == "before")]
        result[phase] = dict(events=sum(at(value, "proposed_hold_seconds", "n") or 0 for value in selected),
            **{field: weighted([value.get(field, {}) for value in selected]) for field in
               ("proposed_hold_seconds", "executed_hold_seconds", "execution_changed")})
    return result


def _physical(directory, summary, check):
    physical = summary.get("physical_intervals") or {}
    coverage, initialization = physical.get("coverage", {}), physical.get("initialization", {})
    check(initialization.get("domain") == "source" and all(
        at(initialization, "used_by_line", line) is not None
        and 5 <= initialization["used_by_line"][line] <= 64 for line in LINES), "local source provenance")
    check(coverage.get("missing_saved_prediction") == 0, "physical saved prediction coverage")
    rows = {model: [] for model in PHYSICAL_MODELS}
    for row in jsonl(directory / "interval_predictions.jsonl"):
        if row.get("model") in rows:
            rows[row["model"]].append(row)
    keys = [[(row.get("line_id"), row.get("sequence")) for row in values] for values in rows.values()]
    check(bool(keys[0]) and all(len(key) == len(set(key)) and set(key) == set(keys[0]) for key in keys),
          "three physical models share distinct actual intervals")
    check(all(row["completed_at"] <= row["available_at"] <= row["scored_at"] for values in rows.values()
              for row in values), "physical outcome availability")
    model_rows = []
    for line in LINES:
        for period, included in PHYSICAL_PERIODS.items():
            for model in PHYSICAL_MODELS:
                selected = [at(physical, "models", model, "by_line", line, "by_period", name) or {}
                            for name in included]
                model_rows.append([line, period, model, sum(row.get("n", 0) for row in selected),
                                   weighted(selected, "mae_seconds"), weighted(selected, "bias_seconds")])
    return dict(coverage=coverage,
        source_initialization=dict(domain=initialization.get("domain"), used_by_line=initialization.get("used_by_line")),
        predictions=dict(columns=["line", "period", "model", "n", "mae_seconds", "bias_seconds"], rows=model_rows),
        world_covariates=physical.get("world_covariates"))


def _world(summary, arm):
    world = at(summary, "worlds", arm) or {}
    adapter = world.get("local_adapter")
    if adapter is None:
        return dict(local_adapter=None)
    return dict(local_adapter={key: adapter.get(key) for key in
        ("enabled", "ridge", "feature_names", "target_heads", "fit_labels", "fit_rows_by_line", "refits",
         "paired_prequential", "target_semantics", "training_covariates")})


def _local_effects(local, seeds, arms, scenarios):
    index = {(row["seed"], row["scenario"], row["arm"]): row for row in local}
    physical, global_effects = [], []
    for arm in arms:
        for scenario in scenarios:
            for line in LINES:
                for period in PHYSICAL_PERIODS:
                    values = []
                    for seed in seeds:
                        table = at(index.get((seed, scenario, arm)), "physical_intervals", "predictions") or {}
                        rows = {row[2]: row for row in table.get("rows", []) if row[:2] == [line, period]}
                        values.append(difference(at({key: value[4] for key, value in rows.items()}, "local_context"),
                                                 at({key: value[4] for key, value in rows.items()}, "online_linear")))
                    physical.append(dict(arm=arm, scenario=scenario, line=line, period=period,
                        contrast="local_context_minus_online_linear_mae_seconds", **paired(values)))
            if arm not in ("action_model", "action_local"):
                continue
            for phase, included in (("before", ("before",)), ("early_after", ("early_after",)),
                    ("late_after", ("late_after",)), ("all_after", ("early_after", "middle_after", "late_after"))):
                fields = {}
                for name, base_key, calibrated_key in (
                        ("duration_mae_seconds", "base_duration_mae_seconds", "calibrated_duration_mae_seconds"),
                        ("cost_mae_passenger_seconds", "base_cost_mae_passenger_seconds", "calibrated_cost_mae_passenger_seconds"),
                        ("global_head_loss", "base_global_head_loss", "calibrated_global_head_loss")):
                    values = []
                    for seed in seeds:
                        stats = at(index.get((seed, scenario, arm)), "local_adapter", "paired_prequential", "by_phase") or {}
                        rows = [stats.get(part, {}) for part in included]
                        values.append(difference(weighted(rows, calibrated_key), weighted(rows, base_key)))
                    fields[name] = paired(values)
                global_effects.append(dict(arm=arm, scenario=scenario, period=phase,
                    contrast="calibrated_minus_base_same_issue_same_family", **fields))
    return dict(physical_context_minus_online=physical, global_calibrated_minus_base=global_effects)


def analyze(root=HERE / "runs", protocol=None):
    root = Path(root)
    p = protocol or read(HERE / "protocol.json")
    seeds, arms, scenarios = p["training_seeds"], p["arms"], p["scenarios"]
    errors, missing, failures, coverage, cells, reproduction = [], [], [], [], [], []
    controls, windows, initializers = {}, {}, {}
    td, actions, local, contract, global_prediction = [], [], [], [], []
    deadlines = list(range(p["feedback_first_deadline"], p["feedback_last_deadline"] + 1,
                           p["feedback_interval_seconds"]))
    for seed in seeds:
        for scenario in scenarios:
            for arm in arms:
                label = f"control/{scenario}/{arm}/seed{seed}"
                directory = root / label
                summary, run_protocol, failure = [read(directory / name) for name in
                                                 ("summary.json", "protocol.json", "failure.json")]
                if failure:
                    failures.append(dict(cell=label, **failure))
                if summary is None or run_protocol is None:
                    missing.append(label)
                    coverage.append(dict(cell=label, status="failed" if failure else "missing"))
                    continue
                before_errors = len(errors)
                def check(condition, name):
                    if not condition:
                        errors.append(dict(cell=label, check=name))
                check(not failure and summary.get("status") == "complete", "completed summary")
                for field, expected in (("seed", seed), ("scenario", scenario), ("arm", arm),
                        ("target_time_seconds", p["horizon_seconds"]), ("full_horizon_completed", True),
                        ("target_online_training_events", p["target_record_budget"]),
                        ("packets", len(deadlines)), ("policy_updates", p["policy_update_budget"]),
                        ("sumo_version", "SUMO 1.25.0")):
                    check(summary.get(field) == expected, field)
                for field, expected in (("initial_updates", p["initializer_updates"]),
                        ("history_rows", p["history_rows"]), ("source_rows", p["source_rows"])):
                    check(run_protocol.get(field) == expected, field)
                if scenario == "local_speed_drop":
                    check(at(summary, "scenario_metadata", "actual_shock_at_seconds") == p["shock_at_seconds"], "shock clock")
                passengers = summary.get("passengers", {})
                check(passengers.get("active_state_count_balance_error") == 0, "passenger ledger")
                cost, departed = summary.get("generalized_passenger_seconds"), passengers.get("departed_passengers")
                wait, ride = passengers.get("accrued_wait_seconds_total"), passengers.get("accrued_in_vehicle_seconds_total")
                check(all(finite(value) for value in (cost, departed, wait, ride)) and departed > 0
                      and math.isclose(cost, 2 * wait + ride, abs_tol=1e-8), "cost components")
                check(finite(cost) and finite(departed) and departed > 0 and math.isclose(
                      summary.get("generalized_seconds_per_departed_passenger", math.inf), cost / departed,
                      abs_tol=1e-8), "cost denominator")
                receipts = summary.get("feedback_receipts", [])
                check([row.get("deadline") for row in receipts] == deadlines, "feedback deadlines")
                check(all(row.get("records") == p["feedback_records_per_packet"] and
                          row.get("max_available_at", math.inf) <= row.get("deadline", -1) <= row.get("processed_at", -1)
                          for row in receipts), "feedback release/count")
                sequences = [sequence for row in receipts for sequence in row.get("sequence_ids", [])]
                check(len(sequences) == len(set(sequences)) == p["target_record_budget"], "unique global feedback")
                initializers.setdefault(str(seed), set()).add(run_protocol.get("initializer"))
                logs = list(jsonl(directory / "td_diagnostics.jsonl"))
                check(len(logs) == len(deadlines), "one TD diagnostic per packet")
                physical = _physical(directory, summary, check)
                check(summary.get("physical_learning_enters_policy") == (arm == "action_local"), "local policy entry arm")
                world = _world(summary, arm)
                if arm in ("action_model", "action_local"):
                    check(at(world, "local_adapter", "enabled") == (arm == "action_local"), "local readout enabled arm")
                    check(at(world, "local_adapter", "fit_labels") == p["target_record_budget"], "local readout global label budget")
                global_logs = list(jsonl(directory / "prequential.jsonl"))
                expected_predictions = 0 if arm == "action_real" else p["target_record_budget"]
                check(len(global_logs) == expected_predictions, "global issue-time prediction count")
                check(len({row.get("sequence") for row in global_logs}) == expected_predictions,
                      "unique global issue-time predictions")
                check(all(row["completed_at"] <= row["available_at"] <= row["scored_at"] for row in global_logs),
                      "global prediction outcome availability")
                valid = len(errors) == before_errors
                coverage.append(dict(cell=label, status="complete" if valid else "invalid"))
                if not valid:
                    continue
                controls[seed, scenario, arm] = dict(summary=summary, directory=directory)
                cells.append(dict(seed=seed, scenario=scenario, arm=arm,
                    generalized_passenger_seconds=cost, seconds_per_departed_passenger=cost / departed,
                    departed=departed, completed=passengers.get("completed_passengers"),
                    unfinished=passengers.get("unfinished_passengers"), wait_passenger_seconds=wait,
                    ride_passenger_seconds=ride))
                td.append(dict(seed=seed, scenario=scenario, arm=arm, periods=_td_summary(logs, summary, arm)))
                actions.append(dict(seed=seed, scenario=scenario, arm=arm, periods=_action_summary(summary)))
                local.append(dict(seed=seed, scenario=scenario, arm=arm, physical_intervals=physical, **world))
                contract.append(dict(seed=seed, scenario=scenario, arm=arm,
                    execution_contract=summary.get("execution_contract")))
                if global_logs:
                    for phase, (start, stop) in PERIODS.items():
                        selected = [row for row in global_logs if start <= row["issued_at"] < stop]
                        global_prediction.append(dict(seed=seed, scenario=scenario, arm=arm, period=phase,
                            n=len(selected), **{field: mean([row.get(field) for row in selected]) for field in
                               ("selected_score", "duration_absolute_error", "cost_absolute_error")}))
                if arm == "old_model":
                    previous = read(REFERENCE_ROOT / "control" / scenario / "model" / f"seed{seed}" / "summary.json")
                    reproduction.append(dict(seed=seed, scenario=scenario, reference_available=previous is not None,
                        exact_cost=cost == previous.get("generalized_passenger_seconds") if previous else None,
                        cost_delta=difference(cost, previous.get("generalized_passenger_seconds")) if previous else None))
                    check(previous is not None and cost == previous.get("generalized_passenger_seconds"),
                          "old-model passive instrumentation reproduces resident A2 cost")
    for seed, paths in initializers.items():
        if len(paths) != 1:
            errors.append(dict(seed=seed, check="different initializer paths"))
    for seed in seeds:
        for arm in arms:
            stable, shock = controls.get((seed, "stable", arm)), controls.get((seed, "local_speed_drop", arm))
            if stable and shock:
                stable_rows = list(jsonl(stable["directory"] / "metrics.jsonl"))
                shock_rows = list(jsonl(shock["directory"] / "metrics.jsonl"))
                prefix = {row["time_seconds"]: _windows._cost(row) for row in stable_rows
                          if row["time_seconds"] <= p["shock_at_seconds"]}
                if any(row["time_seconds"] in prefix and _windows._cost(row) != prefix[row["time_seconds"]]
                       for row in shock_rows if row["time_seconds"] <= p["shock_at_seconds"]):
                    errors.append(dict(seed=seed, arm=arm, check="stable/shock pre-intervention trajectory differs"))
                comparison = _windows.paired_control(stable_rows, shock_rows)
                complete = all(row["status"] == "complete" for row in comparison["bins"])
                if not complete:
                    errors.append(dict(seed=seed, arm=arm, check="missing fixed-clock endpoints"))
                windows[seed, arm] = dict(seed=seed, arm=arm,
                    stable_absolute=sum(row["stable_rate"] * 300 for row in comparison["bins"]) if complete else None,
                    shock_absolute=sum(row["shock_rate"] * 300 for row in comparison["bins"]) if complete else None,
                    **{key: value for key, value in comparison.items() if key != "bins"})
    effects, window_effects = [], []
    for candidate, reference in CONTRASTS:
        for scenario in scenarios:
            effects.append(dict(candidate=candidate, reference=reference, scenario=scenario,
                **paired([difference(at(controls.get((seed, scenario, candidate)), "summary", "generalized_seconds_per_departed_passenger"),
                                      at(controls.get((seed, scenario, reference)), "summary", "generalized_seconds_per_departed_passenger"))
                          for seed in seeds])))
        for field in ("stable_absolute", "shock_absolute", "excess_generalized_passenger_seconds"):
            window_effects.append(dict(candidate=candidate, reference=reference, quantity=field,
                **paired([difference(at(windows.get((seed, candidate)), field), at(windows.get((seed, reference)), field))
                          for seed in seeds])))
    expected = len(seeds) * len(arms) * len(scenarios)
    return dict(status="complete" if len(controls) == expected and not errors and not missing and not failures else "incomplete",
        seeds=seeds, arms=arms, integrity=dict(expected_runs=expected, complete_controls=len(controls),
            errors=errors, missing=missing, failures=failures, coverage=coverage,
            initializer_paths={key: sorted(paths) for key, paths in initializers.items()}),
        control_cells=cells, full_day_paired_effects=effects, window_paired_effects=window_effects,
        window_cells=list(windows.values()), old_model_reproduction=reproduction,
        td_diagnostics=td, action_statistics=actions, local_prediction_and_world=local,
        local_same_flow_effects=_local_effects(local, seeds, arms, scenarios), global_predictions=global_prediction,
        execution_contract=contract,
        interpretation="Two fixed paired training seeds form a development screen. Report both seed effects and stable/shock absolute costs; no confirmatory p value or significance claim. Local label counts may differ with each policy and are reported explicitly; within-run baseline/calibrated comparisons must use identical issue-time predictions and delayed labels. Actor/action changes demonstrate transmission, not passenger benefit.")


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
