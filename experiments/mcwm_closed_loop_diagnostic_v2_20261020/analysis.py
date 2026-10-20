"""Compact, seed-paired analysis of resident diagnostic JSON files."""

import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
REFERENCE_ROOT = HERE.parent / "mcwm_closed_loop_formal_20261020/runs"
_spec = importlib.util.spec_from_file_location("original_mcwm_analysis", HERE.parent / "mcwm_closed_loop_formal_20261020/analysis.py")
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
GLOBAL_MODELS = ("frozen_model_reference", "online", "context")
PHYSICAL_MODELS = ("frozen_source_prior", "online_linear", "local_context")
LINES = ("7X", "102X", "705X")
PERIODS = ("before", "early_after", "late_after", "all_after")
PHYSICAL_PERIODS = dict(before=("before",), early_after=("early",),
                        late_after=("late",), all_after=("early", "middle", "late"))
CONTRASTS = (("real075", "real1"), ("model", "real075"), ("context", "model"),
             ("model", "real1"), ("context", "real1"))
HEADS = ("global", "local", "count", "line", "station", "neighbor")
T95 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def jsonl(path):
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = [value for value in values if finite(value)]
    return statistics.mean(values) if values else None


def paired(values):
    observed = [float(value) for value in values if finite(value)]
    n = len(observed)
    average = mean(observed)
    half = T95[n-1]*statistics.stdev(observed)/math.sqrt(n) if n > 1 else None
    return dict(values=values, n_pairs=n, mean=average,
                ci95=[average-half, average+half] if half is not None else None,
                negative=sum(value < 0 for value in observed), positive=sum(value > 0 for value in observed))


def difference(candidate, reference):
    return candidate-reference if finite(candidate) and finite(reference) else None


def at(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def weighted(records, field):
    count = sum(record.get("n", 0) for record in records)
    if not count or any(record.get("n", 0) and not finite(record.get(field)) for record in records):
        return None
    return sum(record.get("n", 0)*(record.get(field) or 0.) for record in records)/count


def _feedback_coverage(summary):
    metadata = summary.get("global_feedback_coverage")
    result = []
    for phase in ("before", "after"):
        if metadata is None:
            result.append([phase, *([None]*9)])
            continue
        groups = [group for name, group in metadata.items() if (name == "before") == (phase == "before")]
        by_line = [sum(group.get("by_line", {}).get(line, 0) for group in groups) for line in LINES]
        upstream = [sum(group.get("directly_affected_upstream_by_line", {}).get(line, 0) for group in groups)
                    for line in LINES]
        result.append([phase, sum(group.get("records", 0) for group in groups), sum(by_line),
            sum(group.get("directly_affected_upstream_records", 0) for group in groups), *by_line, *upstream])
    return result


def _td_values(row):
    d = row.get("diagnostic") or {}
    real_batches, model_batches = d.get("real_batches", []), d.get("model_batches", [])
    batch_mean = lambda batches, key: weighted([
        dict(n=batch.get("rows", 0), value=at(batch, key, "mean")) for batch in batches], "value")
    return [at(d, "absolute_trust", "mean"), at(d, "absolute_trust", "p05"),
        at(d, "absolute_trust", "p95"), d.get("effective_model_mean_weight_scale"),
        d.get("real_td_loss"), d.get("real_td_contribution"), d.get("model_td_loss_trust_weighted"),
        d.get("model_td_contribution"), at(d, "gradients", "real_scaled_norm"),
        at(d, "gradients", "model_scaled_norm"),
        at(d, "gradients", "model_to_real_scaled_norm_ratio"), at(d, "gradients", "cosine_real_model"),
        at(d, "gradients", "cosine_combined_real"), batch_mean(real_batches, "target_q"),
        batch_mean(model_batches, "target_q"), batch_mean(real_batches, "reward"),
        batch_mean(model_batches, "reward"), batch_mean(real_batches, "dt_seconds"),
        batch_mean(model_batches, "dt_seconds"), at(d, "actor_parameter_change", "l2"),
        at(d, "critic_parameter_change", "l2"), row.get("proposed_action_change_seconds"),
        row.get("executed_action_change_seconds")]


TD_FIELDS = ("trust_mean", "trust_p05", "trust_p95", "effective_model_scale", "real_td_loss",
    "real_td_contribution", "weighted_model_td_loss", "model_td_contribution", "scaled_real_gradient_norm",
    "scaled_model_gradient_norm", "model_real_gradient_norm_ratio", "real_model_gradient_cosine",
    "combined_real_gradient_cosine", "real_target_q_mean", "model_target_q_mean", "real_reward_mean",
    "model_reward_mean", "real_dt_mean", "model_dt_mean", "actor_parameter_change_l2",
    "critic_parameter_change_l2", "proposed_action_change_seconds", "executed_action_change_seconds")


def _sequence_check(directory, filename, models, model_key, errors, label, expected=None):
    rows = {model: [] for model in models}
    for row in jsonl(directory / filename):
        if row.get(model_key) in rows:
            rows[row[model_key]].append(row)
    keys = [[(row.get("line_id"), row["sequence"]) for row in rows[model]] for model in models]
    ok = all(keys) and all(len(key) == len(set(key)) for key in keys) and all(set(key) == set(keys[0]) for key in keys[1:])
    if expected is not None:
        ok = ok and all(len(key) == expected for key in keys)
    for model_rows in rows.values():
        if any(not (row["completed_at"] <= row["available_at"] <= row["scored_at"]) for row in model_rows):
            ok = False
    if ok:
        shared_fields = ("issued_at", "completed_at", "available_at")
        if filename == "interval_predictions.jsonl":
            shared_fields += ("actual_seconds", "holding_seconds", "nonholding_seconds", "trip_id", "vehicle_id")
        reference = {(row.get("line_id"), row["sequence"]): row for row in rows[models[0]]}
        if any(any(row.get(field) != reference[row.get("line_id"), row["sequence"]].get(field)
                   for field in shared_fields) for model in models[1:] for row in rows[model]):
            ok = False
    if not ok:
        errors.append(dict(cell=label, check=filename + " common sequences or availability"))
    return rows, ok


def analyze(root=HERE / "runs"):
    root = Path(root)
    p = PROTOCOL
    seeds, arms, scenarios = p["training_seeds"], p["arms"], p["scenarios"]
    errors, missing, failures, coverage = [], [], [], []
    controls, predictions, global_rows, physical_valid, physical_logs = {}, {}, {}, {}, {}
    td = dict(columns=["seed", "scenario", "arm", "period", "packets", *TD_FIELDS,
        "negative_gradient_cosine_fraction", "model_real_gradient_norm_ratio_max", "actor_updated_fraction"], rows=[])
    actions = dict(columns=["seed", "scenario", "arm", "period", "events", "proposed_hold_seconds",
                           "executed_hold_seconds", "execution_changed_fraction"], rows=[])
    global_coverage = dict(columns=["seed", "scenario", "mode", "arm", "issue_period", "records",
        "affected_line_records", "direct_upstream_records", *[line+"_records" for line in LINES],
        *[line+"_direct_upstream_records" for line in LINES]], rows=[],
        semantics="Only the selected285 global learning records, grouped by issue time before/after9000; missing metadata is null, observed absence is zero. Separate physical diagnostic labels do not enter these counts.")
    control_cells, reproduction, initializers, lags = [], [], {}, []
    deadlines = list(range(p["feedback_first_deadline"], p["feedback_last_deadline"]+1, p["feedback_interval_seconds"]))
    for seed in seeds:
        for scenario in scenarios:
            for mode, arm in [("control", arm) for arm in arms] + [("prediction", "common")]:
                label = f"{mode}/{scenario}/{arm}/seed{seed}"
                directory = root / label
                summary, protocol, failure = read(directory / "summary.json"), read(directory / "protocol.json"), read(directory / "failure.json")
                if failure is not None:
                    failures.append(dict(cell=label, **failure))
                if summary is None or protocol is None:
                    missing.append(label)
                    coverage.append(dict(cell=label, status="failed" if failure else "missing"))
                    continue
                before_errors = len(errors)
                def check(condition, name):
                    if not condition:
                        errors.append(dict(cell=label, check=name))
                check(not failure and summary.get("status") == "complete", "completed summary")
                for field, expected in (("seed", seed), ("scenario", scenario), ("arm", arm), ("mode", mode),
                        ("target_time_seconds", 24000), ("full_horizon_completed", True),
                        ("target_online_training_events", 285), ("packets", 57),
                        ("policy_updates", 456 if mode == "control" else 0), ("sumo_version", "SUMO 1.25.0")):
                    check(summary.get(field) == expected, field)
                for field, expected in (("initial_updates", 50000), ("history_rows", 32768), ("source_rows", 4096)):
                    check(protocol.get(field) == expected, field)
                check(at(summary, "passengers", "active_state_count_balance_error") == 0, "passenger ledger")
                if scenario == "local_speed_drop":
                    check(at(summary, "scenario_metadata", "actual_shock_at_seconds") == 9000, "shock clock")
                passengers = summary.get("passengers", {})
                departed = passengers.get("departed_passengers")
                cost = summary.get("generalized_passenger_seconds")
                wait, ride = passengers.get("accrued_wait_seconds_total"), passengers.get("accrued_in_vehicle_seconds_total")
                check(all(finite(value) for value in (cost, wait, ride, departed)) and departed > 0
                      and math.isclose(cost, 2*wait+ride, abs_tol=1e-8), "cost components")
                if finite(cost) and finite(departed) and departed > 0:
                    check(math.isclose(summary.get("generalized_seconds_per_departed_passenger", math.inf),
                                       cost/departed, abs_tol=1e-8), "cost denominator")
                receipts = summary.get("feedback_receipts", [])
                check([r.get("deadline") for r in receipts] == deadlines, "receipt deadlines")
                check(all(r.get("records") == 5 and r.get("max_available_at", math.inf) <= r.get("deadline", -1)
                    <= r.get("processed_at", -1) for r in receipts), "receipt availability and count")
                sequences = [seq for receipt in receipts for seq in receipt.get("sequence_ids", [])]
                check(len(sequences) == len(set(sequences)) == 285, "distinct feedback records")
                lags.extend(r["processed_at"]-r["deadline"] for r in receipts
                            if finite(r.get("processed_at")) and finite(r.get("deadline")))
                initializers.setdefault(str(seed), set()).add(protocol.get("initializer"))
                valid = len(errors) == before_errors
                coverage.append(dict(cell=label, status="complete" if valid else "invalid"))
                if not valid:
                    continue
                global_coverage["rows"].extend([seed, scenario, mode, arm, *row] for row in _feedback_coverage(summary))
                if mode == "control":
                    controls[seed, scenario, arm] = dict(summary=summary, directory=directory)
                    control_cells.append([seed, scenario, arm, cost,
                        summary["generalized_seconds_per_departed_passenger"], departed,
                        passengers.get("completed_passengers"), passengers.get("unfinished_passengers"), wait, ride])
                    old_arm = {"real1": "replay", "model": "online", "context": "context"}.get(arm)
                    if old_arm:
                        previous = read(REFERENCE_ROOT / "control" / scenario / old_arm / f"seed{seed}" / "summary.json")
                        reproduction.append(dict(seed=seed, scenario=scenario, arm=arm, reference_available=previous is not None,
                            exact_cost=(cost == previous.get("generalized_passenger_seconds")) if previous else None,
                            cost_delta=difference(cost, previous.get("generalized_passenger_seconds")) if previous else None,
                            exact_feedback=(summary["target_online_training_events"] == previous.get("target_online_training_events")) if previous else None))
                    logs = list(jsonl(directory / "td_diagnostics.jsonl"))
                    check(len(logs) == 57, "one TD diagnostic per packet")
                    for phase in ("before", "after"):
                        selected = [row for row in logs if (row.get("processed_at", row["deadline"]) < 9000) == (phase == "before")]
                        values = [_td_values(row) for row in selected]
                        if arm in ("real1", "real075"):
                            for value in values:
                                value[10] = value[11] = None
                        cosines = [value[11] for value in values if finite(value[11])]
                        ratios = [value[10] for value in values if finite(value[10])]
                        update_records = [value.get("train/actor_updated", {}) for name, value in summary.get("update_statistics", {}).items()
                                          if (name == "before") == (phase == "before")]
                        td["rows"].append([seed, scenario, arm, phase, len(selected),
                            *[mean([row[index] for row in values]) for index in range(len(TD_FIELDS))],
                            sum(value < 0 for value in cosines)/len(cosines) if cosines else None,
                            max(ratios) if ratios else None, weighted(update_records, "mean")])
                        action_records = [value for name, value in summary.get("action_statistics", {}).items()
                                          if (name == "before") == (phase == "before")]
                        action_n = sum(at(value, "proposed_hold_seconds", "n") or 0 for value in action_records)
                        actions["rows"].append([seed, scenario, arm, phase, action_n,
                            *[weighted([value.get(field, {}) for value in action_records], "mean") for field in
                              ("proposed_hold_seconds", "executed_hold_seconds", "execution_changed")]])
                else:
                    predictions[seed, scenario] = summary
                    rows, paired_global = _sequence_check(directory, "prequential.jsonl", GLOBAL_MODELS,
                        "model_arm", errors, label, expected=285)
                    global_rows[seed, scenario] = (rows, paired_global)
                    interval_rows, physical_valid[seed, scenario] = _sequence_check(directory, "interval_predictions.jsonl",
                        PHYSICAL_MODELS, "model", errors, label)
                    physical_logs[seed, scenario] = interval_rows[PHYSICAL_MODELS[0]]
                    check(summary.get("physical_learning_enters_policy") is False, "physical isolation")
    for seed, paths in initializers.items():
        if len(paths) != 1:
            errors.append(dict(seed=seed, check="different initializer paths"))
    full_effects, window_effects, windows = [], [], {}
    for seed in seeds:
        for arm in arms:
            stable, shock = controls.get((seed, "stable", arm)), controls.get((seed, "local_speed_drop", arm))
            if stable and shock:
                stable_metrics = list(jsonl(stable["directory"] / "metrics.jsonl"))
                shock_metrics = list(jsonl(shock["directory"] / "metrics.jsonl"))
                prefix = {row["time_seconds"]: _original._cost(row) for row in stable_metrics if row["time_seconds"] <= 9000}
                if any(row["time_seconds"] in prefix and _original._cost(row) != prefix[row["time_seconds"]]
                       for row in shock_metrics if row["time_seconds"] <= 9000):
                    errors.append(dict(seed=seed, arm=arm, check="stable-shock pre-intervention trajectory differs"))
                pair = _original.paired_control(stable_metrics, shock_metrics)
                complete = all(bin["status"] == "complete" for bin in pair["bins"])
                if not complete:
                    errors.append(dict(seed=seed, arm=arm, check="missing fixed-clock metric endpoints"))
                windows[seed, arm] = dict(seed=seed, arm=arm,
                    stable_absolute=sum(bin["stable_rate"]*300 for bin in pair["bins"]) if complete else None,
                    shock_absolute=sum(bin["shock_rate"]*300 for bin in pair["bins"]) if complete else None,
                    **{key: value for key, value in pair.items() if key != "bins"})
    for candidate, reference in CONTRASTS:
        for scenario in scenarios:
            values = []
            for seed in seeds:
                c, r = controls.get((seed, scenario, candidate)), controls.get((seed, scenario, reference))
                values.append(difference(at(c, "summary", "generalized_seconds_per_departed_passenger"),
                                         at(r, "summary", "generalized_seconds_per_departed_passenger")))
            full_effects.append(dict(candidate=candidate, reference=reference, scenario=scenario, **paired(values)))
        for field in ("stable_absolute", "shock_absolute", "excess_generalized_passenger_seconds"):
            window_effects.append(dict(candidate=candidate, reference=reference, quantity=field,
                **paired([difference(at(windows.get((seed, candidate)), field),
                                     at(windows.get((seed, reference)), field)) for seed in seeds])))
    global_columns = ["seed", "scenario", "period", "model", "n", "score", *["head_"+name for name in HEADS],
                      "duration_mae_seconds", "cost_mae", "direct_family_fraction"]
    global_table, global_values = dict(columns=global_columns, rows=[]), {}
    for (seed, scenario), (rows, valid) in global_rows.items():
        for model in GLOBAL_MODELS:
            for phase, (start, stop) in _original.PERIODS.items():
                selected = [row for row in rows[model] if start <= row["issued_at"] < stop]
                fields = [[row.get("selected_score") for row in selected]]
                fields += [[at(row, "selected_head_losses", head) for row in selected] for head in HEADS]
                fields += [[row.get(field) for row in selected] for field in ("duration_absolute_error", "cost_absolute_error")]
                values = [mean(column) if len([v for v in column if finite(v)]) == len(selected) else None for column in fields]
                family = mean([float(row.get("selected_family_at_issue") == "direct") for row in selected])
                global_table["rows"].append([seed, scenario, phase, model, len(selected), *values, family])
                global_values[seed, scenario, phase, model] = values if valid else [None]*len(values)
    global_effects = []
    for scenario in scenarios:
        for phase in PERIODS:
            for candidate, reference in (("online", "frozen_model_reference"), ("context", "frozen_model_reference"), ("context", "online")):
                effects = {}
                for name, index in (("score", 0), ("duration_mae_seconds", 7), ("cost_mae", 8)):
                    effects[name] = paired([difference(global_values.get((seed, scenario, phase, candidate), [None]*9)[index],
                        global_values.get((seed, scenario, phase, reference), [None]*9)[index]) for seed in seeds])
                global_effects.append(dict(scenario=scenario, period=phase, candidate=candidate, reference=reference, **effects))
    family_decomposition = _family_decomposition(global_rows, seeds, scenarios)
    physical = _physical_analysis(predictions, physical_valid, seeds, scenarios, errors)
    physical["matched_trip_shock_effects"] = _matched_trip_analysis(physical_logs, seeds)
    source = read(root.parent / "inputs/source_summary.json")
    if not source or source.get("status") != "complete" or source.get("target_rows") != 0 or source.get("policy_updates") != 0:
        errors.append(dict(check="source-only interval calibration summary missing or invalid"))
    expected = len(seeds)*len(scenarios)*(len(arms)+1)
    return dict(status="complete" if len(coverage) == expected and not errors and not missing and not failures else "incomplete",
        seeds=seeds, integrity=dict(expected_runs=expected, complete_controls=len(controls), complete_predictions=len(predictions),
            coverage=coverage, errors=errors, missing=missing, failures=failures,
            initializer_paths={key: sorted(value) for key, value in initializers.items()},
            feedback_processing_lag_seconds=dict(min=min(lags) if lags else None, max=max(lags) if lags else None)),
        control_cells=dict(columns=["seed", "scenario", "arm", "generalized_passenger_seconds", "seconds_per_departed_passenger",
            "departed", "completed", "unfinished", "wait_passenger_seconds", "ride_passenger_seconds"], rows=control_cells),
        full_day_paired_effects=full_effects, window_paired_effects=window_effects, window_cells=list(windows.values()),
        original_formal_reproduction=reproduction, td_diagnostics=td, actions=actions,
        global_feedback_coverage=global_coverage,
        global_predictions=global_table, global_paired_effects=global_effects,
        global_context_family_decomposition=family_decomposition, physical=physical, source_calibration=source,
        interpretation="Each contrast uses up to five paired seeds in the listed seed order; 95% paired t intervals are unadjusted. Missing values remain null. Absolute shock cost must accompany shock-minus-stable excess. No early degradation is distinct from right censoring. TD telemetry samples the first update of57 packets; actor_updated covers all456 updates and action changes cover each packet's8 updates. TD model/real batch means are unpaired. Local interval context is distinct from global A3 context.")


def _physical_analysis(predictions, valid, seeds, scenarios, errors):
    labels = dict(columns=["seed", "scenario", "line", "period", "n", "dt_seconds", "holding_seconds", "nonholding_seconds"], rows=[])
    models = dict(columns=["seed", "scenario", "line", "period", "model", "n", "mae_seconds", "bias_seconds"], rows=[])
    counts, initializations, label_values, model_values = [], [], {}, {}
    for (seed, scenario), summary in predictions.items():
        physical = summary.get("physical_intervals") or {}
        coverage, initialization = physical.get("coverage", {}), physical.get("initialization", {})
        counts.append(dict(seed=seed, scenario=scenario, **coverage))
        initializations.append(dict(seed=seed, scenario=scenario, domain=initialization.get("domain"),
            available_by_line=initialization.get("available_by_line"), used_by_line=initialization.get("used_by_line")))
        if initialization.get("domain") != "source" or any(at(initialization, "used_by_line", line) is None
                or not 5 <= initialization["used_by_line"][line] <= 64 for line in LINES):
            errors.append(dict(seed=seed, scenario=scenario, check="physical source provenance or coverage"))
        if coverage.get("missing_saved_prediction") != 0:
            errors.append(dict(seed=seed, scenario=scenario, check="missing physical saved predictions"))
        for line in LINES:
            for phase, included in PHYSICAL_PERIODS.items():
                records = [at(physical, "labels", "by_line", line, "by_period", name) or {} for name in included]
                n = sum(record.get("n", 0) for record in records)
                values = [weighted(records, "mean_"+field+"_seconds") for field in ("duration", "holding", "nonholding")]
                labels["rows"].append([seed, scenario, line, phase, n, *values])
                label_values[seed, scenario, line, phase] = values
                for model in PHYSICAL_MODELS:
                    records = [at(physical, "models", model, "by_line", line, "by_period", name) or {} for name in included]
                    n = sum(record.get("n", 0) for record in records)
                    values = [weighted(records, field) for field in ("mae_seconds", "bias_seconds")]
                    models["rows"].append([seed, scenario, line, phase, model, n, *values])
                    model_values[seed, scenario, line, phase, model] = values if valid.get((seed, scenario)) else [None, None]
    shock_effects, adaptation_effects = [], []
    for line in LINES:
        for phase in PHYSICAL_PERIODS:
            effects = {field: paired([difference(label_values.get((seed, "local_speed_drop", line, phase), [None]*3)[index],
                        label_values.get((seed, "stable", line, phase), [None]*3)[index]) for seed in seeds])
                       for index, field in enumerate(("dt_seconds", "holding_seconds", "nonholding_seconds"))}
            shock_effects.append(dict(line=line, period=phase, **effects))
            for scenario in scenarios:
                for candidate, reference in (("online_linear", "frozen_source_prior"),
                        ("local_context", "frozen_source_prior"), ("local_context", "online_linear")):
                    effects = {field: paired([difference(model_values.get((seed, scenario, line, phase, candidate), [None]*2)[index],
                        model_values.get((seed, scenario, line, phase, reference), [None]*2)[index]) for seed in seeds])
                        for index, field in enumerate(("mae_seconds", "bias_seconds"))}
                    adaptation_effects.append(dict(line=line, scenario=scenario, period=phase,
                        candidate=candidate, reference=reference, **effects))
    initialization = at(next(iter(predictions.values()), {}), "physical_intervals", "initialization")
    source_labels = at(initialization, "labels") or {}
    return dict(labels=labels, predictions=models, coverage=counts, initialization_by_run=initializations,
        source_label_statistics=dict(overall=source_labels.get("overall"),
            by_line={line: at(source_labels, "by_line", line, "overall") for line in LINES}), source_features=at(initialization, "features"),
        stable_shock_effects=shock_effects, prediction_paired_effects=adaptation_effects,
        periods="before6k–9k; early9k–12k; late15k–18k; all_after combines early/middle/late, excludes crossing. Coverage retains crossing/drain separately. Bias is predicted minus actual; bias deltas are signed, not an improvement score.")


def _matched_trip_analysis(logs, seeds):
    phases = ("before", "crossing", "early", "middle", "late", "all_after")
    fields = ("actual_seconds", "holding_seconds", "nonholding_seconds")
    rows, values, unmatched = [], {}, []
    for seed in seeds:
        stable = {(row["line_id"], row["trip_id"], row["vehicle_id"]): row
                  for row in logs.get((seed, "stable"), [])}
        shock = {(row["line_id"], row["trip_id"], row["vehicle_id"]): row
                 for row in logs.get((seed, "local_speed_drop"), [])}
        for line in LINES:
            stable_keys = {key for key in stable if key[0] == line}
            shock_keys = {key for key in shock if key[0] == line}
            unmatched.append(dict(seed=seed, line=line, stable_intervals=len(stable_keys),
                shock_intervals=len(shock_keys), stable_without_shock=len(stable_keys-shock_keys),
                shock_without_stable=len(shock_keys-stable_keys)))
            for phase in phases:
                selected = {key: row for key, row in shock.items() if key[0] == line and
                    (row.get("period") in ("early", "middle", "late") if phase == "all_after" else row.get("period") == phase)}
                matched = sorted(set(selected) & set(stable))
                delta = [mean([selected[key][field]-stable[key][field] for key in matched]) for field in fields]
                values[seed, line, phase] = delta
                rows.append([seed, line, phase, len(matched), len(selected)-len(matched), *delta])
    effects = [dict(line=line, period=phase, **{field: paired([values[seed, line, phase][index] for seed in seeds])
        for index, field in enumerate(("dt_seconds", "holding_seconds", "nonholding_seconds"))}) for line in LINES for phase in phases]
    return dict(columns=["seed", "line", "shock_period", "matched_trips", "shock_without_stable",
                         "delta_dt_seconds", "delta_holding_seconds", "delta_nonholding_seconds"], rows=rows,
        paired_seed_effects=effects, unmatched_by_seed_line=unmatched,
        semantics="One frozen_source_prior log per real direct interval; matched by line/trip/vehicle and assigned by shock interval period. Nonholding includes motion/signals/next-stop service, not pure speed. Stable-only trajectories have no shock period and remain outside these period means.")


def _family_decomposition(global_rows, seeds, scenarios):
    rows, values = [], {}
    for (seed, scenario), (models, valid) in global_rows.items():
        online = {row["sequence"]: row for row in models["online"] if 9000 <= row["issued_at"] < 18000}
        context = {row["sequence"]: row for row in models["context"]}
        samples, switched = [], 0
        for sequence, reference in online.items():
            candidate = context.get(sequence, {})
            family = reference.get("selected_family_at_issue")
            head_losses = at(candidate, "family_head_losses", family) or {}
            fixed_score = mean(list(head_losses.values())) if len(head_losses) == 6 else at(candidate, "scores", family)
            samples.append([candidate.get("selected_score"), fixed_score, reference.get("selected_score"),
                candidate.get("cost_absolute_error"), at(candidate, "family_physical_errors", family, "cost_absolute_error"),
                reference.get("cost_absolute_error")])
            switched += candidate.get("selected_family_at_issue") != family
        averaged = [mean([sample[index] for sample in samples]) if samples
            and all(finite(sample[index]) for sample in samples) else None for index in range(6)]
        rows.append([seed, scenario, len(samples), *averaged, switched/len(samples) if samples else None])
        values[seed, scenario] = averaged if valid else [None]*6
    effects = []
    for scenario in scenarios:
        for name, indices in (("actual_context_minus_context_using_online_family", (0, 1, 3, 4)),
                              ("context_using_online_family_minus_online", (1, 2, 4, 5))):
            effects.append(dict(scenario=scenario, contrast=name,
                score=paired([difference(values.get((seed, scenario), [None]*6)[indices[0]],
                    values.get((seed, scenario), [None]*6)[indices[1]]) for seed in seeds]),
                cost_mae=paired([difference(values.get((seed, scenario), [None]*6)[indices[2]],
                    values.get((seed, scenario), [None]*6)[indices[3]]) for seed in seeds])))
    return dict(columns=["seed", "scenario", "n_all_after", "context_actual_score", "context_online_family_score", "online_score",
        "context_actual_cost_mae", "context_online_family_cost_mae", "online_cost_mae", "different_family_fraction"], rows=rows,
        paired_effects=effects, semantics="Secondary all-after comparison on identical saved predictions; the fixed family is A2's issue-time selection, never the best family after observing truth.")
