"""Compact resident analysis of v12 state interactions and native paired truth."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ENDPOINTS = ("global_next", "same_vehicle_next", "fixed300")
MODELS = ("legacy", "baseline", "residual", "interaction")
GLOBAL_MODELS = ("base_analytic", "global58", *MODELS)
METRICS = ("dt", "wait", "ride", "cost")
MAX_BYTES = 200000
PAIR_EFFECT_ORDER = ("predicted_gain", "gain_signed_error", "direction_correct", "regret")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compact(value):
    if isinstance(value, float):
        return float(format(value, ".8g"))
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def errors(values, keys):
    n = len(values)
    return dict(n=n, mae=[sum(abs(row[key]) for row in values)/n for key in keys] if n else None,
        mse=[sum(row[key]**2 for row in values)/n for key in keys] if n else None)


def moments(values):
    return dict(n=len(values), mean=sum(values)/len(values) if values else None,
        mean_absolute=sum(abs(value) for value in values)/len(values) if values else None)


def compact_global_gains(value):
    """One model order replaces repeated moment labels without dropping any value."""
    output = dict(raw_available=[value["raw_available"].get(model, 0) for model in GLOBAL_MODELS])
    for key in ("matched_raw", "matched_decoded"):
        rows = [value[key][model] for model in GLOBAL_MODELS]
        output[key] = dict(n=rows[0]["n"], mean=[row["mean"] for row in rows],
                           mean_absolute=[row["mean_absolute"] for row in rows])
    return output


def endpoint_truth(record, name):
    value = (record.get("truth") or {}).get(name)
    return value if isinstance(value, dict) and value.get("status") == "complete" else None


def prediction_endpoint(prediction, model, name, action):
    if model in ("base_analytic", "global58"):
        if name != "global_next":
            return None
        return prediction.get("global_comparator", {}).get("arms", {}).get(model, {}).get(action)
    return prediction.get(model, {}).get(action, {}).get("endpoints", {}).get(name)


def predicted_gain(prediction, model, name, *, raw=False):
    if model in ("base_analytic", "global58"):
        if name != "global_next":
            return None
        key = "raw_cost_gain_low_minus_high" if raw else "cost_gain_low_minus_high"
        value = prediction.get("global_comparator", {}).get("arms", {}).get(model, {}).get(key)
    else:
        candidate = prediction.get(model, {})
        value = candidate.get("raw_global_next_gain") if raw else candidate.get("gains", {}).get(name)
    return value if finite(value) else None


def duration_group(record):
    actual = endpoint_truth(record, "global_next")
    if actual is None or not finite(actual.get("dt")):
        return "missing_global"
    return "zero_dt" if actual["dt"] == 0 else "positive_dt"


def summarize_rows(rows):
    result = dict(n=len(rows), actions=dict(Counter(
        "low" if abs(row["hold_seconds"]) < 1e-5 else "high" for row in rows)), endpoints={})
    for endpoint in ENDPOINTS:
        models = GLOBAL_MODELS if endpoint == "global_next" else MODELS
        metrics = ("dt", "cost") if endpoint == "global_next" else METRICS
        matched = {model: [] for model in models}
        available = Counter()
        missing_truth = 0
        for row in rows:
            actual = endpoint_truth(row, endpoint)
            if actual is None or not all(finite(actual.get(key)) for key in metrics):
                missing_truth += 1
                continue
            action = "low" if abs(row["hold_seconds"]) < 1e-5 else "high"
            candidates = {model: prediction_endpoint(row.get("prediction", {}), model, endpoint, action)
                          for model in models}
            valid = {model: value for model, value in candidates.items()
                     if isinstance(value, dict) and all(finite(value.get(key)) for key in metrics)}
            available.update(valid.keys())
            if len(valid) == len(models):
                for model, value in valid.items():
                    matched[model].append({key: value[key]-actual[key] for key in metrics})
        result["endpoints"][endpoint] = dict(missing_truth=missing_truth,
            available={model: available[model] for model in models},
            matched_errors={model: errors(values, metrics) for model, values in matched.items()})
    raw, decoded = {model: [] for model in GLOBAL_MODELS}, {model: [] for model in GLOBAL_MODELS}
    raw_available = Counter()
    for row in rows:
        pred = row.get("prediction", {})
        values = {model: predicted_gain(pred, model, "global_next", raw=True) for model in GLOBAL_MODELS}
        raw_available.update(model for model, value in values.items() if value is not None)
        if all(value is not None for value in values.values()):
            for model, value in values.items():
                raw[model].append(value)
        values = {model: predicted_gain(pred, model, "global_next") for model in GLOBAL_MODELS}
        if all(value is not None for value in values.values()):
            for model, value in values.items():
                decoded[model].append(value)
    result["global_gains"] = compact_global_gains(dict(raw_available={model: raw_available[model] for model in GLOBAL_MODELS},
        matched_raw={model: moments(values) for model, values in raw.items()},
        matched_decoded={model: moments(values) for model, values in decoded.items()}))
    return result


def expected_paths(root, p, *, preflight=False):
    root = Path(root)
    if preflight:
        settings = p["preflight"]
        cell = dict(seed=settings["seed"], scenario=settings["scenario"])
        days = [(cell, root/"day")]
        pairs = [(dict(**cell, target_line=settings.get("target_line", "7X"), anchor_index=0, branch=branch),
                  root/"pair"/branch) for branch in ("low", "high")]
    else:
        days = [(dict(seed=seed, scenario=scenario), root/"runs"/scenario/f"seed{seed}")
                for seed, scenario in product(p["training_seeds"], p["scenarios"])]
        seed = int(p.get("paired_seed", 6201))
        pairs = [(dict(seed=seed, scenario=scenario, target_line=line, anchor_index=index, branch=branch),
                  root/"pairs"/scenario/line/f"seed{seed}"/f"state{index}"/branch)
                 for scenario, line, index, branch in product(p["scenarios"], p["target_lines"],
                     range(p.get("paired_anchors_per_line", 2)), ("low", "high"))]
    return days, pairs


def load_result(path):
    if (path/"summary.json").is_file():
        return json.loads((path/"summary.json").read_text())
    if (path/"failure.json").is_file():
        failure = json.loads((path/"failure.json").read_text())
        return dict(status="failed", failure={key: failure.get(key) for key in (
            "type", "message", "time", "error_type", "error", "reason") if key in failure})
    return dict(status="missing")


def preflight_compute_budget(root):
    """Report observed new-version preflight work, including failed attempts."""
    attempts = []
    for attempt in sorted(Path(root).glob("preflight*")):
        if not attempt.is_dir():
            continue
        runs = []
        paths = [attempt/"day", *sorted((attempt/"pair").glob("*"))]
        for path in paths:
            if not (path/"summary.json").is_file() and not (path/"failure.json").is_file():
                continue
            value = load_result(path)
            budget = value.get("budget", {})
            is_day = path == attempt/"day"
            native = budget.get("native_seconds") if is_day else None
            if not is_day and all(finite(budget.get(key)) for key in ("prefix_sim_seconds", "suffix_sim_seconds")):
                native = budget["prefix_sim_seconds"]+budget["suffix_sim_seconds"]
            if native is None:
                native = value.get("failure", {}).get("time")
            runs.append(dict(run=str(path.relative_to(attempt)), status=value.get("status"),
                native_seconds=native,
                target_labels=budget.get("target_labels") if is_day else None,
                replay_target_labels=budget.get("replay_target_labels") if not is_day else None,
                policy_updates=budget.get("policy_updates" if is_day else "replay_policy_updates"),
                suffix_seconds=budget.get("new_native_suffix_seconds" if is_day else "suffix_sim_seconds")))
        if runs:
            attempts.append(dict(attempt=attempt.name, runs=runs))
    return attempts


def fixed_clock(rows):
    observed, truth_bad = 0, 0
    model = {name: dict(predictions=0, incorrect_duration=0, unsupported=0) for name in MODELS}
    for row in rows:
        truth = endpoint_truth(row, "fixed300")
        observed += truth is not None
        truth_bad += truth is not None and truth.get("dt") != 300.
        for name in MODELS:
            for action in ("low", "high"):
                value = prediction_endpoint(row.get("prediction", {}), name, "fixed300", action)
                if value is None:
                    model[name]["unsupported"] += 1
                else:
                    model[name]["predictions"] += 1
                    model[name]["incorrect_duration"] += value.get("dt") != 300.
    return dict(observed=observed, incorrect_truth_duration=truth_bad, models=model)


def direction(gain):
    return 1 if gain > 0 else -1 if gain < 0 else 0


def effect_metrics(predicted, actual):
    if predicted is None or actual is None:
        return None
    chooses_high = predicted > 0
    return dict(predicted_gain=predicted, true_gain=actual, gain_signed_error=predicted-actual,
        direction_correct=direction(predicted) == direction(actual),
        predicted_tie=predicted == 0, true_tie=actual == 0,
        selected_action="high" if chooses_high else "low",
        regret=max(0., -actual) if chooses_high else max(0., actual))


def stored_effect(predicted, actual):
    value = effect_metrics(predicted, actual)
    return {key: value[key] for key in PAIR_EFFECT_ORDER} \
        if value is not None else None


def compact_pair_tables(pairs):
    """Retain every paired metric with one schema rather than thousands of repeated labels."""
    for pair in pairs:
        sections = [*pair["effects"].values(), pair["raw_global_effects"], pair["global_to_fixed300_direction_proxy"]]
        for section in sections:
            for model, value in section.items():
                section[model] = [value[key] for key in PAIR_EFFECT_ORDER] if value is not None else None


def state_match(left, right):
    return bool(left) and left == right


def training_zero(value):
    delta = value.get("suffix", {}).get("training_delta")
    if isinstance(delta, dict):
        return bool(delta) and all(finite(number) and number == 0 for number in delta.values())
    return finite(delta) and delta == 0


def interaction_diagnostic(prediction):
    """Keep additive contrasts, never fitted coefficients or native states."""
    value = prediction.get("interaction_diag")
    if not isinstance(value, dict):
        return dict(state_values=None, endpoints={endpoint: dict(status="missing") for endpoint in ENDPOINTS})
    endpoints = value.get("endpoints", value)
    result = {}
    for endpoint in ENDPOINTS:
        item = endpoints.get(endpoint)
        if not isinstance(item, dict):
            result[endpoint] = dict(status="missing")
            continue
        result[endpoint] = {key: item.get(key) for key in ("status", "analytic_zero_duration") if key in item}
        for key in ("baseline", "line_holding", "raw_total", "analytic_adjustment", "total",
                    "fixed_weight_no_state_interactions", "raw_fixed_weight_no_state_interactions"):
            component = item.get(key)
            result[endpoint][key] = component.get("cost") if isinstance(component, dict) else None
        result[endpoint]["state_interactions"] = ({
            key: value.get("cost") for key, value in item["state_interactions"].items()}
            if isinstance(item.get("state_interactions"), dict) else None)
        components = [item.get("baseline"), item.get("line_holding")]
        interactions = item.get("state_interactions")
        if isinstance(interactions, dict):
            components.extend(interactions.values())
        target = item.get("raw_total", item.get("total"))
        valid = bool(components) and isinstance(interactions, dict) and all(
            isinstance(value, dict) and all(finite(value.get(key)) for key in METRICS)
            for value in components+[target])
        reconstruction = {key: sum(value[key] for value in components)-target[key] for key in METRICS} if valid else None
        result[endpoint]["reconstruction_error"] = reconstruction["cost"] if reconstruction is not None else None
        result[endpoint]["max_component_reconstruction_error"] = max(abs(value) for value in reconstruction.values()) \
            if reconstruction is not None else None
        adjustment = item.get("analytic_adjustment")
        total = item.get("total")
        result[endpoint]["analytic_reconstruction_error"] = target["cost"]+adjustment["cost"]-total["cost"] if valid and all(
                isinstance(value, dict) and all(finite(value.get(key)) for key in METRICS)
                for value in (adjustment, total)) else None
    return dict(state_values=value.get("state_values"), endpoints=result)


def compact_design(model):
    """The actual support/rank evidence is small; raw training arrays are resident."""
    result = {key: model.get(key) for key in (
        "design", "endpoint_designs", "interaction_designs", "contrast_designs", "contrast_design", "endpoint_contrast_designs",
        "interaction_features", "feature_names", "contrast_feature_names", "perline_state_ranges") if key in model}
    for key in ("design", "contrast_design"):
        if isinstance(result.get(key), dict):
            result[key] = {name: value for name, value in result[key].items()
                           if name not in ("feature_names", "shared_interaction_columns")}
    for key in ("endpoint_designs", "interaction_designs", "contrast_designs", "endpoint_contrast_designs"):
        if isinstance(result.get(key), dict):
            result[key] = {endpoint: {name: value for name, value in design.items()
                if name not in ("feature_names", "shared_interaction_columns")}
                for endpoint, design in result[key].items()}
    return result


def analyze_pair(cell, low, high, anchor):
    predictions = low.get("prediction", {})
    complete = low.get("status") == high.get("status") == "complete"
    checks = dict(branches_complete=complete,
        branch_integrity=complete and all(value.get("integrity", {}).get("all_pass") is True for value in (low, high)),
        full_native_prefix_exact=all(value.get("integrity", {}).get("checks", {}).get("full_native_prefix_exact") is True
                                     for value in (low, high)),
        anchor_present=isinstance(anchor, dict),
        common_state=state_match(low.get("selected", {}), high.get("selected", {})),
        common_prefix=bool(low.get("prefix")) and low.get("prefix") == high.get("prefix"),
        common_prediction=bool(predictions) and predictions == high.get("prediction"),
        anchor_state=isinstance(anchor, dict) and all(state_match(value.get("selected", {}), anchor.get("selected", {})) for value in (low, high)),
        anchor_prefix=isinstance(anchor, dict) and bool(anchor.get("prefix")) and all(value.get("prefix") == anchor["prefix"] for value in (low, high)),
        anchor_prediction=isinstance(anchor, dict) and bool(anchor.get("prediction")) and all(value.get("prediction") == anchor["prediction"] for value in (low, high)),
        suffix_zero_training=all(training_zero(value) for value in (low, high)),
        suffix_exact300=all(value.get("suffix", {}).get("seconds") == 300. for value in (low, high)),
        fixed300_truth=all(endpoint_truth(value, "fixed300") is not None and endpoint_truth(value, "fixed300").get("dt") == 300. for value in (low, high)),
        global_truth=all(endpoint_truth(value, "global_next") is not None for value in (low, high)))
    checks["actual_low_high_commands"] = all(
        value.get("execution", {}).get("requested_hold_seconds") == expected
        and value.get("execution", {}).get("executed_hold_seconds") == expected
        for value, expected in ((low, 0.), (high, 7.5)))
    checks["same_external_seed"] = low.get("prefix", {}).get("external_seed") is not None \
        and low.get("prefix", {}).get("external_seed") == high.get("prefix", {}).get("external_seed")
    if cell.get("anchor_index", 0) == 0:
        checks["v11_reference_truth_exact"] = all(value.get("integrity", {}).get("checks", {}).get(
            "v11_reference_truth_exact") is True for value in (low, high))
    # Sibling controls within the intervention batch must keep their nominal actions.
    siblings = [value.get("native", {}).get("other_batch_actions") for value in (low, high)]
    checks["same_batch_other_actions"] = siblings[0] is not None and siblings[0] == siblings[1]
    outcome, effects = {}, {}
    for endpoint in ENDPOINTS:
        actual = {branch: endpoint_truth(value, endpoint) for branch, value in (("low", low), ("high", high))}
        truth_gain = actual["low"]["cost"]-actual["high"]["cost"] if all(
            value is not None and finite(value.get("cost")) for value in actual.values()) else None
        outcome[endpoint] = dict(status="complete" if truth_gain is not None else "missing",
            branches={branch: {key: value.get(key) for key in METRICS} if value is not None else None
                      for branch, value in actual.items()}, true_gain=truth_gain)
        models = GLOBAL_MODELS if endpoint == "global_next" else MODELS
        effects[endpoint] = {model: stored_effect(predicted_gain(predictions, model, endpoint), truth_gain)
                             for model in models}
    low_global, high_global = endpoint_truth(low, "global_next"), endpoint_truth(high, "global_next")
    if low_global is None or high_global is None:
        duration = "missing_global"
    elif low_global["dt"] == high_global["dt"] == 0:
        duration = "zero_dt"
    elif low_global["dt"] > 0 and high_global["dt"] > 0:
        duration = "positive_dt"
    else:
        duration = "mixed_dt"
    raw = {model: stored_effect(predicted_gain(predictions, model, "global_next", raw=True),
                                 outcome["global_next"]["true_gain"]) for model in GLOBAL_MODELS}
    proxy = {model: stored_effect(predicted_gain(predictions, model, "global_next"),
                                   outcome["fixed300"]["true_gain"])
             for model in ("base_analytic", "global58")}
    return dict(**cell, status="complete" if complete else "missing_or_failed", duration=duration,
        integrity=dict(all_pass=all(checks.values()), checks=checks),
        v11_reference_truth_exact=checks.get("v11_reference_truth_exact"),
        v11_reference_truth_status="checked" if cell.get("anchor_index", 0) == 0 else "not_previously_paired",
        branch_status={"low": low.get("status"), "high": high.get("status")},
        common_prefix={key: low.get("prefix", {}).get(key) for key in (
            "external_seed", "initial_updates", "freeze_clock", "native_steps", "issued_count")},
        prefix_matched_issues={branch: value.get("native", {}).get("prefix_matched_issues")
            for branch, value in (("low", low), ("high", high))},
        suffix_training_delta={branch: value.get("suffix", {}).get("training_delta")
            for branch, value in (("low", low), ("high", high))},
        execution={branch: {key: value.get("execution", {}).get(key) for key in (
            "requested_hold_seconds", "executed_hold_seconds")}
            for branch, value in (("low", low), ("high", high))},
        truth=outcome, effects=effects, raw_global_effects=raw,
        interaction_diagnostic=interaction_diagnostic(predictions),
        global_to_fixed300_direction_proxy=proxy,
        budget={branch: value.get("budget", {}) for branch, value in (("low", low), ("high", high))})


def summarize_effects(pairs):
    output = {}
    for endpoint in ENDPOINTS:
        models = GLOBAL_MODELS if endpoint == "global_next" else MODELS
        common = [pair for pair in pairs if pair["integrity"]["all_pass"]
                  and all(pair["effects"][endpoint][model] is not None for model in models)]
        output[endpoint] = dict(matched_n=len(common), unsupported_missing_or_invalid=len(pairs)-len(common), models={})
        for model in models:
            values = [pair["effects"][endpoint][model] for pair in common]
            n = len(values)
            output[endpoint]["models"][model] = dict(n=n,
                gain_mae=sum(abs(value["gain_signed_error"]) for value in values)/n if n else None,
                gain_mse=sum(value["gain_signed_error"]**2 for value in values)/n if n else None,
                direction_correct=sum(value["direction_correct"] for value in values),
                predicted_ties=sum(value["predicted_gain"] == 0 for value in values),
                true_ties=sum(pair["truth"][endpoint]["true_gain"] == 0 for pair in common),
                regret_total=sum(value["regret"] for value in values),
                regret_mean=sum(value["regret"] for value in values)/n if n else None)
    return output


def state_changes(pairs):
    """Within each fitted flow, compare both anchors on one line without requiring a truth flip."""
    grouped = defaultdict(dict)
    for pair in pairs:
        grouped[(pair["seed"], pair["scenario"], pair["line"])][pair["anchor_index"]] = pair
    changes = []
    for (seed, scenario, line), values in sorted(grouped.items()):
        first, second = values.get(0), values.get(1)
        endpoints = {}
        for endpoint in ("fixed300",):
            actuals = [item["truth"][endpoint]["true_gain"] if item is not None
                       and item["integrity"]["all_pass"] else None for item in (first, second)]
            actual_supported = all(value is not None for value in actuals)
            models = GLOBAL_MODELS if endpoint == "global_next" else MODELS
            model_changes = {}
            for model in models:
                effects = [item["effects"][endpoint][model] if item is not None
                           and item["integrity"]["all_pass"] else None for item in (first, second)]
                predicted = [effect["predicted_gain"] if effect is not None else None for effect in effects]
                supported = actual_supported and all(value is not None for value in predicted)
                delta = predicted[1]-predicted[0] if supported else None
                model_changes[model] = dict(status="complete" if supported else "unsupported_or_missing",
                    predicted_gain_change=delta,
                    change_direction_correct=direction(delta) == direction(actuals[1]-actuals[0]) if supported else None,
                    predicted_direction_flip=direction(predicted[0]) != direction(predicted[1]) if supported else None)
            endpoints[endpoint] = dict(status="complete" if actual_supported else "missing_or_invalid",
                true_gain_change=actuals[1]-actuals[0] if actual_supported else None,
                true_direction_flip=direction(actuals[0]) != direction(actuals[1]) if actual_supported else None,
                models=model_changes)
        changes.append(dict(seed=seed, scenario=scenario, line=line, anchors_present=sorted(values), endpoints=endpoints))
    return changes


def stability_comparison(rows, pairs, target_lines):
    """Matched stable heldout prediction errors and paired action regret determine separate risks."""
    result = []
    for line in target_lines:
        selected = [row for row in rows if row["scenario"] == "stable" and row["line_id"] == line
                    and row["role"] == "heldout"]
        paired = [pair for pair in pairs if pair["scenario"] == "stable" and pair["line"] == line]
        errors_by_endpoint = summarize_rows(selected)["endpoints"]
        gains = summarize_effects(paired)
        endpoints = {}
        for endpoint in ENDPOINTS:
            err = errors_by_endpoint[endpoint]["matched_errors"]
            metrics = ("dt", "cost") if endpoint == "global_next" else METRICS
            cost_index = metrics.index("cost")
            comparisons = {}
            for comparator in ("baseline", "residual"):
                current, prior = err["interaction"], err[comparator]
                current_gain = gains[endpoint]["models"]["interaction"]
                prior_gain = gains[endpoint]["models"][comparator]
                comparisons[comparator] = dict(
                    matched_prediction_n=current["n"],
                    cost_mae_change=current["mae"][cost_index]-prior["mae"][cost_index]
                        if current["mae"] is not None else None,
                    matched_pair_n=current_gain["n"],
                    direction_correct_change=current_gain["direction_correct"]-prior_gain["direction_correct"],
                    regret_change=current_gain["regret_total"]-prior_gain["regret_total"]
                        if current_gain["n"] else None)
            endpoints[endpoint] = comparisons
        result.append(dict(line=line, endpoints=endpoints))
    return result


def summarize_fixed_weight_ablation(pairs):
    """Do the new interaction terms improve the same fitted model's action contrast?"""
    result = {}
    for endpoint in ENDPOINTS:
        matched = []
        for pair in pairs:
            if not pair["integrity"]["all_pass"]:
                continue
            diag = pair["interaction_diagnostic"]["endpoints"][endpoint]
            actual = pair["truth"][endpoint]["true_gain"]
            deleted = diag.get("fixed_weight_no_state_interactions")
            full = pair["effects"][endpoint]["interaction"]
            if full is not None and finite(deleted) and actual is not None:
                matched.append((full, effect_metrics(deleted, actual)))
        models = {}
        for index, name in enumerate(("interaction", "fixed_weight_without_interactions")):
            values = [row[index] for row in matched]
            models[name] = dict(n=len(values),
                gain_mae=sum(abs(value["gain_signed_error"]) for value in values)/len(values) if values else None,
                direction_correct=sum(value["direction_correct"] for value in values),
                regret_total=sum(value["regret"] for value in values))
        result[endpoint] = dict(matched_n=len(matched), models=models,
            action_choice_changed=sum((full["predicted_gain"] > 0) != (deleted["predicted_gain"] > 0)
                                      for full, deleted in matched))
    return result


def analyze(root, p=None, *, preflight=False):
    p = json.loads((HERE/"protocol.json").read_text()) if p is None else p
    day_paths, pair_paths = expected_paths(root, p, preflight=preflight)
    days = [(cell, load_result(path)) for cell, path in day_paths]
    branches = [(cell, load_result(path)) for cell, path in pair_paths]
    rows, cells, anchors = [], [], {}
    for cell, value in days:
        evaluations = value.get("evaluations", [])
        anchors[(cell["seed"], cell["scenario"])] = value.get("anchors", {})
        rows.extend(dict(**row, scenario=cell["scenario"], seed=cell["seed"]) for row in evaluations)
        model = value.get("residual_model", {})
        heldout_actions = {line: {action: sum(row["line_id"] == line and row["role"] == "heldout"
            and (abs(row["hold_seconds"]) < 1e-5) == (action == "low") for row in evaluations)
            for action in ("low", "high")} for line in p["target_lines"]}
        coverage = value.get("coverage", {})
        coverage = dict(train=coverage.get("train"), heldout={key: coverage.get("heldout", {}).get(key)
            for key in ("start_after", "latest_issue", "required_per_line", "complete", "coverage", "missing_by_line", "statuses")}
            if coverage.get("heldout") else None)
        budget = {key: item for key, item in value.get("budget", {}).items() if key != "pending_sequence_ids"}
        cells.append(dict(**cell, status=value.get("status"), integrity=value.get("integrity", {}),
            budget=budget, initializer=value.get("initializer"), coverage=coverage,
            heldout_actions=heldout_actions,
            residual_design=model.get("design"), endpoint_designs=model.get("endpoint_designs"),
            endpoint_coverage=model.get("endpoint_coverage"),
            interaction_model=compact_design(value.get("interaction_model", {})),
            action_difference=value.get("action_difference"),
            anchors_present={line: len(value.get("anchors", {}).get(line, [])) for line in p["target_lines"]},
            v11_matched_issues=value.get("native", {}).get("v11_matched_issues"),
            fixed300_clock=fixed_clock(evaluations), failure=value.get("failure")))
    grouped = defaultdict(list)
    for row in rows:
        phase = "before_shock_clock" if row.get("issued_at", 0) < p.get("shock_at_seconds", 9000) else "after_shock_clock"
        grouped[(row["scenario"], row["line_id"], row["role"], phase, duration_group(row))].append(row)
    strata = [dict(scenario=scenario, line=line, role=role, phase=phase, duration=duration, **summarize_rows(values))
              for (scenario, line, role, phase, duration), values in sorted(grouped.items())]
    paired = defaultdict(dict)
    for cell, value in branches:
        paired[(cell["seed"], cell["scenario"], cell["target_line"], cell["anchor_index"])][cell["branch"]] = value
    pair_results = []
    for (seed, scenario, line, index), values in sorted(paired.items()):
        candidates = anchors.get((seed, scenario), {}).get(line, [])
        anchor = candidates[index] if index < len(candidates) else None
        pair_results.append(analyze_pair(dict(seed=seed, scenario=scenario, line=line, anchor_index=index),
            values.get("low", {}), values.get("high", {}), anchor))
    pair_strata = [dict(scenario=scenario, line=line, duration=duration,
                       **summarize_effects([pair for pair in pair_results if pair["scenario"] == scenario
                                            and pair["line"] == line and pair["duration"] == duration]))
                   for scenario, line, duration in sorted({(pair["scenario"], pair["line"], pair["duration"]) for pair in pair_results})]
    checks = dict(all_days_complete=all(value.get("status") == "complete" for _, value in days),
        all_days_integrity=all(value.get("integrity", {}).get("all_pass") is True for _, value in days),
        all_days_native_prefix_recorded=all(value.get("integrity", {}).get("checks", {}).get("prefix_trace_recorded") is True
                                            for _, value in days),
        v11_common_flow_exact=all(value.get("integrity", {}).get("checks", {}).get("v11_common_flow_exact") is True
                                 for _, value in days),
        v11_old_predictions_exact=all(value.get("integrity", {}).get("checks", {}).get("v11_old_predictions_exact") is True
                                    for _, value in days),
        equal_training_slot_budget=all(value.get("budget", {}).get("target_labels") ==
            (p["preflight"]["target_record_budget"] if preflight else p["target_record_budget"]) for _, value in days),
        all_pairs_integrity=bool(pair_results) and all(pair["integrity"]["all_pass"] for pair in pair_results),
        fixed300_truth_exact=all(cell["fixed300_clock"]["incorrect_truth_duration"] == 0 for cell in cells),
        fixed300_residual_exact=all(cell["fixed300_clock"]["models"][model]["incorrect_duration"] == 0
                                   for cell in cells for model in ("baseline", "residual", "interaction")))
    # Same-car endpoints may lie beyond the finite suffix and remain missing.
    missing_same_car = sum(pair["truth"]["same_vehicle_next"]["status"] != "complete" for pair in pair_results)
    main_budget = sum(value.get("budget", {}).get("target_labels", 0) for _, value in days)
    extra_budget = {key: sum(value.get("budget", {}).get(key, 0) for _, value in branches)
                    for key in ("prefix_sim_seconds", "suffix_sim_seconds", "replay_target_labels",
                                "replay_policy_updates", "truth_labels_enter_training")}
    review = dict(all_pairs_have_same_car_truth=missing_same_car == 0,
        all_pairs_have_fixed300_truth=all(pair["truth"]["fixed300"]["status"] == "complete" for pair in pair_results),
        all_lines_scenes_have_pairs=bool(pair_results) and all(pair["status"] == "complete" for pair in pair_results),
        heldout_both_actions_every_line_day=all(cell["heldout_actions"][line][action] > 0
            for cell in cells for line in p["target_lines"] for action in ("low", "high")),
        fixed300_model_supported=all(pair["effects"]["fixed300"]["interaction"] is not None for pair in pair_results))
    terminal = all(value.get("status") != "missing" for _, value in days+branches)
    result = dict(status="complete" if terminal else "incomplete", mode="preflight" if preflight else "development",
        integrity=dict(all_pass=all(checks.values()), checks=checks),
        schema=dict(global_metric_order=["dt", "cost"], endpoint_metric_order=list(METRICS),
                    global_gain_model_order=list(GLOBAL_MODELS),
                    pair_effect_order=list(PAIR_EFFECT_ORDER),
                    interaction_diagnostic_unit="physical low-minus-high passenger cost gain; clocks verified separately",
                    gain_sign="low cost minus high cost; positive chooses 7.5 seconds; ties choose zero",
                    numeric_precision_significant_digits=8),
        cells=cells, strata=strata, pairs=pair_results, pair_strata=pair_strata,
        paired_overall=summarize_effects(pair_results),
        paired_by_state_index={str(index): {"fixed300": summarize_effects([pair for pair in pair_results if pair["anchor_index"] == index])["fixed300"]}
                               for index in sorted({pair["anchor_index"] for pair in pair_results})},
        within_line_state_changes=state_changes(pair_results),
        stable_degradation=stability_comparison(rows, pair_results, p["target_lines"]),
        fixed_weight_ablation=dict(overall=summarize_fixed_weight_ablation(pair_results),
            by_scenario={scenario: summarize_fixed_weight_ablation([pair for pair in pair_results if pair["scenario"] == scenario])
                         for scenario in p["scenarios"]}),
        budget=dict(main_unique_training_slots=main_budget, expected_unique_training_slots=
                    len(days)*(p["preflight"]["target_record_budget"] if preflight else p["target_record_budget"]),
                    extra_native_branch_budget=extra_budget,
                    expected_suffix_seconds=len(branches)*300,
                    repeated_prefix_labels_are_replay_not_new_training_labels=True),
        missing=dict(day_statuses=dict(Counter(value.get("status") for _, value in days)),
                     branch_statuses=dict(Counter(value.get("status") for _, value in branches)),
                     same_car_pair_truth=missing_same_car),
        independent_confirmation=dict(status="not_submitted", review_ready=all(checks.values()) and all(review.values()),
                                      checks=review, decision="Review the small paired pilot before authorizing independent confirmation"),
        interpretation=dict(comparator="Passive 58-column global model, old endpoint fit, issue baseline, 13-column residual, and state-interaction residual on each exactly replayed v11 flow.",
            matching="Chosen MAE/MSE and aggregate gain errors compare identical supported records within each endpoint; availability and missing truth remain explicit.",
            endpoints="58-column/base global predictions have no same-car or fixed300 target. Their direction against fixed300 is a cross-endpoint proxy only.",
            raw="Zero-duration raw global response is separate from analytic zero. Positive-duration day gains lack paired truth and are descriptive.",
            scope="Twelve development states at one seed; no deployed candidate control comparison or independent confirmation.",
            interactions="State 0/1 changes compare anchors under one fitted flow. Fixed-weight deletion is an action-contrast decomposition, not a separately refitted model; truth is not required to flip sign."))
    if preflight:
        result = {key: result[key] for key in ("status", "mode", "integrity", "budget", "missing")}
        result["day_integrity"] = [cell["integrity"] for cell in cells]
        result["pairs"] = [{key: pair[key] for key in ("seed", "scenario", "line", "anchor_index", "status", "duration",
                           "integrity", "execution", "branch_status", "prefix_matched_issues")} for pair in pair_results]
    else:
        result["preflight_compute_budget"] = preflight_compute_budget(root)
        compact_pair_tables(result["pairs"])
    return compact(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    p = json.loads((HERE/"protocol.json").read_text())
    started = time.monotonic()
    if args.wait:
        days, pairs = expected_paths(args.root, p, preflight=args.preflight)
        paths = [path for _, path in days+pairs]
        while True:
            terminal = sum((path/"summary.json").is_file() or (path/"failure.json").is_file() for path in paths)
            print("SCHEDULEURM_PROGRESS "+json.dumps(dict(phase="waiting_v12_native_truth", current=terminal,
                total=len(paths), unit="runs", elapsed_seconds=time.monotonic()-started)), flush=True)
            if terminal == len(paths) or time.monotonic()-started >= 72*3600:
                break
            time.sleep(60)
    value = analyze(args.root, p, preflight=args.preflight)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
    if len(payload.encode()) > (20000 if args.preflight else MAX_BYTES):
        raise RuntimeError("Reduce the resident analysis before collection; compact summary exceeds its byte budget")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(json.dumps(dict(phase="summary_written", status=value["status"], bytes=args.out.stat().st_size)), flush=True)


if __name__ == "__main__":
    main()
