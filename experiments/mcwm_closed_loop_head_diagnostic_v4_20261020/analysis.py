"""Resident-only head diagnosis and explicitly separated development cohorts."""

import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent / "mcwm_closed_loop_action_local_v3_20261020"
V2_ROOT = HERE.parent / "mcwm_closed_loop_diagnostic_v2_20261020"
spec = importlib.util.spec_from_file_location("mcwm_v3_readonly_analysis", V3_ROOT / "analysis.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
read, jsonl, at, finite, mean, difference = v3.read, v3.jsonl, v3.at, v3.finite, v3.mean, v3.difference
COHORTS = {"original_two_screened": [6201, 6202], "fresh_three": [6203, 6204, 6205],
           "all_five_development": [6201, 6202, 6203, 6204, 6205]}
T975 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052}
COST_WINDOWS = {"before": (0., 9000.), "adaptation": (9000., 18000.),
                "post_training_frozen": (18300., 24000.)}


def run_cells(p):
    return p["run_cells"]


def paired(values, seeds):
    result = v3.paired(values)
    result["seeds"] = seeds
    observed = [float(value) for value in values if finite(value)]
    n = len(observed)
    half = T975.get(n - 1, math.nan) * statistics.stdev(observed) / math.sqrt(n) if n >= 2 else math.nan
    result["descriptive_95pct_t_ci"] = [result["mean"] - half, result["mean"] + half] if finite(half) else None
    return result


def window_cost(rows, start, stop):
    costs = {float(row["time_seconds"]): v3._windows._cost(row) for row in rows}
    missing = [time for time in (start, stop) if time not in costs]
    if missing:
        return dict(status="missing_endpoint", missing_times=missing, generalized_passenger_seconds=None)
    value = costs[stop] - costs[start]
    if not finite(value) or value < 0:
        return dict(status="invalid_ledger_increment", generalized_passenger_seconds=None)
    return dict(status="complete", start_seconds=start, stop_seconds=stop,
                generalized_passenger_seconds=value)


def cost_cell(directory, seed, scenario, arm, origin):
    summary = read(directory / "summary.json")
    if summary is None:
        return None
    rows = list(jsonl(directory / "metrics.jsonl"))
    departed = at(summary, "passengers", "departed_passengers")
    last = max((receipt["processed_at"] for receipt in summary.get("feedback_receipts", [])), default=math.inf)
    windows = {}
    for name, (start, stop) in COST_WINDOWS.items():
        window = window_cost(rows, start, stop)
        if name == "post_training_frozen" and last > start:
            window = dict(status="updates_overlap_window", generalized_passenger_seconds=None)
        window["seconds_per_full_day_departed_passenger"] = (
            window["generalized_passenger_seconds"] / departed
            if window["status"] == "complete" and finite(departed) and departed > 0 else None)
        windows[name] = window
    return dict(seed=seed, scenario=scenario, arm=arm, origin=origin,
        seconds_per_departed_passenger=summary.get("generalized_seconds_per_departed_passenger"),
        generalized_passenger_seconds=summary.get("generalized_passenger_seconds"), departed=departed,
        completed=at(summary, "passengers", "completed_passengers"),
        unfinished=at(summary, "passengers", "unfinished_passengers"),
        last_policy_update_at_seconds=last if finite(last) else None, windows=windows)


def _subset(root, p, seeds, arms):
    copy = dict(p, training_seeds=seeds, arms=arms)
    return v3.analyze(root, copy)


def _effect_rows(index, contrasts, scenarios, cohorts=COHORTS):
    result = []
    for cohort, seeds in cohorts.items():
        for candidate, reference in contrasts:
            for scenario in scenarios:
                for quantity, path in [("full_day_seconds_per_departed_passenger", ("seconds_per_departed_passenger",)),
                        *[(name + "_seconds_per_full_day_departed_passenger",
                           ("windows", name, "seconds_per_full_day_departed_passenger")) for name in COST_WINDOWS]]:
                    values = [difference(at(index.get((seed, scenario, candidate)), *path),
                                         at(index.get((seed, scenario, reference)), *path)) for seed in seeds]
                    result.append(dict(cohort=cohort, candidate=candidate, reference=reference,
                        scenario=scenario, quantity=quantity, **paired(values, seeds)))
    return result


def _head_summary(summary, arm, detailed_lines=False):
    heads = at(summary, "worlds", arm, "head_diagnostics")
    if not isinstance(heads, dict):
        return None
    # Every collected statistic answers a per-head failure question. Rich
    # all-family contribution matrices remain in resident issue-time logs.
    result = {key: heads.get(key) for key in ("overall", "by_phase")}
    if detailed_lines:
        result["by_phase_and_line"] = heads.get("by_phase_and_line")
    def convert_metrics(value):
        if not isinstance(value, dict):
            return value
        return {("rmse" if key == "mse" else key): (
            [math.sqrt(max(item, 0.)) if finite(item) else None for item in item_value]
            if key == "mse" and isinstance(item_value, list) else convert_metrics(item_value))
            for key, item_value in value.items()}
    result = convert_metrics(result)
    return result


def _probe_summary(directory, active):
    groups = {}
    for row in jsonl(directory / "td_diagnostics.jsonl"):
        now = row.get("processed_at", row["deadline"])
        phase = ("before" if now < 9000 else "early_after" if now < 12000
                 else "middle_after" if now < 15000 else "late_after")
        probe = row.get("head_counterfactual") or at(row, "diagnostic", "head_counterfactual")
        if probe is None:
            continue
        groups.setdefault(phase, []).append(probe)
    result = {}
    for phase, rows in groups.items():
        variants = {}
        for name in sorted({name for row in rows for name in (row.get("variants") or {})}):
            values = [row["variants"][name] for row in rows if name in row.get("variants", {})]
            total = sum(row.get("rows", 0) for row in values)
            applied = sum(row.get("applied_rows", 0) for row in values)
            metrics = {}
            for field in sorted({field for row in values for field, value in row.items() if isinstance(value, dict)}):
                metrics[field] = {}
                for key in ("mean_delta", "mean_absolute_delta", "max_absolute_delta", "mean", "mean_absolute"):
                    observed = [(row.get("rows", 0), row[field].get(key)) for row in values
                                if isinstance(row.get(field), dict) and finite(row[field].get(key))]
                    if not observed:
                        continue
                    metrics[field][key] = (max(value for n, value in observed) if key == "max_absolute_delta" else
                        sum(n * value for n, value in observed) / sum(n for n, value in observed))
            if active:
                variants[name] = dict(rows=total, applied_rows=applied, **metrics)
            else:
                deltas = [abs(value) for metric in metrics.values() for value in metric.values() if finite(value)]
                variants[name] = dict(rows=total, applied_rows=applied,
                                      max_absolute_delta=max(deltas) if deltas else None)
        result[phase] = dict(packets=len(rows), rows=sum(row.get("rows", 0) for row in rows), variants=variants)
        if not active:
            result[phase]["all_variant_deltas_exact_zero"] = bool(variants) and all(
                value["max_absolute_delta"] == 0 and value["applied_rows"] == 0 for value in variants.values())
    return result


def analyze(root=HERE / "runs", protocol=None, v3_root=V3_ROOT, v2_root=V2_ROOT):
    root, v3_root, v2_root = Path(root), Path(v3_root), Path(v2_root)
    p = protocol or read(HERE / "protocol.json")
    declared = {(row["seed"], row["scenario"], row["arm"]) for row in run_cells(p)}
    expected = {(seed, scenario, arm) for seed in (6201, 6202) for scenario in p["scenarios"]
                for arm in ("action_model", "action_local")} | {
        (seed, scenario, arm) for seed in (6203, 6204, 6205) for scenario in p["scenarios"]
        for arm in ("action_real", "action_model")}
    if declared != expected or len(run_cells(p)) != len(expected):
        raise ValueError("The v4 protocol must declare exactly the twenty development cells")
    head = _subset(root, p, [6201, 6202], ["action_model", "action_local"])
    extension = _subset(root, p, [6203, 6204, 6205], ["action_real", "action_model"])
    original = _subset(v3_root / "runs", p, [6201, 6202], ["action_real", "action_model"])
    errors, references, reproduction, new_cells, stability_cells, head_rows, probes = [], [], [], [], [], [], []
    head_metadata = None
    for cohort, result in (("head_replay", head), ("fresh_extension", extension), ("resident_original", original)):
        if result["status"] != "complete":
            errors.append(dict(scope=cohort, integrity=result["integrity"]))
    for row in run_cells(p):
        seed, scenario, arm = row["seed"], row["scenario"], row["arm"]
        directory = root / "control" / scenario / arm / f"seed{seed}"
        cell = cost_cell(directory, seed, scenario, arm, "v4_passive_replay" if seed <= 6202 else "v4_fresh_extension")
        if cell:
            new_cells.append(cell)
            summary = read(directory / "summary.json")
            if arm != "action_real":
                diagnostic = _head_summary(summary, arm, detailed_lines=seed <= 6202)
                raw_diagnostic = at(summary, "worlds", arm, "head_diagnostics") or {}
                if head_metadata is None:
                    original_heads = at(summary, "worlds", arm, "head_diagnostics") or {}
                    head_metadata = {key: original_heads.get(key) for key in ("head_order", "physical_units", "error_sign",
                        "pairing", "physical_semantics", "decoded_physical_semantics", "covariate_line_order", "feature_names")}
                head_rows.append(dict(seed=seed, scenario=scenario, arm=arm, diagnostics=diagnostic))
                if diagnostic is None:
                    errors.append(dict(scope="head_diagnostic", seed=seed, scenario=scenario, arm=arm, check="missing four-head summary"))
                elif at(raw_diagnostic, "overall", "n") != p["target_record_budget"] or sum(
                        value.get("n", 0) for value in raw_diagnostic.get("by_phase", {}).values()) != p["target_record_budget"]:
                    errors.append(dict(scope="head_diagnostic", seed=seed, scenario=scenario, arm=arm, check="four-head score count"))
                probe_periods = _probe_summary(directory, active=arm == "action_local")
                probes.append(dict(seed=seed, scenario=scenario, arm=arm, periods=probe_periods))
                expected_variants = {"full_base", "omit_logdt", "omit_logcost", "omit_dWaiting", "omit_dOnboard"}
                if sum(value["packets"] for value in probe_periods.values()) != 57 or any(
                        set(value["variants"]) != expected_variants or value["rows"] <= 0
                        or (arm == "action_model" and not value.get("all_variant_deltas_exact_zero"))
                        for value in probe_periods.values()):
                    errors.append(dict(scope="head_counterfactual", seed=seed, scenario=scenario, arm=arm,
                                       check="complete matched probes and disabled zero control"))
            if seed <= 6202:
                prior = read(v3_root / "runs/control" / scenario / arm / f"seed{seed}" / "summary.json")
                fields = ("generalized_passenger_seconds", "generalized_seconds_per_departed_passenger",
                          "policy_updates", "target_online_training_events")
                same = prior is not None and all(summary.get(field) == prior.get(field) for field in fields)
                reproduction.append(dict(seed=seed, scenario=scenario, arm=arm, exact_cost_and_budget=same,
                    cost_delta=difference(summary.get("generalized_passenger_seconds"),
                                          prior.get("generalized_passenger_seconds") if prior else None)))
                if not same:
                    errors.append(dict(scope="passive_reproduction", seed=seed, scenario=scenario, arm=arm))
    for seed in range(6201, 6206):
        for scenario in p["scenarios"]:
            for arm in ("action_real", "action_model"):
                origin = "resident_v3_original" if seed <= 6202 else "v4_fresh_extension"
                directory = (v3_root / "runs" if seed <= 6202 else root) / "control" / scenario / arm / f"seed{seed}"
                cell = cost_cell(directory, seed, scenario, arm, origin)
                if cell:
                    stability_cells.append(cell)
            directory = v2_root / "runs/control" / scenario / "model" / f"seed{seed}"
            prior = read(directory / "summary.json")
            check = prior is not None and prior.get("status") == "complete" and all(prior.get(field) == value for field, value in (
                ("target_time_seconds", p["horizon_seconds"]), ("policy_updates", p["policy_update_budget"]),
                ("target_online_training_events", p["target_record_budget"]), ("full_horizon_completed", True)))
            references.append(dict(seed=seed, scenario=scenario, available=prior is not None, contract_valid=check))
            if not check:
                errors.append(dict(scope="resident_old_A2", seed=seed, scenario=scenario))
            cell = cost_cell(directory, seed, scenario, "old_model", "resident_v2_old_A2")
            if cell:
                stability_cells.append(cell)
    for cell in [*new_cells, *stability_cells]:
        for window, value in cell["windows"].items():
            if value["status"] != "complete":
                errors.append(dict(scope="fixed_clock_cost", seed=cell["seed"], scenario=cell["scenario"],
                                   arm=cell["arm"], origin=cell["origin"], window=window, **value))
    stability = {(row["seed"], row["scenario"], row["arm"]): row for row in stability_cells}
    replay = {(row["seed"], row["scenario"], row["arm"]): row for row in new_cells}
    # Check shared initialization across the original and new development cohorts.
    initializer_paths = {}
    for result in (head, extension, original):
        for seed, paths in result["integrity"]["initializer_paths"].items():
            initializer_paths.setdefault(seed, set()).update(paths)
    for seed, paths in initializer_paths.items():
        if len(paths) != 1:
            errors.append(dict(scope="initializer_paths", seed=seed, paths=sorted(paths)))
    return dict(status="complete" if not errors and len(new_cells) == 20 and len(stability_cells) == 30 else "incomplete",
        integrity=dict(expected_new_runs=20, complete_new_runs=len(new_cells), errors=errors,
            new_batches={name: result["integrity"] for name, result in (("head_replay", head), ("fresh_extension", extension))},
            resident_original_status=original["status"], old_A2_references=references,
            initializer_paths={seed: sorted(paths) for seed, paths in initializer_paths.items()}),
        passive_reproduction=reproduction, new_control_cells=new_cells, stability_control_cells=stability_cells,
        stability_paired_effects=_effect_rows(stability, (("action_model", "action_real"), ("action_model", "old_model")), p["scenarios"]),
        local_replay_paired_effects=_effect_rows(replay, (("action_local", "action_model"),), p["scenarios"],
                                               {"original_two_passive_replay": [6201, 6202]}),
        head_diagnostic_metadata=head_metadata, head_diagnostics=head_rows, head_counterfactuals=probes,
        head_counterfactual_metadata=dict(sign="actual calibrated minus full-base or single-head removal",
            phase_clock="optimizer processed_at; final18000 receipt with processing lag belongs to late_after",
            heads=["logdt", "logcost", "dWaiting", "dOnboard"],
            modeled_duration_enters_bellman_discount=False,
            semantics="Matched generated states, executed actions, decoder/target draws and fixed actual trust. Next action differences respond to predicted next state and do not include a shadow optimizer update."),
        td_diagnostics=head["td_diagnostics"] + extension["td_diagnostics"],
        action_statistics=head["action_statistics"] + extension["action_statistics"],
        local_prediction_and_world=head["local_prediction_and_world"] + extension["local_prediction_and_world"],
        local_same_flow_effects=head["local_same_flow_effects"],
        recovery_window_cells=head["window_cells"] + extension["window_cells"],
        interpretation="Development evidence only. Original two seeds were screened in v3; fresh three are reported separately before the five-seed aggregate. Original references are read-only. Four-head residuals use saved issue-time predictions before outcome release or fitting. Per-head TD omissions are passive counterfactuals on the same model batch and cannot establish passenger benefit. Post-training18300–24000 is an actual2wait+ride ledger increment after all optimizer updates, normalized by the full-day departed cohort.")


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
