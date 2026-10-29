"""Resident-only matched context and eight-update shadow diagnosis."""

import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4_ROOT = HERE.parent / "mcwm_closed_loop_head_diagnostic_v4_20261020"
spec = importlib.util.spec_from_file_location("mcwm_v4_readonly_analysis_v5", V4_ROOT / "analysis.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
read, jsonl, at, finite, difference = v4.read, v4.jsonl, v4.at, v4.finite, v4.difference
WINDOWS = {"adaptation": (9000., 18000.), "post_training_frozen": (18300., 24000.)}
BRANCHES = {"Base", "Local", "Local-W", "Local-C"}


def run_cells(p, stage="stage_a"):
    cells = p["run_cells" if stage == "stage_a" else "frozen_run_cells"]
    arm = "action_local" if stage == "stage_a" else "frozen_world"
    expected = {(seed, scenario, arm) for seed in (6201, 6202)
                for scenario in ("stable", "local_speed_drop")}
    declared = {(row["seed"], row["scenario"], row["arm"]) for row in cells}
    if declared != expected or len(cells) != 4:
        raise ValueError("Each v5 stage must retain the four fixed seed/scenario cells")
    return cells


def result_name(stage):
    return "context_shadow_summary.json" if stage == "stage_a" else "frozen_world_summary.json"


def cost_cell(directory, summary):
    departed = at(summary, "passengers", "departed_passengers")
    rows = list(jsonl(directory / "metrics.jsonl"))
    last = max((row["processed_at"] for row in summary.get("feedback_receipts", [])), default=math.inf)
    windows = {}
    for name, (start, stop) in WINDOWS.items():
        value = v4.window_cost(rows, start, stop)
        if name == "post_training_frozen" and last > start:
            value = dict(status="updates_overlap_window", generalized_passenger_seconds=None)
        value["seconds_per_full_day_departed_passenger"] = (
            value["generalized_passenger_seconds"] / departed
            if value["status"] == "complete" and finite(departed) and departed > 0 else None)
        windows[name] = value
    return dict(generalized_passenger_seconds=summary.get("generalized_passenger_seconds"),
        seconds_per_departed_passenger=summary.get("generalized_seconds_per_departed_passenger"),
        departed=departed, completed=at(summary, "passengers", "completed_passengers"),
        unfinished=at(summary, "passengers", "unfinished_passengers"),
        wait_passenger_seconds=at(summary, "passengers", "accrued_wait_seconds_total"),
        ride_passenger_seconds=at(summary, "passengers", "accrued_in_vehicle_seconds_total"),
        last_active_update_at=last if finite(last) else None, windows=windows)


def diagnostic_checks(context, shadow, anchors):
    """Incomplete matching cannot support a context or optimizer conclusion."""
    checks = {}
    historical = context.get("historical", [])
    future_context = context.get("next_packet_frozen_anchor", [])
    checks["context_branches"] = context.get("branches") == ["base", "saved", "current"]
    checks["context_four_historical_anchors"] = [row.get("deadline") for row in historical] == anchors
    checks["context_historical_rows"] = all(row.get("n", 0) > 0 for row in historical)
    checks["context_four_future_packets"] = [row.get("deadline") for row in future_context] == anchors
    checks["context_future_rows"] = bool(future_context) and all(row.get("n") == 5 for row in future_context)
    checks["context_future_evaluator_frozen"] = bool(future_context) and all(
        row.get("current_context_fixed_at_anchor") is True and row.get("evaluator_fixed_at_anchor") is True
        and row.get("scored_before_next_packet_training") is True for row in future_context)
    checks["context_no_pending_packet"] = context.get("pending_anchor_deadline") is None
    forks = shadow.get("anchors", [])
    future_shadow = shadow.get("next_packet", [])
    checks["shadow_four_anchors"] = [row.get("deadline") for row in forks] == anchors
    checks["shadow_eight_updates"] = bool(forks) and all(
        row.get("steps") == 8 and row.get("active_model_batches") == 8 for row in forks)
    checks["shadow_local_exact_active"] = bool(forks) and all(row.get("local_exact_active") is True for row in forks)
    checks["shadow_fixed_actual_trust"] = bool(forks) and all(row.get("trust_weights_fixed_to_active") is True for row in forks)
    checks["shadow_four_branches"] = bool(forks) and all(set(row.get("branches", {})) == BRANCHES for row in forks)
    checks["shadow_four_future_packets"] = [row.get("anchor_deadline") for row in future_shadow] == anchors
    checks["shadow_common_future_targets"] = bool(future_shadow) and all(
        row.get("rows") == 5 and row.get("common_frozen_anchor_target") is True
        and set(row.get("branches", {})) == BRANCHES for row in future_shadow)
    checks["shadow_no_pending_packets"] = shadow.get("pending_anchor_count") == 0
    return checks


def _compact_context(context):
    # Detailed physical/decoder moments belong to the overall comparison.
    # Secondary slices retain count and head losses without duplicating them.
    def compact_slices(groups):
        return {name: dict(n=value.get("n"), branches={branch: {
            key: metric.get(key) for key in ("six_loss_mean", "g4_mse")}
            for branch, metric in value.get("branches", {}).items()})
            for name, value in groups.items()}
    result = {key: value for key, value in context.items()
              if key not in ("historical", "next_packet_frozen_anchor")}
    for field in ("historical", "next_packet_frozen_anchor"):
        result[field] = []
        for row in context.get(field, []):
            retained = {key: value for key, value in row.items()
                        if key not in ("by_line", "by_age", "by_delta_z", "row_records")}
            if field == "historical":
                retained.update({key: compact_slices(row[key]) for key in ("by_line", "by_age", "by_delta_z") if key in row})
            result[field].append(retained)
    return result


def _run_checks(directory, summary, recorded, p, cell, *, preflight=False):
    config = dict(p, **p["preflight"]) if preflight else p
    deadline_values = list(range(p["feedback_first_deadline"], config["feedback_last_deadline"] + 1,
                                 p["feedback_interval_seconds"]))
    checks = {"summary_completed": summary.get("status") == "complete",
              "native_runtime": summary.get("sumo_version") == "SUMO 1.25.0"}
    for field, value in (("seed", cell["seed"]), ("scenario", cell["scenario"]), ("arm", cell["arm"]),
            ("target_online_training_events", config["target_record_budget"]),
            ("packets", len(deadline_values)), ("policy_updates", config["policy_update_budget"])):
        checks[field] = summary.get(field) == value
    if preflight:
        now = summary.get("target_time_seconds")
        checks["target_time_seconds"] = finite(now) and config["horizon_seconds"] <= now < p["horizon_seconds"]
        checks["preflight"] = summary.get("preflight") is True
        checks["full_horizon_completed"] = summary.get("full_horizon_completed") is False
    else:
        checks["target_time_seconds"] = summary.get("target_time_seconds") == p["horizon_seconds"]
        checks["full_horizon_completed"] = summary.get("full_horizon_completed") is True
    for field, value in (("initial_updates", p["initializer_updates"]), ("history_rows", p["history_rows"]),
                         ("source_rows", p["source_rows"])):
        checks[field] = recorded.get(field) == value
    for field in ("model_init_updates", "model_refit_updates", "policy_updates_per_packet"):
        checks[field] = at(recorded, "protocol", field) == config[field]
    if cell["scenario"] == "local_speed_drop":
        checks["shock_clock"] = at(summary, "scenario_metadata", "actual_shock_at_seconds") == (1350 if preflight else p["shock_at_seconds"])
    passengers = summary.get("passengers", {})
    cost, departed = summary.get("generalized_passenger_seconds"), passengers.get("departed_passengers")
    wait, ride = passengers.get("accrued_wait_seconds_total"), passengers.get("accrued_in_vehicle_seconds_total")
    checks["passenger_balance"] = passengers.get("active_state_count_balance_error") == 0
    checks["cost_components"] = all(finite(value) for value in (cost, departed, wait, ride)) and departed > 0 and math.isclose(cost, 2 * wait + ride, abs_tol=1e-8)
    checks["cost_denominator"] = finite(cost) and finite(departed) and departed > 0 and math.isclose(
        summary.get("generalized_seconds_per_departed_passenger", math.inf), cost / departed, abs_tol=1e-8)
    receipts = summary.get("feedback_receipts", [])
    checks["feedback_deadlines"] = [row.get("deadline") for row in receipts] == deadline_values
    checks["feedback_release_count"] = bool(receipts) and all(
        row.get("records") == p["feedback_records_per_packet"]
        and row.get("max_available_at", math.inf) <= row.get("deadline", -1) <= row.get("processed_at", -1) for row in receipts)
    sequences = [value for row in receipts for value in row.get("sequence_ids", [])]
    checks["distinct_feedback"] = len(sequences) == len(set(sequences)) == config["target_record_budget"]
    logs = list(jsonl(directory / "prequential.jsonl"))
    checks["saved_global_predictions"] = len(logs) == len({row.get("sequence") for row in logs}) == config["target_record_budget"]
    checks["global_prediction_causality"] = bool(logs) and all(
        row["completed_at"] <= row["available_at"] <= row["scored_at"] for row in logs)
    checks["one_active_TD_probe_per_packet"] = len(list(jsonl(directory / "td_diagnostics.jsonl"))) == len(deadline_values)
    if cell["arm"] == "action_local":
        world = at(summary, "worlds", "action_local") or {}
        checks["local_policy_entry"] = summary.get("physical_learning_enters_policy") is True
        checks["local_readout_labels"] = at(world, "local_adapter", "fit_labels") == config["target_record_budget"]
        checks["four_head_labels"] = at(world, "head_diagnostics", "overall", "n") == config["target_record_budget"]
        checks.update(diagnostic_checks(world.get("context_diagnostics") or {}, summary.get("shadow_diagnostics") or {},
                                        config["context_probe_deadlines"]))
    elif cell["arm"] == "frozen_world":
        world = at(summary, "worlds", "frozen_world") or {}
        frozen = world.get("frozen_parameters") or {}
        models = frozen.get("models") or {}
        checks.update(
            frozen_initial_fit=frozen.get("initial_fit_updates") == config["model_init_updates"],
            frozen_neural_parameters=frozen.get("enabled") is True
                and set(models) == {"source", "direct", "residual"}
                and all(row.get("changed_tensor_count") == 0 and row.get("max_absolute_change") == 0
                        for row in models.values()),
            frozen_no_neural_updates=frozen.get("online_neural_gradient_updates") == 0
                and world.get("gradient_updates") == 0 and world.get("target_refits") == 0,
            frozen_all_refits_skipped=frozen.get("skipped_refit_packets") == len(deadline_values),
            frozen_family_selection_active=frozen.get("family_selection_adapts") is True,
            frozen_trust_learning_active=frozen.get("trust_update_calls") == config["policy_update_budget"],
            frozen_readout_disabled=frozen.get("local_readout_enabled") is False
                and summary.get("physical_learning_enters_policy") is False,
            frozen_prequential_rows=world.get("prequential_rows") == config["target_record_budget"],
            frozen_refit_random_draws_preserved=frozen.get("nominal_refit_rng_draws_preserved") is True)
    return checks


def native_preflight_gate(root=None, protocol=None):
    p = protocol or read(HERE / "protocol.json")
    root = Path(root) if root is not None else HERE / p["preflight_directory"]
    cell = dict(seed=p["preflight"]["seed"], scenario=p["preflight"]["scenario"], arm="action_local")
    directory = Path(root) / "control" / cell["scenario"] / cell["arm"] / f"seed{cell['seed']}"
    summary, recorded, failure = [read(directory / name) for name in ("summary.json", "protocol.json", "failure.json")]
    checks = _run_checks(directory, summary, recorded, p, cell, preflight=True) if summary and recorded else {}
    checks["summary_exists"] = summary is not None
    checks["protocol_exists"] = recorded is not None
    checks["no_failure"] = failure is None
    return dict(cell=cell, checks=checks, all_gates_pass=all(checks.values()))


def native_frozen_preflight_gate(root=None, protocol=None):
    p = protocol or read(HERE / "protocol.json")
    root = Path(root) if root is not None else HERE / "preflight_frozen_world"
    cell = dict(seed=p["preflight"]["seed"], scenario=p["preflight"]["scenario"], arm="frozen_world")
    directory = root / "control" / cell["scenario"] / cell["arm"] / f"seed{cell['seed']}"
    summary, recorded, failure = [read(directory / name) for name in ("summary.json", "protocol.json", "failure.json")]
    checks = _run_checks(directory, summary, recorded, p, cell, preflight=True) if summary and recorded else {}
    checks.update(summary_exists=summary is not None, protocol_exists=recorded is not None, no_failure=failure is None)
    return dict(cell=cell, checks=checks, all_gates_pass=all(checks.values()))


def prequential_heads(directory):
    logs = list(jsonl(directory / "prequential.jsonl"))
    result = {}
    for name, included in (("before", lambda row: row["issued_at"] < 9000),
                           ("after", lambda row: row["issued_at"] >= 9000)):
        rows = [row for row in logs if included(row)]
        result[name] = dict(n=len(rows), selected_head_loss={head: v4.v3.mean([
            at(row, "selected_head_losses", head) for row in rows])
            for head in ("global", "local", "count", "line", "station", "neighbor")})
    return result


def analyze(root=HERE / "runs", protocol=None, *, stage="stage_a", v4_root=V4_ROOT):
    root, v4_root = Path(root), Path(v4_root)
    p = protocol or read(HERE / "protocol.json")
    errors, missing, failures, cells, reproduction, contexts, shadows, local, td, actions = [], [], [], [], [], [], [], [], [], []
    initializers, frozen_diagnostics = {}, []
    for cell in run_cells(p, stage):
        seed, scenario, arm = cell["seed"], cell["scenario"], cell["arm"]
        label = f"control/{scenario}/{arm}/seed{seed}"
        directory = root / label
        summary, recorded, failure = [read(directory / name) for name in ("summary.json", "protocol.json", "failure.json")]
        if failure:
            failures.append(dict(cell=label, **failure))
        if summary is None or recorded is None:
            missing.append(label)
            continue
        checks = _run_checks(directory, summary, recorded, p, cell)
        errors.extend(dict(cell=label, check=key) for key, valid in checks.items() if not valid)
        physical = v4.v3._physical(directory, summary, lambda valid, key: errors.append(dict(cell=label, check=key)) if not valid else None)
        initializers.setdefault(str(seed), set()).add(recorded.get("initializer"))
        cost = cost_cell(directory, summary)
        cells.append(dict(**cell, **cost))
        errors.extend(dict(cell=label, check="fixed_clock_cost", window=name, **value)
                      for name, value in cost["windows"].items() if value["status"] != "complete")
        reference_arm = "action_local" if stage == "stage_a" else "action_model"
        prior_path = v4_root / "runs/control" / scenario / reference_arm / f"seed{seed}"
        prior, prior_protocol = read(prior_path / "summary.json"), read(prior_path / "protocol.json")
        budget_fields = ("generalized_passenger_seconds", "generalized_seconds_per_departed_passenger",
                         "policy_updates", "target_online_training_events", "target_time_seconds", "packets")
        same = prior is not None and all(summary.get(key) == prior.get(key) for key in budget_fields)
        same_initializer = prior_protocol is not None and recorded.get("initializer") == prior_protocol.get("initializer")
        if stage == "stage_a":
            reproduction.append(dict(**cell, exact_cost_and_budget=same, same_initializer=same_initializer,
                cost_delta=difference(cost["generalized_passenger_seconds"], prior.get("generalized_passenger_seconds") if prior else None)))
            if not same or not same_initializer:
                errors.append(dict(cell=label, check="passive_v4_reproduction"))
        else:
            if not prior or not same_initializer:
                errors.append(dict(cell=label, check="resident_v4_action_model_reference"))
            prior_cost = cost_cell(prior_path, prior) if prior else None
            reproduction.append(dict(**cell, reference_arm=reference_arm, same_initializer=same_initializer,
                full_day_difference=difference(cost["seconds_per_departed_passenger"], at(prior_cost, "seconds_per_departed_passenger")),
                window_differences={name: difference(at(cost, "windows", name, "seconds_per_full_day_departed_passenger"),
                    at(prior_cost, "windows", name, "seconds_per_full_day_departed_passenger")) for name in WINDOWS}))
            world = at(summary, "worlds", arm) or {}
            frozen_diagnostics.append(dict(**cell,
                frozen_parameters=world.get("frozen_parameters"),
                selected_family=world.get("selected_family"),
                prequential_heads=prequential_heads(directory),
                reference_prequential_heads=prequential_heads(prior_path)))
        if arm == "action_local":
            world = at(summary, "worlds", arm) or {}
            contexts.append(dict(**cell, diagnostics=_compact_context(world.get("context_diagnostics") or {})))
            shadows.append(dict(**cell, diagnostics=summary.get("shadow_diagnostics")))
            adapter = world.get("local_adapter") or {}
            paired = adapter.get("paired_prequential") or {}
            local.append(dict(**cell, physical_intervals={key: physical.get(key)
                    for key in ("coverage", "source_initialization")},
                local_adapter={key: adapter.get(key)
                    for key in ("enabled", "fit_labels", "fit_rows_by_line", "refits")},
                global_paired_prequential={key: paired.get(key) for key in ("overall", "by_phase")}))
        logs = list(jsonl(directory / "td_diagnostics.jsonl"))
        td.append(dict(**cell, periods=v4.v3._td_summary(logs, summary, arm)))
        actions.append(dict(**cell, periods=v4.v3._action_summary(summary)))
    for seed, paths in initializers.items():
        if len(paths) != 1:
            errors.append(dict(seed=seed, check="different_initializers"))
    return dict(status="complete" if len(cells) == 4 and not errors and not missing and not failures else "incomplete",
        stage=stage, integrity=dict(expected_runs=4, complete_runs=len(cells), errors=errors, missing=missing,
            failures=failures, initializer_paths={seed: sorted(paths) for seed, paths in initializers.items()}),
        passive_reproduction=reproduction if stage == "stage_a" else None,
        frozen_world_contrasts=reproduction if stage == "stage_b" else None,
        frozen_world_diagnostics=frozen_diagnostics if stage == "stage_b" else None,
        control_cells=cells, context_diagnostics=contexts, shadow_diagnostics=shadows,
        local_prediction_and_world=local, td_diagnostics=td, action_statistics=actions,
        head_order=["logdt", "logcost", "dWaiting", "dOnboard"],
        interpretation=("Four fixed development days. Context scores pair the same current parameters and selected family; next-packet scores use frozen anchor models before outcome fitting. Shadows execute eight optimizer steps using matched initial states and fixed actual trust. Local must exactly reproduce active parameters. Other forks are never deployed; action/value changes do not establish realized passenger benefit. Adaptation9000–18000 and frozen18300–24000 use actual ledger differences; full-day departed cohort is the common denominator."
            if stage == "stage_a" else
            "Four real frozen-neural-world control days versus resident v4 action_model, local readout disabled in both. Family selection, trust, replay and policy updates remain active. Initializers, target duration and label/update budget match; realized trajectories and labels may differ. Full-day and9000–18000/18300–24000 ledger costs use each full-day departed cohort. Original two development seeds, not independent confirmation."))


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
