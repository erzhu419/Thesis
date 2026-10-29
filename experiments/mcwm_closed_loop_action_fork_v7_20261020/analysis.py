"""Resident paired branch integrity and action-benefit evidence; no trajectory pull."""

from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (6201, 6202)
SCENARIOS = ("stable", "local_speed_drop")
LINES = ("7X", "102X", "705X")
BRANCHES = ("low", "high")
WORLDS = ("Online", "Frozen")


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def result_name():
    return "action_fork_summary.json"


def state_cells(p, *, preflight=False):
    if preflight:
        row = p["preflight"]
        return [dict(seed=row["seed"], scenario=row["scenario"], target_line=row["target_line"])]
    expected = [dict(seed=seed, scenario=scenario, target_line=line)
                for seed, scenario, line in product(SEEDS, SCENARIOS, LINES)]
    if "state_cells" in p and p["state_cells"] != expected:
        raise ValueError("Retain all 12 fixed development states without outcome-based selection")
    return expected


def branch_cells(p, *, preflight=False):
    return [dict(**row, branch=branch) for row in state_cells(p, preflight=preflight) for branch in BRANCHES]


def cell_path(cell):
    return Path(cell["scenario"]) / cell["target_line"] / f"seed{cell['seed']}" / cell["branch"]


def sign(value):
    return None if value is None else (1 if value > 0 else (-1 if value < 0 else 0))


def difference(high, low):
    return None if high is None or low is None else high - low


def direction_agreement(predicted, observed):
    left, right = sign(predicted), sign(observed)
    if left is None or right is None:
        return "missing"
    if left == 0 or right == 0:
        return "tie"
    return "agree" if left == right else "disagree"


def branch_checks(row, p, *, preflight=False):
    selected = row.get("selected") or {}
    prefix, suffix, budget, native = [row.get(key) or {} for key in ("prefix", "suffix", "budget", "native")]
    configuration = dict(p, **p["preflight"]) if preflight else p
    window = configuration["selection_window_seconds"]
    receipts = prefix.get("feedback_receipts", [])
    initializer = prefix.get("initializer_receipt") or {}
    packets = len(receipts)
    cell = row.get("cell") or {}
    prediction = row.get("prediction") or {}
    factual = row.get("next_event") or {}
    chosen = selected.get(cell.get("branch", "") + "_action")
    return dict(
        completed=row.get("status") == "complete",
        correct_preflight=row.get("preflight") is preflight,
        selected_in_fixed_window=selected.get("time") is not None and window[0] <= selected["time"] <= window[1],
        selected_target_line=selected.get("line") == cell.get("target_line"),
        same_affected_lines=cell.get("target_line") in LINES,
        distinguishable_actions=selected.get("low_action") is not None and selected.get("high_action") is not None
            and selected["high_action"] > selected["low_action"],
        full_policy_packet_updates=prefix.get("policy_updates") == packets * p["policy_updates_per_packet"],
        same_v6_reference_feedback_prefix=prefix.get("reference_feedback_prefix_exact") is True,
        prefix_target_labels=prefix.get("target_labels") == packets * p["feedback_records_per_packet"]
            and prefix.get("target_labels") == sum(receipt.get("records", 0) for receipt in receipts),
        prefix_world_refits=prefix.get("world_refits") == packets,
        prefix_world_updates=prefix.get("world_updates") == 2 * configuration["model_refit_updates"] * packets,
        prefix_trust_updates=prefix.get("trust_updates") == packets * p["policy_updates_per_packet"],
        same_exogenous_seed=prefix.get("external_seed") == cell.get("seed", 0) + p["target_seed_offset"],
        same_initializer_updates=initializer.get("initial_updates") == p["initializer_updates"],
        same_initializer_history=initializer.get("history_rows") == p["history_rows"],
        same_initializer_source=initializer.get("source_rows") == p["source_rows"],
        same_model_init_updates=initializer.get("model_init_updates") == configuration["model_init_updates"],
        same_model_refit_updates=initializer.get("model_refit_updates") == configuration["model_refit_updates"],
        frozen_suffix_policy=suffix.get("training_updates") == 0,
        frozen_suffix_world=suffix.get("world_updates") == 0,
        frozen_suffix_trust=suffix.get("trust_updates") == 0,
        frozen_suffix_physical=suffix.get("physical_updates") == 0,
        exact_native_steps=suffix.get("native_steps") == 300,
        cutoff_nonterminal=native.get("cutoff_not_terminal") is True,
        exact_300_second_suffix=suffix.get("seconds") == 300 and suffix.get("end_time") is not None
            and suffix.get("start_time") == selected.get("time") and suffix["end_time"] - suffix["start_time"] == 300,
        balanced_passenger_ledger=native.get("passenger_ledger_balanced") is True,
        first_action_applied=native.get("first_action_applied") == chosen and chosen is not None,
        other_batch_actions_unchanged=native.get("other_batch_actions_unchanged") is True,
        native_sumo_version=bool(native.get("sumo_version")),
        added_suffix_budget=budget.get("suffix_sim_seconds") == 300,
        next_global_truth_one=budget.get("truth_next_global_records") == 1,
        truth_never_used_for_training=budget.get("truth_labels_enter_training") == 0,
        explicit_prefix_budget=budget.get("prefix_sim_seconds") == selected.get("time"),
        explicit_prefix_label_budget=budget.get("prefix_target_labels") == prefix.get("target_labels"),
        explicit_prefix_update_budget=budget.get("prefix_policy_updates") == prefix.get("policy_updates"),
        valid_saved_candidate_pair=prediction.get("valid_pair") is True,
        correct_branch_truth=factual.get("candidate") == cell.get("branch"),
        outcome_from_selected_clock=factual.get("started_at") == selected.get("time")
            and factual.get("completed_at") is not None and selected.get("time") is not None
            and selected["time"] <= factual["completed_at"] <= selected["time"] + 300,
        fixed_prefix_value_evaluator=all(prediction.get(key) is not None
            and factual.get(key) == prediction[key]
            for key in ("anchor_update", "world_version", "common_family", "target_seed")),
        explicit_initializer=isinstance(prefix.get("initializer_receipt"), dict)
            and bool(prefix.get("initializer_receipt")),
    )


def pair_checks(low, high, p, *, preflight=False):
    checks = {}
    for name, row in (("low", low), ("high", high)):
        checks.update({name + "/" + key: value for key, value in branch_checks(row, p, preflight=preflight).items()})
    checks.update(
        selected_state_and_remaining_batch_identical=low.get("selected") == high.get("selected"),
        initializer_training_and_exogenous_seed_identical=low.get("prefix") == high.get("prefix"),
        model_predictions_and_prefix_evaluator_identical=low.get("prediction") == high.get("prediction"),
        native_sumo_version_identical=(low.get("native") or {}).get("sumo_version")
            == (high.get("native") or {}).get("sumo_version"),
    )
    return checks


def native_preflight_gate(root=None, protocol=None):
    p = protocol or read(HERE / "protocol.json")
    root = Path(root) if root is not None else HERE / p["preflight_directory"]
    rows = branch_cells(p, preflight=True)
    records = [read(root / cell_path(row) / "summary.json") for row in rows]
    failures = [str(cell_path(row)) for row in rows if read(root / cell_path(row) / "failure.json")]
    checks = dict(both_branch_summaries=all(value is not None for value in records), no_failures=not failures)
    if checks["both_branch_summaries"]:
        checks.update(pair_checks(*records, p, preflight=True))
    return dict(cells=rows, checks=checks, all_gates_pass=all(checks.values()), failures=failures)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _delta(low, high):
    global_effect = [difference(h, l) for l, h in zip(low["global_decoded"], high["global_decoded"])]
    bellman = {key: difference(high["bellman"][key], low["bellman"][key])
               for key in ("scaled_reward", "bootstrap", "target")}
    return dict(global_high_minus_low=global_effect, bellman_high_minus_low=bellman,
                gains=dict(next_event_cost=-global_effect[1], scaled_reward=bellman["scaled_reward"],
                           bootstrap=bellman["bootstrap"], bellman_target=bellman["target"]))


def pair_evidence(low, high):
    prediction = low["prediction"]
    actual = _delta(low["next_event"], high["next_event"])
    actual["gains"]["cost_300_seconds"] = low["suffix"]["cost"] - high["suffix"]["cost"]
    actual["cost_300_high_minus_low"] = -actual["gains"]["cost_300_seconds"]
    worlds = {}
    for world in ("online", "frozen_common"):
        candidates = prediction["branches"][world]["candidates"]
        predicted = _delta(candidates["low"], candidates["high"])
        outcomes = {role: row["next_event"]["branches"][world] for role, row in (("low", low), ("high", high))}
        scores = dict(
            global_mae=[mean([abs(outcomes[role]["global_signed_error"][i]) for role in BRANCHES]) for i in range(4)],
            g4_mse=mean([value*value for role in BRANCHES
                         for value in outcomes[role]["standardized_global_signed_error"]]),
            bellman_mse={key: mean([outcomes[role]["bellman_signed_error"][key]**2 for role in BRANCHES])
                         for key in ("scaled_reward", "bootstrap", "target")},
            bellman_mae={key: mean([abs(outcomes[role]["bellman_signed_error"][key]) for role in BRANCHES])
                         for key in ("scaled_reward", "bootstrap", "target")},
            gain_signed_error={key: predicted["gains"][key]-actual["gains"][key] for key in predicted["gains"]},
            next_event_cost_direction=direction_agreement(predicted["gains"]["next_event_cost"], actual["gains"]["next_event_cost"]),
            bellman_direction=direction_agreement(predicted["gains"]["bellman_target"], actual["gains"]["bellman_target"]),
            bellman_vs_cost_300_direction=direction_agreement(predicted["gains"]["bellman_target"], actual["gains"]["cost_300_seconds"]),
        )
        worlds[world] = dict(predicted_effect=predicted, scores=scores,
            candidates={role: dict(global_raw=candidates[role]["global_raw"], global_decoded=candidates[role]["global_decoded"],
                bellman=candidates[role]["bellman"], trust=candidates[role]["trust"],
                six_losses=outcomes[role]["six_losses"], trust_outcome=outcomes[role]["trust_outcome"])
                for role in BRANCHES})
    return dict(factual_effect=actual, worlds=worlds,
        actual_next_events={role: {key: row["next_event"].get(key) for key in (
            "global_decoded", "bellman", "started_at", "completed_at", "terminal", "zero_duration",
            "vehicle_id", "next_vehicle_id", "next_topology")} for role, row in (("low", low), ("high", high))},
        suffixes={role: {key: row["suffix"].get(key) for key in (
            "start_time", "end_time", "seconds", "cost", "wait", "ride", "departed", "ledger_balance")}
            for role, row in (("low", low), ("high", high))},
        next_global_same_vehicle=low["next_event"].get("next_vehicle_id") == high["next_event"].get("next_vehicle_id"),
        next_global_completed_time_difference=high["next_event"]["completed_at"]-low["next_event"]["completed_at"])


def aggregate_pairs(pairs):
    output = {}
    for name, selected in (("overall", pairs), *((scenario, [row for row in pairs if row["scenario"] == scenario])
                                               for scenario in SCENARIOS)):
        groups = {}
        for world in ("online", "frozen_common"):
            scores = [row["evidence"]["worlds"][world]["scores"] for row in selected]
            directions = {}
            for key in ("next_event_cost_direction", "bellman_direction", "bellman_vs_cost_300_direction"):
                directions[key] = {value: sum(row[key] == value for row in scores)
                                   for value in ("agree", "disagree", "tie", "missing")}
            candidates = [row["evidence"]["worlds"][world]["candidates"][role] for row in selected for role in BRANCHES]
            groups[world] = dict(n_pairs=len(selected), n_branches=2*len(selected),
                global_mae=[mean([row["global_mae"][i] for row in scores]) for i in range(4)],
                g4_mse=mean([row["g4_mse"] for row in scores]),
                bellman_mse={key: mean([row["bellman_mse"][key] for row in scores]) for key in ("scaled_reward", "bootstrap", "target")},
                bellman_mae={key: mean([row["bellman_mae"][key] for row in scores]) for key in ("scaled_reward", "bootstrap", "target")},
                gain_mse={key: mean([row["gain_signed_error"][key]**2 for row in scores])
                          for key in ("next_event_cost", "scaled_reward", "bootstrap", "bellman_target")},
                mean_trust=mean([row["trust"]["weight"] for row in candidates]),
                directions=directions)
        output[name] = groups
    return output


def analyze(root=HERE / "runs", protocol=None):
    root = Path(root)
    p = protocol or read(HERE / "protocol.json")
    errors, missing, failures, pairs, statuses, budgets = [], [], [], [], [], []
    for cell in state_cells(p):
        records = []
        state_label = f"{cell['scenario']}/{cell['target_line']}/seed{cell['seed']}"
        for branch in BRANCHES:
            branch_cell = dict(**cell, branch=branch)
            directory = root / cell_path(branch_cell)
            failure = read(directory / "failure.json")
            row = read(directory / "summary.json")
            if failure:
                failures.append(dict(cell=str(cell_path(branch_cell)), failure=failure))
            if row is None:
                missing.append(str(cell_path(branch_cell)))
            elif row.get("cell") != branch_cell:
                errors.append(dict(cell=str(cell_path(branch_cell)), check="fixed_cell_identity"))
            if row is not None:
                statuses.append(dict(**branch_cell, status=row.get("status")))
                if row.get("budget"):
                    budgets.append(dict(**branch_cell, **row["budget"]))
            records.append(row)
        if any(row is None for row in records):
            continue
        if any(row.get("status") != "complete" for row in records):
            errors.append(dict(cell=state_label, check="branch_status", statuses=[row.get("status") for row in records]))
            continue
        checks = pair_checks(*records, p)
        errors.extend(dict(cell=state_label, check=key) for key, valid in checks.items() if not valid)
        if not all(checks.values()):
            continue
        low, high = records
        selected = low["selected"]
        prefix = {key: value for key, value in low['prefix'].items() if key != 'feedback_receipts'}
        receipts = low['prefix']['feedback_receipts']
        prefix['feedback_packet_count'] = len(receipts)
        prefix['first_feedback_deadline'] = receipts[0]['deadline'] if receipts else None
        prefix['last_feedback_deadline'] = receipts[-1]['deadline'] if receipts else None
        pairs.append(dict(**cell, selected={key: selected.get(key) for key in (
            "time", "sequence", "line", "vehicle_id", "trip_id", "station", "low_action", "high_action",
            "remaining_actions", "prefix_cost", "prefix_departed")},
            prefix=prefix, evidence=pair_evidence(low, high)))
    suffix_seconds = sum(row.get("suffix_sim_seconds", 0) for row in budgets)
    preflight_records = [read(root.parent / p["preflight_directory"] / cell_path(row) / "summary.json")
                         for row in branch_cells(p, preflight=True)]
    preflight_budgets = [row["budget"] for row in preflight_records if row and row.get("budget")]
    preflight_budget = dict(expected_branches=2, recorded_branches=len(preflight_budgets),
        planned_suffix_sim_seconds=600,
        prefix_sim_seconds_replayed=sum(row.get("prefix_sim_seconds", 0) for row in preflight_budgets),
        suffix_sim_seconds=sum(row.get("suffix_sim_seconds", 0) for row in preflight_budgets),
        truth_next_global_records=sum(row.get("truth_next_global_records", 0) for row in preflight_budgets),
        truth_labels_enter_training=sum(row.get("truth_labels_enter_training", 0) for row in preflight_budgets))
    return dict(status="complete" if len(pairs) == 12 and not (errors or failures or missing) else "incomplete",
        integrity=dict(expected_pairs=12, complete_pairs=len(pairs), expected_branches=24,
                       recorded_branches=len(statuses), errors=errors, failures=failures, missing=missing),
        budget=dict(prefix_sim_seconds_replayed=sum(row.get("prefix_sim_seconds", 0) for row in budgets),
            suffix_sim_seconds=suffix_seconds, planned_suffix_sim_seconds=7200,
            truth_next_global_records=sum(row.get("truth_next_global_records", 0) for row in budgets),
            truth_labels_enter_training=sum(row.get("truth_labels_enter_training", 0) for row in budgets),
            observed_branches=len(budgets), prefix_and_suffix_budgets_are_additional=True,
            preflight_budget_separate=True, preflight=preflight_budget, branch_budgets=budgets),
        branch_statuses=statuses, pairs=pairs, aggregates=aggregate_pairs(pairs),
        interpretation=dict(gain_sign="Positive gain favors the higher executed holding action.",
            next_event="The endpoint is the actual next global control event and may differ across branches, including zero elapsed time.",
            cost_300="Raw generalized passenger seconds over the same 300-second intervention-clock window; low cost minus high cost is the gain.",
            scope="Two original development seeds, fixed lines and eligibility window. No branch outcome enters replay, policy, world, or trust training."))
