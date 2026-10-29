"""Two fixed same-state holding interventions; resident exact replay and costs."""
import argparse
from contextlib import ExitStack
from itertools import chain, zip_longest
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE = Path("experiments") / HERE.name
sys.path.insert(0, str(ROOT))
from experiments.mcwm_historical_coverage_v111_20261029 import analyze as original
from experiments.mcwm_behavior_replay_v113_20261029 import review_analysis as behavior

read, finite = original.read, original.finite


def write_once(path, value, limit):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(text.encode()) > limit:
        raise ValueError("Summary exceeds the declared byte limit")
    with Path(path).open("x") as handle:
        handle.write(text)
    return value


def compare_segments(paths, reference):
    """Stream parent plus suffix as one closed JSONL, without making a copy."""
    counts, unequal, first = [0, 0], 0, None
    absent = object()
    try:
        with ExitStack() as stack:
            parts = [stack.enter_context(Path(path).open()) for path in paths]
            target = stack.enter_context(Path(reference).open())
            for index, pair in enumerate(zip_longest(chain.from_iterable(parts), target,
                                                       fillvalue=absent), 1):
                values = []
                for side, line in enumerate(pair):
                    if line is absent:
                        values.append(absent)
                    else:
                        counts[side] += 1
                        values.append(json.loads(line))
                if values[0] != values[1]:
                    unequal += 1
                    if first is None:
                        first = index
    except (OSError, ValueError) as error:
        return dict(all_pass=False, observed_rows=counts, error=str(error))
    return dict(all_pass=unequal == 0, observed_rows=counts,
                unequal_rows=unequal, first_difference=first)


def exact_fields(current, reference, fields):
    checks = {field: field in current and field in reference
              and current[field] == reference[field] for field in fields}
    return dict(all_pass=all(checks.values()), checks=checks)


def run_errors(row, p, scenario, arm):
    """v111 checks unchanged except its obsolete zero-counterfactual field."""
    errors = []
    if row.get("status") != "complete" or row.get("integrity", {}).get("all_pass") is not True:
        errors.append("control_integrity_failed")
    if not original.identity(row, p, scenario, arm):
        errors.append("cell_identity_failed")
    if row.get("counts") != original.expected_counts(p["policy_update_budget"]):
        errors.append("optimizer_or_model_counts_failed")
    if (not original.mechanism_valid(row, arm, p["policy_update_budget"])
            or row.get("replay", {}).get("mechanism") != row.get("mechanism")):
        errors.append("mechanism_channel_evidence_failed")
    budget = row.get("budget", {})
    expected = dict(native_seconds=p["horizon_seconds"], packets=57,
        feedback_slots=p["target_record_budget"], policy_updates=p["policy_update_budget"],
        update_attempts=p["policy_update_budget"], source_collection_events=0, model_refits=0,
        offline_training_updates=0,
        target_training_labels=p["target_record_budget"] if arm == "fresh" else 0)
    if any(budget.get(key) != value for key, value in expected.items()) or budget.get("partial_update_possible") is not False:
        errors.append("actual_budget_failed")
    action = row.get("action_audit", {})
    if action.get("raw_execution_exact") is not True or action.get("canonical_observation_exact") is not True:
        errors.append("execution_contract_failed")
    tail = row.get("post_budget_counts", {})
    if (not finite(tail.get("frozen_at")) or tail["frozen_at"] > p["periods"]["post_budget"][0]
            or not finite(tail.get("last_update_clock")) or tail["last_update_clock"] > p["periods"]["post_budget"][0]
            or not finite(tail.get("post18300_observed_at")) or tail["post18300_observed_at"] < 18300
            or not (tail.get("before") == tail.get("after") == tail.get("post18300") == row.get("counts"))):
        errors.append("post_budget_updates_changed")
    return errors


def original_cost_gates(cells):
    """Exactly the original five conditions, now applied to the low days."""
    if any(cell is None or any(cell["comparisons"]["full_day"].get(name) is None
            for name in ("low_minus_matched", "low_minus_frozen")) for cell in cells.values()):
        return None
    matched = {scene: cell["comparisons"]["full_day"]["low_minus_matched"]["cost_per_passenger"]
               for scene, cell in cells.items()}
    frozen = [cell["comparisons"]["full_day"]["low_minus_frozen"]["cost_per_passenger"]
              for cell in cells.values()]
    return dict(matched_mean_negative=sum(matched.values()) / len(matched) < 0,
        stable_matched_nonpositive=matched["stable"] <= 0,
        shock_matched_negative=matched["local_speed_drop"] < 0,
        each_frozen_nonpositive=all(value <= 0 for value in frozen),
        some_frozen_negative=any(value < 0 for value in frozen))


def unique_work(prefix_clock, prefix_counts, branches, horizon):
    """A shared prefix is paid once; each child pays only its own suffix."""
    prefix_known = finite(prefix_clock) and 0 <= prefix_clock <= horizon
    native, counters, errors = {}, {}, []
    for arm in ("high", "low"):
        row = branches.get(arm)
        end = row.get("ledger", {}).get("time_seconds", row.get("budget", {}).get("native_seconds")) if row else None
        native[arm] = end - prefix_clock if prefix_known and finite(end) and end >= prefix_clock else None
    complete_native = prefix_known and all(value is not None for value in native.values())
    for name in original.expected_counts(0):
        before = prefix_counts.get(name)
        known_prefix = finite(before) and before >= 0
        suffix = {}
        for arm in ("high", "low"):
            total = branches.get(arm, {}).get("counts", {}).get(name)
            if known_prefix and finite(total) and total >= before:
                suffix[arm] = total - before
            else:
                suffix[arm] = None
                errors.append(arm + ":unknown_or_decreasing_" + name)
        complete = known_prefix and all(value is not None for value in suffix.values())
        known = ([before] if known_prefix else []) + [value for value in suffix.values() if value is not None]
        counters[name] = dict(prefix=before if known_prefix else None, suffix=suffix,
            total=before + sum(suffix.values()) if complete else None,
            known_subtotal=sum(known) if known else None)
    known_native = ([prefix_clock] if prefix_known else []) + [value for value in native.values() if value is not None]
    return dict(prefix_native_seconds=prefix_clock if prefix_known else None, suffix_native_seconds=native,
        new_native_seconds=prefix_clock + sum(native.values()) if complete_native else None,
        known_native_seconds=sum(known_native) if known_native else None,
        native_seconds_is_lower_bound=not complete_native or any(
            row.get("budget", {}).get("native_seconds_is_lower_bound") is True for row in branches.values()),
        counters=counters, errors=errors)


def command_record(path, sequence):
    with Path(path).open() as handle:
        matches = [row for line in handle if (row := json.loads(line)).get("sequence") == sequence]
    if len(matches) != 1:
        raise ValueError(f"Expected one applied root command, found {len(matches)}")
    return matches[0]


def root_gate(folder, anchor, parent, branches):
    records = anchor["records"]["fresh"]["actions"]
    checks, actions, commands = {}, {}, {}
    try:
        fork = read(folder / "branches/live_fork.json")
        state = read(folder / "root_state.json")
        checks["live_fork"] = fork.get("all_pass") is True
        checks["declared_root_clock"] = (fork.get("native_before") == fork.get("native_after")
            == parent.get("budget", {}).get("native_seconds") == records["started_at"])
        for arm in ("high", "low"):
            child = folder / "branches" / arm
            checks[arm + "_root_state_exact"] = bool(state) and read(child / "root_state.json") == state
            action = actions[arm] = read(child / "root_action.json")
            checks[arm + "_action"] = (action.get("branch") == arm
                and action.get("sequence") == records["sequence"]
                and action.get("started_at") == records["started_at"]
                and action.get("trip_id") == anchor["trip_id"]
                and action.get("station_id") == anchor["stop_id"]
                and action.get("expected_hold_seconds") == action.get("actual_hold_seconds") == anchor[arm]
                and all(action.get(key) is True for key in (
                    "raw_F_exact", "root_state_exact", "only_root_override")))
            command = commands[arm] = command_record(child / "commands.jsonl", records["sequence"])
            checks[arm + "_native_command"] = (command.get("accepted") is True
                and command.get("requested_additional_hold_seconds") == anchor[arm]
                and command.get("trip_id") == anchor["trip_id"]
                and command.get("stop_id") == anchor["stop_id"]
                and command.get("applied_at") == records["started_at"])
            metadata = branches.get(arm, {}).get("branch", {})
            checks[arm + "_branch_identity"] = (metadata.get("name") == arm
                and metadata.get("root_clock") == records["started_at"]
                and metadata.get("root_counts") == parent.get("counts")
                and metadata.get("root_feedback_slots") == parent.get("budget", {}).get("feedback_slots")
                and metadata.get("root_issue_count") == 1
                and metadata.get("proposal_overrides") == (1 if arm == "low" else 0))
        checks["physical_doses_distinct"] = anchor["low"] < anchor["high"]
        checks["high_original_proposal"] = actions["high"].get("proposed_a") == records["proposed_a"]
        checks["high_original_execution"] = actions["high"].get("executed_a") == records["a"]
    except (OSError, ValueError, KeyError) as error:
        return dict(all_pass=False, checks=checks, error=str(error), actions=actions, commands=commands)
    return dict(all_pass=all(checks.values()), checks=checks, actions=actions, commands=commands,
        live_fork=dict(all_pass=fork["all_pass"], overlap_seconds=fork.get("overlap_seconds"),
            child_exitcodes=fork.get("child_exitcodes"), parent_rng_unchanged=fork.get("parent_rng_unchanged")))


def high_reproduction(folder, old_folder, high):
    try:
        old = read(old_folder / "summary.json")
    except (OSError, ValueError) as error:
        return dict(all_pass=False, error=str(error))
    # The simulation, learning and passenger quantities must all agree. Only
    # declared output paths, wall time, host placement and branch receipts differ.
    current = dict(high)
    current.pop("branch", None)
    # v113's no-counterfactual declaration does not describe this intervention.
    # Compare every other budget field; neither mutate history nor invent zero.
    for row in (current, old):
        row["budget"] = {key: value for key, value in row.get("budget", {}).items()
                         if key != "counterfactual_queries"}
    summary = behavior.summary_comparison(current, old)
    child = folder / "branches/high"
    logs = {name: original.compare_json(child / name, old_folder / name)
            for name in ("metrics.json", "packet_receipts.json")}
    for name in ("actions.jsonl", "updates.jsonl"):
        logs[name] = compare_segments([folder / name, child / name], old_folder / name)
    for name in ("passengers.jsonl", "visits.jsonl"):
        logs[name] = compare_segments([child / name], old_folder / name)
    return dict(all_pass=summary["all_pass"] and all(row["all_pass"] for row in logs.values()),
                summary=summary, logs=logs)


def scalar_consumption(prefix, branches, budget_field):
    values, known = {}, []
    before = prefix.get("budget", {}).get(budget_field)
    if finite(before):
        known.append(before)
    for arm in ("high", "low"):
        total = branches.get(arm, {}).get("budget", {}).get(budget_field)
        value = total - before if finite(total) and finite(before) and total >= before else None
        values[arm] = value
        if value is not None:
            known.append(value)
    complete = finite(before) and all(value is not None for value in values.values())
    return dict(prefix=before, suffix=values, total=sum(known) if complete else None,
                known_subtotal=sum(known) if known else None)


def comparison_cell(root, p, scene, branches):
    values, errors = {}, []
    for arm in ("high", "low"):
        if arm in branches:
            values[arm], bad = original.period_values(branches[arm], p)
            errors.extend(arm + ":" + value for value in bad)
        else:
            values[arm] = {name: None for name in p["periods"]}
    for kind, path in (
        ("matched", root / p["v111_root"] / "runs" / scene / "no_fresh/summary.json"),
        ("frozen", root / p["frozen_reference_root"] / scene / "summary.json")):
        try:
            row = read(path)
            values[kind], bad = original.period_values(row, p)
            valid = row.get("status") == "complete" and row.get("integrity", {}).get("all_pass") is True
            valid &= (original.identity(row, p, scene, "no_fresh") if kind == "matched" else
                row.get("actor_seed") == p["training_seed"] and row.get("external_seed") == p["external_seed"]
                and row.get("scenario") == scene)
            if not valid or bad:
                errors.append(kind + ":invalid_reference")
        except (OSError, ValueError) as error:
            values[kind] = {name: None for name in p["periods"]}
            errors.append(kind + ":" + str(error))
    comparisons = {name: {
        key: original.differences(values[a][name], values[b][name]) for key, a, b in (
            ("low_minus_high", "low", "high"), ("low_minus_matched", "low", "matched"),
            ("low_minus_frozen", "low", "frozen"), ("high_minus_matched", "high", "matched"),
            ("matched_minus_frozen", "matched", "frozen"))} for name in p["periods"]}
    return dict(values=values, comparisons=comparisons), errors


def analyze(root):
    root = Path(root)
    experiment = root / RELATIVE
    destination = experiment / "summary.json"
    if destination.exists():
        return read(destination)
    p = read(experiment / "protocol.json")
    anchors = {row["scenario"]: row for row in read(experiment / "anchor_binding.json")["anchors"]}
    missing_parents = [scene for scene in p["scenarios"]
                       if not (experiment / "runs" / scene / "summary.json").exists()]
    if missing_parents:
        return dict(status="pending", missing_parent_summaries=missing_parents)
    parents, cells, root_checks, high_checks, observations, budgets, errors = {}, {}, {}, {}, {}, {}, []
    for scene in p["scenarios"]:
        folder = experiment / "runs" / scene
        parent = parents[scene] = read(folder / "summary.json")
        frozen = p["anchors"][scene]
        binding = anchors[scene]
        action_binding = binding["records"]["fresh"]["actions"]
        if (frozen["trip_id"] != binding["trip_id"] or frozen["station_id"] != binding["stop_id"]
                or frozen["sequence"] != action_binding["sequence"] or frozen["clock"] != action_binding["started_at"]
                or frozen["high_hold_seconds"] != binding["high"] or frozen["low_hold_seconds"] != binding["low"]):
            errors.append(scene + ":protocol_anchor_binding_mismatch")
        if parent.get("status") != "prefix_complete":
            errors.append(scene + ":parent_prefix_incomplete")
        branches = {arm: read(folder / "branches" / arm / "summary.json") for arm in ("high", "low")
                    if (folder / "branches" / arm / "summary.json").exists()}
        root_checks[scene] = root_gate(folder, anchors[scene], parent, branches)
        if not root_checks[scene]["all_pass"]:
            errors.append(scene + ":same_state_or_root_action_failed")
        for arm in ("high", "low"):
            key = scene + "/" + arm
            if arm not in branches:
                observations[key] = dict(all_pass=False, error="missing_branch")
                errors.append(key + ":missing_branch")
                continue
            bad = run_errors(branches[arm], p, scene, "fresh")
            errors.extend(key + ":" + error for error in bad)
            try:
                observations[key], _, _ = behavior.observed_cell(folder / "branches" / arm, branches[arm], p)
                if not observations[key]["all_pass"]:
                    errors.append(key + ":passenger_or_visit_ledger_failed")
            except (OSError, ValueError, KeyError, TypeError) as error:
                observations[key] = dict(all_pass=False, error=str(error))
                errors.append(key + ":observer_missing_or_invalid")
        high_checks[scene] = (high_reproduction(folder,
            root / p["v113_root"] / "runs" / scene / "fresh", branches["high"])
            if "high" in branches else dict(all_pass=False, error="missing_high"))
        if not high_checks[scene]["all_pass"]:
            errors.append(scene + ":high_does_not_reproduce_original_fresh")
        cells[scene], bad = comparison_cell(root, p, scene, branches)
        errors.extend(scene + ":" + error for error in bad)
        high, low = observations[scene + "/high"], observations[scene + "/low"]
        cells[scene]["low_minus_high_phases"] = (behavior.report_blocks(
            behavior.delta_blocks(low["totals"], high["totals"]), p["full_day_departed_passengers"])
            if high["all_pass"] and low["all_pass"] else None)
        work = budgets[scene] = unique_work(parent.get("budget", {}).get("native_seconds"),
            parent.get("counts", {}), branches, p["horizon_seconds"])
        work["fresh_admissions"] = scalar_consumption(parent, branches, "target_training_labels")
        errors.extend(scene + ":" + error for error in work["errors"])
        for arm, row in branches.items():
            branch = row.get("branch", {})
            if (branch.get("suffix_native_seconds") != work["suffix_native_seconds"][arm]
                    or branch.get("suffix_updates") != work["counters"]["policy"]["suffix"][arm]
                    or branch.get("suffix_feedback_slots") != work["fresh_admissions"]["suffix"][arm]):
                errors.append(scene + "/" + arm + ":suffix_consumption_mismatch")
    technical = not errors
    gates = original_cost_gates(cells) if technical else None
    means = {}
    for period in p["periods"]:
        means[period] = {}
        for name in next(iter(cells.values()))["comparisons"][period]:
            rows = [cell["comparisons"][period][name] for cell in cells.values()]
            means[period][name] = (None if any(row is None for row in rows) else
                {field: sum(row[field] for row in rows) / len(rows) for field in rows[0]})
    def panel_sum(values):
        return sum(values) if all(value is not None for value in values) else None
    result = dict(status="complete" if technical else "technical_hold",
        integrity=dict(all_pass=technical, errors=errors),
        decision=dict(state="COMPLETE_FIXED_ANCHOR_INTERVENTIONS" if technical else "HOLD_TECHNICAL_INCOMPLETE",
            low_intervention_cost_gate_pass=all(gates.values()) if gates is not None else None,
            method_candidate_passed=False, v111_candidate_reopened=False, automatic_new_experiments=False),
        cost_gates=gates, root_checks=root_checks, high_reproduction=high_checks,
        observer_checks={key: {field: value for field, value in row.items() if field != "totals"}
                         for key, row in observations.items()}, cells=cells, means=means,
        budget=dict(by_scene=budgets,
            new_native_seconds=panel_sum([row["new_native_seconds"] for row in budgets.values()]),
            known_native_seconds=panel_sum([row["known_native_seconds"] for row in budgets.values()]),
            native_seconds_is_lower_bound=any(row["native_seconds_is_lower_bound"] for row in budgets.values()),
            policy_updates=panel_sum([row["counters"]["policy"]["total"] for row in budgets.values()]),
            fresh_admissions=panel_sum([row["fresh_admissions"]["total"] for row in budgets.values()])),
        interpretation=dict(estimand="Low-minus-high full-day cost from the same complete fresh root, followed by the original continued-learning recipe. Includes later traffic, feedback and learning responses.",
            comparison="Matched no_fresh and frozen MC-WM remain separate whole-day references; the original five conditions are unchanged.",
            scope="Two previously exposed, purposefully selected single-action interventions; not independent sampling or a qualified general policy constraint.",
            action="Low is a fixed physical dose passed through original raw F, not the old no_fresh actor evaluated at this state.",
            units="(2W+R)/15079; all full-day and window costs retain the same denominator. Phase raw totals are passenger-seconds.",
            work="One prefix plus two suffixes per scene. Child cumulative counts are differenced against the shared root; historical training and cloned prefix counters are not charged again."))
    return write_once(destination, result, p["summary_max_bytes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), ensure_ascii=False, allow_nan=False))
