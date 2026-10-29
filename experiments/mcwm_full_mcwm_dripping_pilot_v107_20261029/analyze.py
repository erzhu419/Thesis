"""Resident prefix equality and the fixed matched/frozen v107 comparisons."""

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELATIVE = Path("experiments") / HERE.name


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value, limit=32000):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > limit:
        raise ValueError("Small summary exceeds the declared byte limit")
    with Path(path).open("x") as handle:
        handle.write(text)
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identity(row, p, scenario, arm):
    return row.get("cell") == dict(training_seed=p["training_seed"],
        external_seed=p["external_seed"], scenario=scenario, arm=arm)


def expected_counts(updates):
    return dict(policy=updates, actor=updates, critic=updates, value=updates, alpha=updates,
        ratio=0, trust=updates // 4, trust_optimizer=updates // 4,
        model_batches=updates, model_rows=updates * 128, world_refit=0)


def overlap(rows, end_key):
    intervals = []
    for row in rows:
        wall = row.get("execution_wall", {})
        start, end = wall.get("native_started"), wall.get(end_key)
        if not finite(start) or not finite(end) or end <= start:
            return dict(all_pass=False, overlap_seconds=None, reason="missing_or_invalid_native_interval")
        intervals.append([start, end])
    seconds = min(x[1] for x in intervals) - max(x[0] for x in intervals)
    return dict(all_pass=seconds > 0, overlap_seconds=seconds, intervals=intervals)


def compare_prefix(left, right, expected_left, expected_right):
    """Read both closed logs in place; compare every JSON object without hashes."""
    counts, unequal, first = [0, 0], 0, None
    absent = object()
    try:
        with Path(left).open() as a, Path(right).open() as b:
            for index, pair in enumerate(zip_longest(a, b, fillvalue=absent), 1):
                values = []
                for side, line in enumerate(pair):
                    if line is absent:
                        values.append(absent)
                    else:
                        counts[side] += 1
                        values.append(json.loads(line))
                if values[0] != values[1]:
                    unequal += 1
                    first = index if first is None else first
    except (OSError, ValueError) as error:
        return dict(all_pass=False, observed_rows=counts, error=str(error))
    return dict(all_pass=unequal == 0 and counts == [expected_left, expected_right],
        observed_rows=counts, declared_rows=[expected_left, expected_right],
        unequal_rows=unequal, first_difference=first)


def preflight(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    gate_path = out_dir / "preflight_gate.json"
    if gate_path.exists():
        return read(gate_path)
    experiment = root / RELATIVE
    p = read(experiment / "protocol.json")
    rows, missing, errors = {}, [], []
    for arm in p["arms"]:
        folder = experiment / "runs" / "stable" / arm
        path = folder / "preflight.json"
        if not path.exists():
            final = folder / "summary.json"
            if final.exists() and read(final).get("status") != "complete":
                errors.append(arm + ":run_failed_before_preflight")
            else:
                missing.append(str(path))
            continue
        row = rows[arm] = read(path)
        valid = (row.get("status") == "complete" and row.get("integrity", {}).get("all_pass") is True
            and identity(row, p, "stable", arm)
            and row.get("initial_updates") == 70000 and row.get("initial_online_critic_updates") == 20000
            and row.get("counts") == expected_counts(p["policy_updates_per_packet"])
            and row.get("mix") == dict(real=.75, model=.25) and row.get("model_batch_sizes") == [128]
            and row.get("feedback", {}).get("records_received") == p["feedback_records_per_packet"]
            and row.get("feedback", {}).get("training_outcomes") == (5 if arm == "fresh" else 0))
        if not valid:
            errors.append(arm + ":first_packet_receipt_failed")
    if missing and not errors:
        return dict(status="pending", missing=missing)
    prefixes, parallel = {}, None
    if not errors:
        a, b = (rows[arm] for arm in p["arms"])
        for name, filename, count_key in (("controls", "prefix_trace.jsonl", "trace_rows"),
                                          ("services", "prefix_services.jsonl", "service_rows")):
            prefixes[name] = compare_prefix(
                experiment / "runs/stable" / p["arms"][0] / filename,
                experiment / "runs/stable" / p["arms"][1] / filename,
                a["prefix"][count_key], b["prefix"][count_key])
            if not prefixes[name]["all_pass"]:
                errors.append(name + ":prefix_mismatch")
        if (a["prefix"]["clock"] != b["prefix"]["clock"]
                or a["prefix"]["service_events"] != b["prefix"]["service_events"]):
            errors.append("prefix_clock_or_service_count_mismatch")
        # Both native days continue after this receipt. Nonoverlapping prefix
        # intervals cannot establish that the complete days ran serially.
        parallel = dict(status="pending_final_native_intervals",
            prefix_interval_observation=overlap([a, b], "first_packet_finished"))
    result = dict(status="complete" if not errors else "failed", all_pass=not errors,
        decision="PASS_LAUNCH_SHOCK_PAIR" if not errors else "HOLD_TECHNICAL_FAILURE_NO_EXPANSION",
        errors=errors, missing=missing, prefixes=prefixes, parallel=parallel,
        receipts={arm: dict(counts=row["counts"], mix=row["mix"], feedback=row["feedback"],
                           execution_wall=row["execution_wall"]) for arm, row in rows.items()})
    return write_once(gate_path, result, p["summary_max_bytes"])


def period_values(row, p):
    values, errors = {}, []
    for name, bounds in p["periods"].items():
        item = row.get("periods", {}).get(name, {})
        if (item.get("status") != "complete" or [item.get("start"), item.get("end")] != bounds
                or item.get("departed") != p["full_day_departed_passengers"]
                or not all(finite(item.get(key)) for key in ("wait", "ride", "cost", "cost_per_passenger"))):
            errors.append(name + ":missing_or_invalid_period")
            values[name] = None
            continue
        denominator = p["full_day_departed_passengers"]
        if item["cost"] != 2 * item["wait"] + item["ride"] or item["cost_per_passenger"] != item["cost"] / denominator:
            errors.append(name + ":cost_identity_failed")
            values[name] = None
            continue
        values[name] = dict(wait_per_passenger=item["wait"] / denominator,
            ride_per_passenger=item["ride"] / denominator, cost_per_passenger=item["cost_per_passenger"])
    return values, errors


def differences(left, right):
    if left is None or right is None:
        return None
    return {key: left[key] - right[key] for key in left}


def run_errors(row, p, scenario, arm):
    errors = []
    if row.get("status") != "complete" or row.get("integrity", {}).get("all_pass") is not True:
        errors.append("control_integrity_failed")
    if not identity(row, p, scenario, arm):
        errors.append("cell_identity_failed")
    if row.get("counts") != expected_counts(p["policy_update_budget"]):
        errors.append("optimizer_or_model_counts_failed")
    budget = row.get("budget", {})
    expected = dict(native_seconds=p["horizon_seconds"], packets=57,
        feedback_slots=p["target_record_budget"], policy_updates=p["policy_update_budget"],
        update_attempts=p["policy_update_budget"], source_collection_events=0, model_refits=0,
        offline_training_updates=0, counterfactual_queries=0,
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


def analyze(root):
    root = Path(root)
    experiment = root / RELATIVE
    destination = experiment / "summary.json"
    if destination.exists():
        return read(destination)
    p = read(experiment / "protocol.json")
    runs, missing, errors = {}, [], []
    for scene in p["scenarios"]:
        for arm in p["arms"]:
            key = scene + "/" + arm
            path = experiment / "runs" / key / "summary.json"
            if path.exists():
                runs[key] = read(path)
            else:
                missing.append(key)
    gate_path = experiment / "preflight_gate.json"
    gate = read(gate_path) if gate_path.exists() else None
    failed = (gate is not None and gate.get("all_pass") is not True) or any(
        row.get("status") != "complete" for row in runs.values())
    if missing and not failed:
        return dict(status="pending", missing=missing)
    if gate is None or gate.get("all_pass") is not True:
        errors.append("stable_preflight_gate_not_passed")
    cells, pairs, references = [], {}, {}
    for scene in p["scenarios"]:
        reference_path = root / p["frozen_reference_root"] / scene / "summary.json"
        reference = read(reference_path) if reference_path.exists() else {}
        frozen, bad = period_values(reference, p)
        reference_valid = (reference.get("status") == "complete"
            and reference.get("integrity", {}).get("all_pass") is True
            and reference.get("actor_seed") == p["training_seed"]
            and reference.get("external_seed") == p["external_seed"] and reference.get("scenario") == scene)
        if not reference_valid or bad:
            errors.append(scene + ":invalid_frozen_reference")
        references[scene] = dict(path=str(reference_path), values=frozen)
        values = {}
        for arm in p["arms"]:
            key = scene + "/" + arm
            row = runs.get(key)
            if row is None:
                values[arm] = {name: None for name in p["periods"]}
                continue
            values[arm], bad_periods = period_values(row, p)
            bad_run = run_errors(row, p, scene, arm) + bad_periods
            errors.extend(key + ":" + error for error in bad_run)
        if all(scene + "/" + arm in runs for arm in p["arms"]):
            pairs[scene] = overlap([runs[scene + "/" + arm] for arm in p["arms"]], "native_finished")
            if not pairs[scene]["all_pass"]:
                errors.append(scene + ":pair_not_parallel")
        else:
            pairs[scene] = None
        cells.append(dict(scenario=scene, values=dict(values, frozen=frozen),
            comparisons={name: dict(
                fresh_minus_matched=differences(values["fresh"][name], values["no_fresh"][name]),
                no_fresh_minus_frozen=differences(values["no_fresh"][name], frozen[name]),
                fresh_minus_frozen=differences(values["fresh"][name], frozen[name])) for name in p["periods"]}))
    technical = not missing and not errors
    means, gates = {}, None
    for period in p["periods"]:
        means[period] = {}
        for comparison in ("fresh_minus_matched", "no_fresh_minus_frozen", "fresh_minus_frozen"):
            rows = [cell["comparisons"][period][comparison] for cell in cells]
            means[period][comparison] = (None if any(row is None for row in rows) else
                {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]})
    if technical:
        full = {cell["scenario"]: cell["comparisons"]["full_day"] for cell in cells}
        matched = {scene: full[scene]["fresh_minus_matched"]["cost_per_passenger"] for scene in p["scenarios"]}
        frozen = [full[scene]["fresh_minus_frozen"]["cost_per_passenger"] for scene in p["scenarios"]]
        gates = dict(matched_mean_negative=sum(matched.values()) / len(matched) < 0,
            stable_matched_nonpositive=matched["stable"] <= 0,
            shock_matched_negative=matched["local_speed_drop"] < 0,
            each_frozen_nonpositive=all(value <= 0 for value in frozen),
            some_frozen_negative=any(value < 0 for value in frozen))
    native = [row.get("budget", {}).get("native_seconds") for row in runs.values()]
    known = [value for value in native if finite(value)]
    complete_budget = len(known) == len(p["arms"]) * len(p["scenarios"])
    passed = technical and all(gates.values())
    result = dict(status="complete" if technical else "technical_hold", complete=technical,
        decision=dict(state="PASS_FIXED_PILOT" if passed else "CLOSE_FIXED_PILOT" if technical else "HOLD_TECHNICAL_INCOMPLETE",
                      cost_gate_pass=passed if technical else None, automatic_new_experiments=False),
        integrity=dict(all_pass=technical, errors=errors), missing=missing, cost_gates=gates,
        cells=cells, means=means, pairs=pairs, frozen_references=references,
        budget=dict(new_native_seconds=sum(known) if complete_budget else None,
            known_native_seconds=sum(known) if known else None,
            native_seconds_is_lower_bound=not complete_budget or any(
                row.get("budget", {}).get("native_seconds_is_lower_bound") for row in runs.values()),
            declared_native_seconds=p["new_native_seconds"],
            actual_by_cell={key: row.get("budget", {}) for key, row in runs.items()}),
        interpretation=dict(metric=p["primary_metric"], scope=p["inference_scope"],
            estimand=p["scientific_estimand"], frozen="Same actor6201 and environment106206; no five-actor averaging.",
            windows="All periods use15079; early and post_budget are descriptive, not additional gates."))
    return write_once(destination, result, p["summary_max_bytes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result = preflight(args.root, args.out_dir or args.root / RELATIVE) if args.preflight else analyze(args.root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
