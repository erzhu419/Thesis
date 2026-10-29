"""Resident, small-pull description of cumulative same-stream shadow updates."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
SEEDS = (6201, 6202)
SCENARIOS = ("stable", "local_speed_drop")
LINES = ("7X", "102X", "705X")
DEADLINES = tuple(range(9000, 18001, 300))
CANDIDATES = ["initial", "fresh", "no_fresh", "zero", "7.5s"]
PAIR_METRICS = [
    "proposal_fresh_minus_no_seconds", "holding_fresh_minus_no_seconds",
    "fresh_critic_mean_adv_fresh_minus_no", "fresh_critic_min_adv_fresh_minus_no",
    "no_critic_mean_adv_fresh_minus_no", "no_critic_min_adv_fresh_minus_no",
    "fresh_critic_mean_adv_7.5_minus_zero", "fresh_critic_min_adv_7.5_minus_zero",
    "no_critic_mean_adv_7.5_minus_zero", "no_critic_min_adv_7.5_minus_zero",
]
TRAJECTORY_METRICS = PAIR_METRICS+[
    "mean_absolute_holding_fresh_minus_no_seconds", "maximum_absolute_holding_fresh_minus_no_seconds"]
BASELINE_METRICS = [f"{arm}_{stat}_{contrast}" for arm in ("fresh", "no_fresh")
                    for stat in ("mean", "min")
                    for contrast in ("zero_level_shift", "fresh_minus_initial", "no_minus_initial")]
BELLMAN_METRICS = ("reward_scaled", "bootstrap_contribution", "bellman_target")
EXPECTED_BUDGET = dict(native_replay_seconds=24000, feedback_slots=285, main_updates=456,
    shadow_packets=31, shadow_arms=2, shadow_updates=496, extra_target_labels=0,
    real_control_branches=0)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values):
    finite = [x for x in values if _finite(x)]
    return sum(finite)/len(finite) if finite else None


def _diff(values, a, b):
    if len(values) <= max(a, b) or not all(_finite(values[i]) for i in (a, b)):
        return None
    return values[a]-values[b]


def _bank_id(row):
    return str(row.get("bank_id", row.get("index")))


def _descriptor(row):
    return [_bank_id(row), row.get("line_id"), row.get("bank_deadline"),
            row.get("role"), row.get("canonical_clock"), row.get("issue_time")]


def _metrics(row):
    values = [_diff(row.get("proposal_seconds", []), 1, 2),
              _diff(row.get("holding_seconds", []), 1, 2)]
    for arm in ("fresh", "no_fresh"):
        for stat in ("mean", "min"):
            values.append(_diff(row.get("q_"+stat, {}).get(arm, []), 1, 2))
    for arm in ("fresh", "no_fresh"):
        for stat in ("mean", "min"):
            values.append(_diff(row.get("q_"+stat, {}).get(arm, []), 4, 3))
    return values


def _display_metrics(values):
    # Micro-Q shortens small action contrasts without reducing significant digits.
    return [value*1e6 if 2 <= i < 10 and value is not None else value
            for i, value in enumerate(values)]


def _baseline_metrics(row, initial):
    result = []
    for arm in ("fresh", "no_fresh"):
        for stat in ("mean", "min"):
            current = row.get("q_"+stat, {}).get(arm, [])
            first = initial.get("q_"+stat, {}).get(arm, []) if initial else []
            shift = current[3]-first[3] if len(current) == len(first) == 5 and all(
                _finite(x) for x in (current[3], first[3])) else None
            result.extend([shift, _diff(current, 1, 0), _diff(current, 2, 0)])
    return result


def persistence(values):
    """Describe sign/range without assigning a scientific success threshold."""
    finite = [x for x in values if _finite(x)]
    runs, last_sign, length = {-1: 0, 0: 0, 1: 0}, None, 0
    for value in values:
        if not _finite(value):
            last_sign, length = None, 0
            continue
        sign = int(value > 0)-int(value < 0)
        length = length+1 if sign == last_sign else 1
        runs[sign] = max(runs[sign], length)
        last_sign = sign
    return [len(finite), min(finite) if finite else None, max(finite) if finite else None,
            values[-1] if values and _finite(values[-1]) else None,
            runs[-1], runs[0], runs[1]]


def summarize_bank(records):
    """Each checkpoint uses the same bank; no repeated checkpoint is a new state."""
    grouped, all_states, baselines, alias_counts = defaultdict(list), {}, [], Counter()
    invalid, consistency, zero_duration, state_only = 0, True, 0, 0
    feedback_counts = Counter()
    initial = {_bank_id(row): row for row in records[0].get("initial_probe_rows", [])} if records else {}
    first_descriptors, first_ids = None, None
    for record in records:
        deadline = record["metadata"]["deadline"]
        for feedback in record.get("packet_feedback", []):
            duration = feedback.get("dt_seconds")
            stratum = "missing_dt" if not _finite(duration) else "negative_dt" if duration < 0 else (
                "zero_dt" if duration == 0 else "positive_dt")
            feedback_counts["rows"] += 1
            feedback_counts[stratum] += 1
            feedback_counts["terminal"] += bool(feedback.get("terminal"))
        probes = record.get("probe_rows", [])
        descriptors = {_bank_id(row): _descriptor(row) for row in probes}
        if first_descriptors is None:
            first_descriptors, first_ids = descriptors, set(descriptors)
        consistency &= descriptors == first_descriptors and len(descriptors) == len(probes)
        per_line = defaultdict(list)
        for row in probes:
            bank_id, line = _bank_id(row), row.get("line_id")
            if line not in LINES:
                line = "other"
            values = _metrics(row)
            invalid += sum(value is None for value in values)
            per_line[line].append(values)
            all_states[bank_id] = row
            h, p = row.get("holding_seconds", []), row.get("proposal_seconds", [])
            if len(h) == len(p) == 5:
                alias_counts[line, "fresh_no_same_F"] += h[1] == h[2]
                alias_counts[line, "fresh_no_distinct_proposal_same_F"] += p[1] != p[2] and h[1] == h[2]
                alias_counts[line, "zero_7.5_same_F"] += h[3] == h[4]
                alias_counts[line, "fresh_zero_holding"] += h[1] == 0
                alias_counts[line, "no_zero_holding"] += h[2] == 0
                alias_counts[line, "scored_occurrences"] += 1
            if row.get("dt_seconds") == 0:
                zero_duration += 1
            if not row.get("has_transition", False):
                state_only += 1
        for line in LINES:
            group = per_line[line]
            values = [_mean([row[i] for row in group]) for i in range(len(PAIR_METRICS))]
            magnitudes = [abs(row[1]) for row in group if _finite(row[1])]
            values.extend([_mean(magnitudes), max(magnitudes) if magnitudes else None])
            grouped[line].append(_display_metrics(values))

    trajectories, sustained = [], []
    for line in LINES:
        rows = grouped[line]
        trajectories.append([line, sum(row.get("line_id") == line for row in all_states.values()), rows])
        for metric_index in (1, 3, 5, 10):
            sustained.append([line, metric_index, persistence([row[metric_index] for row in rows])])
    final = []
    for bank_id, row in sorted(all_states.items()):
        final.append([bank_id, row.get("line_id"), row.get("bank_deadline"),
                      row.get("canonical_clock"), *_display_metrics(_metrics(row))])
    for line in LINES:
        states = [row for row in all_states.values() if row.get("line_id") == line]
        metrics = [_baseline_metrics(row, initial.get(_bank_id(row))) for row in states]
        means = [_mean([value[i] for value in metrics]) for i in range(len(BASELINE_METRICS))]
        baselines.append([line, len(states), [value*1e6 if value is not None else None for value in means]])
    return dict(trajectories=trajectories, persistence=sustained, final_by_state=final,
        final_baseline_contrasts=baselines,
        F_alias_counts=[[line, [alias_counts[line, key] for key in (
            "scored_occurrences", "fresh_no_same_F", "fresh_no_distinct_proposal_same_F",
            "zero_7.5_same_F", "fresh_zero_holding", "no_zero_holding")]] for line in LINES],
        fixed_bank_consistent=bool(consistency), initial_bank_complete=set(initial) == (first_ids or set()),
        distinct_states=len(first_ids or ()),
        missing_metric_values=invalid, state_only_occurrences=state_only,
        zero_duration_occurrences=zero_duration, packet_feedback_counts=dict(feedback_counts))


def summarize_training(records):
    groups = defaultdict(list)
    nonadditive = 0
    for record in records:
        for arm, branch in record["arms"].items():
            packet = defaultdict(lambda: [0, [0., 0., 0.]])
            for step in branch["training_steps"]:
                for batch in step["batches"]:
                    parts = [batch[field]["mean"] for field in BELLMAN_METRICS]
                    if not all(_finite(value) for value in parts):
                        continue
                    nonadditive += abs(parts[0]+parts[1]-parts[2]) > 1e-6*(1+abs(parts[2]))
                    n = batch["n"]
                    group = packet[batch["partition"]]
                    group[0] += n
                    for i, value in enumerate(parts):
                        group[1][i] += n*value
            for partition, (n, sums) in packet.items():
                groups[arm, partition].append((n, sums))
    rows = []
    for (arm, partition), packets in sorted(groups.items()):
        n = sum(value[0] for value in packets)
        combined = [sum(value[1][i] for value in packets)/n for i in range(3)]
        first, last = packets[0], packets[-1]
        rows.append([arm, partition, n, combined,
                     [x/first[0] for x in first[1]], [x/last[0] for x in last[1]]])
    return dict(rows=rows, bellman_nonadditive_batches=nonadditive)


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _round(value, precision=4):
    if isinstance(value, float):
        rounded = float(format(value, f".{precision}g"))
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, list):
        return [_round(item, precision) for item in value]
    if isinstance(value, dict):
        return {key: _round(item, precision) for key, item in value.items()}
    return value


def analyze(root):
    cells, pending, failures, missing = [], [], [], []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            path = root/"runs"/f"seed{seed}"/scenario
            cell = f"{seed}/{scenario}"
            if not (path/"summary.json").exists():
                (failures if (path/"failure.json").exists() else pending).append(cell)
                continue
            summary = json.loads((path/"summary.json").read_text())
            checks = {"native_complete": summary.get("status") == "complete",
                "native_integrity": summary.get("integrity", {}).get("all_pass") is True,
                "cell": summary.get("cell") == dict(seed=seed, scenario=scenario, origin_arm="fresh"),
                "budget": summary.get("budget") == EXPECTED_BUDGET,
                "full_reference_exact": summary.get("replay_exact") == dict(
                    trace=True, updates=True, receipts=True, cost=True, metrics=True),
                "post_budget_zero_training": summary.get("post_budget_counts") == dict(before=456, after=456)}
            checks["post_budget_zero_shadow_training"] = summary.get("post_budget_shadow_counts") == dict(
                before=dict(fresh=248, no_fresh=248), after=dict(fresh=248, no_fresh=248))
            checks.update({"native/"+key: value is True for key, value in
                           summary.get("integrity", {}).get("checks", {}).items()})
            if not (path/"shadows.jsonl").exists():
                failures.append(cell+"/missing_shadows")
                continue
            records = _read_jsonl(path/"shadows.jsonl")
            checks["31_packet_deadlines"] = [r["metadata"]["deadline"] for r in records] == list(DEADLINES)
            checks["shadow_cell_match"] = all(r["metadata"]["seed"] == seed and
                r["metadata"]["scenario"] == scenario for r in records)
            checks["persistent_update_counts"] = all(r.get("packet_index") == i and
                r.get("cumulative_updates") == 8*i for i, r in enumerate(records, 1))
            checks["eight_steps_each_arm"] = all(set(r.get("arms", {})) == {"fresh", "no_fresh"} and
                all(b.get("updates") == len(b.get("training_steps", [])) == 8 for b in r["arms"].values())
                for r in records)
            checks["shadow_integrity"] = all(r.get("integrity") and
                all(value is True for value in r["integrity"].values()) for r in records)
            checks["fresh_main_parameters_Adam_RNG"] = all(r.get("fresh_main_replay", {}).get("exact") is True and
                r["fresh_main_replay"].get("checks", {}).get("global_rng") is True for r in records)
            checks["common_candidates"] = all(r.get("candidates") == CANDIDATES for r in records)
            checks["twelve_states_or_explicit_missing"] = bool(records) and len(
                records[0].get("probe_rows", []))+len(records[0]["metadata"].get("missing_probes", [])) == 12
            bank = summarize_bank(records)
            training = summarize_training(records)
            checks["fixed_bank_and_clocks"] = bank.pop("fixed_bank_consistent")
            checks["initial_bank_complete"] = bank.pop("initial_bank_complete")
            checks["finite_candidate_metrics"] = bank["missing_metric_values"] == 0
            checks["bellman_additivity"] = training["bellman_nonadditive_batches"] == 0
            for record in records:
                absent = record["metadata"].get("missing_probes", [])
                if absent and not any(row[0] == cell for row in missing):
                    missing.append([cell, absent])
            failed = [key for key, value in checks.items() if not value]
            if failed:
                failures.extend(cell+"/"+key for key in failed)
            cells.append(dict(cell=cell, integrity=dict(all_pass=not failed,
                checked=len(checks), failed=failed, exact_main_packets=sum(
                    r.get("fresh_main_replay", {}).get("exact") is True for r in records)),
                bank=bank, bellman=training))
    status = "integrity_failed" if failures else "pending" if pending else "complete"
    return _round(dict(status=status, integrity=dict(all_pass=status == "complete",
        cells=len(cells), failed=failures), pending=pending,
        budget=dict(main_replay_days=4, main_replay_seconds=96000, main_updates=1824,
            main_feedback_slots=1140, diagnostic_optimizer_updates=1984,
            extra_target_labels=0, real_control_branches=0), cells=cells,
        missing_state_probes=missing, columns=dict(deadlines=list(DEADLINES),
            trajectory=["line", "distinct_bank_states", "checkpoint_metric_means"],
            trajectory_metrics=TRAJECTORY_METRICS,
            persistence=["line", "trajectory_metric_index", "statistics"],
            persistence_statistics=["finite_checkpoints", "min", "max", "last", "longest_negative_run",
                "longest_zero_run", "longest_positive_run"],
            final_by_state=["bank_id", "line", "bank_deadline", "canonical_clock",
                *PAIR_METRICS],
            bank_roles={"9000": "future_reference_state", "9300": "future_reference_state",
                        "11700": "future_reference_state", "18000": "tail_heldout_state"},
            final_baseline_contrasts=["line", "states", "metric_means"],
            baseline_metrics=BASELINE_METRICS,
            units="Proposal/holding columns are seconds. Trajectory/final-state Q contrasts and baseline metrics are micro-Q (multiply native training Q by 1e6); Bellman means remain native training units.",
            F_alias_counts=["line", ["scored_occurrences", "fresh_no_same_F",
                "fresh_no_distinct_proposal_same_F", "zero_7.5_same_F", "fresh_zero_holding", "no_zero_holding"]],
            bellman=["arm", "partition", "sample_uses", "all_packets_mean", "first_packet_mean", "last_packet_mean"],
            bellman_metrics=list(BELLMAN_METRICS)),
        interpretation="Same fresh-origin prefix and factual stream; independent cumulative parameters and Adam after 9000. Fixed real-clock states are evaluation-only. Line means reuse the same bank states at every checkpoint; state/window endpoint rows remain separate. Q/holding differences are not realized passenger-cost effects. Training sample uses are not independent labels.",
        next_decision="Review sustained same-candidate action advantages and F-holding separation before selecting a few informative real branches. This analysis submits no branches and applies no scientific sign gate.",
        summary_precision="4 significant digits; full-precision records remain resident. Exact replay/RNG checks precede rounding."))


def encode(result):
    value = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
    if len(value.encode("utf-8")) > 50000:
        reduced = _round(result, 3)
        reduced["summary_precision"] = reduced["summary_precision"].replace(
            "4 significant digits", "3 significant digits (50 KB size fallback)")
        value = json.dumps(reduced, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
        if len(value.encode("utf-8")) > 50000:
            raise ValueError("Resident summary exceeds the 50 KB small-pull contract at 3 significant digits")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            result = analyze(args.root)
            if result["status"] != "pending" or not args.wait:
                break
            time.sleep(30)
        encoded = encode(result)
        with args.out.open("x") as handle:
            handle.write(encoded)
    except Exception as error:
        with (args.root/"analysis_failure.json").open("x") as handle:
            json.dump(dict(status="analysis_failed", error_type=type(error).__name__,
                           error=str(error)), handle)
        raise
    print(json.dumps(dict(status=result["status"], output=str(args.out), bytes=len(encoded.encode("utf-8")))))
    return 0 if result["status"] in ("complete", "pending") else 1


if __name__ == "__main__":
    raise SystemExit(main())
