"""Small resident summary of fixed-state, full-packet update attribution."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
SEEDS = (6201, 6202)
SCENARIOS = ("stable", "local_speed_drop")
DEADLINES = (9000, 9300, 11700, 18000)
METRICS = (
    "reward_scaled", "bootstrap_contribution", "bellman_target",
    "q_observed_mean", "q_on_before_actor_mean", "q_on_post_actor_mean",
    "proposal_seconds", "holding_seconds",
)
POINTS = ("before", "fresh", "no_fresh")
PHASES = {9000: "pre_shock_labels", 9300: "early_labels", 11700: "early_labels", 18000: "last_update"}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _stratum(row):
    if not row["has_transition"]:
        return "state_only"
    duration = row.get("dt_seconds")
    if not _finite(duration):
        return "missing_dt"
    if duration < 0:
        return "negative_dt"
    if duration == 0:
        return "terminal_zero_dt" if row.get("terminal") else "zero_dt"
    return "terminal" if row.get("terminal") else "positive_dt"


def summarize_probes(records):
    """Paired descriptive means; state-only probes have no invented outcomes."""
    grouped = defaultdict(list)
    counts = Counter()
    target_checks = dict(compared=0, nonadditive=0, max_absolute_residual=0.)
    for record in records:
        meta = record["metadata"]
        for row in record["probe_rows"]:
            role, raw_line = row["role"], row.get("line_id")
            line = raw_line if raw_line in ("7X", "102X", "705X", None) else "other"
            stratum = _stratum(row)
            key = (meta["seed"], meta["scenario"], line, PHASES[meta["deadline"]], role, stratum)
            grouped[key].append(row)
            counts["probe_occurrences"] += 1
            counts[stratum] += 1
            if raw_line is None:
                counts["missing_line"] += 1
            expected = METRICS if row["has_transition"] else METRICS[4:]
            for branch in POINTS:
                point = row[branch]
                for field in expected:
                    if not _finite(point.get(field)):
                        counts["unexpected_missing_metric_values"] += 1
                if row["has_transition"]:
                    parts = [point.get(field) for field in METRICS[:3]]
                    if all(_finite(value) for value in parts):
                        residual = abs(parts[0]+parts[1]-parts[2])
                        target_checks["compared"] += 1
                        target_checks["max_absolute_residual"] = max(
                            target_checks["max_absolute_residual"], residual)
                        target_checks["nonadditive"] += residual > 1e-6*(1+abs(parts[2]))

    rows = []
    for key, probes in sorted(grouped.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        n, before, fresh_delta, arm_delta = [], [], [], []
        for field in METRICS:
            values = [[row[branch].get(field) for branch in POINTS] for row in probes]
            valid = [value for value in values if all(_finite(x) for x in value)]
            n.append(len(valid))
            before.append(sum(value[0] for value in valid)/len(valid) if valid else None)
            fresh_delta.append(sum(value[1]-value[0] for value in valid)/len(valid) if valid else None)
            arm_delta.append(sum(value[1]-value[2] for value in valid)/len(valid) if valid else None)
        rows.append([*key, len(probes), n, before, fresh_delta, arm_delta])
    return dict(
        group_columns=["seed", "scenario", "line", "phase", "role", "stratum",
                       "occurrences", "paired_finite_n", "before_mean", "fresh_minus_before",
                       "fresh_minus_no_fresh"],
        metric_columns=list(METRICS), rows=rows, counts=dict(counts),
        bellman_additivity=target_checks,
        difference_definition="no_fresh-minus-before equals fresh-minus-before minus fresh-minus-no_fresh.",
        unit_definition="proposal_seconds/holding_seconds are seconds; reward and Q use resident training units.",
        phase_deadlines=PHASES,
        cohort_definition="Packet feedback is in-sample. Future/tail reference states are evaluation-only; their target and observed-Q nulls are by design.",
    )


def summarize_training(records):
    """Weight each Bellman component by actual sample uses, including all 8 steps."""
    accum = defaultdict(lambda: dict(n=0, sums=[0., 0., 0.], available=[0, 0, 0]))
    steps = Counter()
    parameters = defaultdict(list)
    for record in records:
        meta = record["metadata"]
        for arm, branch in record["arms"].items():
            steps[arm] += len(branch["training_steps"])
            for step in branch["training_steps"]:
                for batch in step["batches"]:
                    key = (meta["seed"], meta["scenario"], PHASES[meta["deadline"]], arm, batch["partition"])
                    group = accum[key]
                    n = batch["n"]
                    group["n"] += n
                    for i, field in enumerate(METRICS[:3]):
                        value = batch[field].get("mean")
                        if _finite(value):
                            group["sums"][i] += n*value
                            group["available"][i] += n
            for network, change in branch.get("parameter_change", {}).items():
                parameters[meta["seed"], meta["scenario"], arm, network].append(
                    [change["l2"], change["max_absolute"]])
    rows = []
    for key, value in sorted(accum.items()):
        means = [total/n if n else None for total, n in zip(value["sums"], value["available"])]
        rows.append([*key, value["n"], value["available"], means])
    return dict(group_columns=["seed", "scenario", "phase", "arm", "partition", "sample_uses",
                               "finite_sample_uses", "component_means"],
                metric_columns=list(METRICS[:3]), rows=rows, recorded_steps=dict(steps),
                parameter_columns=["seed", "scenario", "arm", "network", "anchors",
                                   "l2_mean", "max_absolute_max"],
                parameter_changes=[[*key, len(values), sum(x[0] for x in values)/len(values),
                                    max(x[1] for x in values)] for key, values in sorted(parameters.items())],
                interpretation="Sample uses repeat across 8 optimizer steps and anchors; these are descriptive, not independent labels.")


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rounded(value):
    if isinstance(value, float):
        return float(format(value, ".8g"))
    if isinstance(value, list):
        return [_rounded(part) for part in value]
    if isinstance(value, dict):
        return {key: _rounded(part) for key, part in value.items()}
    return value


def analyze(root):
    records, checks, runs, pending, missing = [], {}, {}, [], []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            directory = root/"runs"/f"seed{seed}"/scenario
            summary_path = directory/"summary.json"
            if not summary_path.exists():
                if (directory/"failure.json").exists():
                    checks[f"{seed}/{scenario}/native_complete"] = False
                else:
                    pending.append(f"{seed}/{scenario}")
                continue
            summary = json.loads(summary_path.read_text())
            key = f"{seed}/{scenario}"
            checks[key+"/native_complete"] = summary["status"] == "complete"
            checks[key+"/native_integrity"] = summary["integrity"]["all_pass"] is True
            checks.update({key+"/"+name: value is True
                           for name, value in summary["integrity"]["checks"].items()})
            checks[key+"/cell"] = summary["cell"] == dict(seed=seed, scenario=scenario, origin_arm="fresh")
            expected_budget = dict(native_replay_seconds=24000, feedback_slots=285, main_updates=456,
                shadow_anchors=4, shadow_arms=2, shadow_updates=64, extra_target_labels=0,
                real_control_branches=0)
            checks[key+"/budget"] = summary["budget"] == expected_budget
            checks[key+"/full_reference_exact"] = summary["replay_exact"] == dict(
                trace=True, updates=True, receipts=True, cost=True, metrics=True)
            checks[key+"/post_budget_456_frozen"] = summary["post_budget_counts"] == dict(before=456, after=456)
            shadow_path = directory/"shadows.jsonl"
            if not shadow_path.exists():
                checks[key+"/resident_shadows_present"] = False
                continue
            cell_records = _read_jsonl(shadow_path)
            checks[key+"/four_packet_deadlines"] = [row["metadata"]["deadline"] for row in cell_records] == list(DEADLINES)
            checks[key+"/shadow_cells_match"] = all(row["metadata"]["seed"] == seed
                and row["metadata"]["scenario"] == scenario for row in cell_records)
            checks[key+"/eight_steps_each_arm"] = all(
                set(row["arms"]) == {"fresh", "no_fresh"}
                and all(branch["updates"] == len(branch["training_steps"]) == 8
                        for branch in row["arms"].values()) for row in cell_records)
            checks[key+"/shadow_integrity"] = all(row["integrity"]
                and all(value is True for value in row["integrity"].values()) for row in cell_records)
            for row in cell_records:
                absent = row["metadata"].get("missing_probes", [])
                if absent:
                    missing.append(dict(seed=seed, scenario=scenario,
                                        deadline=row["metadata"]["deadline"], probes=absent))
            records.extend(cell_records)
            runs[key] = dict(budget=summary["budget"], replay_exact=summary["replay_exact"],
                             post_budget_counts=summary["post_budget_counts"])
    if pending:
        return dict(status="pending", pending=pending, integrity=dict(all_pass=False, checks=checks))
    probes = summarize_probes(records)
    checks["probe_bellman_additivity"] = probes["bellman_additivity"]["nonadditive"] == 0
    checks["recorded_all_four_days"] = len(runs) == 4
    passed = all(checks.values())
    return _rounded(dict(status="complete" if passed else "integrity_failed",
        integrity=dict(all_pass=passed, checks=checks), runs=runs,
        budget=dict(main_replay_days=4, main_replay_seconds=96000, main_updates=1824,
                    main_feedback_slots=1140, diagnostic_optimizer_updates=256,
                    extra_target_labels=0, real_control_branches=0),
        fixed_state_probes=probes, training_batches=summarize_training(records),
        missing_state_probes=missing,
        next_decision="Review paired actor and F-holding changes, Bellman components and action-value shifts before deciding whether a few same-prefix control branches are informative. No branches submitted by this analysis.",
        interpretation="Fresh-origin prefixes only; no_fresh shadows change optimizer supervision at a common snapshot. Q, Bellman targets and proposal changes are not observed passenger-cost benefits. Future reference states are evaluated only, never supplied to learning.",
        summary_precision="8 significant digits in this small summary; full-precision resident records retained."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        result = analyze(args.root)
        if result["status"] != "pending" or not args.wait:
            break
        time.sleep(30)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
    if len(encoded.encode("utf-8")) > 60000:
        raise ValueError("Resident summary exceeds the 60 KB small-pull contract")
    with args.out.open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=result["status"], output=str(args.out), bytes=len(encoded.encode("utf-8")))))
    return 0 if result["status"] in ("complete", "pending") else 1


if __name__ == "__main__":
    raise SystemExit(main())
