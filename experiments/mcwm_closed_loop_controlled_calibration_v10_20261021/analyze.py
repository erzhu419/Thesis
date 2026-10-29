"""Small resident analysis of four controlled common-flow development days."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ENDPOINTS = ("global_next", "same_vehicle_next", "fixed300")
METRICS = ("dt", "wait", "ride", "cost")
GLOBAL_METRICS = ("dt", "cost")
ARMS = ("endpoint_candidate", "base_analytic", "global58")
MAX_BYTES = 200000


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _low(value):
    return abs(value) <= 1e-5


def _compact_numbers(value):
    """Keep eight significant digits in collected metrics; source stays resident."""
    if isinstance(value, float):
        return float(format(value, ".8g"))
    if isinstance(value, dict):
        return {key: _compact_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_numbers(item) for item in value]
    return value


def _moments(values):
    values = [float(value) for value in values if _finite(value)]
    return [len(values), sum(values)/len(values), sum(abs(x) for x in values)/len(values)] if values else [0, None, None]


def _errors(rows, metrics):
    valid = [row for row in rows if all(_finite(row.get(name)) for name in metrics)]
    n = len(valid)
    return dict(n=n,
        mae=[sum(abs(row[name]) for row in valid)/n for name in metrics] if n else None,
        mse=[sum(row[name]**2 for row in valid)/n for name in metrics] if n else None)


def _phase(row):
    return f"{row['role']}_{row['issue_phase']}"


def _duration(row):
    actual = row.get("truth", {}).get("global_next", {})
    if actual.get("status") != "complete" or not _finite(actual.get("dt")):
        return "missing_global"
    return "zero_dt" if actual["dt"] == 0 else "positive_dt"


def _selected_coverage(records, lines, *, positive_global=False):
    return {line: {action: sum(row["line_id"] == line and row.get("selected_for_training")
        and (not positive_global or row.get("endpoints", {}).get("global_next", {}).get("dt", 0) > 0)
        and _low(row["hold_seconds"]) == (action == "low") for row in records)
        for action in ("low", "high")} for line in lines}


def _heldout_coverage(records, lines):
    return {line: {action: sum(row["line_id"] == line and row["role"] == "heldout"
        and row["status"] == "complete" and _low(row["hold_seconds"]) == (action == "low")
        for row in records) for action in ("low", "high")} for line in lines}


def summarize_group(rows):
    """All fair global errors use exactly the same supported issue records."""
    endpoint_errors = {name: [] for name in ENDPOINTS}
    fair_errors = {name: [] for name in ARMS}
    unmatched = Counter()
    gains = defaultdict(list)
    fair_gains = defaultdict(list)
    groups = {name: {action: [] for action in ("low", "high")} for name in ENDPOINTS}
    nonfinite = 0
    for row in rows:
        candidate = row.get("endpoint_prediction", {})
        error = row.get("endpoint_error", {}).get("endpoints") or {}
        supported = candidate.get("status") == "supported"
        for name in ENDPOINTS:
            signed = error.get(name, {}).get("signed_errors")
            if signed is not None:
                endpoint_errors[name].append(signed)
                nonfinite += sum(not _finite(signed.get(metric)) for metric in METRICS)
            actual = row.get("truth", {}).get(name, {})
            if actual.get("status") == "complete":
                cost = actual.get("cost")
                if _finite(cost):
                    groups[name]["low" if _low(row["hold_seconds"]) else "high"].append(cost)
                else:
                    nonfinite += 1
        global_error = error.get("global_next", {}).get("signed_errors")
        common = row.get("global_comparator", {})
        reasons = []
        if not supported:
            reasons.append("candidate_unsupported")
        if _duration(row) == "positive_dt" and candidate.get("endpoint_status", {}).get("global_next") != "supported":
            reasons.append("candidate_positive_global_unsupported")
        if not global_error or not all(_finite(global_error.get(k)) for k in GLOBAL_METRICS):
            reasons.append("candidate_global_missing_or_nonfinite")
        for arm in ("base_analytic", "global58"):
            value = common.get(arm, {})
            if not all(_finite(value.get(k+"_signed_error")) for k in GLOBAL_METRICS):
                reasons.append(arm+"_missing_or_nonfinite")
                nonfinite += sum(k+"_signed_error" in value and not _finite(value[k+"_signed_error"])
                                 for k in GLOBAL_METRICS)
        if reasons:
            unmatched.update(reasons)
            unmatched["total"] += 1
        else:
            fair_errors["endpoint_candidate"].append(global_error)
            for arm in ("base_analytic", "global58"):
                fair_errors[arm].append({k: common[arm][k+"_signed_error"] for k in GLOBAL_METRICS})
            decoded = dict(endpoint_candidate=(candidate.get("gains") or {}).get("global_next"),
                **{arm: common[arm].get("cost_gain_low_minus_high") for arm in ("base_analytic", "global58")})
            raw = dict(endpoint_candidate=candidate.get("raw_global_next_gain"),
                **{arm: common[arm].get("raw_cost_gain_low_minus_high") for arm in ("base_analytic", "global58")})
            for label, values in (("decoded", decoded), ("raw", raw)):
                if all(_finite(value) for value in values.values()):
                    for arm, value in values.items():
                        fair_gains[arm+"_"+label].append(value)
                elif label == "raw" and values["endpoint_candidate"] is None:
                    unmatched["raw_global_unsupported"] += 1
                else:
                    nonfinite += sum(value is not None and not _finite(value) for value in values.values())
        if supported:
            for name, value in (candidate.get("gains") or {}).items():
                if name != "global_next" and value is not None:
                    gains["endpoint_"+name].append(value)
                    nonfinite += not _finite(value)
    descriptive = {}
    for name, actions in groups.items():
        means = {action: [len(values), sum(values)/len(values) if values else None]
                 for action, values in actions.items()}
        descriptive[name] = dict(**means, low_minus_high=
            means["low"][1]-means["high"][1] if all(means[a][0] for a in actions) else None)
    return dict(n=len(rows), statuses=dict(Counter(row["status"] for row in rows)),
        actions={action: sum(_low(row["hold_seconds"]) == (action == "low") for row in rows)
                 for action in ("low", "high")},
        selected_train=sum(row.get("selected_for_training", False) for row in rows),
        endpoint_errors={name: _errors(values, METRICS) for name, values in endpoint_errors.items()},
        fair_global_errors={name: _errors(values, GLOBAL_METRICS) for name, values in fair_errors.items()},
        fair_global_gains={name: _moments(values) for name, values in fair_gains.items()},
        unmatched=dict(unmatched), predicted_gains={name: _moments(values) for name, values in gains.items()},
        descriptive_unpaired_cost=descriptive, nonfinite_values=nonfinite)


def expected_paths(root, protocol):
    return [(dict(seed=seed, scenario=scenario), Path(root)/scenario/f"seed{seed}")
            for seed, scenario in product(protocol["training_seeds"], protocol["scenarios"])]


def analyze(root, protocol=None):
    p = json.loads((HERE/"protocol.json").read_text()) if protocol is None else protocol
    cells, strata = [], []
    all_rows = []
    coverage_gaps = []
    for cell, path in expected_paths(root, p):
        if not (path/"summary.json").is_file():
            failure = json.loads((path/"failure.json").read_text()) if (path/"failure.json").is_file() else None
            cells.append(dict(**cell, status="failed" if failure else "missing", failure=failure))
            continue
        value = json.loads((path/"summary.json").read_text())
        records = value.get("controlled_records", [])
        selected = {row["sequence"]: row for row in records}
        rows = [{**row, "selected_for_training": bool(selected.get(row["sequence"], {}).get("selected_for_training"))}
                for row in value.get("evaluations", [])]
        coverage = _selected_coverage(records, p["target_lines"])
        positive_coverage = _selected_coverage(records, p["target_lines"], positive_global=True)
        heldout = _heldout_coverage(records, p["target_lines"])
        for label, counts, minimum in (("selected_train", coverage, 2),
                ("selected_positive_global", positive_coverage, 2), ("complete_heldout", heldout, 1)):
            for line, actions in counts.items():
                for action, n in actions.items():
                    if n < minimum:
                        coverage_gaps.append(dict(**cell, scope=label, line=line, action=action, n=n, minimum=minimum))
        budget = value.get("budget", {})
        cells.append(dict(**cell, status=value["status"], integrity=value.get("integrity", {}),
            initializer=value.get("initializer"), budget={key: budget.get(key) for key in (
                "target_labels", "packets", "calibration_admitted", "zero_global_admitted",
                "calibration_pending_or_unused", "completed_calibration_unused", "native_seconds",
                "policy_updates", "world_updates", "world_refits", "trust_updates", "global58_labels",
                "endpoint_labels", "heldout_evaluation_issues", "training_stopped_at")},
            selected_train_coverage=coverage, selected_positive_global_coverage=positive_coverage,
            complete_heldout_coverage=heldout,
            endpoint_statuses=value.get("endpoints", {}),
            design=value.get("endpoint_model", {}).get("design", {}),
            endpoint_designs=value.get("endpoint_model", {}).get("endpoint_designs", {}),
            global58_design_columns=value.get("global58", {}).get("columns"),
            global58_vs_issue_slots=dict(global58_labels=budget.get("global58_labels"),
                issue_slots=budget.get("target_labels"), planned_issue_slots=p["target_record_budget"],
                zero_global_slots=budget.get("zero_global_admitted")),
            cost=value.get("cost"), missing_evaluations=len(set(selected)-{row["sequence"] for row in rows})))
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["line_id"], _phase(row), _duration(row))].append(row)
        for (line, phase, duration), group in sorted(grouped.items()):
            strata.append(dict(**cell, line=line, phase=phase, duration=duration,
                               **summarize_group(group)))
        all_rows.extend(dict(**cell, **row) for row in rows)
    valid = [cell for cell in cells if cell["status"] == "complete" and cell.get("integrity", {}).get("all_pass") is True]
    aggregate = defaultdict(list)
    for row in all_rows:
        aggregate[(row["scenario"], _phase(row), _duration(row))].append(row)
    overall = [dict(scenario=scenario, phase=phase, duration=duration, **summarize_group(group))
               for (scenario, phase, duration), group in sorted(aggregate.items())]
    finite = all(group["nonfinite_values"] == 0 for group in strata)
    gate = dict(all_four_days_complete=len(valid) == len(cells),
        selected_train_two_actions_per_line=not any(gap["scope"] == "selected_train" for gap in coverage_gaps),
        selected_positive_global_two_actions_per_line=not any(gap["scope"] == "selected_positive_global" for gap in coverage_gaps),
        complete_heldout_two_actions_per_line=not any(gap["scope"] == "complete_heldout" for gap in coverage_gaps),
        finite_reported_errors=finite,
        at_least_one_supported_heldout_comparison=any(group["phase"] == "heldout_after"
            and group["duration"] == "positive_dt"
            and group["fair_global_errors"]["endpoint_candidate"]["n"] > 0 for group in strata))
    result = dict(status="complete" if gate["all_four_days_complete"] else "incomplete",
        integrity=dict(expected_days=len(cells), complete_days=len(valid),
            failed_days=sum(c["status"] == "failed" for c in cells),
            missing_days=sum(c["status"] == "missing" for c in cells),
            invalid_days=sum(c["status"] not in ("complete", "missing", "failed") for c in cells)),
        schema=dict(endpoint_metric_order=list(METRICS), global_metric_order=list(GLOBAL_METRICS),
            numeric_precision_significant_digits=8,
            gain_statistics_order=["n", "mean", "mean_absolute"],
            descriptive_action_order=["n", "mean_cost"],
            gain_sign="low cost minus high cost; positive favors 7.5 seconds"),
        cells=cells, strata=strata, overall=overall,
        independent_confirmation=dict(status="not_submitted", review_ready=all(gate.values()),
            checks=gate, coverage_gaps=coverage_gaps,
            paired_native_truth_required=True),
        interpretation=dict(
            comparator="Passive base, 58-column residual and endpoint readouts on each identical new controlled flow.",
            fairness="Global MAE/MSE use the identical candidate-supported records in all three arms; unmatched counts remain explicit.",
            endpoints="Same-car and 300-second responses remain separate targets; only true global transitions fit the global comparator.",
            zero_duration="Zero-duration strata are separate. Decoded analytic zero does not establish learned slope improvement; inspect raw gains.",
            randomized_groups="Descriptive assigned-action means across different states are not paired same-state counterfactual truth; no causal direction-success fraction is computed.",
            scope="Four development days; no candidate controller deployment, full-day control benefit or automatic independent confirmation."))
    return _compact_numbers(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    p = json.loads((HERE/"protocol.json").read_text())
    started = time.monotonic()
    if args.wait:
        while True:
            paths = [path for _, path in expected_paths(args.root, p)]
            terminal = sum((path/"summary.json").is_file() or (path/"failure.json").is_file() for path in paths)
            print("SCHEDULEURM_PROGRESS "+json.dumps(dict(phase="waiting_controlled_days", current=terminal,
                total=len(paths), unit="days", elapsed_seconds=time.monotonic()-started)), flush=True)
            if terminal == len(paths) or time.monotonic()-started >= 72*3600:
                break
            time.sleep(60)
    value = analyze(args.root, p)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
    if len(payload.encode()) > MAX_BYTES:
        raise RuntimeError("Resident analysis exceeds the 200KB collection budget; reduce summary before collecting")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(json.dumps(dict(phase="summary_written", status=value["status"], bytes=args.out.stat().st_size)), flush=True)


if __name__ == "__main__":
    main()
