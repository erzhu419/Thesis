"""Resident paired300 analysis; passive prediction evidence is not deployed control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_matched_response_v59_20261025.difference import (
    DirectDifferenceReadout, LINES, actual_holds, physical_cost,
)

MODELS = ("readout_line", "readout_lowdim", "fresh_Q", "no_fresh_Q", "fixed0", "fixedhigh")
LEARNERS = {"fresh_Q": "fresh", "no_fresh_Q": "no_fresh"}
AGGREGATE_COLUMNS = ("n", "direction_n", "direction_correct", "high_choices", "regret",
                     "selected_wait", "selected_ride", "selected_cost")
STATE_MODEL_COLUMNS = ("delta", "high_choice", "direction_correct", "regret",
                       "selected_wait", "selected_ride", "selected_cost")


def read(path):
    return json.loads(Path(path).read_text())


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(value, message):
    if not value:
        raise ValueError(message)


def truth(record):
    try:
        values = [physical_cost(record["outcomes"][arm]) for arm in ("low", "high")]
    except ValueError as error:
        if "exactly300" in str(error):
            return None
        raise
    return {key: values[0][key] - values[1][key] for key in ("wait", "ride", "cost")}


def decision(predicted_gain, benefit, costs):
    high = int(predicted_gain > 0.0)
    selected = costs[high]
    return dict(delta=float(predicted_gain), high_choice=high,
        direction_correct=((predicted_gain > 0) - (predicted_gain < 0) ==
                           (benefit > 0) - (benefit < 0)) if benefit != 0 else None,
        regret=selected["cost"] - min(value["cost"] for value in costs),
        selected_wait=selected["wait"], selected_ride=selected["ride"],
        selected_cost=selected["cost"])


def readers():
    return dict(readout_line=DirectDifferenceReadout(line_only=True),
                readout_lowdim=DirectDifferenceReadout())


def fit_readouts(admitted, dataset):
    require(all(row["role"] == "train" and row.get("selected_for_training") is True
                for row in admitted), "Admitted records contain heldout or unadmitted truth")
    by_id = {row["id"]: row for row in admitted}
    require(len(by_id) == len(admitted), "Duplicate admitted pair")
    fitted, fitted_ids, log = readers(), set(), []
    last = float("-inf")
    for packet in dataset["packets"]:
        receipt, index = packet["receipt"], packet["index"]
        now, ids = float(receipt["processed_at"]), receipt["pair_ids"]
        require(now >= last, "Packet receipts are not in causal order")
        last = now
        require(len(set(ids)) == len(ids) and not fitted_ids.intersection(ids),
                "A pair was released twice")
        require(all(key in by_id for key in ids), "Receipt refers to absent admitted pair")
        require(receipt["paired_records"] == 2 * len(ids), "A pair must consume two feedback slots")
        new = [by_id[key] for key in ids]
        for row in new:
            require(float(row["admitted_at"]) == now, "Pair admission differs from its packet receipt")
        if new:
            for model in fitted.values():
                model.accept(new, now=now)
                model.refit()
            fitted_ids.update(ids)
            log.append(dict(packet=index, processed_at=now, pair_ids=ids, cumulative_pairs=len(fitted_ids)))
    require(fitted_ids == set(by_id), "An admitted pair has no budget receipt")
    return fitted, log


def aggregate(rows):
    result = dict(states=len(rows), complete_truth=sum(row["benefit"] is not None for row in rows), models={})
    for name in MODELS:
        values = [row["models"][name] for row in rows if name in row["models"]]
        result["models"][name] = list(dict(n=len(values),
            direction_n=sum(value["direction_correct"] is not None for value in values),
            direction_correct=sum(value["direction_correct"] is True for value in values),
            high_choices=sum(value["high_choice"] for value in values),
            regret=sum(value["regret"] for value in values),
            selected_wait=sum(value["selected_wait"] for value in values),
            selected_ride=sum(value["selected_ride"] for value in values),
            selected_cost=sum(value["selected_cost"] for value in values)).values())
    result["readout_comparisons"] = {}
    for name in ("readout_line", "readout_lowdim"):
        result["readout_comparisons"][name] = {}
        for comparator in ("fixed0", "no_fresh_Q"):
            paired = [row for row in rows if name in row["models"] and comparator in row["models"]]
            result["readout_comparisons"][name][comparator] = dict(n=len(paired),
                selected_cost_difference=sum(row["models"][name]["selected_cost"] -
                    row["models"][comparator]["selected_cost"] for row in paired),
                wait_difference=sum(row["models"][name]["selected_wait"] -
                    row["models"][comparator]["selected_wait"] for row in paired),
                ride_difference=sum(row["models"][name]["selected_ride"] -
                    row["models"][comparator]["selected_ride"] for row in paired))
    paired = [row for row in rows if all(name in row["models"] for name in LEARNERS)]
    result["fresh_Q_minus_no_fresh_Q"] = dict(n=len(paired), **{
        key + "_difference": sum(row["models"]["fresh_Q"]["selected_" + key] -
            row["models"]["no_fresh_Q"]["selected_" + key] for row in paired)
        for key in ("wait", "ride", "cost")})
    return result


def case_analysis(root, scenario, protocol):
    collection, learning = root / "runs/collection" / scenario, root / "runs/learning" / scenario
    dataset = read(collection / "dataset.json")
    native = read(collection / "summary.json")
    admitted = read(collection / "admitted_pair_records.json")
    records = read(collection / "pair_truth.json")
    fitted, fit_log = fit_readouts(admitted, dataset)
    heldout = [row for row in records if row["role"] == "heldout"]
    expected = {line + "/" + window for line in protocol["focus_lines"] for window in protocol["heldout_windows"]}
    by_slot = {row["id"].removeprefix("heldout/"): row for row in heldout}
    require(len(by_slot) == len(heldout), "Duplicate heldout physical state")
    require(set(by_slot) <= expected, "Unexpected heldout state outside declared coverage")
    summaries, scores, missing = {}, {}, []
    for name, arm in LEARNERS.items():
        directory = learning / arm
        if not (directory / "summary.json").exists():
            summaries[name], scores[name] = None, {}
            missing.append([scenario, arm, "learning_summary_missing"])
            continue
        summaries[name] = read(directory / "summary.json")
        values = jsonlines(directory / "heldout.jsonl") if (directory / "heldout.jsonl").exists() else []
        scores[name] = {row["slot"]: row for row in values}
        require(len(scores[name]) == len(values), "Duplicate heldout model score")
        require(set(scores[name]) <= expected, "Model score invents an undeclared heldout")
    metadata = dataset["metadata"]
    audit_path = learning / "matched_audit.json"
    audit = read(audit_path) if audit_path.exists() else None
    checks = dict(collection=bool(native["integrity"]["all_pass"]),
        dataset=bool(dataset["proof"]["all_pass"]),
        feedback285=metadata["feedback_slots"] == 285,
        packets57=len(dataset["packets"]) == 57,
        rows285=sum(len(packet["rows"]) for packet in dataset["packets"]) == 285,
        ordinary267=sum(packet["receipt"]["ordinary_records"] for packet in dataset["packets"]) == 267,
        paired18=sum(packet["receipt"]["paired_records"] for packet in dataset["packets"]) == 18,
        training9=len(admitted) == 9 and {row["index"] for row in admitted} == set(range(9)),
        three_line_training=all(sum(row["line_id"] == line for row in admitted) == 3 for line in LINES),
        heldout_coverage=set(by_slot) == expected,
        training_window_coverage={(row["line_id"], row["window_name"]) for row in admitted} ==
            {(line, window) for line in LINES for window in protocol["train_windows"]},
        matched_learning_audit=audit is not None and bool(audit["all_pass"]),
        heldout_never_admitted=not set(row["id"] for row in admitted).intersection(row["id"] for row in heldout))
    for name, summary in summaries.items():
        budget = summary["budget"] if summary is not None else {}
        checks[name + "_integrity"] = summary is not None and bool(summary["integrity"]["all_pass"])
        checks[name + "_285_456_zero_after"] = (budget.get("feedback_slots") == 285 and
            budget.get("optimizer_updates") == 456 and budget.get("post_stop_updates") == 0)
        checks[name + "_coverage"] = set(scores[name]) == expected
    latest_fit = max((row["processed_at"] for row in fit_log), default=0.0)
    checks["fits_before_heldout"] = all(float(row["issued_at"]) >= latest_fit for row in heldout)
    require(checks["fits_before_heldout"], "Final readout would use feedback from after heldout issue")
    rows = []
    for slot in sorted(expected):
        if slot not in by_slot:
            missing.append([scenario, slot, "no_native_issue_in_declared_window"])
            continue
        record = by_slot[slot]
        executed = actual_holds(record)
        benefit = truth(record)
        row = dict(scenario=scenario, slot=slot, line=record["line_id"], window=record["window_name"],
            issue=record["issued_at"], sequence=record["source_sequence"], benefit=benefit,
            models={}, finite_predictions={}, support={},
            quality=dict(actual_holds=executed, overlap_seconds=record["pair_receipt"]["overlap_seconds"],
                physical_separation=record["comparison"]["physical_separation"],
                zero_next_event=[arm for arm in ("low", "high") if
                    record["outcomes"][arm]["next_event"].get("zero_duration") is True]))
        checks["parallel_" + slot] = (record["pair_receipt"]["all_pass"] and
                                      record["pair_receipt"]["overlap_seconds"] > 0.0)
        costs = [physical_cost(record["outcomes"][arm]) for arm in ("low", "high")] if benefit is not None else None
        if benefit is None:
            missing.append([scenario, slot, "incomplete_or_truncated300_truth"])
        for name, model in fitted.items():
            prediction = model.predict(record["features"])
            row["finite_predictions"][name] = prediction["gain"]
            row["support"][name] = {key: prediction[key] for key in
                ("status", "support_pairs", "headway_range", "headway_extrapolation")}
            if prediction["status"] != "supported":
                missing.append([scenario, slot, name, "unsupported_line"])
            elif benefit is not None:
                row["models"][name] = decision(prediction["gain"]["cost"], benefit["cost"], costs)
        for name in LEARNERS:
            if slot not in scores[name]:
                missing.append([scenario, slot, name, "prediction_missing"])
                continue
            saved = scores[name][slot]
            require(saved["sequence"] == record["source_sequence"] and saved["issue"] == record["issued_at"],
                    "Model prediction belongs to a different physical prefix issue")
            require(saved["line_id"] == record["line_id"], "Model prediction belongs to a different line")
            require(saved["prediction"]["holding_seconds"] == [0.0, 7.5], "Q comparison has different executed actions")
            if benefit is not None:
                row["models"][name] = decision(saved["prediction"]["delta_min"], benefit["cost"], costs)
        if benefit is not None:
            row["models"]["fixed0"] = decision(0.0, benefit["cost"], costs)
            row["models"]["fixedhigh"] = decision(1.0, benefit["cost"], costs)
            row["models"]["fixedhigh"]["delta"] = None
        rows.append(row)
    checks["complete300_truth"] = all(row["benefit"] is not None for row in rows) and len(rows) == len(expected)
    checks["all_models_supported"] = all(len(row["models"]) == len(MODELS) for row in rows)
    fit = {name: model.summary() for name, model in fitted.items()}
    mae = {}
    for name in fitted:
        valid = [row for row in rows if row["benefit"] is not None and row["finite_predictions"][name] is not None]
        mae[name] = dict(n=len(valid), **{key: sum(abs(row["finite_predictions"][name][key] - row["benefit"][key])
            for row in valid) / len(valid) if valid else None for key in ("wait", "ride", "cost")})
    return dict(integrity=dict(all_pass=all(checks.values()), checks=checks),
        matched_learning_audit=None if audit is None else dict(all_pass=audit["all_pass"],
            failed_checks=[key for key, value in audit["checks"].items() if not value],
            optimizer_overlap_seconds=audit["optimizer_overlap_seconds"]),
        metadata=metadata, heldout_coverage=dict(expected=len(expected), actual=len(heldout),
            missing=sorted(expected - set(by_slot))), finite_fit=fit, finite_fit_packets=fit_log,
        native_budget={key: value for key, value in native.get("budget", {}).items()
            if isinstance(value, (int, float, str)) or key == "pairs_by_line"},
        training_pairs=[[row["index"], row["line_id"], row["window_name"], row["issued_at"],
            row["available_at"], row["admitted_at"], truth(row)] for row in sorted(admitted, key=lambda row: row["index"])],
        finite300_heldout_mae=mae, overall=aggregate(rows),
        by_line={line: aggregate([row for row in rows if row["line"] == line]) for line in LINES},
        by_window={window: aggregate([row for row in rows if row["window"] == window])
                   for window in protocol["heldout_windows"]},
        zero_duration=[row["slot"] for row in rows if row["quality"]["zero_next_event"]],
        zero_truth=[row["slot"] for row in rows if row["benefit"] is not None and row["benefit"]["cost"] == 0.0],
        extrapolation=[row["slot"] for row in rows if row["support"]["readout_lowdim"]["headway_extrapolation"]]), rows, missing


def analyze(root=HERE):
    root = Path(root)
    protocol = read(root / "protocol.json")
    cases, states, missing = {}, [], []
    for scenario in protocol["scenarios"]:
        directory = root / "runs/collection" / scenario
        if not all((directory / name).exists() for name in
                   ("dataset.json", "summary.json", "admitted_pair_records.json", "pair_truth.json")):
            missing.append([scenario, "collection_outputs_missing"])
            continue
        case, rows, absent = case_analysis(root, scenario, protocol)
        cases[scenario] = case
        states.extend(rows)
        missing.extend(absent)
    complete = len(cases) == len(protocol["scenarios"]) and all(case["integrity"]["all_pass"] for case in cases.values())
    overall = aggregate(states)
    stable = aggregate([row for row in states if row["scenario"] == "stable"])
    eligible = []
    for name in ("readout_line", "readout_lowdim"):
        if complete and all(overall["readout_comparisons"][name][comparator]["selected_cost_difference"] < 0.0 and
            stable["readout_comparisons"][name][comparator]["selected_cost_difference"] <= 0.0
            for comparator in ("fixed0", "no_fresh_Q")):
            eligible.append(name)
    return dict(status="complete" if complete else "partial_complete", integrity=dict(all_pass=complete),
        models=list(MODELS), cases=cases, overall=overall,
        states=[dict(row, models={name: [value[key] for key in STATE_MODEL_COLUMNS]
            for name, value in row["models"].items()}) for row in states], missing=missing,
        columns=dict(aggregate_model=list(AGGREGATE_COLUMNS),
            state_model=list(STATE_MODEL_COLUMNS),
            training_pairs=["index", "line", "window", "issue", "available", "admitted", "wait_ride_cost_gain"]),
        next_control_gate=dict(eligible_models=eligible, passed=bool(eligible),
            decision="eligible_for_separate_small_control_validation" if eligible else "stop_before_control",
            criterion="Complete matched-budget coverage; heldout aggregate cost below fixed0 and no_fresh_Q, stable aggregate not increased.",
            model_selection="Both predeclared readouts are reported; this gate does not create an independent confirmation sample."),
        units=dict(wait="passenger_seconds", ride="passenger_seconds", cost="2*wait+ride",
            finite_gain="J300_low-J300_high", scaled_gain="1e-4*finite_gain",
            Q_delta="long_soft_Q_high-minus-low; finite300 outcome is a separate estimand"),
        limitations=["Paired300 branch costs do not establish lower realized full-day deployed passenger cost than MC-WM.",
            "Readouts are passive; actor, V, Q, F and trust have no response-readout gradients.",
            "Zero total-benefit pairs have regret but no unique direction; missing truth is never zero-imputed.",
            "No heldout physical response is used to fit either readout."],
        compute=dict(readout_optimizer_updates=0, added_native_control_deployments=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "analysis/summary.json")
    args = parser.parse_args()
    from experiments.mcwm_closed_loop_matched_response_v59_20261025.audit import write_audit_if_complete
    for scenario in read(args.root / "protocol.json")["scenarios"]:
        write_audit_if_complete(args.root / "runs/learning" / scenario)
    result = analyze(args.root)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    require(len(payload.encode()) <= 48000, "Summary exceeds48000-byte transfer budget")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(payload)
    print(json.dumps(dict(status=result["status"], bytes=len(payload.encode()),
        next_control_gate=result["next_control_gate"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
