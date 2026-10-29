"""Small paired-action truth readout; no model loading or trajectory transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = ("one_step", "trace")
STATE_COLUMNS = ["scenario", "slot", "sequence", "issue", "trip", "status", "holding_seconds",
    "physical_status_departure_gap_arrival_gap", "native_registered_seconds",
    "next_dt_low_high_benefit_wait_ride_J_reward_high_minus_low", "benefit300_wait_ride_J",
    "models", "trace_minus_one_step_cost", "quality"]
MODEL_COLUMNS = ["Qmin_high_minus_low", "Qmean_high_minus_low", "choice_0low_1high", "regret300",
    "strict_sign_matches_benefit300", "own_actor_F_holding", "cost_minus_fixed0_wait_ride_J"]
GROUP_COLUMNS = ["slots", "selected", "scored300", "truth_ties", "physical_separable_nonseparable_unknown",
    "zero_next_event_states", "models", "trace_minus_one_step_cost"]
GROUP_MODEL_COLUMNS = ["nonzero_truth_direction_correct", "nonzero_truth_states", "Q_ties", "high_choices",
    "regret_mean", "regret_max", "regret_sum", "cost_minus_fixed0_mean_wait_ride_J", "worse_than_fixed0"]
DELTA_COLUMNS = ["states", "mean", "sum", "improved", "worsened", "tied"]


def read(path):
    return json.loads(Path(path).read_text())


def sign(value):
    return (value > 0) - (value < 0)


def resolve(path, relative):
    path = Path(path)
    return path if path.is_absolute() else relative/path


def benefit(low, high, fixed=False):
    if low["status"] != "complete" or high["status"] != "complete":
        return None
    if fixed and (low["dt"] != 300. or high["dt"] != 300.):
        return None
    return [low[key] - high[key] for key in ("wait", "ride", "cost")]


def delta_stats(values):
    values = list(values)
    return [len(values), sum(values)/len(values) if values else None, sum(values) if values else None,
        sum(value < 0 for value in values), sum(value > 0 for value in values), sum(value == 0 for value in values)]


def common_prefix(root, before=9000.):
    """Compare existing native rows directly, without reconstructing a state."""
    paths = [Path(root)/scenario/"prefix_trace.jsonl" for scenario in ("stable", "local_speed_drop")]
    if not all(path.is_file() for path in paths):
        return dict(status="unavailable", exact=None, rows=None)
    count = 0
    with paths[0].open() as left, paths[1].open() as right:
        while True:
            raw = [left.readline(), right.readline()]
            if any(not value for value in raw):
                return dict(status="short_before_boundary", exact=False, rows=count)
            low, high = map(json.loads, raw)
            if low["started_at"] >= before or high["started_at"] >= before:
                reached = low["started_at"] >= before and high["started_at"] >= before
                return dict(status="exact" if reached and count else "different_boundary", exact=reached and count > 0,
                    rows=count, first_boundary_clocks=[low["started_at"], high["started_at"]], before_seconds=before)
            if low != high:
                return dict(status="different", exact=False, rows=count,
                    sequences=[low["sequence"], high["sequence"]],
                    differing_fields=sorted(key for key in low.keys() | high.keys() if low.get(key) != high.get(key)))
            count += 1


def aggregate(rows):
    rows = list(rows)
    scored = [row for row in rows if row[5] == "scored"]
    models = []
    for index in range(2):
        points = [row[11][index] for row in scored]
        regrets = [point[3] for point in points]
        count = len(points)
        fixed0 = [sum(point[6][j] for point in points)/count for j in range(3)] if count else None
        models.append([sum(point[4] for row, point in zip(scored, points) if row[10][2] != 0),
            sum(row[10][2] != 0 for row in scored), sum(point[0] == 0 for point in points),
            sum(point[2] == 1 for point in points), sum(regrets)/count if count else None,
            max(regrets) if regrets else None, sum(regrets) if regrets else None, fixed0,
            sum(point[6][2] > 0 for point in points)])
    physical = [sum(row[7] is not None and row[7][0] == status for row in rows)
        for status in ("separable", "nonseparable", "physical_unknown")]
    return [len(rows), sum(row[2] is not None for row in rows), len(scored),
        sum(row[10][2] == 0 for row in scored), physical,
        sum(bool(row[13].get("zero_next_event_arms")) for row in rows), models,
        delta_stats(row[12] for row in scored)]


def state_result(scenario, slot, reference, parent, protocol, case_root):
    row = [scenario, slot, None, None, None, "unselected", None, None, None, None, None, None, None, {}]
    selection = parent["selection"][slot]
    selected = selection.get("selected")
    if selected is None:
        row[13]["selection"] = selection
        return row, dict(unselected_has_no_saved_state=reference is None)
    row[2:4] = [selected["sequence"], selected["issue"]]
    if reference is None or not resolve(reference["path"], case_root).is_file():
        row[5] = "missing_state_summary"
        return row, dict(selected_file_complete=False)
    path = resolve(reference["path"], case_root)
    record = read(path)
    issue = record["selected"]
    row[2:7] = [issue["sequence"], issue["started_at"], issue["trip_id"], "missing_branch",
        [record["holds"][arm] for arm in ("low", "high")]]
    line, window = slot.split("/")
    checks = dict(parent_integrity=parent["integrity"]["all_pass"],
        declared_slot=record["slot"] == slot and issue["line_id"] == line and
            protocol["windows"][window][0] <= issue["started_at"] < protocol["windows"][window][1],
        selected_receipt=issue["sequence"] == selected["sequence"] and issue["started_at"] == selected["issue"],
        actual_pair_parallel=record["pair_receipt"]["all_pass"] is True and
            record["pair_receipt"]["overlap_seconds"] > 0.,
        predeclared_holds=row[6] == [0., 7.5])
    branches = {}
    for arm in ("low", "high"):
        branch_path = resolve(record["branch_paths"][arm], path.parent)
        if not branch_path.is_file():
            row[13].setdefault("missing_branches", []).append(arm)
            checks[arm+"_present"] = False
            continue
        value = branches[arm] = read(branch_path)
        execution = value["execution"]
        hold = record["holds"][arm]
        checks.update({arm+"_integrity": value["integrity"]["all_pass"],
            arm+"_same_issue": value["selected"] == issue,
            arm+"_same_exogenous_seed": value["external_seed"] == protocol["native_external_seed"],
            arm+"_actual_action": execution["executed_action"] == record["actions"][arm],
            arm+"_actual_holding": abs(execution["executed_hold_seconds"]-hold) <= 1e-5 and
                abs(execution["native_applied_hold_seconds"]-hold) <= 1e-5,
            arm+"_suffix_zero_training": value["integrity"]["checks"]["suffix_zero_training"] is True})
    if len(branches) != 2:
        return row, checks
    low, high = (branches[arm] for arm in ("low", "high"))
    physical = record["comparison"]["physical_separation"]
    row[7] = [physical["status"], physical["departure_gap_seconds"], physical["next_same_vehicle_arrival_gap_seconds"]]
    row[8] = [branches[arm]["execution"].get("registered_hold_seconds") for arm in ("low", "high")]
    row[13]["registered_missing_arms"] = [arm for arm, value in zip(("low", "high"), row[8]) if value is None]
    row[13]["physical_endpoint_status"] = [physical.get(key) for key in
        ("low_departure_status", "high_departure_status", "low_next_arrival_status", "high_next_arrival_status")]
    next_pair = [value["truth"]["next_event"] for value in (low, high)]
    row[13]["zero_next_event_arms"] = [arm for arm, value in zip(("low", "high"), next_pair)
        if value.get("zero_duration") is True]
    rewards = [value.get("global_transition", {}).get("r") if value.get("global_transition") else None
        for value in (low, high)]
    row[9] = [value.get("dt") for value in next_pair] + [benefit(*next_pair),
        rewards[1]-rewards[0] if None not in rewards else None]
    endpoints = [value["truth"]["fixed300"] for value in (low, high)]
    row[10] = benefit(*endpoints, fixed=True)
    if row[10] is None:
        row[13]["fixed300_status"] = [value["status"] for value in endpoints]
    for value in endpoints:
        if value["status"] == "complete":
            checks["truth_J_equals_2W_plus_R"] = checks.get("truth_J_equals_2W_plus_R", True) and \
                abs(value["cost"] - 2*value["wait"] - value["ride"]) <= 1e-6
    row[11] = []
    for model in MODELS:
        point = record["prediction"][model]
        q, mean = point["q_min"], point["q_mean"]
        choice = int(q[1] > q[0])
        checks[model+"_common_actions"] = point["executed_actions"] == [record["actions"][arm][0] for arm in ("low", "high")]
        checks[model+"_choice_consistent"] = point["predicted_choice"] == choice
        truth = row[10]
        regret = max((-1 if choice else 1)*truth[2], 0.) if truth is not None else None
        cost0 = [-choice*value for value in truth] if truth is not None else None
        row[11].append([q[1]-q[0], mean[1]-mean[0], choice, regret,
            sign(q[1]-q[0]) == sign(truth[2]) if truth is not None else None,
            point["actor_holding_seconds"], cost0])
    row[12] = row[11][1][3]-row[11][0][3] if row[10] is not None else None
    if not all(checks.values()):
        row[5], row[12] = "integrity_failed", None
        for point in row[11]:
            point[3:5], point[6] = [None, None], None
    else:
        row[5] = "scored" if row[10] is not None else "missing300"
    return row, checks


def analyze(root):
    root = Path(root)
    protocol = read(root.parent/"protocol.json")
    expected = [line+"/"+window for line in protocol["focus_lines"] for window in protocol["windows"]]
    rows, cases, checks, parents, failure_costs = [], {}, {}, [], {}
    for scenario in protocol["scenarios"]:
        case_root = root/scenario
        if not (case_root/"summary.json").is_file():
            failure_path = case_root/"failure.json"
            failure = read(failure_path) if failure_path.is_file() else None
            failure_costs[scenario] = (failure.get("failure_budget", failure.get("budget"))
                if failure else None)
            cases[scenario] = dict(status="missing_parent", missing=expected,
                failure=failure, accounting="failed_or_incomplete_prefix; missing costs remain unknown")
            checks[scenario+"_parent_present"] = False
            rows.extend([scenario, slot, None, None, None, "missing_parent", None, None, None,
                None, None, None, None, {}] for slot in expected)
            continue
        parent = read(case_root/"summary.json")
        parents.append(parent)
        if parent.get("failure_budget") is not None:
            failure_costs[scenario] = parent["failure_budget"]
        references = {value["slot"]: value for value in parent["states"]}
        checks[scenario+"_parent_integrity"] = parent["integrity"]["all_pass"]
        checks[scenario+"_predeclared_slots"] = set(parent["selection"]) == set(expected)
        checks[scenario+"_unique_saved_slots"] = len(references) == len(parent["states"])
        checks[scenario+"_native_seed"] = parent["native_external_seed"] == protocol["native_external_seed"]
        checks[scenario+"_initializer_and_action_seed"] = parent.get("initializer_seed") == 6202 and \
            parent.get("private_action_seed") == 526255
        controller = parent.get("frozen_model_gate", {}).get("continuation_mu14_gate", {})
        checks[scenario+"_same_frozen_mu14_controller"] = all(controller.get(key) is True for key in
            ("all_pass", "shared_frozen_actor_exact", "F_contract_exact")) and \
            controller.get("actor_updates") == 50024 and parent.get("controller_snapshot") == "v53_actual_post24_from14"
        budget = parent["budget"]
        checks[scenario+"_zero_training"] = all(budget[key] == 0 for key in
            ("new_training_labels", "new_prefix_optimizer_updates", "suffix_optimizer_updates"))
        state_checks, case_rows, seen = {}, [], set()
        for slot in expected:
            row, evidence = state_result(scenario, slot, references.get(slot), parent, protocol, case_root)
            if row[4] is not None:
                identity = tuple(row[2:5])
                evidence["unique_physical_state"] = identity not in seen
                seen.add(identity)
                if not evidence["unique_physical_state"]:
                    row[5], row[12] = "integrity_failed", None
            state_checks[slot] = dict(all_pass=all(evidence.values()), failed=[key for key, value in evidence.items() if not value])
            case_rows.append(row)
        checks[scenario+"_state_evidence"] = all(value["all_pass"] for value in state_checks.values())
        rows.extend(case_rows)
        cases[scenario] = dict(parent_checks=parent["integrity"]["checks"], state_checks=state_checks,
            frozen_model_gate=parent.get("frozen_model_gate", parent.get("model_rebuild")),
            overall=aggregate(case_rows), separable_only=aggregate(row for row in case_rows if row[7] and row[7][0] == "separable"),
            by_line={line: aggregate(row for row in case_rows if row[1].split("/")[0] == line) for line in protocol["focus_lines"]},
            by_window={window: aggregate(row for row in case_rows if row[1].split("/")[1] == window) for window in protocol["windows"]},
            unsupported_visible_attempts={slot: item.get("reasons", {}) for slot, item in parent["selection"].items() if item.get("reasons")})
    known = len(parents) == len(protocol["scenarios"])
    budget = dict(declared=protocol["compute_budget"],
        native_prefix_seconds=sum(parent["budget"]["native_prefix_seconds"] for parent in parents) if known else None,
        suffix_native_seconds=sum(parent["budget"]["suffix_sim_seconds"] for parent in parents) if known else None,
        selected_states=sum(parent["budget"]["states"] for parent in parents),
        new_truth_labels=2*sum(parent["budget"]["states"] for parent in parents),
        new_training_labels=sum(parent["budget"]["new_training_labels"] for parent in parents) if known else None,
        new_optimizer_updates=sum(parent["budget"]["new_prefix_optimizer_updates"]+parent["budget"]["suffix_optimizer_updates"] for parent in parents) if known else None,
        failed_or_incomplete_prefix_costs=failure_costs)
    prefix_audit = common_prefix(root)
    checks["stable_shock_native_prefix_before9000_exact"] = prefix_audit["exact"] is True
    valid = all(checks.values())
    return dict(status="complete" if valid and all(row[5] == "scored" for row in rows) else "partial" if valid else "integrity_failed",
        integrity=dict(all_pass=valid, checks=checks), columns=dict(states=STATE_COLUMNS, models=MODEL_COLUMNS,
            model_order=list(MODELS), groups=GROUP_COLUMNS, group_models=GROUP_MODEL_COLUMNS, paired_cost=DELTA_COLUMNS,
            benefit="low minus high; J=2W+R; positive favors7.5s", units="aggregate passenger-seconds over300s"),
        budget=budget, common_prefix_before9000=prefix_audit, states=rows, cases=cases,
        stable_trace_minus_one_step_cost=delta_stats(row[12] for row in rows if row[0] == "stable" and row[5] == "scored"),
        missing_or_partial=[row[:6]+[row[13]] for row in rows if row[5] != "scored"],
        zero_duration_next_events=[dict(scenario=row[0], slot=row[1], arms=row[13]["zero_next_event_arms"])
            for row in rows if row[13].get("zero_next_event_arms")],
        nonseparable_or_unknown=[dict(scenario=row[0], slot=row[1], physical=row[7])
            for row in rows if row[7] and row[7][0] != "separable"],
        limitations=["Frozen Q is a long-term mixed V/SAC target containing entropy; it is not the300s passenger-cost target.",
            "Truth compares only0/7.5s with a common frozen continuation. It does not measure either actor's deployed cost.",
            "One new exogenous seed and eight predeclared states are action-value diagnosis, not independent policy confirmation.",
            "Strict floating signs retain ties; unequal next-event durations stay separate from fixed300s truth."])


def report(result):
    lines = ["# v55 冻结模型动作真值", "", f"状态：{result['status']}。", "",
        "|场景/线路/窗口|状态|300秒收益 W / R / J|单步regret|trace regret|trace−单步成本|", "|---|---|---|---:|---:|---:|"]
    fmt = lambda value: "缺失" if value is None else format(value, ".7g")
    for row in result["states"]:
        truth = "缺失" if row[10] is None else " / ".join(map(fmt, row[10]))
        regrets = [None, None] if row[11] is None else [point[3] for point in row[11]]
        lines.append(f"|{row[0]}/{row[1]}|{row[5]}|{truth}|{fmt(regrets[0])}|{fmt(regrets[1])}|{fmt(row[12])}|")
    lines.extend(["", "正收益表示7.5秒动作降低成本；负trace−单步成本表示trace选择更好。固定0对照、等待/乘车分解、缺失、零时长和物理不可分离见小汇总JSON。", "",
        "新原生前缀和后缀预算单列；真值未入训。长期混合Q与300秒乘客成本的目标不同，本轮用于动作方向诊断。"])
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE/"runs")
    parser.add_argument("--out", type=Path, default=HERE/"analysis"/"summary.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = analyze(args.root)
        compact = json.dumps(result, separators=(",", ":"), allow_nan=False)
        maximum = read(args.root.parent/"protocol.json")["summary_max_bytes"]
        if len(compact.encode())+1 > maximum:
            raise RuntimeError(f"v55 summary exceeds{maximum}byte transfer limit")
        with args.out.open("x") as handle:
            handle.write(compact+"\n")
        with (args.out.parent/"RESULTS.md").open("x") as handle:
            handle.write(report(result))
    except Exception as error:
        with (args.out.parent/"failure.json").open("x") as handle:
            json.dump(dict(type=type(error).__name__, message=str(error)), handle)
        raise
