"""Small resident paired-truth summary; missing states never become zero costs."""
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = ("zero_memory", "memory")
STATE_COLUMNS = ["scenario", "slot", "sequence", "issue", "trip", "status",
    "holding_seconds", "physical_status_departure_gap_arrival_gap",
    "native_registered_seconds", "next_dt_low_high_then_wait_ride_cost_benefit",
    "benefit300_wait_ride_cost", "models", "memory_minus_zero_regret", "quality"]
MODEL_COLUMNS = ["Qmin_high_minus_low", "Qmean_high_minus_low", "choice_0low_1high",
    "regret300", "strict_sign_matches_benefit300", "own_actor_F_holding"]
GROUP_COLUMNS = ["slots", "selected", "scored300", "truth_ties", "physical_separable_nonseparable_unknown",
    "zero_next_event_states", "registered_missing_states", "benefit300_sum_wait_ride_cost",
    "models", "paired_memory_minus_zero_regret"]
GROUP_MODEL_COLUMNS = ["nonzero_truth_direction_correct", "nonzero_truth_states", "Q_ties",
    "high_choices", "regret_mean", "regret_max", "regret_sum"]
REGRET_COLUMNS = ["paired_states", "mean", "sum", "improved", "worsened", "tied"]


def read(path):
    with Path(path).open() as source:
        return json.load(source)


def sign(value):
    return (value > 0)-(value < 0)


def _path(value, relative):
    path = Path(value)
    return path if path.is_absolute() else relative/path


def _benefit(left, right, *, fixed=False):
    if left["status"] != "complete" or right["status"] != "complete":
        return None
    if fixed and (left["dt"] != 300. or right["dt"] != 300.):
        return None
    return [left[key]-right[key] for key in ("wait", "ride", "cost")]


def _regret_delta(rows):
    values = [row[12] for row in rows if row[12] is not None]
    if not values:
        return [0, None, None, 0, 0, 0]
    return [len(values), sum(values)/len(values), sum(values),
        sum(x < 0 for x in values), sum(x > 0 for x in values), sum(x == 0 for x in values)]


def aggregate(rows):
    scored = [row for row in rows if row[5] == "scored"]
    physical = [sum(row[7][0] == kind for row in rows if row[7] is not None)
                for kind in ("separable", "nonseparable", "physical_unknown")]
    totals = [sum(row[10][index] for row in scored) for index in range(3)] if scored else None
    models = []
    for index in range(2):
        values = [row[11][index] for row in scored]
        regrets = [value[3] for value in values]
        models.append([sum(value[4] for row, value in zip(scored, values) if row[10][2] != 0),
            sum(row[10][2] != 0 for row in scored), sum(value[0] == 0 for value in values),
            sum(value[2] == 1 for value in values), sum(regrets)/len(regrets) if regrets else None,
            max(regrets) if regrets else None, sum(regrets) if regrets else None])
    return [len(rows), sum(row[2] is not None for row in rows), len(scored),
        sum(row[10][2] == 0 for row in scored), physical,
        sum(bool(row[13].get("zero_next_event_arms")) for row in rows),
        sum(bool(row[13].get("registered_missing_arms")) for row in rows), totals, models,
        _regret_delta(rows)]


def _state(scenario, slot, reference, parent, protocol, case_root):
    """Return a compact row and evidence checks, without re-running models."""
    empty = [scenario, slot, None, None, None, "unselected", None, None, None,
             None, None, None, None, {}]
    selected = parent["selection"][slot].get("selected")
    if selected is None:
        empty[13]["selection"] = parent["selection"][slot]
        return empty, dict(selected_files_complete=reference is None)
    if reference is None or not _path(reference["path"], case_root).exists():
        empty[2:6] = [selected["sequence"], selected["issue"], None, "missing_state_summary"]
        return empty, dict(selected_files_complete=False)
    path = _path(reference["path"], case_root)
    record = read(path)
    issue = record["selected"]
    row = [scenario, slot, issue["sequence"], issue["started_at"], issue["trip_id"], "missing_branch",
           [record["holds"][arm] for arm in ("low", "high")], None, None, None, None, None, None, {}]
    line, window = slot.split("/")
    checks = dict(selected_files_complete=True, parent_integrity=parent["integrity"]["all_pass"],
        declared_slot=record["slot"] == slot and issue["line_id"] == line and
            protocol["windows"][window][0] <= issue["started_at"] < protocol["windows"][window][1],
        selected_receipt=issue["sequence"] == selected["sequence"] and issue["started_at"] == selected["issue"],
        paused_parent_and_parallel=record["pair_receipt"]["all_pass"])
    branches = {}
    for arm in ("low", "high"):
        branch_path = _path(record["branch_paths"][arm], path.parent)
        if not branch_path.exists():
            checks["selected_files_complete"] = False
            row[13].setdefault("missing_branches", []).append(arm)
            continue
        value = read(branch_path)
        branches[arm] = value
        checks[arm+"_integrity"] = value["integrity"]["all_pass"]
        checks[arm+"_same_selected_prefix"] = value["selected"] == issue
        checks[arm+"_actual_action"] = value["execution"]["executed_action"] == record["actions"][arm]
        checks[arm+"_same_exogenous_seed"] = value["external_seed"] == protocol["native_external_seed"]
    if len(branches) != 2:
        return row, checks
    low, high = branches["low"], branches["high"]
    registered = [branches[arm]["execution"].get("registered_hold_seconds") for arm in ("low", "high")]
    row[8] = registered
    row[13]["registered_missing_arms"] = [arm for arm, value in zip(("low", "high"), registered) if value is None]
    physical = record["comparison"]["physical_separation"]
    row[7] = [physical["status"], physical["departure_gap_seconds"], physical["next_same_vehicle_arrival_gap_seconds"]]
    missing = {arm: branches[arm]["physical"].get("missing", {}) for arm in ("low", "high")}
    if any(missing.values()):
        row[13]["physical_missing_api_counts"] = missing
    row[13]["physical_endpoint_status"] = [physical.get(key) for key in
        ("low_departure_status", "high_departure_status", "low_next_arrival_status", "high_next_arrival_status")]
    row[13]["zero_next_event_arms"] = [arm for arm in ("low", "high")
        if branches[arm]["truth"]["next_event"].get("zero_duration") is True]
    next_low, next_high = low["truth"]["next_event"], high["truth"]["next_event"]
    next_benefit = _benefit(next_low, next_high)
    row[9] = [next_low.get("dt"), next_high.get("dt"), next_benefit]
    if next_benefit is None:
        row[13]["next_endpoint_status"] = [next_low["status"], next_high["status"]]
    fixed_low, fixed_high = low["truth"]["fixed300"], high["truth"]["fixed300"]
    benefit = _benefit(fixed_low, fixed_high, fixed=True)
    row[10] = benefit
    if benefit is None:
        row[13]["fixed300_endpoint_status"] = [fixed_low["status"], fixed_high["status"]]
    row[11] = []
    actions = [record["actions"][arm][0] for arm in ("low", "high")]
    for model in MODELS:
        prediction = record["prediction"][model]
        q, mean = prediction["q_min"], prediction["q_mean"]
        choice = 0 if q[0] >= q[1] else 1
        checks[model+"_common_actions"] = prediction["executed_actions"] == actions
        checks[model+"_choice_consistent"] = prediction["predicted_choice"] == choice
        regret = (max(-benefit[2], 0.) if choice else max(benefit[2], 0.)) if benefit is not None else None
        row[11].append([q[1]-q[0], mean[1]-mean[0], choice, regret,
            sign(q[1]-q[0]) == sign(benefit[2]) if benefit is not None else None,
            prediction["actor_holding_seconds"]])
    row[12] = row[11][1][3]-row[11][0][3] if benefit is not None else None
    if not all(checks.values()):
        row[5] = "integrity_failed"
        # Invalid paired evidence contributes neither costs nor regret to aggregates.
        row[12] = None
        for model in row[11]:
            model[3:5] = [None, None]
    else:
        row[5] = "scored" if benefit is not None else "missing300"
    return row, checks


def analyze(root):
    root = Path(root)
    protocol = read(root.parents[1]/"protocol.json")
    rows, cases, parents, checks = [], {}, {}, {}
    for scenario in protocol["scenarios"]:
        case_root = root/scenario
        if not (case_root/"summary.json").exists():
            checks[scenario+"_parent_present"] = False
            continue
        parent = read(case_root/"summary.json")
        parents[scenario] = parent
        checks[scenario+"_parent_integrity"] = parent["integrity"]["all_pass"]
        checks[scenario+"_new_exogenous_seed"] = parent["native_external_seed"] == protocol["native_external_seed"]
        references = {value["slot"]: value for value in parent["states"]}
        expected = [line+"/"+window for line in protocol["focus_lines"] for window in protocol["windows"]]
        checks[scenario+"_predeclared_slots"] = set(parent["selection"]) == set(expected)
        checks[scenario+"_unique_state_slots"] = len(references) == len(parent["states"])
        case_rows, state_checks, physical_states = [], {}, set()
        for slot in expected:
            row, evidence = _state(scenario, slot, references.get(slot), parent, protocol, case_root)
            if row[2] is not None and row[4] is not None:
                identity = row[2], row[3], row[4]
                evidence["unique_physical_state"] = identity not in physical_states
                physical_states.add(identity)
                if not evidence["unique_physical_state"]:
                    row[5], row[12] = "integrity_failed", None
                    if row[11] is not None:
                        for model in row[11]:
                            model[3:5] = [None, None]
            case_rows.append(row)
            state_checks[slot] = dict(all_pass=all(evidence.values()),
                failed=[key for key, value in evidence.items() if not value])
        rows.extend(case_rows)
        checks[scenario+"_state_evidence"] = all(value["all_pass"] for value in state_checks.values())
        cases[scenario] = dict(parent_checks=parent["integrity"]["checks"],
            state_checks=state_checks, overall=aggregate(case_rows),
            separable_only=aggregate([r for r in case_rows if r[7] is not None and r[7][0] == "separable"]),
            by_line={line: aggregate([r for r in case_rows if r[1].split("/")[0] == line])
                     for line in protocol["focus_lines"]},
            by_window={window: aggregate([r for r in case_rows if r[1].split("/")[1] == window])
                       for window in protocol["windows"]})
    all_parents = len(parents) == len(protocol["scenarios"])
    budget = dict(declared=protocol["compute_budget"],
        reconstruction_updates=sum(p["budget"]["reconstruction_updates"] for p in parents.values()) if all_parents else None,
        prefix_native_seconds=sum(p["budget"]["native_prefix_seconds"] for p in parents.values()) if all_parents else None,
        selected_states=sum(p["budget"]["states"] for p in parents.values()),
        suffix_native_seconds=sum(p["budget"]["suffix_sim_seconds"] for p in parents.values()) if all_parents else None,
        new_truth_labels=sum(2*p["budget"]["states"] for p in parents.values()) if all_parents else None,
        new_training_labels=0, new_prefix_optimizer_updates=0, suffix_optimizer_updates=0)
    invalid = any(not value for value in checks.values())
    return dict(status="integrity_failed" if invalid else "complete",
        integrity=dict(all_pass=not invalid, checks=checks), columns=dict(states=STATE_COLUMNS,
            model_order=list(MODELS), models=MODEL_COLUMNS, groups=GROUP_COLUMNS,
            group_models=GROUP_MODEL_COLUMNS, paired_regret=REGRET_COLUMNS,
            benefit="low minus high; C=2W+R; positive favors high", physical_order=["separable", "nonseparable", "unknown"]),
        budget=budget, states=rows, cases=cases,
        stable_paired_memory_minus_zero_regret=_regret_delta([r for r in rows if r[0] == "stable" and r[5] == "scored"]),
        missing=[r[:6]+[r[13]] for r in rows if r[5] != "scored"],
        limitations=["Frozen critics use their own actor/V long-term backup; paired300s truth uses the common frozen zero-memory continuation.",
            "One new exogenous seed and at most12 predeclared states; this is action-value diagnosis, not policy confirmation.",
            "Strict floating signs retain ties. Unequal next-event durations are separate from the common300s endpoint."])


def report(result):
    lines = ["# v34 冻结记忆模型配对真值", "", f"完整性：{result['status']}。新增场景与状态按预声明槽位保留。", "",
        "|场景/线路/窗口|状态|300秒收益 W / R / C|zero regret|memory regret|memory−zero|", "|---|---|---|---:|---:|---:|"]
    for row in result["states"]:
        benefit = "缺失" if row[10] is None else " / ".join(format(x, ".7g") for x in row[10])
        regrets = [None, None] if row[11] is None else [row[11][i][3] for i in range(2)]
        fmt = lambda x: "缺失" if x is None else format(x, ".7g")
        lines.append(f"|{row[0]}/{row[1]}|{row[5]}|{benefit}|{fmt(regrets[0])}|{fmt(regrets[1])}|{fmt(row[12])}|")
    paired = result["stable_paired_memory_minus_zero_regret"]
    lines.extend(["", f"稳定场景配对状态 {paired[0]}：改善 {paired[3]}、退化 {paired[4]}、持平 {paired[5]}。", "",
        "正收益表示高动作降低成本；负 regret 差表示记忆模型在这两个动作中的选择更好。零时长下一事件、缺失、注册时长不可见及物理不可分离均在 JSON 单列。", "",
        "新增预算见 JSON：重建更新与新原生前缀/后缀分别计费，配对真值未入训。", "", "## 解释边界", ""])
    lines.extend("- "+value for value in result["limitations"])
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE/"runs"/"seed6202")
    parser.add_argument("--out", type=Path, default=HERE/"truth_summary.json")
    args = parser.parse_args()
    try:
        result = analyze(args.root)
        compact = json.dumps(result, separators=(",", ":"), allow_nan=False)
        if len(compact.encode())+1 > 20000:
            raise RuntimeError("v34 summary exceeds the20000byte transfer limit")
        with args.out.open("x") as handle:
            handle.write(compact+"\n")
        with (args.out.parent/"RESULTS.md").open("x") as handle:
            handle.write(report(result))
    except Exception as error:
        with (args.out.parent/"truth_summary_failure.json").open("x") as handle:
            json.dump(dict(type=type(error).__name__, message=str(error)), handle)
        raise
