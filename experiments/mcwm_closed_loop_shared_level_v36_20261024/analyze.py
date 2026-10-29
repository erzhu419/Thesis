"""Resident three-arm shadow analysis and existing development-truth join."""
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("zero_memory", "per_head_level", "shared_level")
CONTRASTS = (("shared_level", "per_head_level"), ("shared_level", "zero_memory"),
             ("per_head_level", "zero_memory"))
TRAIN_METRICS = ("target", "value_contribution", "sac_contribution", "actor_holding_seconds", "proposal_seconds")
SHARED_VARIANTS = ("base", "level_only", "action_only", "full")
API_FIELDS = ("action_step_length_seconds", "is_at_bus_stop", "lane_id", "lane_position_m",
              "last_action_time_seconds", "position_xy", "speed_mps", "stop_state")
STATE_COLUMNS = ["scenario", "slot", "sequence", "benefit300_wait_ride_cost", "models",
    "shared_variants", "shared_level_structure", "shared_minus_perhead_regret", "quality"]
MODEL_COLUMNS = ["delta_min", "delta_mean", "choice", "regret300", "direction_correct",
                 "min_head_low", "min_head_high", "actor_F_holding"]
VARIANT_COLUMNS = ["delta_min", "delta_mean", "choice", "min_head_low", "min_head_high"]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def display_numbers(value):
    if isinstance(value, float):
        return float(format(value, ".7g"))
    if isinstance(value, list):
        return [display_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key:display_numbers(item) for key, item in value.items()}
    return value


def stats(values):
    return [sum(values)/len(values), sum(abs(x) for x in values)/len(values), max(abs(x) for x in values)] if values else None


def train_aggregate(rows):
    result = []
    for left, right in CONTRASTS:
        result.append([stats([row[left][key]-row[right][key] for row in rows]) for key in TRAIN_METRICS])
    return dict(n=len(rows), changed_actor_F=[sum(row[a]["actor_holding_seconds"] != row[b]["actor_holding_seconds"]
        for row in rows) for a, b in CONTRASTS], differences=result)


def truth_aggregate(rows):
    valid = [row for row in rows if row[3] is not None]
    models = []
    for index in range(3):
        values = [row[4][index] for row in valid]
        models.append([len(values), sum(value[4] for value in values), sum(value[2] == 1 for value in values),
            sum(value[3] for value in values) if values else None,
            sum(value[3] for value in values)/len(values) if values else None,
            sum(value[5] != value[6] for value in values)])
    differences = [row[7] for row in valid]
    return dict(models=models, shared_minus_perhead=[len(differences), sum(differences) if differences else None,
        sum(max(-value, 0.) for value in differences) if differences else None,
        sum(max(value, 0.) for value in differences) if differences else None,
        sum(value < 0 for value in differences), sum(value > 0 for value in differences), sum(value == 0 for value in differences)])


def _training(case_root):
    by_arm = {}
    for arm in ARMS:
        selected = {}
        for chunk in lines(case_root/arm/"scores.jsonl"):
            if chunk["phase"] != "packet_after":
                continue
            for row in chunk["rows"]:
                key = chunk["packet"], row["sequence"]
                require(key not in selected, "Duplicate factual packet-after score")
                selected[key] = row
        by_arm[arm] = selected
    keys = by_arm[ARMS[0]].keys()
    require(len(keys) == 285 and all(by_arm[arm].keys() == keys for arm in ARMS), "Three arms did not score the same285 admitted states")
    matched = []
    for key in keys:
        row = {arm: by_arm[arm][key] for arm in ARMS}
        reference = row[ARMS[0]]
        for arm in ARMS[1:]:
            require(all(reference[field] == row[arm][field] for field in
                ("line_id", "window", "issue", "context_observed_at", "holding_seconds", "executed_actions", "reward_scaled")),
                "Factual covariates/actions/rewards differ between treatments")
        matched.append(row)
    return matched


def _truth_state(scenario, slot, records):
    reference = records[ARMS[0]]
    require(all(all(record[field] == reference[field] for field in ("slot", "sequence", "issue", "line_id", "source"))
                for record in records.values()), "Frozen treatments scored different old issue states")
    original = read(reference["source"])
    require(original["slot"] == slot and original["selected"]["sequence"] == reference["sequence"], "Old truth joined to the wrong physical issue")
    saved_actions = [original["actions"][arm][0] for arm in ("low", "high")]
    for arm, record in records.items():
        require(record["prediction"]["executed_actions"] == saved_actions, "Frozen common executed actions differ")
        if arm != "shared_level":
            old_arm = "memory" if arm == "per_head_level" else "zero_memory"
            saved = original["prediction"][old_arm]
            require(all(record["prediction"][key] == saved[key] for key in
                ("q_min", "q_mean", "predicted_choice", "actor_holding_seconds")),
                "Old twelve-state frozen predictions did not reproduce exactly")
    difference = original["comparison"]["high_minus_low"]["fixed300"]
    benefit = [-difference[key] for key in ("wait", "ride", "cost")] if difference["status"] == "complete" else None
    quality = dict(physical=original["comparison"]["physical_separation"]["status"])
    if benefit is None:
        quality["truth_status"] = difference["status"]
    branches = {}
    for arm in ("low", "high"):
        path = Path(original["branch_paths"][arm])
        if not path.is_absolute():
            path = Path(reference["source"]).parent/path
        if not path.exists():
            quality.setdefault("missing_truth_branches", []).append(arm)
            benefit = None
            continue
        branches[arm] = read(path)
    if len(branches) == 2:
        quality["zero_next_event_arms"] = [arm for arm in ("low", "high")
            if branches[arm]["truth"]["next_event"].get("zero_duration") is True]
        quality["registered_missing_arms"] = [arm for arm in ("low", "high")
            if branches[arm]["execution"].get("registered_hold_seconds") is None]
        missing = [[branches[arm]["physical"].get("missing", {}).get(field, 0) for field in API_FIELDS] for arm in ("low", "high")]
        if any(any(values) for values in missing):
            quality["api_missing_counts"] = missing
        if any(value["truth"]["fixed300"]["status"] != "complete" or value["truth"]["fixed300"]["dt"] != 300.
               for value in branches.values()):
            benefit = None
            quality["truth_status"] = "missing_or_truncated300"
    models = []
    for arm in ARMS:
        prediction = records[arm]["prediction"]
        choice = prediction["predicted_choice"]
        delta = prediction["delta_min"]
        regret = (max(-benefit[2], 0.) if choice else max(benefit[2], 0.)) if benefit is not None else None
        correct = ((delta > 0)-(delta < 0) == (benefit[2] > 0)-(benefit[2] < 0)) if benefit is not None else None
        models.append([delta, prediction["delta_mean"], choice, regret, correct,
            *prediction["min_heads"], prediction["actor_holding_seconds"]])
    shared = records["shared_level"]
    variants = [[shared["variants"][name]["delta_min"], shared["variants"][name]["delta_mean"],
        shared["variants"][name]["choice"], *shared["variants"][name]["min_heads"]] for name in SHARED_VARIANTS]
    level = shared["components"]["level_heads"]
    scale = max(1., *(abs(x) for values in shared["prediction"]["q_heads"] for x in values),
                *(abs(x) for values in level for x in values))
    tolerance = 4.*2.**-23*scale
    full_action = variants[3][0]-variants[2][0]
    level_base = variants[1][0]-variants[0][0]
    require(len(level) == 1 and abs(full_action) <= tolerance and abs(level_base) <= tolerance,
        "Shared level directly altered action advantage beyond float32 roundoff")
    structure = [full_action, level_base, variants[3][3:] == variants[2][3:], variants[1][3:] == variants[0][3:], tolerance]
    paired = models[2][3]-models[1][3] if benefit is not None else None
    return [scenario, slot, reference["sequence"], benefit, models, variants, structure, paired, quality]


def analyze(root):
    root = Path(root)
    protocol = read(HERE/"protocol.json")
    cases, states, checks = {}, [], {}
    for scenario in protocol["scenarios"]:
        case_root = root/scenario
        parent = read(case_root/"summary.json")
        workers = {arm: read(case_root/arm/"summary.json") for arm in ARMS}
        require(parent["integrity"]["all_pass"] and all(value["integrity"]["all_pass"] for value in workers.values()),
                "A parent/worker failed; scientific comparisons are unavailable")
        require(all(value["budget"]["feedback_slots"] == 285 and value["budget"]["shadow_updates"] == 456
                    for value in workers.values()), "Matched285/456 worker budgets failed")
        overlap = parent["optimizer_overlap_wall_seconds"]
        require(overlap > 0, "The three actual optimizer intervals never overlapped")
        factual = _training(case_root)
        training = dict(overall=train_aggregate(factual),
            by_line={line: train_aggregate([row for row in factual if row[ARMS[0]]["line_id"] == line])
                     for line in protocol["focus_lines"]},
            by_window={window: train_aggregate([row for row in factual if row[ARMS[0]]["window"] == window])
                       for window in ("before9000", "9000-12000", "12000-18000")})
        heldout = {}
        for arm in ARMS:
            values = lines(case_root/arm/"heldout.jsonl")
            heldout[arm] = {row["slot"]: row for row in values}
            require(len(values) == len(heldout[arm]) == 6, "Frozen issue slots are missing or duplicated")
        expected = {line+"/"+window for line in protocol["focus_lines"] for window in ("early", "tail")}
        require(all(set(values) == expected for values in heldout.values()), "The predeclared twelve issue slots changed")
        case_rows = [_truth_state(scenario, slot, {arm: heldout[arm][slot] for arm in ARMS}) for slot in sorted(expected)]
        states.extend(case_rows)
        checks[scenario] = True
        cases[scenario] = dict(parent_checks=parent["integrity"]["checks"],
            worker_gates={arm:[value["integrity"]["all_pass"], value["initial_preflight"]["all_pass"],
                value["budget"]["feedback_slots"], value["budget"]["shadow_updates"],
                value["integrity"]["checks"]["original_sampling_and_global_private_rng_exact"],
                value["integrity"]["checks"]["optimizer_objects_retained"],
                value["integrity"]["checks"]["post_budget_zero_training"], value["heldout"]["rows"],
                value["old_reference_arm"], value["old_final_prediction_rows"]] for arm, value in workers.items()},
            training_overlap_seconds=parent["training_overlap_wall_seconds"], optimizer_overlap_seconds=overlap,
            training=training, truth=truth_aggregate(case_rows),
            truth_by_line={line:truth_aggregate([row for row in case_rows if row[1].split("/")[0] == line])
                           for line in protocol["focus_lines"]},
            truth_by_window={window:truth_aggregate([row for row in case_rows if row[1].endswith("/"+window)])
                             for window in ("early", "tail")})
    return display_numbers(dict(status="complete", integrity=dict(all_pass=all(checks.values()), checks=checks),
        columns=dict(states=STATE_COLUMNS, arms=list(ARMS), models=MODEL_COLUMNS, shared_variants=list(SHARED_VARIANTS),
            variant=VARIANT_COLUMNS, shared_structure=["full_minus_action_only_delta", "level_only_minus_base_delta",
                "full_action_min_heads_equal", "level_base_min_heads_equal", "float32_tolerance"],
            training_contrasts=[a+" minus "+b for a, b in CONTRASTS], training_metrics=list(TRAIN_METRICS),
            statistics=["signed_mean", "mean_absolute", "max_absolute"],
            truth_model=["n", "correct", "high_choices", "regret_sum", "regret_mean", "head_switches"],
            paired_regret=["n", "net", "gain", "loss", "improved", "worsened", "tied"],
            api_fields=list(API_FIELDS), worker_gates=["all_pass", "preflight", "slots", "updates", "sampling_rng",
                "Adam_retained", "post_budget_zero", "heldout_rows", "old_reference", "old_final_rows"],
            display_precision="7 significant digits after exact parity, direction and regret computations; full records remain resident"),
        budget=protocol["compute_budget"], states=states, cases=cases,
        stable_gains_losses=truth_aggregate([row for row in states if row[0] == "stable"])["shared_minus_perhead"],
        missing=[row[:3]+[row[8]] for row in states if row[3] is None],
        limitations=["All three learners use the original285 labels; selected old12 truths enter final analysis only.",
            "Same-weight shared-level checks isolate a direct path; refitting still changes base/actor/V through Bellman learning.",
            "Old12 states and overlapping300s outcomes are development evidence, not independent confirmation or all-day controller costs.",
            "Critic own-policy long-term values and common zero-memory continuation300s truth have different horizons and policy operators."]))


def report(result):
    lines = ["# v36 共享记忆水平项结果", "", "三路同流累计影子及冻结后的旧12状态发展诊断；本轮没有新的原生控制实验。", "",
        "|场景/状态|300秒真实收益|zero regret|per-head regret|shared regret|shared−per-head|",
        "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "缺失" if value is None else format(value, ".7g")
    for row in result["states"]:
        lines.append("|"+row[0]+"/"+row[1]+"|"+fmt(row[3][2] if row[3] else None)+"|"+
            "|".join(fmt(value[3]) for value in row[4])+"|"+fmt(row[7])+"|")
    paired = result["stable_gains_losses"]
    lines += ["", f"稳定场景：shared 对 per-head 改善 {paired[4]}、退化 {paired[5]}、持平 {paired[6]}，净 regret 差 {fmt(paired[1])}。", "",
        "正真实收益支持高动作；负 regret 差支持共享 level。逐线/时窗方向与代价、训练 target/V/SAC 和 actor/F 的有符号及绝对变化见紧凑 JSON。", "",
        "新增影子更新2736次，重复使用旧标签；新增训练标签、原生前缀与后缀均0。原两路复现、三路抽样/RNG、Adam持续及实际优化区间重叠均先通过。", "", "## 解释边界", ""]
    lines += ["- "+value for value in result["limitations"]]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE/"runs")
    parser.add_argument("--out", type=Path, default=HERE/"shared_level_summary.json")
    args = parser.parse_args()
    try:
        result = analyze(args.root)
        compact = json.dumps(result, separators=(",", ":"), allow_nan=False)
        require(len(compact.encode())+1 <= 20000, "Shared-level summary exceeds20000byte transfer limit")
        with args.out.open("x") as output:
            output.write(compact+"\n")
        with (args.out.parent/"RESULTS.md").open("x") as output:
            output.write(report(result))
    except Exception as error:
        with (args.out.parent/"shared_level_summary_failure.json").open("x") as output:
            json.dump(dict(type=type(error).__name__, message=str(error)), output)
        raise
