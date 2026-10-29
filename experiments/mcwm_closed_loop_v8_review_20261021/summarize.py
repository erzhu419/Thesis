"""Read-only v7/v8 comparison; write derived review files in this new directory."""

import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "mcwm_closed_loop_action_response_v8_20261021/response_summary.json"
V7 = HERE.parent / "mcwm_closed_loop_action_fork_v7_20261020/action_fork_summary.json"


def aggregate(rows):
    result = dict(n=len(rows))
    if not rows:
        return result
    for version in ("legacy", "response"):
        result[version] = dict(
            false_immediate_gain_mae=mean(abs(r[version + "_cost_gain"]-r["true_next_cost_gain"]) for r in rows),
            signed_immediate_gain=mean(r[version + "_cost_gain"] for r in rows),
            g4_mse=mean(r[version + "_g4_mse"] for r in rows),
            bellman_mse=mean(r[version + "_bellman_mse"] for r in rows),
            bellman_gain_mae=mean(abs(r[version + "_bellman_gain"]-r["true_next_bellman_gain"]) for r in rows),
            bellman_vs_300s_correct=sum(r[version + "_direction_vs_300s"] == "agree" for r in rows),
            hindsight_choice_gain_vs_all_low=mean(r[version + "_choice_gain"] for r in rows))
    result["false_immediate_error_improved"] = sum(
        abs(r["response_cost_gain"]-r["true_next_cost_gain"]) < abs(r["legacy_cost_gain"]-r["true_next_cost_gain"])
        for r in rows)
    result["hindsight_cost_saved_vs_legacy"] = sum(r["response_choice_gain"]-r["legacy_choice_gain"] for r in rows)
    return result


def main():
    source = json.loads(SOURCE.read_text())
    old = json.loads(V7.read_text())
    index = {(r["seed"], r["scenario"], r["target_line"]): r for r in old["pairs"]}
    rows = []
    for pair in source["pairs"]:
        cell, truth = pair["cell"], pair["factual"]["gains"]
        key = (cell["seed"], cell["scenario"], cell["target_line"])
        prior = index[key]
        # Detect swapped cells/branches or changed reused truth before interpreting effects.
        assert truth == prior["evidence"]["factual_effect"]["gains"]
        assert all(pair["integrity"]["checks"].values())
        for world in ("online", "frozen_common"):
            legacy = pair["versions"]["legacy"][world]
            expected = prior["evidence"]["worlds"][world]
            assert legacy["gains"] == expected["predicted_effect"]["gains"]
            for candidate in ("low", "high"):
                for field in ("global_decoded", "bellman"):
                    assert legacy["candidates"][candidate][field] == expected["candidates"][candidate][field]
            if not pair["zero_duration"]:
                updated = pair["versions"]["response"]["frozen_common"]
                assert pair["versions"]["legacy"]["frozen_common"]["gains"] == updated["gains"]
        row = dict(**cell, zero_duration=pair["zero_duration"], time=pair["selected"]["time"],
                   true_next_cost_gain=truth["next_event_cost"], true_next_bellman_gain=truth["bellman_target"],
                   true_300s_gain=truth["cost_300_seconds"])
        for version in ("legacy", "response"):
            p = pair["versions"][version]["online"]
            row.update({version + "_" + name: value for name, value in dict(
                cost_gain=p["gains"]["next_event_cost"], bellman_gain=p["gains"]["bellman_target"],
                reward_gain=p["gains"]["scaled_reward"], bootstrap_gain=p["gains"]["bootstrap"],
                direction_vs_300s=p["direction_vs_300s"], g4_mse=p["g4_mse"],
                bellman_mse=p["bellman_target_mse"],
                choice_gain=truth["cost_300_seconds"] if p["gains"]["bellman_target"] > 0 else 0.).items()})
        if not row["zero_duration"]:
            base = pair["versions"]["legacy"]["online"]["candidates"]
            response = pair["versions"]["response"]["online"]["candidates"]
            # Counterfactual low has zero holding interactions. Its total correction
            # contains both memory levels and physical levels, not memory alone.
            factor = (response["low"]["global_decoded"][1]+1)/(base["low"]["global_decoded"][1]+1)
            level_gain = factor*row["legacy_cost_gain"]
            row.update(common_logcost_level_scale=factor,
                       gain_with_common_level_only=level_gain,
                       incremental_gain_from_holding_interactions=row["response_cost_gain"]-level_gain)
        else:
            row.update(common_logcost_level_scale=None, gain_with_common_level_only=None,
                       incremental_gain_from_holding_interactions=None)
        rows.append(row)
    groups = dict(overall=aggregate(rows), positive_dt=aggregate([r for r in rows if not r["zero_duration"]]))
    for scenario in ("stable", "local_speed_drop"):
        selected = [r for r in rows if r["scenario"] == scenario]
        groups[scenario] = aggregate(selected)
        positive = [r for r in selected if not r["zero_duration"]]
        groups[scenario + "/positive_dt"] = aggregate(positive)
        for line in ("7X", "102X", "705X"):
            groups[scenario + "/" + line + "/positive_dt"] = aggregate([r for r in positive if r["target_line"] == line])
    result = dict(source=str(SOURCE), source_bytes=SOURCE.stat().st_size,
                  status=source["status"], integrity=source["integrity"], budget=source["budget"],
                  validation=dict(legacy_predictions_match_v7=True, reused_truth_matches_v7=True,
                                  all_12_integrity_checks_pass=True),
                  groups=groups, zero_duration=[r for r in rows if r["zero_duration"]],
                  hindsight_scope="Choose between the existing two fork outcomes at fixed states; no deployed controller improvement is implied.",
                  decision="Do not advance this candidate to independent confirmation; retain analytic dt=0 and diagnose response feature groups first.")
    positive = [r for r in rows if not r["zero_duration"]]
    result["cost_level_decomposition"] = dict(
        n=len(positive),
        baseline_false_gain_mae=mean(abs(r["legacy_cost_gain"]) for r in positive),
        common_level_only_false_gain_mae=mean(abs(r["gain_with_common_level_only"]) for r in positive),
        full_response_false_gain_mae=mean(abs(r["response_cost_gain"]) for r in positive),
        scope="Common shift includes additive memory AND physical levels. Holding interactions also include non-memory features; this does not isolate memory causally.")
    (HERE / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    with (HERE / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(status=result["status"], integrity=result["integrity"],
                         validation=result["validation"], overall=groups["overall"], positive_dt=groups["positive_dt"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
