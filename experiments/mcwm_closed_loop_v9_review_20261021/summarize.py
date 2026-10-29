"""Read-only v9 attribution review; derive tables from the existing small files."""

from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "mcwm_closed_loop_response_ablation_v9_20261021/ablation_summary.json"
V8 = HERE.parent / "mcwm_closed_loop_action_response_v8_20261021/response_summary.json"


def aggregate(pairs):
    out = dict(n=len(pairs), arms={})
    if not pairs:
        return out
    for arm in pairs[0]["arms"]:
        rows = [p["arms"][arm] for p in pairs]
        out["arms"][arm] = dict(
            false_immediate_gain_mae=mean(r["false_immediate_gain_error"] for r in rows),
            direction_correct=sum(r["direction_vs_300s"] == "agree" for r in rows),
            g4_mse=mean(r["g4_mse"] for r in rows),
            bellman_mse=mean(r["bellman_target_mse"] for r in rows),
            hindsight_gain_total=sum(r["hindsight_choice_gain"] for r in rows),
            hindsight_saved_vs_base=sum(r["hindsight_choice_gain"]-p["arms"]["base_analytic"]["hindsight_choice_gain"]
                                        for r,p in zip(rows,pairs)))
    return out


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = json.loads(SOURCE.read_text())
    previous = json.loads(V8.read_text())
    prior = {tuple(p["cell"][k] for k in ("seed", "scenario", "target_line")):p for p in previous["pairs"]}
    pairs, rows, support, contributions = source["pairs"], [], [], []
    checks = Counter()
    for p in pairs:
        cell = p["cell"]
        old = prior[tuple(cell[k] for k in ("seed", "scenario", "target_line"))]
        assert p["factual_gains"] == old["factual"]["gains"]
        for name, passed in p["integrity"]["checks"].items():
            assert passed, (cell, name)
            checks[name] += 1
        full, expected = p["arms"]["memory_interaction"], old["versions"]["response"]["online"]
        for key in ("gains", "g4_mse", "global_mae", "bellman_target_mse", "direction_vs_300s"):
            assert full[key] == expected[key], (cell, key)
        if p["zero_duration"]:
            assert all(m["g4_mse"] == m["bellman_target_mse"] == 0 for m in p["arms"].values())
        for arm,m in p["arms"].items():
            rows.append(dict(**cell, zero_duration=p["zero_duration"], arm=arm,
                true_300s_gain=p["factual_gains"]["cost_300_seconds"],
                predicted_cost_gain=m["gains"]["next_event_cost"], predicted_bellman_gain=m["gains"]["bellman_target"],
                predicted_reward_gain=m["gains"]["scaled_reward"], predicted_bootstrap_gain=m["gains"]["bootstrap"],
                false_immediate_gain_error=m["false_immediate_gain_error"], direction_vs_300s=m["direction_vs_300s"],
                g4_mse=m["g4_mse"], bellman_mse=m["bellman_target_mse"], hindsight_gain=m["hindsight_choice_gain"]))
        coverage = p["fits"]["holding_coverage"]
        own = coverage["by_line"][cell["target_line"]]
        entry = dict(**cell, retained_labels=p["fits"]["retained_labels"],
                     own_line_n=own["all"]["n"], own_line_hold_min=own["all"]["hold_seconds_min"],
                     own_line_hold_max=own["all"]["hold_seconds_max"], own_line_zero_holding=own["all"]["zero_holding"],
                     own_station_n=own["at_diagnostic_station"]["n"],
                     own_station_hold_min=own["at_diagnostic_station"]["hold_seconds_min"],
                     own_station_hold_max=own["at_diagnostic_station"]["hold_seconds_max"],
                     target_lines_fraction=sum(v["all"]["n"] for v in coverage["by_line"].values())/p["fits"]["retained_labels"])
        for arm,d in p["fits"]["designs"].items():
            entry[arm + "_rank"] = d["rank"]
            entry[arm + "_ridge_condition"] = d["ridge_precision_condition"]
        support.append(entry)
        for group,g in p["full_group_contributions"]["groups"].items():
            if group == "memory_level":
                assert g["high_minus_low"] == [0.]*4
            contributions.append(dict(**cell, group=group,
                low_logcost_correction=g["low"][1], high_logcost_correction=g["high"][1],
                **{head + "_high_minus_low":g["high_minus_low"][i]
                   for i,head in enumerate(("logdt", "logcost", "dWaiting", "dOnboard"))}))
    positive = [p for p in pairs if not p["zero_duration"]]
    groups = dict(overall=aggregate(pairs), positive_dt=aggregate(positive))
    for scenario in ("stable", "local_speed_drop"):
        selected = [p for p in pairs if p["cell"]["scenario"] == scenario]
        groups[scenario] = aggregate(selected)
        for line in ("7X", "102X", "705X"):
            groups[scenario + "/" + line + "/positive_dt"] = aggregate([
                p for p in selected if p["cell"]["target_line"] == line and not p["zero_duration"]])
    ranges = {}
    for arm in ("action_physical", "memory_level", "memory_interaction"):
        designs = [p["fits"]["designs"][arm] for p in pairs]
        ranges[arm] = dict(columns=designs[0]["columns"], rank_min=min(d["rank"] for d in designs),
            rank_max=max(d["rank"] for d in designs),
            ridge_condition_min=min(d["ridge_precision_condition"] for d in designs),
            ridge_condition_max=max(d["ridge_precision_condition"] for d in designs))
    result = dict(source=str(SOURCE), source_bytes=SOURCE.stat().st_size, status=source["status"],
        integrity=source["integrity"], verification=dict(checks_passed_by_name=dict(checks),full_matches_local_v8=True),
        budget=source["budget"], groups=groups, design_ranges=ranges, support=support,
        own_line_station_empty=sum(r["own_station_n"] == 0 for r in support),
        drop_interaction_improves_immediate=sum(p["arms"]["drop_memory_interaction"]["false_immediate_gain_error"] <
            p["arms"]["memory_interaction"]["false_immediate_gain_error"] for p in positive),
        zero_duration=[dict(cell=p["cell"],factual=p["factual_gains"],arms=p["arms"]) for p in pairs if p["zero_duration"]],
        decision="Retain the58column candidate as a development comparator. Establish local action support and a valid event-response model before independent confirmation.")
    (HERE / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    write_csv("arms.csv", rows)
    write_csv("support.csv", support)
    write_csv("contributions.csv", contributions)
    print(json.dumps(dict(status=result["status"], integrity=result["integrity"],
                         checks=dict(checks), own_line_station_empty=result["own_line_station_empty"],
                         positive_dt=groups["positive_dt"]),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
