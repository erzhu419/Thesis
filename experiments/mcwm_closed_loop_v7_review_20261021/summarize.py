"""Derive small v7 review tables from the existing compact summary, read-only."""

import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "mcwm_closed_loop_action_fork_v7_20261020/action_fork_summary.json"
WORLDS = ("online", "frozen_common")
SCENARIOS = ("stable", "local_speed_drop")
HEADS = ("duration", "cost", "dWaiting", "dOnboard")


def aggregate(pairs):
    output = {"pairs": len(pairs)}
    if not pairs:
        return output
    actual = [p["evidence"]["factual_effect"]["gains"] for p in pairs]
    output["actual"] = {
        "mean_300s_gain": mean(a["cost_300_seconds"] for a in actual),
        "high_better_300s": sum(a["cost_300_seconds"] > 0 for a in actual),
        "low_better_300s": sum(a["cost_300_seconds"] < 0 for a in actual),
        "ties_300s": sum(a["cost_300_seconds"] == 0 for a in actual),
        "next_event_cost_ties": sum(a["next_event_cost"] == 0 for a in actual),
        "bellman_ties": sum(a["bellman_target"] == 0 for a in actual),
    }
    for world in WORLDS:
        rows = [p["evidence"]["worlds"][world] for p in pairs]
        candidates = [row["candidates"][role] for row in rows for role in ("low", "high")]
        output[world] = {
            "mean_predicted_gain": {
                key: mean(row["predicted_effect"]["gains"][key] for row in rows)
                for key in ("next_event_cost", "scaled_reward", "bootstrap", "bellman_target")
            },
            "mean_predicted_head_high_minus_low": {
                head: mean(row["predicted_effect"]["global_high_minus_low"][i] for row in rows)
                for i, head in enumerate(HEADS)
            },
            "high_predicted_better": sum(row["predicted_effect"]["gains"]["bellman_target"] > 0 for row in rows),
            "direction_vs_300s": {
                value: sum(row["scores"]["bellman_vs_cost_300_direction"] == value for row in rows)
                for value in ("agree", "disagree", "tie", "missing")
            },
            "g4_mse": mean(row["scores"]["g4_mse"] for row in rows),
            "bellman_mse": {
                key: mean(row["scores"]["bellman_mse"][key] for row in rows)
                for key in ("scaled_reward", "bootstrap", "target")
            },
            "mean_trust": mean(c["trust"]["weight"] for c in candidates),
            "mean_tau": mean(c["trust"]["tau"] for c in candidates),
            "mean_trust_outcome": {
                key: mean(c["trust_outcome"][key] for c in candidates)
                for key in ("six_loss_mean", "log_c_step", "bootstrap", "target", "prediction_minus_target")
            },
        }
    return output


def main():
    source = json.loads(SOURCE.read_text())
    pairs = source["pairs"]
    groups = {"overall": aggregate(pairs)}
    for scenario in SCENARIOS:
        selected = [p for p in pairs if p["scenario"] == scenario]
        groups[scenario] = aggregate(selected)
        groups[scenario + "/positive_dt"] = aggregate([
            p for p in selected if not p["evidence"]["actual_next_events"]["low"]["zero_duration"]])
        for line in ("7X", "102X", "705X"):
            groups[scenario + "/" + line] = aggregate([p for p in selected if p["target_line"] == line])
    groups["positive_dt"] = aggregate([
        p for p in pairs if not p["evidence"]["actual_next_events"]["low"]["zero_duration"]])
    rows = []
    for p in pairs:
        e, prefix = p["evidence"], p["prefix"]
        actual = e["factual_effect"]["gains"]
        row = dict(seed=p["seed"], scenario=p["scenario"], line=p["target_line"],
                   selected_time=p["selected"]["time"], external_seed=prefix["external_seed"],
                   low_action=p["selected"]["low_action"], high_action=p["selected"]["high_action"],
                   target_labels=prefix["target_labels"], policy_updates=prefix["policy_updates"],
                   world_updates=prefix["world_updates"],
                   zero_duration=e["actual_next_events"]["low"]["zero_duration"],
                   next_vehicle_low=e["actual_next_events"]["low"]["next_vehicle_id"],
                   next_vehicle_high=e["actual_next_events"]["high"]["next_vehicle_id"])
        row.update({"actual_" + key: value for key, value in actual.items()})
        for world in WORLDS:
            w = e["worlds"][world]
            row.update({world + "_gain_" + key: value for key, value in w["predicted_effect"]["gains"].items()})
            row[world + "_direction_vs_300s"] = w["scores"]["bellman_vs_cost_300_direction"]
            for i, head in enumerate(HEADS):
                row[world + "_delta_" + head] = w["predicted_effect"]["global_high_minus_low"][i]
                row["actual_delta_" + head] = e["factual_effect"]["global_high_minus_low"][i]
        rows.append(row)
    review = dict(source=str(SOURCE), source_bytes=SOURCE.stat().st_size,
                  status=source["status"], integrity=source["integrity"], budget=source["budget"],
                  groups=groups,
                  zero_duration_cells=[{k: row[k] for k in ("seed", "scenario", "line")} for row in rows if row["zero_duration"]])
    (HERE / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "pairs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(status=review["status"], integrity=review["integrity"],
                         groups={key: groups[key] for key in ("overall", "stable", "local_speed_drop", "positive_dt")}),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
