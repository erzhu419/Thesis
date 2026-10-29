"""Derive compact descriptive metrics from the collected Stage A summary."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def context_score(rows):
    count = sum(row["n"] for row in rows)
    result = {"rows": count}
    for branch in ("base", "saved", "current"):
        scores = [(row["n"], row["overall"]["branches"][branch]) for row in rows]
        value = {key: [sum(n * score[key][i] for n, score in scores) / count
                       for i in range(4)] for key in ("g4_mse", "physical_mae")}
        value["g4_mean_mse"] = mean(value["g4_mse"])
        if "bellman_target_error" in scores[0][1]:
            value["bellman_target_mae"] = sum(
                n * score["bellman_target_error"]["mae"] for n, score in scores) / count
        result[branch] = value
    return result


def review(value):
    result = dict(status=value["status"], integrity=value["integrity"],
        passive_reproduction=value["passive_reproduction"], context={}, shadow={})
    for scenario in ("stable", "local_speed_drop"):
        cells = [row["diagnostics"] for row in value["context_diagnostics"]
                 if row["scenario"] == scenario]
        historical = [row for cell in cells for row in cell["historical"] if row["deadline"] > 9000]
        future = [row for cell in cells for row in cell["next_packet_frozen_anchor"]
                  if row["deadline"] > 9000]
        designs = [row["identifiability"]["branches"]["saved"] for row in historical]
        result["context"][scenario] = dict(
            historical_post_anchor_repeated=context_score(historical),
            next_packet_post_anchor=context_score(future),
            historical_saved_ranks=[row["rank"] for row in designs],
            historical_saved_condition_range=[min(row["condition_number_nonzero"] for row in designs),
                                               max(row["condition_number_nonzero"] for row in designs)],
            historical_by_anchor={str(deadline): context_score([
                row for row in historical if row["deadline"] == deadline])
                for deadline in (10200., 13800., 17100.)})
        shadows = [row["diagnostics"] for row in value["shadow_diagnostics"]
                   if row["scenario"] == scenario]
        for phase, after in (("all", False), ("post_anchor", True)):
            anchors = [row for cell in shadows for row in cell["anchors"]
                       if not after or row["deadline"] > 9000]
            next_rows = [row for cell in shadows for row in cell["next_packet"]
                         if not after or row["anchor_deadline"] > 9000]
            metrics = dict(anchors=len(anchors), packet_states=sum(row["packet_rows"] for row in anchors),
                next_packet_rows=sum(row["rows"] for row in next_rows),
                local_exact_active=all(row["local_exact_active"] for row in anchors),
                ordinary_local_executed_shift=mean(row["branches"]["Local"]["executed_versus_before"]
                                                    ["mean_absolute_delta"] for row in anchors),
                actual_trust_mean=mean(row["active_trust_mean"] for row in anchors), branches={})
            for branch in ("Base", "Local", "Local-W", "Local-C"):
                metrics["branches"][branch] = dict(
                    executed_gap_versus_local=mean(row["branches"][branch]
                        ["executed_versus_local"]["mean_absolute_delta"] for row in anchors)
                        if branch != "Local" else 0.,
                    future_td_mse=mean(row["branches"][branch]["mean_squared_td_error"] for row in next_rows),
                    future_td_mae=mean(row["branches"][branch]["mean_absolute_td_error"] for row in next_rows))
            geometry = [row["branches"]["Local"]["action_geometry"] for row in anchors]
            metrics["positive_amplitude_derivative"] = mean(row["positive_amplitude_derivative_mean"] for row in geometry)
            metrics["zero_amplitude_derivative_fraction"] = mean(row["zero_amplitude_derivative_fraction"] for row in geometry)
            result["shadow"][scenario + "/" + phase] = metrics
    result["interpretation"] = (
        "Historical anchors reuse cumulative labels: descriptive repeated observations, not independent samples. "
        "Next-packet post-anchor results contain30 labels per scenario from two development seeds. "
        "Shadows fix actual trust and never control the environment. TD changes are not realized passenger benefits.")
    return result


if __name__ == "__main__":
    summary = json.loads((HERE / "context_shadow_summary.json").read_text())
    result = review(summary)
    (HERE / "stage_a_review_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], scenarios=list(result["context"]))))
