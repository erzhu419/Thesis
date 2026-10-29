"""Derive same-flow v6 comparisons from the existing small local summary."""

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "mcwm_closed_loop_paired_world_v6_20261020/paired_world_summary.json"
BRANCHES = ("online", "frozen_common", "frozen_own")


def weighted(rows, key):
    count = sum(row["n"] for row in rows)
    first = rows[0][key]
    if isinstance(first, list):
        return [sum(row["n"] * row[key][i] for row in rows) / count for i in range(len(first))]
    return sum(row["n"] * row[key] for row in rows) / count


def scores(groups):
    result = dict(n=sum(row["n"] for row in groups), branches={})
    for branch in groups[0]["branches"]:
        rows = [dict(n=row["n"], **row["branches"][branch]) for row in groups]
        value = {key: weighted(rows, key) for key in ("six_loss_mean", "g4_mse", "physical_mae")}
        value.update(g4_mean_mse=mean(value["g4_mse"]), six_head_mean=mean(value["six_loss_mean"]))
        for key in ("bellman_target_error", "scaled_reward_error", "bootstrap_error"):
            if key in rows[0]:
                parts = [dict(n=row["n"], **row[key]) for row in rows]
                value[key] = {name: weighted(parts, name) for name in ("signed_mean", "mae", "mse")}
        if "bellman_target_error" in value:
            value["twice_reward_bootstrap_cross_moment"] = (
                value["bellman_target_error"]["mse"] - value["scaled_reward_error"]["mse"]
                - value["bootstrap_error"]["mse"])
        result["branches"][branch] = value
    return result


def trust_mean(groups):
    result = dict(n=sum(row["n"] for row in groups))
    for branch in ("online", "frozen"):
        rows = [dict(n=row["n"], **row[branch]) for row in groups]
        result[branch] = {key: weighted(rows, key) for key in groups[0][branch]}
    return result


def shadow_mean(anchors, future):
    return dict(anchors=len(anchors), packet_states=sum(row["packet_rows"] for row in anchors),
        future_rows=sum(row["rows"] for row in future),
        online_exact=all(row["online_exact_active"] for row in anchors),
        executed_gap_mean=mean(row["branches"]["Frozen"]["executed_versus_online"]
                               ["mean_absolute_delta"] for row in anchors),
        ordinary_online_executed_shift=mean(row["branches"]["Online"]["executed_versus_before"]
                                           ["mean_absolute_delta"] for row in anchors),
        candidate_valid_rows=sum(row["branches"]["Online"]["candidate_q_contrast"]["valid_rows"]
                                 for row in anchors),
        ranking_flip_fractions=[row["branches"]["Frozen"]["q_ranking_flip_versus_online_fraction"]
                                for row in anchors],
        future_td={branch: {key: mean(row["branches"][branch][key] for row in future)
                            for key in ("mean_absolute_td_error", "mean_squared_td_error")}
                   for branch in ("Online", "Frozen")})


def review(value):
    result = dict(status=value["status"], integrity=value["integrity"],
        passive_reproduction=value["passive_reproduction"], scenarios={})
    for scenario in ("stable", "local_speed_drop"):
        worlds = [row["diagnostics"] for row in value["paired_world_diagnostics"] if row["scenario"] == scenario]
        shadows = [row["diagnostics"] for row in value["shadow_diagnostics"] if row["scenario"] == scenario]
        issue = dict(all_selected=scores([row["issue_time"]["overall"] for row in worlds]),
            before=scores([row["issue_time"]["by_issue_phase"]["before"] for row in worlds]),
            after=scores([part for row in worlds for name, part in row["issue_time"]["by_issue_phase"].items()
                          if name != "before"]),
            by_family={family: scores([row["issue_time"]["by_model_family"][family] for row in worlds])
                       for family in ("direct", "residual")})
        future = {name: scores([part["overall"] for row in worlds for part in row["next_packet_frozen_anchor"]
                               if (part["deadline"] > 9000) == after])
                  for name, after in (("before_anchor", False), ("after_anchor", True))}
        trust = {name: trust_mean([part for row in worlds for phase, part in row["trust_period_means"].items()
                                  if phase in phases])
                 for name, phases in (("all_updates", ("before", "early_after", "middle_after", "late_after", "drain")),
                    ("adaptation_before_final_packet", ("early_after", "middle_after", "late_after")),
                    ("all_post_packets", ("early_after", "middle_after", "late_after", "drain")))}
        trust_anchors = {str(deadline): trust_mean([dict(n=1, online=step["online"], frozen=step["frozen"])
            for row in worlds for anchor in row["trust_anchor_steps"] if anchor["deadline"] == deadline
            for step in anchor["steps"]]) for deadline in (8400., 10200., 13800., 17100.)}
        shadow = {name: shadow_mean([part for row in shadows for part in row["anchors"]
                                    if (part["deadline"] > 9000) == after],
                                   [part for row in shadows for part in row["next_packet"]
                                    if (part["anchor_deadline"] > 9000) == after])
                  for name, after in (("before_anchor", False), ("after_anchor", True))}
        result["scenarios"][scenario] = dict(issue_time=issue, held_next_packet=future,
            trust=trust, trust_anchors=trust_anchors, shadow=shadow)
    result["scope"] = (
        "Two original development seeds. Before9000 paths are identical across scenarios; do not count them twice. "
        "Each scenario has570 issue labels including300 after9000; heldfuture after-anchor30 is a subset, not extra labels. "
        "Trust n counts actual updates, not additional outcomes. Adaptation trust excludes final18000s packet, "
        "while all_post_packets includes it. Shadows never drive the environment; fixed trust isolates neural-model effects. "
        "Common Bellman targets use the current value networks, not true policy returns. No post-stop prediction labels.")
    return result


if __name__ == "__main__":
    result = review(json.loads(SOURCE.read_text()))
    (HERE / "comparison_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for scenario, row in result["scenarios"].items():
        print(scenario, json.dumps(dict(issue_after=row["issue_time"]["after"],
            held_next_packet=row["held_next_packet"]["after_anchor"], shadow=row["shadow"]["after_anchor"])))
