"""Small local tables from the unmodified resident v4 summary."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASES = ("early_after", "middle_after", "late_after")


def summarize_heads(source, scenario):
    flows = [r for r in source["head_diagnostics"]
             if r["scenario"] == scenario and r["arm"] == "action_local"]
    heads = {}
    for phase, selected in [("all_after", PHASES), *[(p, (p,)) for p in PHASES]]:
        rows = [r["diagnostics"]["by_phase"][p] for r in flows for p in selected]
        total = sum(r["n"] for r in rows)
        value = {"n": total}
        for branch in ("base", "calibrated"):
            physical = {k: [sum(r["n"] * r[branch]["physical"][k][i] for r in rows) / total
                            for i in range(4)] for k in ("mae", "signed_mean")}
            physical["rmse"] = [(sum(r["n"] * r[branch]["physical"]["rmse"][i] ** 2
                                      for r in rows) / total) ** .5 for i in range(4)]
            mse = [sum(r["n"] * r[branch]["standardized"]["rmse"][i] ** 2
                       for r in rows) / total for i in range(4)]
            value[branch] = dict(physical=physical, standardized_mse=mse, g4_mse=sum(mse) / 4)
        value["delta_standardized_mse"] = [b - a for a, b in zip(
            value["base"]["standardized_mse"], value["calibrated"]["standardized_mse"])]
        heads[phase] = value
    return heads


def main():
    source = json.loads((HERE / "v4_summary.json").read_text())
    heads = summarize_heads(source, "local_speed_drop")
    probes = [p for r in source["head_counterfactuals"]
              if r["scenario"] == "local_speed_drop" and r["arm"] == "action_local"
              for phase, p in r["periods"].items() if phase != "before"]
    td = {}
    for variant in probes[0]["variants"]:
        rows = [p["variants"][variant] for p in probes]
        total = sum(r["rows"] for r in rows)
        td[variant] = dict(rows=total, applied_rows=sum(r["applied_rows"] for r in rows),
            **{k: {m: sum(r["rows"] * r[k][m] for r in rows) / total
                    for m in rows[0][k] if m != "max_absolute_delta"}
               for k in rows[0] if isinstance(rows[0][k], dict)})
    recovery = {}
    for r in source["recovery_window_cells"]:
        recovery[r["recovery_status"]] = recovery.get(r["recovery_status"], 0) + 1
    result = dict(head_order=source["head_diagnostic_metadata"]["head_order"],
        shock_active_four_heads=heads, stable_active_four_heads=summarize_heads(source, "stable"),
        shock_active_td=td,
        paired_costs=[r for key in ("stability_paired_effects", "local_replay_paired_effects")
                      for r in source[key] if not r["quantity"].startswith("before_")],
        recovery_status_counts=recovery,
        departed_cohort_sizes=sorted({r["departed"] for r in source["stability_control_cells"]}),
        actual_last_update_times=sorted({r["last_policy_update_at_seconds"] for r in source["new_control_cells"]}))
    (HERE / "review_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                           allow_nan=False) + "\n")
    print(json.dumps(dict(head_rows=heads["all_after"]["n"], td_rows=td["full_base"]["rows"],
                          departed=result["departed_cohort_sizes"], recovery=recovery)))


if __name__ == "__main__":
    main()
