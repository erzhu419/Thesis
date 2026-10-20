"""Post-hoc comparison to the existing zero-policy-update common behavior days."""

import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
T975 = {1: 12.7062047364, 4: 2.7764451052}


def main():
    formal = json.loads((HERE.parent / "mcwm_closed_loop_formal_20261020/campaign_analysis.json").read_text())
    v4 = json.loads((HERE / "v4_summary.json").read_text())
    frozen = {(r["seed"], r["scenario"]): r for r in formal["runs"]
              if r["mode"] == "prediction" and r["arm"] == "common"}
    assert len(frozen) == 10
    assert all(r["status"] == "complete" and r["policy_updates"] == 0
               and r["target_time_seconds"] == 24000 for r in frozen.values())
    rows = []
    for arm in ("replay", "online", "context", "action_real", "action_model", "action_local"):
        source = ([r for r in formal["runs"] if r["mode"] == "control" and r["arm"] == arm]
                  if not arm.startswith("action_") else
                  [r for r in (v4["new_control_cells"] if arm == "action_local" else v4["stability_control_cells"])
                   if r["arm"] == arm])
        for scenario in ("stable", "local_speed_drop"):
            controls = sorted((r for r in source if r["scenario"] == scenario), key=lambda r: r["seed"])
            values, baselines = [], []
            for r in controls:
                b = frozen[r["seed"], scenario]["generalized_seconds_per_departed_passenger"]
                c = r["generalized_seconds_per_departed_passenger"] if "mode" in r else r["seconds_per_departed_passenger"]
                baselines.append(b)
                values.append(c - b)
            n = len(values)
            mean = statistics.mean(values)
            half = T975[n - 1] * statistics.stdev(values) / n ** .5
            rows.append(dict(arm=arm, scenario=scenario, seeds=[r["seed"] for r in controls],
                frozen_mean=statistics.mean(baselines), values=values, mean=mean,
                descriptive_95pct_t_ci=[mean - half, mean + half], negative=sum(x < 0 for x in values)))
    result = dict(analysis="Post-hoc full-day contrast to existing common-stream frozen initializer behavior",
        interpretation="Continued policy learning versus no policy updates; does not isolate fresh labels from extra optimizer steps. Shared-F branches additionally changed execution objectives.",
        contrasts=rows)
    (HERE / "frozen_behavior_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r in rows:
        print(r["arm"], r["scenario"], round(r["mean"], 6), r["negative"], len(r["values"]))


if __name__ == "__main__":
    main()
