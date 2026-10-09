#!/usr/bin/env python3
"""Compare the 32 unique F-2 cases and preserve the figure's 40 displayed cells."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COST = "generalized_seconds_per_departed_passenger"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scale", type=float, choices=(0.25, 1.0), default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "cases.json").read_text())
    folder = args.out_root / ("scale_%g" % args.scale)
    pairs, missing, missing_original = [], [], []
    for case in manifest["cases"]:
        new_path = folder / ("case_%02d" % case["id"]) / "evaluation" / Path(case["result"]).name
        old_path = args.original_root / case["result"]
        if not new_path.is_file():
            missing.append(case["id"])
        if not old_path.is_file():
            missing_original.append(case["id"])
        if not new_path.is_file() or not old_path.is_file():
            continue
        old, new = (json.loads(p.read_text()) for p in (old_path, new_path))
        for result in (old, new):
            if (result["method"] != case["tag"] or result["demand_scale"] != case["demand_scale"]
                    or result["sumo_seed"] != 13301 or result["demand_seed"] != 13401
                    or result["horizon_seconds"] != 24000 or result["sumo_version"][1] != "SUMO 1.25.0"):
                raise ValueError("evaluation identity mismatch in case %s" % case["id"])
        pairs.append(dict(case_id=case["id"], arm=case["arm"], blocks=case["blocks"],
                          demand_scale=case["demand_scale"], policy_updates=case["policy_updates"],
                          nominal_source_events=case["nominal_source_events"],
                          old_cost=old[COST], new_cost=new[COST], cost_change_seconds=new[COST]-old[COST],
                          old_completion=old["passengers"]["completion_rate"],
                          new_completion=new["passengers"]["completion_rate"],
                          completion_change_pp=100*(new["passengers"]["completion_rate"]-old["passengers"]["completion_rate"]),
                          old_mean_hold_seconds=old["mean_hold_seconds"], new_mean_hold_seconds=new["mean_hold_seconds"]))
    if missing or missing_original:
        print(json.dumps(dict(status="incomplete", paired_cases=len(pairs), total_cases=32,
                              missing_new=missing, missing_original=missing_original)))
        return
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["arm"], row["demand_scale"], row["policy_updates"]].append(row)
    summaries = []
    for (arm, demand, budget), rows in sorted(grouped.items()):
        summaries.append(dict(arm=arm, demand_scale=demand, policy_updates=budget,
                              data_conditions=len(rows),
                              **{key: statistics.mean(r[key] for r in rows)
                                 for key in ("old_cost", "new_cost", "cost_change_seconds", "completion_change_pp")}))
    by_id = {r["case_id"]: r for r in pairs}
    cells = [dict(mapping, **{key: by_id[mapping["case_id"]][key]
                             for key in ("old_cost", "new_cost", "cost_change_seconds", "completion_change_pp")})
             for mapping in manifest["figure_cells"]]
    lookup = {(c["arm"], c["demand_scale"], c["blocks"], c["policy_updates"]): c for c in cells}
    contrasts = []
    for demand in (1.0, 1.25):
        for blocks in (2, 6):
            for updates in (512, 2048):
                for candidate, control in (("h2o", "online"), ("mcwm", "h2o"), ("mcwm", "online")):
                    a, b = (lookup[arm, demand, blocks, updates] for arm in (candidate, control))
                    contrasts.append(dict(candidate=candidate, control=control, demand_scale=demand,
                                          blocks=blocks, policy_updates=updates,
                                          old_delta_seconds=a["old_cost"]-b["old_cost"],
                                          new_delta_seconds=a["new_cost"]-b["new_cost"]))
    report = dict(status="complete", unique_evaluations=32, displayed_figure_cells=40,
                  old_extra_hold_scale=0.25, new_extra_hold_scale=args.scale,
                  interpretation="One frozen training seed and one common environment block per demand; retrospective execution-scale diagnostic, not independent statistical confirmation. Group means average data-budget conditions.",
                  per_condition=pairs, by_method_demand_budget=summaries,
                  method_contrasts=contrasts, figure_cells=cells)
    output = args.output or folder / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    csv_path = output.with_name(output.stem + "_figure_cells.csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    print(json.dumps(dict(status="complete", cases=32, figure_cells=40,
                          analysis=str(output), figure_csv=str(csv_path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
