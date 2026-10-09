#!/usr/bin/env python3
"""Generate new paired passenger scenarios with the original official generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2o-root", type=Path, required=True)
    parser.add_argument("--sumo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--passenger-objects", type=Path,
                        help="Original passenger-object template if the execution bundle omits it")
    parser.add_argument("--sumo-seeds", type=int, nargs="+", default=list(range(108101, 108106)))
    parser.add_argument("--demand-seeds", type=int, nargs="+", default=list(range(108201, 108206)))
    parser.add_argument("--demand-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    parser.add_argument("--departure-jitter-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if len(args.sumo_seeds) != len(args.demand_seeds):
        parser.error("each SUMO block must have one corresponding demand realization seed")
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new scenario directory; existing inputs are retained: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    root = args.h2o_root.resolve(strict=True)
    sumo = args.sumo_root.resolve(strict=True)
    sys.path.insert(0, str(root))
    from reproducibility.build_passenger_scenarios import build_scenario
    from reproducibility.prepare_h2oplus_v8_development_stage_a import _portable_scenario

    cases = []
    for block, (sumo_seed, demand_seed) in enumerate(zip(args.sumo_seeds, args.demand_seeds)):
        for scale in args.demand_scales:
            original = build_scenario(
                base_config=sumo / "online_control/control_sim_traci_period.sumocfg",
                passenger_routes=(sumo / "e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml",
                                  sumo / "e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml"),
                passenger_objects=(args.passenger_objects.resolve() if args.passenger_objects else
                                   sumo / "online_control/initialize_obj/save_obj_passenger.add.xml"),
                output_root=output, scale=scale, seed=demand_seed,
                depart_jitter_seconds=args.departure_jitter_seconds)
            portable, evidence = _portable_scenario(original, force=False)
            cases.append(dict(case=len(cases), block_id=block, sumo_seed=sumo_seed,
                demand_seed=demand_seed, demand_scale=scale,
                scenario_manifest=str(portable.relative_to(output)),
                scaled_total_persons=evidence["scaled_total_persons"]))
            print(json.dumps(cases[-1]), flush=True)
    index = dict(evaluation_horizon_seconds=24000,
        departure_jitter_seconds=args.departure_jitter_seconds, cases=cases)
    (output / "scenario_index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
