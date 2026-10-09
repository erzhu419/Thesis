"""Evaluate fixed rules through the unchanged full-day SUMO execution path."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_usage_v3"))
from usage_common import module, write
from rules import hold_seconds

P = json.loads((HERE / "protocol.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=P["methods"], required=True)
    parser.add_argument("--layer", choices=P["execution_layers"], required=True)
    parser.add_argument("--case", type=int, choices=range(10), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runtime = module("daganzo_runtime", HERE.parent / "sumo_eval/attribution_v1/run.py")
    config, _ = runtime.contract()
    assert config.max_hold_seconds == P["max_hold_seconds"]
    out = HERE / ("smoke" if args.smoke else "runs") / args.method / args.layer / f"case{args.case:02d}"
    output = out / "results" / f"case{args.case:02d}.json"
    if output.exists():
        row = json.loads(output.read_text())
        if row.get("comparison_protocol") != P["name"]:
            raise RuntimeError(f"Incomplete result metadata: {output}")
        print(f"Already complete: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = defaultdict(float)

    def action(method, raw, cfg):
        event = runtime.current_event_v7(raw)
        fields = ("forward_headway_seconds", "backward_headway_seconds",
                  "target_forward_headway_seconds", "target_backward_headway_seconds",
                  "forward_bus_present", "backward_bus_present")
        values = [float(event[runtime.RAW_EVENT_STATE_INDEX[key]]) for key in fields]
        hold = hold_seconds(method, *values)
        previous = float(runtime.headway_balance_hold_seconds_v4(event[:runtime.V4_DIM], 1.0, cfg))
        difference = abs(hold - previous)
        diagnostic["decisions"] += 1
        diagnostic["different_from_adjusted_headway"] += difference > 1e-4
        diagnostic["absolute_hold_difference_seconds"] += difference
        diagnostic["unequal_scheduled_targets"] += abs(values[2] - values[3]) > 1e-4
        diagnostic["missing_neighbor"] += not (values[4] and values[5])
        return runtime.np.asarray([runtime.hold_seconds_to_policy_action_v7(hold, cfg)], runtime.np.float32)

    runtime.rule_action = action
    p = dict(P, policy_exports={}, training_seed=None)
    if args.smoke:
        p["evaluation_horizon_seconds"] = 1200
    block = P["evaluation_blocks"][args.case % 5]
    scale = P["evaluation_demand_scales"][args.case // 5]
    case = dict(method=args.method, execution_layer=args.layer, demand_scale=scale,
                **block, case_id=f"case{args.case:02d}")
    cfg, objects = runtime.common.build_scenario(out, scale, block["demand_seed"], p["departure_jitter_seconds"])
    row = runtime.evaluate((case, p, str(out), cfg, objects))
    assert row["sumo_version"][1] == "SUMO 1.25.0"
    row.update(comparison_protocol=P["name"], deployment_hold_scale=P["deployment_hold_scale"],
               rule_diagnostic=dict(diagnostic))
    write(output, row)
    print(f"DONE {args.method} {args.layer} case {args.case}: {row['horizon_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
