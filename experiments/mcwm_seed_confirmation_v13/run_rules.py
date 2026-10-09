"""Existing headway and TransitDuet Daganzo-style rules on the new paired days."""

import argparse
from confirmation_common import HERE, P, module, write, evaluation_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=P["rules"], required=True)
    parser.add_argument("--case", type=int, choices=range(20), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runtime = module("confirmation_rule_runtime", HERE.parent / "sumo_eval/attribution_v1/run.py")
    if args.rule == "daganzo_duet":
        rules = module("confirmation_duet_rule", HERE.parent / "daganzo_baseline_v1/rules.py")

        def action(method, raw, cfg):
            event = runtime.current_event_v7(raw)
            fields = ("forward_headway_seconds", "backward_headway_seconds",
                "target_forward_headway_seconds", "target_backward_headway_seconds",
                "forward_bus_present", "backward_bus_present")
            values = [float(event[runtime.RAW_EVENT_STATE_INDEX[key]]) for key in fields]
            hold = rules.hold_seconds("daganzo_duet", *values)
            return runtime.np.asarray([runtime.hold_seconds_to_policy_action_v7(hold, cfg)], runtime.np.float32)

        runtime.rule_action = action
    p, local = evaluation_protocol(P, args.case)
    p = dict(p, policy_exports={}, training_seed=None)
    if args.smoke:
        p["evaluation_horizon_seconds"] = 1200
    out = HERE / ("rule_smoke" if args.smoke else "rules") / args.rule / f"case{args.case:02d}"
    result = out / "results" / f"case{args.case:02d}.json"
    if result.exists():
        return
    result.parent.mkdir(parents=True, exist_ok=True)
    block = p["evaluation_blocks"][local % 5]
    scale = p["evaluation_demand_scales"][local // 5]
    case = dict(method=args.rule, execution_layer="early60", demand_scale=scale,
        **block, case_id=f"case{args.case:02d}")
    cfg, objects = runtime.common.build_scenario(out, scale, block["demand_seed"], p["departure_jitter_seconds"])
    row = runtime.evaluate((case, p, str(out), cfg, objects))
    row.update(case=args.case, split="development" if args.case < 10 else "test",
        comparison_protocol=P["name"], deployment_hold_scale=1.)
    write(result, row)
    print(f"DONE fixed rule {args.rule} case{args.case:02d}", flush=True)


if __name__ == "__main__":
    main()
