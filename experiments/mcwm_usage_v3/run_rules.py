"""Evaluate existing rule controllers on the frozen model-use test days."""

import argparse

from usage_common import HERE, EXPERIMENTS, P, module, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=("zero", "headway"), required=True)
    parser.add_argument("--case", type=int, choices=range(10), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runtime = module("usage_rule_runtime", EXPERIMENTS/"sumo_eval/attribution_v1/run.py")
    p = dict(P, policy_exports={}, training_seed=None)
    if args.smoke:
        p["evaluation_horizon_seconds"] = 1200
    out = HERE/("rule_smoke" if args.smoke else "rules")/args.rule/f"case{args.case:02d}"
    (out/"results").mkdir(parents=True, exist_ok=True)
    block = P["evaluation_blocks"][args.case % 5]
    scale = P["evaluation_demand_scales"][args.case // 5]
    case = dict(method=args.rule, execution_layer="early60", demand_scale=scale,
                **block, case_id=f"case{args.case:02d}")
    cfg, objects = runtime.common.build_scenario(out, scale, block["demand_seed"], p["departure_jitter_seconds"])
    row = runtime.evaluate((case, p, str(out), cfg, objects))
    assert row["sumo_version"][1] == "SUMO 1.25.0"
    row["deployment_hold_scale"] = p["deployment_hold_scale"]
    write(out/"results"/f"case{args.case:02d}.json", row)
    print(f"DONE rule {args.rule} case {args.case}: {row['horizon_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
