#!/usr/bin/env python3
"""Evaluate frozen controllers with the original v7 loop and common v19 execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h2o-root", "checkpoint", "dataset", "scenario-manifest", "sumo-dir",
                 "edge-xml", "sumo-runtime-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--method", required=True,
                        choices=("mcwm_aligned", "h2oplus_final", "sac_guarded"))
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--sumo-seed", required=True, type=int)
    parser.add_argument("--training-record", type=Path)
    parser.add_argument("--reference-result", type=Path,
                        help="Compare completed replay with an existing guarded-controller result")
    parser.add_argument("--verified-dataset-sha256")
    parser.add_argument("--verified-dataset-size-bytes", type=int)
    parser.add_argument("--verified-dataset-mtime-ns", type=int)
    parser.add_argument("--job-signature")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def check_replay(reference, result):
    """Compare physical outcomes; timestamps and measured runtime are irrelevant."""
    if "passenger_welfare" in reference:
        fields = ("contract", "episode", "event_identity", "operations", "passenger_welfare",
                  "passengers", "service", "ledger_audit", "bus_service_semantics_audit")
        checks = {key: reference[key] == result[key] for key in fields}
        checks["runtime"] = (reference["runtime"]["sumo_runtime_signature"] ==
                             result["runtime"]["sumo_runtime_signature"])
        checks["torch_version"] = (reference["runtime"]["torch_version"] ==
                                   result["runtime"]["torch_version"])
    else:
        # Earlier MC-WM files contain the same service quantities in a compact layout.
        pairs = {
            "passengers": (reference["passengers"], result["passengers"]),
            "service": (reference["service"], result["service"]),
            "generalized_time": (reference["generalized_seconds_per_departed_passenger"],
                result["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"]),
            "horizon": (reference["horizon_seconds"], result["episode"]["observed_end_time_seconds"]),
            "decisions": (reference["decision_events"], result["episode"]["control_decisions"]),
            "mean_hold": (reference["mean_hold_seconds"], result["operations"]["hold_seconds"]["mean"]),
        }
        checks = {key: old == new for key, (old, new) in pairs.items()}
    checks["sumo_seed"] = reference["sumo_seed"] == result["sumo_seed"]
    old_scenario = reference.get("scenario")
    if isinstance(old_scenario, dict):
        checks["demand_realization"] = all(old_scenario[key] == result["scenario"][key]
            for key in ("selection_seed", "demand_scale", "depart_jitter_seconds"))
    else:
        checks["demand_realization"] = (
            reference["demand_seed"] == result["scenario"]["selection_seed"] and
            reference["demand_scale"] == result["scenario"]["demand_scale"])
    return dict(passed=all(checks.values()), checks=checks)


def main():
    args = parse_args()
    os.environ["LIBSUMO_AS_TRACI"] = "quiet"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    sys.path.insert(0, str(args.h2o_root.resolve(strict=True)))
    import torch
    from SimpleSAC import evaluate_transit_v20_formal_confirmation as v20
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
    from bus_h2o.sumo_provenance import load_sumo_runtime_lock

    v19, base = v20.v19, v20.v19.base
    _, lock = load_sumo_runtime_lock(args.sumo_runtime_lock)
    official = argparse.Namespace(**vars(args))
    official.method_name = "%s-seed%d" % (args.method, args.training_seed)
    official.sumo_backend = "libsumo"
    official.expected_sumo_runtime_version = "1.25.0"
    official.expected_sumo_runtime_signature = lock["runtime_signature"]
    official.expected_sumo_runtime_lock_sha256 = lock["sha256"]
    official.device = "cpu"
    official.max_events = official.max_sim_time_seconds = 0
    official.zero_hold = official.myopic_welfare = official.force = False
    official.headway_balance_gain = official.event_log = None

    def build_policy(official_args, config, station_max, dataset_hash):
        actor, extra = load_actor_checkpoint_v7(
            args.checkpoint, "cpu", require_policy_artifact=True)
        actor_seed = extra.get("training_seed", extra.get("seed"))
        if actor_seed != args.training_seed:
            raise ValueError("actor training seed differs from the requested case")
        if "contract_config" in extra and extra["contract_config"] != config.to_dict():
            raise ValueError("actor and evaluation MDP contracts differ")
        if "dataset_sha256" in extra and extra["dataset_sha256"] != dataset_hash:
            raise ValueError("actor and evaluation datasets differ")

        def policy(raw):
            canonical = raw_state_to_canonical_v7(raw, station_max, config)
            with torch.inference_mode():
                action, _ = actor(torch.as_tensor(canonical[None], dtype=torch.float32),
                                  deterministic=True)
            return v19.completion_safety_action(
                raw, action.cpu().numpy()[0], "quarter-load-sixty", config)

        metadata = dict(type="frozen-v7-actor-common-v19-execution",
            path=str(args.checkpoint.resolve()), artifact_schema=extra["policy_artifact_schema"],
            training_metadata=extra, additional_training_updates=0,
            execution_transform="v19.completion_safety_action:quarter-load-sixty",
            deployment_hold_scale=1.0, evaluation_entry=str(Path(__file__).resolve()))
        if args.training_record:
            metadata["training_record"] = dict(path=str(args.training_record.resolve()),
                record=json.loads(args.training_record.read_text()))
        return policy, metadata

    base.parse_args = lambda: official
    base.build_policy = build_policy
    base.evaluation_source_paths = v20.evaluation_source_paths
    base.main()
    result = json.loads(args.output.read_text())
    result["comparison_case"] = dict(method=args.method, training_seed=args.training_seed,
        sumo_seed=args.sumo_seed, demand_seed=result["scenario"]["selection_seed"],
        demand_scale=result["scenario"]["demand_scale"],
        execution_transform="v19:quarter-load-sixty", deployment_hold_scale=1.0)
    if args.reference_result:
        reference = json.loads(args.reference_result.read_text())
        result["replay_check"] = dict(check_replay(reference, result),
            reference_result=str(args.reference_result.resolve()))
    temporary = args.output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(dict(case=result["comparison_case"],
        cost_seconds=result["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"],
        completion_rate=result["passengers"]["completion_rate"],
        replay_check=result.get("replay_check")), ensure_ascii=False), flush=True)
    if args.reference_result and not result["replay_check"]["passed"]:
        raise RuntimeError("replay differs from the original physical outcome; see replay_check")


if __name__ == "__main__":
    main()
