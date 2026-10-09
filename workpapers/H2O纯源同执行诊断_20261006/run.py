#!/usr/bin/env python3
"""Run one frozen 25k SAC actor through the original v20 SUMO evaluator.

Each invocation uses one existing formal SAC result as its immutable case
identity. Run all 5 training seeds x 10 blocks x 3 demand levels externally.
Only the action transform changes in source-guarded mode; source-raw replays
the original controller for a full-episode environment consistency check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2o-root", type=Path, default=os.environ.get("H2O_ROOT"))
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--mode", choices=("source-guarded", "source-raw"),
                        default="source-guarded")
    for name in ("checkpoint", "scenario-manifest", "dataset", "sumo-dir",
                 "edge-xml", "sumo-runtime-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--verified-dataset-size-bytes", type=int)
    parser.add_argument("--verified-dataset-mtime-ns", type=int,
                        help="Use with verified size only after parent verification of the original dataset")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.h2o_root is None:
        parser.error("--h2o-root or H2O_ROOT is required")
    if (args.verified_dataset_size_bytes is None) != (args.verified_dataset_mtime_ns is None):
        parser.error("verified dataset size and mtime must be supplied together")
    return args


def main():
    args = parse_args()
    root = args.h2o_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(root))
    from SimpleSAC import evaluate_transit_v20_formal_confirmation as v20
    from SimpleSAC import evaluate_transit_v11_actor_hold as v11

    v19, base = v20.v19, v20.v19.base
    reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
    method = reference["method"]
    match = re.fullmatch(r"sac-online-seed(620[1-5])-events000025000", method)
    if match is None:
        raise ValueError("reference must be one of the original five frozen SAC 25k actors")
    if reference["sumo_seed"] not in range(8101, 8111):
        raise ValueError("reference is outside the original ten formal blocks")
    scenario = reference["scenario"]
    if (scenario["selection_seed"] != reference["sumo_seed"] + 100
            or scenario["demand_scale"] not in (0.75, 1.0, 1.25)
            or scenario["depart_jitter_seconds"] != 120.0):
        raise ValueError("reference does not use the frozen formal demand design")
    runtime = reference["runtime"]
    if runtime["sumo_backend"] != "libsumo" or runtime["sumo_version"] != "1.25.0":
        raise ValueError("reference must use the formal libsumo 1.25.0 runtime")
    extra = reference["policy"]["training_metadata"]
    if extra["fewshot_learner_update_steps"] != 20000:
        raise ValueError("reference is not the frozen 25k-event / 20k-update SAC endpoint")
    os.environ[v11.V8_PROTOCOL_SHA_ENV] = extra["fewshot_protocol"]["sha256"]

    # Populate the original evaluator arguments from the original case identity.
    official = argparse.Namespace(**vars(args))
    official.method_name = (method if args.mode == "source-raw" else
                            method.replace("sac-online-", "sac-online-v19-quarter-load-sixty-", 1))
    official.expected_algorithm = "sac-online"
    official.expected_training_seed = int(match.group(1))
    official.expected_training_run_signature = extra["run_signature"]
    official.expected_training_admission_sha256 = extra["training_admission"]["sha256"]
    official.expected_training_checkpoint_sha256 = extra["training_checkpoint_sha256"]
    official.expected_actor_tensor_sha256 = reference["policy"]["actor_tensor_sha256"]
    official.expected_learner_update_steps = extra["fewshot_learner_update_steps"]
    official.expected_ratio_optimizer_steps = extra["fewshot_ratio_optimizer_steps"]
    official.sumo_seed = reference["sumo_seed"]
    official.sumo_backend = runtime["sumo_backend"]
    official.expected_sumo_runtime_version = runtime["sumo_version"]
    official.expected_sumo_runtime_signature = runtime["sumo_runtime_signature"]
    official.expected_sumo_runtime_lock_sha256 = runtime["sumo_runtime_lock"]["sha256"]
    official.verified_dataset_sha256 = (reference["dataset"]["sha256"]
                                       if args.verified_dataset_size_bytes is not None else None)
    official.device = "cpu"
    official.max_events = official.max_sim_time_seconds = 0
    official.zero_hold = official.myopic_welfare = official.force = False
    official.headway_balance_gain = official.job_signature = None

    original_scenario_loader = base.load_validated_scenario

    def matched_scenario_loader(path):
        scenario_runtime, observed = original_scenario_loader(path)
        for key in ("generation_config", "scenario_sha256", "scaled_total_persons"):
            if observed[key] != scenario[key]:
                raise ValueError("scenario differs from the original formal case: " + key)
        if sorted(observed["runtime_input_sha256"].values()) != sorted(scenario["runtime_input_sha256"].values()):
            raise ValueError("SUMO runtime inputs differ from the original formal case")
        return scenario_runtime, observed

    def build_policy(official_args, config, station_max, dataset_hash):
        if dataset_hash != reference["dataset"]["sha256"]:
            raise ValueError("dataset differs from the original formal case")
        frozen_args = argparse.Namespace(**vars(official_args))
        frozen_args.method_name = method
        actor_policy, metadata = v11.build_development_policy(
            frozen_args, config, station_max, dataset_hash)
        metadata["execution_diagnostic"] = {
            "reference_result": str(args.reference_result.resolve()),
            "original_method": method,
            "mode": args.mode,
            "frozen_actor": True,
            "additional_training_updates": 0,
            "execution_transform": ("v19.completion_safety_action:quarter-load-sixty"
                                    if args.mode == "source-guarded" else "identity"),
            "design": "retrospective same-execution diagnostic on existing formal blocks",
        }
        if args.mode == "source-raw":
            return actor_policy, metadata

        def guarded_policy(raw):
            return v19.completion_safety_action(
                raw, actor_policy(raw), "quarter-load-sixty", config)

        return guarded_policy, metadata

    base.parse_args = lambda: official
    base.build_policy = build_policy
    base.load_validated_scenario = matched_scenario_loader
    base.evaluation_source_paths = v20.evaluation_source_paths
    base.main()


if __name__ == "__main__":
    main()
