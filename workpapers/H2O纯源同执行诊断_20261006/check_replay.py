"""Compare a complete raw-SAC replay with its original evaluation."""
import argparse
import json
from pathlib import Path

FIELDS = ("contract", "episode", "event_identity", "operations", "passenger_welfare",
          "passengers", "service", "ledger_audit", "bus_service_semantics_audit")


def check(original_path, replay_path):
    original = json.loads(original_path.read_text())
    replay = json.loads(replay_path.read_text())
    checks = {key: original[key] == replay[key] for key in FIELDS}
    checks["frozen_actor"] = original["policy"]["actor_tensor_sha256"] == replay["policy"]["actor_tensor_sha256"]
    checks["source_contents"] = sorted(original["source_sha256"].values()) == sorted(replay["source_sha256"].values())
    checks["runtime"] = original["runtime"]["sumo_runtime_signature"] == replay["runtime"]["sumo_runtime_signature"]
    checks["torch_version"] = original["runtime"]["torch_version"] == replay["runtime"]["torch_version"]
    return {"passed": all(checks.values()), "checks": checks,
            "original": str(original_path), "replay": str(replay_path),
            "cost_seconds": replay["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"],
            "completion_rate": replay["passengers"]["completion_rate"],
            "dataset": replay["dataset"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.original, args.replay)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "dataset"}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
