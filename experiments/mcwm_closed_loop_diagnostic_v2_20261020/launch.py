"""Stage and launch the finite development diagnosis without editing old experiments."""

import argparse
import json
from pathlib import Path

import infra

HERE = Path(__file__).resolve().parent
PREFLIGHT_ARMS = (("control", "real075"), ("control", "model"), ("prediction", "common"))


def protocol():
    return json.loads((HERE/"protocol.json").read_text())


def make_job(mode, arm, seed, scenario, *, preflight=False):
    label = f"{mode}/{scenario}/{arm}/seed{seed}"
    out = HERE/("preflight" if preflight else "runs")/label
    argv = ["--mode", mode, "--arm", arm, "--seed", str(seed), "--scenario", scenario,
            "--out", infra.remote(out)]
    if preflight:
        argv.append("--preflight")
    return infra.task(protocol(), ("preflight/" if preflight else "")+label, argv)


def source_ready():
    record = infra.read_remote_json("inputs/source_summary.json")
    if record.get("status") != "complete":
        raise RuntimeError("Source interval calibration is incomplete")
    return record


def submit_source():
    return infra.submit([infra.task(protocol(), "source-calibration", runner="source_calibration.py")], "source")


def submit_preflight():
    source_ready()
    rows = [make_job(mode, arm, 6201, "local_speed_drop", preflight=True)
            for mode, arm in PREFLIGHT_ARMS]
    for row, node in zip(rows, ("node001", "node004", "node005")):
        row["preferred_node"] = node
    return infra.submit(rows, "preflight")


def submit_campaign():
    source_ready()
    for mode, arm in PREFLIGHT_ARMS:
        r = infra.read_remote_json(f"preflight/{mode}/local_speed_drop/{arm}/seed6201/summary.json")
        if (r.get("status") != "complete" or r.get("target_online_training_events") != 15
                or r.get("sumo_version") != "SUMO 1.25.0"
                or r.get("scenario_metadata", {}).get("actual_shock_at_seconds") != 1350):
            raise RuntimeError(f"Native preflight is incomplete: {mode}/{arm}")
    p = protocol()
    rows = []
    for seed in p["training_seeds"]:
        for scenario in p["scenarios"]:
            rows.extend(make_job("control", arm, seed, scenario) for arm in p["arms"])
            rows.append(make_job("prediction", "common", seed, scenario))
    return infra.submit(rows, "campaign")


def submit_finalizer():
    json.loads((HERE/"campaign_submission.json").read_text())
    row = infra.task(protocol(), "finalize", runner="finalize.py", ram_mb=512)
    return infra.submit([row], "finalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "source", "preflight", "submit", "finalize", "status", "collect"))
    parser.add_argument("--batch", choices=("source", "preflight", "campaign", "finalize"), default="campaign")
    args = parser.parse_args()
    if args.operation == "stage":
        files = sorted(path for path in HERE.glob("*.py") if not path.name.startswith("test_"))
        files.append(HERE/"protocol.json")
        result = infra.stage(files)
    elif args.operation == "source":
        result = submit_source()
    elif args.operation == "preflight":
        result = submit_preflight()
    elif args.operation == "submit":
        result = submit_campaign()
    elif args.operation == "finalize":
        result = submit_finalizer()
    elif args.operation == "status":
        rows = json.loads((HERE/f"{args.batch}_submission.json").read_text())["submitted"]
        result = infra.status([row["id"] for row in rows])
    else:
        result = {}
        for name in ("diagnostic_summary.json",):
            value = infra.collect_json(name)
            result[name] = dict(status=value.get("status"), bytes=(HERE/name).stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
