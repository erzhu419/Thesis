"""Launch only the isolated finite action/local development pilot."""

import argparse
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
PREFLIGHT_ARMS = ("old_model", "action_real", "action_local")


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def make_job(arm, seed, scenario, *, preflight=False):
    label = f"control/{scenario}/{arm}/seed{seed}"
    output = HERE / ("preflight" if preflight else "runs") / label
    argv = ["--arm", arm, "--seed", str(seed), "--scenario", scenario,
            "--out", infra.remote(output)]
    if preflight:
        argv.append("--preflight")
    return infra.task(protocol(), ("preflight/" if preflight else "") + label, argv)


def source_ready():
    command = "import json;from pathlib import Path;p=Path(" + repr(str(infra.RESIDENT_SOURCE)) + ");s=json.loads((p/'source_summary.json').read_text());assert s.get('status')=='complete';assert (p/'source_intervals.json').is_file();print(json.dumps({'status':s['status'],'source_intervals_bytes':(p/'source_intervals.json').stat().st_size}))"
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", command]), timeout=45, check=True)
    return json.loads(output)


def submit_preflight():
    source_ready()
    return infra.submit([make_job(arm, 6201, "local_speed_drop", preflight=True)
                         for arm in PREFLIGHT_ARMS], "preflight")


def submit_campaign():
    source_ready()
    for arm in PREFLIGHT_ARMS:
        result = infra.read_remote_json(f"preflight/control/local_speed_drop/{arm}/seed6201/summary.json")
        if (result.get("status") != "complete" or result.get("target_online_training_events") != 50
                or result.get("sumo_version") != "SUMO 1.25.0"
                or result.get("scenario_metadata", {}).get("actual_shock_at_seconds") != 1350):
            raise RuntimeError(f"Native preflight is incomplete: {arm}")
    p = protocol()
    return infra.submit([make_job(arm, seed, scenario) for seed in p["training_seeds"]
                         for scenario in p["scenarios"] for arm in p["arms"]], "campaign")


def submit_finalizer():
    json.loads((HERE / "campaign_submission.json").read_text())
    return infra.submit([infra.task(protocol(), "finalize", runner="finalize.py", ram_mb=768)], "finalize")


def build_holdout():
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([
        infra.PYTHON, infra.remote(HERE / "holdout.py")]), timeout=45, check=True)
    return json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "preflight", "submit", "finalize", "holdout", "status", "collect"))
    parser.add_argument("--batch", choices=("preflight", "campaign", "finalize"), default="campaign")
    args = parser.parse_args()
    if args.operation == "stage":
        files = sorted(path for path in HERE.glob("*.py") if not path.name.startswith("test_"))
        result = infra.stage([*files, HERE / "protocol.json"])
    elif args.operation == "preflight":
        result = submit_preflight()
    elif args.operation == "submit":
        result = submit_campaign()
    elif args.operation == "finalize":
        result = submit_finalizer()
    elif args.operation == "holdout":
        result = build_holdout()
    elif args.operation == "status":
        rows = json.loads((HERE / f"{args.batch}_submission.json").read_text())["submitted"]
        result = infra.status([row["id"] for row in rows])
    else:
        value = infra.collect_json("pilot_summary_with_holdout.json")
        result = dict(status=value.get("status"), holdout_status=value.get("post_training_holdout", {}).get("status"),
                      bytes=(HERE / "pilot_summary_with_holdout.json").stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
