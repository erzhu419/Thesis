"""Launch only the declared passive diagnosis and development stability extension."""

import argparse
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
PREFLIGHT_CELLS = (("action_local", 6201), ("action_model", 6203))


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def run_cells(p):
    return p["run_cells"]


def make_job(arm, seed, scenario, *, preflight=False):
    label = f"control/{scenario}/{arm}/seed{seed}"
    output = HERE / ("preflight" if preflight else "runs") / label
    argv = ["--arm", arm, "--seed", str(seed), "--scenario", scenario,
            "--out", infra.remote(output)]
    if preflight:
        argv.append("--preflight")
    return infra.task(protocol(), ("preflight/" if preflight else "") + label, argv)


def source_ready():
    script = ("import json;from pathlib import Path;p=Path(" + repr(str(infra.RESIDENT_SOURCE)) + ");"
        "s=json.loads((p/'source_summary.json').read_text());"
        "rows=json.loads((p/'source_intervals.json').read_text());"
        "assert s.get('status')=='complete' and len(rows)==102;"
        "q=Path(" + repr(str(infra.RESIDENT_INPUTS)) + ");"
        "assert all((q/('h2oplus_seed%d.pt'%seed)).is_file() for seed in range(6201,6206));"
        "assert (q/'history.npz').is_file() and (q/'source_9101.npz').is_file();"
        "print(json.dumps({'status':s['status'],'source_intervals':len(rows),'five_initializers':True}))")
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]), timeout=45, check=True)
    return json.loads(output)


def submit_preflight():
    source_ready()
    return infra.submit([make_job(arm, seed, "local_speed_drop", preflight=True)
                         for arm, seed in PREFLIGHT_CELLS], "preflight")


def submit_campaign():
    source_ready()
    for arm, seed in PREFLIGHT_CELLS:
        result = infra.read_remote_json(f"preflight/control/local_speed_drop/{arm}/seed{seed}/summary.json")
        if (result.get("status") != "complete" or result.get("target_online_training_events") != 50
                or result.get("sumo_version") != "SUMO 1.25.0"
                or result.get("scenario_metadata", {}).get("actual_shock_at_seconds") != 1350):
            raise RuntimeError(f"Native preflight is incomplete: {arm}")
    return infra.submit([make_job(**cell) for cell in run_cells(protocol())], "campaign")


def submit_finalizer():
    json.loads((HERE / "campaign_submission.json").read_text())
    return infra.submit([infra.task(protocol(), "finalize", runner="finalize.py", ram_mb=1024)], "finalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "preflight", "submit", "finalize", "status", "collect"))
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
    elif args.operation == "status":
        rows = json.loads((HERE / f"{args.batch}_submission.json").read_text())["submitted"]
        result = infra.status([row["id"] for row in rows])
    else:
        value = infra.collect_json("head_diagnostic_summary.json")
        result = dict(status=value.get("status"), bytes=(HERE / "head_diagnostic_summary.json").stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
