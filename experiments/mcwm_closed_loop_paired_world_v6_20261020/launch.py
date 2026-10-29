"""Submit one paired-world preflight and its four passive diagnostic days."""

import argparse
import json
from pathlib import Path
import shlex

import infra
from analysis import native_preflight_gate, result_name, run_cells

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def make_job(seed, scenario, arm="action_model", *, preflight=False):
    directory = protocol()["preflight_directory"] if preflight else "runs"
    label = f"control/{scenario}/{arm}/seed{seed}"
    argv = ["--arm", arm, "--seed", str(seed), "--scenario", scenario,
            "--out", infra.remote(HERE / directory / label)]
    if preflight:
        argv.append("--preflight")
    return infra.task(protocol(), directory + "/" + label, argv)


def source_ready():
    script = ("import json;from pathlib import Path;p=Path(" + repr(str(infra.RESIDENT_SOURCE)) + ");"
        "s=json.loads((p/'source_summary.json').read_text());"
        "rows=json.loads((p/'source_intervals.json').read_text());"
        "assert s.get('status')=='complete' and len(rows)==102;"
        "q=Path(" + repr(str(infra.RESIDENT_INPUTS)) + ");"
        "assert all((q/('h2oplus_seed%d.pt'%seed)).is_file() for seed in (6201,6202));"
        "assert (q/'history.npz').is_file() and (q/'source_9101.npz').is_file();"
        "print(json.dumps({'status':s['status'],'source_intervals':len(rows),'initializers':2}))")
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def resident_preflight_gate():
    script = ("import sys,json;sys.path.insert(0," + repr(infra.remote(HERE)) + ");"
              "from analysis import native_preflight_gate;"
              "print(json.dumps(native_preflight_gate(),separators=(',',':')))")
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    value = json.loads(output)
    infra.write(HERE / "native_preflight_gate.json", value)
    return value


def submit_preflight():
    source_ready()
    p = protocol()
    return infra.submit([make_job(p["preflight"]["seed"], p["preflight"]["scenario"], preflight=True)],
                        p["preflight_directory"])


def submit_campaign():
    gate = resident_preflight_gate()
    if not gate["all_gates_pass"]:
        raise RuntimeError("Native paired-world eight-step shadow preflight has not passed")
    source_ready()
    return infra.submit([make_job(**cell) for cell in run_cells(protocol())], "diagnostic")


def submit_finalizer():
    json.loads((HERE / "diagnostic_submission.json").read_text())
    return infra.submit([infra.task(protocol(), "diagnostic/finalize", runner="finalize.py", ram_mb=1024)],
                        "diagnostic_finalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "preflight", "submit", "finalize", "status", "collect"))
    parser.add_argument("--batch", default="diagnostic")
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
        value = infra.collect_json(result_name())
        result = dict(status=value.get("status"), bytes=(HERE / result_name()).stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
