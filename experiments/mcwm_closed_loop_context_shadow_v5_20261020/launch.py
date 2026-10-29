"""Launch fixed v5 diagnostics first; the frozen-world stage requires a decision."""

import argparse
import json
from pathlib import Path
import shlex

import infra
from analysis import native_preflight_gate, native_frozen_preflight_gate, result_name, run_cells

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def make_job(arm, seed, scenario, *, preflight=False, stage="stage_a"):
    label = f"control/{scenario}/{arm}/seed{seed}"
    directory = ("preflight_frozen_world" if arm == "frozen_world" else protocol()["preflight_directory"]) if preflight else "runs" if stage == "stage_a" else "frozen_runs"
    output = HERE / directory / label
    argv = ["--arm", arm, "--seed", str(seed), "--scenario", scenario,
            "--out", infra.remote(output)]
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
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]), timeout=45, check=True)
    return json.loads(output)


def resident_preflight_gate():
    script = ("import sys,json;sys.path.insert(0," + repr(infra.remote(HERE)) + ");"
              "from analysis import native_preflight_gate;"
              "print(json.dumps(native_preflight_gate(),separators=(',',':')))" )
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]), timeout=45, check=True)
    value = json.loads(output)
    infra.write(HERE / "native_preflight_gate.json", value)
    return value


def submit_preflight():
    source_ready()
    return infra.submit([make_job("action_local", 6201, "local_speed_drop", preflight=True)], protocol()["preflight_directory"])


def submit_campaign():
    source_ready()
    gate = resident_preflight_gate()
    if not gate["all_gates_pass"]:
        raise RuntimeError("Native eight-step packet-shadow preflight has not passed")
    return infra.submit([make_job(**cell) for cell in run_cells(protocol(), "stage_a")], "stage_a")


def submit_frozen():
    decision = json.loads((HERE / "decision.json").read_text())
    summary = json.loads((HERE / result_name("stage_a")).read_text())
    if decision.get("go") is not True or summary.get("status") != "complete":
        raise RuntimeError("Read complete Stage A results and record go=true before the frozen-world comparison")
    script = ("import sys,json;sys.path.insert(0," + repr(infra.remote(HERE)) + ");"
              "from analysis import native_frozen_preflight_gate;"
              "print(json.dumps(native_frozen_preflight_gate(),separators=(',',':')))" )
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]), timeout=45, check=True)
    gate = json.loads(output)
    infra.write(HERE / "native_frozen_preflight_gate.json", gate)
    if not gate["all_gates_pass"]:
        raise RuntimeError("The native frozen-world preflight has not passed")
    source_ready()
    infra.stage([HERE / "decision.json"])
    return infra.submit([make_job(**cell, stage="stage_b") for cell in run_cells(protocol(), "stage_b")], "stage_b")


def submit_frozen_preflight():
    decision = json.loads((HERE / "decision.json").read_text())
    if decision.get("go") is not True:
        raise RuntimeError("Record the Stage A decision before the frozen-world preflight")
    source_ready()
    return infra.submit([make_job("frozen_world", 6201, "local_speed_drop", preflight=True)], "preflight_frozen_world")


def submit_finalizer(stage):
    json.loads((HERE / (stage + "_submission.json")).read_text())
    return infra.submit([infra.task(protocol(), stage + "/finalize", ["--stage", stage],
                                  runner="finalize.py", ram_mb=1024)], stage + "_finalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "preflight", "preflight-frozen", "submit", "submit-frozen", "finalize", "status", "collect"))
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), default="stage_a")
    parser.add_argument("--batch", choices=("preflight", "preflight_device_fix", "preflight_frozen_world", "stage_a", "stage_b", "stage_a_finalize", "stage_b_finalize"), default="stage_a")
    args = parser.parse_args()
    if args.operation == "stage":
        files = sorted(path for path in HERE.glob("*.py") if not path.name.startswith("test_"))
        result = infra.stage([*files, HERE / "protocol.json"])
    elif args.operation == "preflight":
        result = submit_preflight()
    elif args.operation == "submit":
        result = submit_campaign()
    elif args.operation == "submit-frozen":
        result = submit_frozen()
    elif args.operation == "preflight-frozen":
        result = submit_frozen_preflight()
    elif args.operation == "finalize":
        result = submit_finalizer(args.stage)
    elif args.operation == "status":
        rows = json.loads((HERE / f"{args.batch}_submission.json").read_text())["submitted"]
        result = infra.status([row["id"] for row in rows])
    else:
        path = result_name(args.stage)
        value = infra.collect_json(path)
        result = dict(status=value.get("status"), bytes=(HERE / path).stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
