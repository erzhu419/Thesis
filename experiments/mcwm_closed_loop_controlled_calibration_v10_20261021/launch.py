"""Stage v10, submit one native preflight, then four new controlled calibration days."""

import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
RESULT_NAME = "calibration_summary.json"
PREFLIGHT_MAX_BYTES = 20000
SUMMARY_MAX_BYTES = 200000


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def preflight_directory(p=None):
    p = protocol() if p is None else p
    return Path(p.get("preflight_directory", "preflight"))


def preflight_batch(p=None):
    return str(preflight_directory(p)).replace("/", "_")


def cells(p=None):
    p = protocol() if p is None else p
    return [dict(seed=seed, scenario=scenario) for seed, scenario in product(p["training_seeds"], p["scenarios"])]


def cell_path(cell):
    return Path(cell["scenario"]) / f"seed{cell['seed']}"


def make_job(seed, scenario, *, preflight=False):
    p = protocol()
    cell = dict(seed=seed, scenario=scenario)
    directory = preflight_directory(p) if preflight else Path("runs") / cell_path(cell)
    argv = ["--seed", str(seed), "--scenario", scenario, "--out", infra.remote(HERE / directory)]
    if preflight:
        argv.append("--preflight")
    return infra.task(p, str(directory), argv)


def source_ready():
    """Inspect resident inputs and imported code; v7 truth is not a v10 data source."""
    script = '''import json
from pathlib import Path
p=Path(__SOURCE__)
s=json.loads((p/'source_summary.json').read_text())
rows=json.loads((p/'source_intervals.json').read_text())
assert s.get('status')=='complete' and len(rows)==102
q=Path(__INPUTS__)
assert all((q/('h2oplus_seed%d.pt'%seed)).is_file() for seed in (6201,6202))
assert (q/'history.npz').is_file() and (q/'history_metadata.json').is_file()
assert (q/'source_9101.npz').is_file()
assert (Path(__V7__)/'bootstrap.py').is_file()
assert (Path(__V8__)/'response_world.py').is_file()
assert (Path(__V9__)/'ablation_world.py').is_file()
print(json.dumps(dict(source_intervals=len(rows),initializers=2,resident_code_available=True)))
'''.replace("__SOURCE__", repr(str(infra.RESIDENT_SOURCE))).replace(
        "__INPUTS__", repr(str(infra.RESIDENT_INPUTS))).replace("__V7__", repr(str(infra.RESIDENT_V7))).replace(
        "__V8__", repr(str(infra.RESIDENT_V8))).replace("__V9__", repr(str(infra.RESIDENT_V9)))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def submit_preflight():
    source_ready()
    p = protocol()
    settings = p["preflight"]
    return infra.submit([make_job(settings["seed"], settings["scenario"], preflight=True)], preflight_batch(p))


def submit_campaign():
    relative = preflight_directory() / "gate.json"
    path = HERE / relative
    value = json.loads(path.read_text()) if path.is_file() else infra.collect_json(
        str(relative), max_bytes=PREFLIGHT_MAX_BYTES)
    if value.get("status") != "complete" or value.get("integrity", {}).get("all_pass") is not True:
        raise RuntimeError("Native v10 preflight must complete with integrity.all_pass=true")
    return infra.submit([make_job(**row) for row in cells()], "diagnostic")


def submit_finalizer():
    json.loads((HERE / "diagnostic_submission.json").read_text())
    argv = ["--root", infra.remote(HERE / "runs"), "--out", infra.remote(HERE / RESULT_NAME), "--wait"]
    return infra.submit([infra.task(protocol(), "diagnostic/finalize", argv, runner="analyze.py", ram_mb=1024)],
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
        value = infra.collect_json(RESULT_NAME, max_bytes=SUMMARY_MAX_BYTES)
        result = dict(status=value.get("status"), bytes=(HERE / RESULT_NAME).stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
