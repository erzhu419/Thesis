"""Stage v8, submit one native preflight, then 12 prefix-only response probes."""

import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
RESULT_NAME = "response_summary.json"


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def cells(p=None):
    p = protocol() if p is None else p
    return [dict(seed=seed, scenario=scenario, target_line=line)
            for seed, scenario, line in product(p["training_seeds"], p["scenarios"], p["target_lines"])]


def cell_path(cell):
    return Path(cell["scenario"]) / cell["target_line"] / f"seed{cell['seed']}"


def make_job(seed, scenario, target_line, *, preflight=False):
    p = protocol()
    cell = dict(seed=seed, scenario=scenario, target_line=target_line)
    directory = Path("preflight") if preflight else Path("runs") / cell_path(cell)
    argv = ["--seed", str(seed), "--scenario", scenario, "--target-line", target_line,
            "--out", infra.remote(HERE / directory)]
    if preflight:
        argv.append("--preflight")
    return infra.task(p, str(directory), argv)


def source_ready():
    script = '''import json
from pathlib import Path
p=Path(__SOURCE__)
s=json.loads((p/'source_summary.json').read_text())
rows=json.loads((p/'source_intervals.json').read_text())
assert s.get('status')=='complete' and len(rows)==102
q=Path(__INPUTS__)
assert all((q/('h2oplus_seed%d.pt'%seed)).is_file() for seed in (6201,6202))
assert (q/'history.npz').is_file() and (q/'source_9101.npz').is_file()
assert (Path(__V7__)/'action_fork_summary.json').is_file()
print(json.dumps(dict(source_intervals=len(rows),initializers=2,v7_truth_available=True)))
'''.replace("__SOURCE__", repr(str(infra.RESIDENT_SOURCE))).replace(
        "__INPUTS__", repr(str(infra.RESIDENT_INPUTS))).replace("__V7__", repr(str(infra.RESIDENT_V7)))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def submit_preflight():
    source_ready()
    p = protocol()["preflight"]
    return infra.submit([make_job(p["seed"], p["scenario"], p["target_line"], preflight=True)], "preflight")


def submit_campaign():
    path = HERE / "preflight/gate.json"
    value = json.loads(path.read_text()) if path.is_file() else infra.collect_json(
        "preflight/gate.json", max_bytes=12000)
    if value.get("status") != "complete" or value.get("integrity", {}).get("all_pass") is not True:
        raise RuntimeError("Native v8 preflight must complete with integrity.all_pass=true")
    return infra.submit([make_job(**row) for row in cells()], "diagnostic")


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
        value = infra.collect_json(RESULT_NAME)
        result = dict(status=value.get("status"), bytes=(HERE / RESULT_NAME).stat().st_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
