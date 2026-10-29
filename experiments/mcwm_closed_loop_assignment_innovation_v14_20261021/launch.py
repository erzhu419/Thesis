"""Stage v14, gate native day/pair preflights, and submit the small paired pilot."""

import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
RESULT_NAME = "innovation_summary.json"
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


def pair_cells(p=None):
    p = protocol() if p is None else p
    seed = int(p.get("paired_seed", 6201))
    return [dict(seed=seed, scenario=scenario, target_line=line,
                 anchor_index=anchor_index, branch=branch)
            for scenario, line, anchor_index, branch in product(
                p["scenarios"], p["target_lines"], (0, 1), ("low", "high"))]


def cell_path(cell):
    return Path(cell["scenario"]) / f"seed{cell['seed']}"


def pair_path(cell):
    return (Path(cell["scenario"]) / cell["target_line"] / f"seed{cell['seed']}" /
            f"state{cell['anchor_index']}" / cell["branch"])


def make_job(seed, scenario, *, mode="day", preflight=False, target_line=None, branch=None,
             anchor_index=0):
    p = protocol()
    cell = dict(seed=seed, scenario=scenario, target_line=target_line,
                anchor_index=anchor_index, branch=branch)
    if mode == "pair":
        if target_line is None or branch not in ("low", "high") or anchor_index not in (0, 1):
            raise ValueError("A native branch requires a target line, state 0/1 and low/high action")
        directory = preflight_directory(p) / "pair" / branch if preflight else Path("pairs") / pair_path(cell)
    else:
        directory = preflight_directory(p) / "day" if preflight else Path("runs") / cell_path(cell)
    argv = ["--mode", mode, "--seed", str(seed), "--scenario", scenario,
            "--out", infra.remote(HERE / directory)]
    if mode == "pair":
        argv.extend(["--target-line", target_line, "--anchor-index", str(anchor_index),
                     "--branch", branch])
    if preflight:
        argv.append("--preflight")
    return infra.task(p, str(directory), argv)


def source_ready():
    """Stat directly reused v12 evidence in place; copy no historical results."""
    p = protocol()
    reference = HERE / p.get("reference_v13_root", "../mcwm_closed_loop_separated_contrast_v13_20261021")
    pref = Path(p.get("reference_v13_preflight_directory", "preflight"))
    snapshots = [Path("runs") / cell_path(row) for row in cells(p)]
    snapshots.append(pref / "day")
    truths = [Path("pairs") / pair_path(row) / "summary.json" for row in pair_cells(p)]
    truths.extend(pref / "pair" / branch / "summary.json" for branch in ("low", "high"))
    references = [dict(version="v13", root=infra.remote(reference.resolve()),
                       snapshots=[str(path) for path in snapshots],
                       truth_paths=[str(path) for path in truths])]
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
for directory, filename in __IMPORTS__:
 assert (Path(directory)/filename).is_file()
counts={}
for reference in __REFERENCES__:
 root=Path(reference['root'])
 for relative in reference['snapshots']:
  for filename in ('summary.json','prefix_trace.jsonl'):
   assert (root/relative/filename).is_file(), str(root/relative/filename)
 for relative in reference['truth_paths']:
  assert (root/relative).is_file(), str(root/relative)
 counts[reference['version']]=dict(snapshot_files=2*len(reference['snapshots']),
  paired_truth_files=len(reference['truth_paths']))
print(json.dumps(dict(source_intervals=len(rows),initializers=2,resident_code_available=True,
 references=counts)))
'''.replace("__SOURCE__", repr(str(infra.RESIDENT_SOURCE))).replace(
        "__INPUTS__", repr(str(infra.RESIDENT_INPUTS))).replace("__IMPORTS__", repr([
            (str(infra.RESIDENT_V7), "bootstrap.py"),
            (str(infra.RESIDENT_V8), "response_world.py"),
            (str(infra.RESIDENT_V9), "ablation_world.py"),
            (str(infra.RESIDENT_V10), "endpoints.py"),
            *[(str(infra.RESIDENT_V11), filename) for filename in
              ("residual.py", "paired_stream.py", "coverage.py", "prefix_trace.py", "run.py")],
            *[(str(infra.RESIDENT_V12), filename) for filename in
              ("bootstrap.py", "interaction.py", "diagnostics.py", "run.py")],
            *[(str(infra.RESIDENT_V13), filename) for filename in ("separated.py", "diagnostics.py")],
        ])).replace("__REFERENCES__", repr(references))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def require_gate(relative):
    path = HERE / relative
    value = json.loads(path.read_text()) if path.is_file() else infra.collect_json(
        str(relative), max_bytes=PREFLIGHT_MAX_BYTES)
    if value.get("status") != "complete" or value.get("integrity", {}).get("all_pass") is not True:
        raise RuntimeError(f"Native v14 gate {relative} must complete with integrity.all_pass=true")
    return value


def submit_preflight():
    source_ready()
    p = protocol()
    settings = p["preflight"]
    return infra.submit([make_job(settings["seed"], settings["scenario"], preflight=True)],
                        preflight_batch(p))


def submit_preflight_pairs():
    p = protocol()
    require_gate(preflight_directory(p) / "day" / "gate.json")
    settings = p["preflight"]
    tasks = [make_job(settings["seed"], settings["scenario"], mode="pair", preflight=True,
                      target_line=settings.get("target_line", "7X"), branch=branch)
             for branch in ("low", "high")]
    return infra.submit(tasks, preflight_batch(p) + "_pairs")


def submit_preflight_finalizer():
    p = protocol()
    json.loads((HERE / (preflight_batch(p) + "_pairs_submission.json")).read_text())
    root = HERE / preflight_directory(p)
    argv = ["--preflight", "--root", infra.remote(root), "--out", infra.remote(root / "gate.json"), "--wait"]
    return infra.submit([infra.task(p, str(preflight_directory(p) / "finalize"), argv,
                                   runner="analyze.py", ram_mb=1024)],
                        preflight_batch(p) + "_finalize")


def submit_campaign():
    require_gate(preflight_directory() / "gate.json")
    return infra.submit([make_job(**row) for row in cells()], "diagnostic")


def days_status():
    paths = [str(Path("runs") / cell_path(row)) for row in cells()]
    script = '''import json
from pathlib import Path
p=Path(__HERE__)
out=[]
for relative in __PATHS__:
 q=p/relative
 f=q/'summary.json'
 v=json.loads(f.read_text()) if f.is_file() else {}
 out.append(dict(path=relative,present=f.is_file(),status=v.get('status'),
  all_pass=v.get('integrity',{}).get('all_pass'),failed=(q/'failure.json').is_file()))
print(json.dumps(out))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace("__PATHS__", repr(paths))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def submit_pairs():
    values = days_status()
    if len(values) != len(cells()) or not all(row["status"] == "complete" and row["all_pass"] is True
                                            and not row["failed"] for row in values):
        raise RuntimeError("All four new v14 day summaries must pass integrity before native paired replay")
    return infra.submit([make_job(mode="pair", **row) for row in pair_cells()], "paired")


def submit_finalizer():
    for batch in ("diagnostic", "paired"):
        json.loads((HERE / (batch + "_submission.json")).read_text())
    argv = ["--root", infra.remote(HERE), "--out", infra.remote(HERE / RESULT_NAME), "--wait"]
    return infra.submit([infra.task(protocol(), "diagnostic/finalize", argv,
                                   runner="analyze.py", ram_mb=1024)], "diagnostic_finalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "preflight", "preflight-pairs", "preflight-finalize",
                                              "submit", "pairs", "finalize", "status", "collect"))
    parser.add_argument("--batch", default="diagnostic")
    args = parser.parse_args()
    if args.operation == "stage":
        files = sorted(path for path in HERE.glob("*.py") if not path.name.startswith("test_"))
        result = infra.stage([*files, HERE / "protocol.json"])
    elif args.operation == "preflight":
        result = submit_preflight()
    elif args.operation == "preflight-pairs":
        result = submit_preflight_pairs()
    elif args.operation == "preflight-finalize":
        result = submit_preflight_finalizer()
    elif args.operation == "submit":
        result = submit_campaign()
    elif args.operation == "pairs":
        result = submit_pairs()
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
