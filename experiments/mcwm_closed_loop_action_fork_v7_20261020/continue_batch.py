"""Run only native preflight -> 24 fixed forks -> small resident summary."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch
from analysis import branch_cells, cell_path, result_name

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
                     ensure_ascii=False, allow_nan=False), flush=True)


def metadata():
    p = launch.protocol()
    preflight = [str(Path(p["preflight_directory"]) / cell_path(row)) for row in branch_cells(p, preflight=True)]
    runs = [str(Path("runs") / cell_path(row)) for row in branch_cells(p)]
    script = '''import json
from pathlib import Path
p=Path(__HERE__)
pre=[p/q for q in __PREFLIGHT_PATHS__]
runs=[p/q for q in __RUN_PATHS__]
f=p/__RESULT_NAME__
out=dict(preflight_started=sum(q.is_dir() for q in pre),preflight_complete=sum((q/'summary.json').is_file() for q in pre),
 preflight_failures=[str(q.relative_to(p)) for q in pre if (q/'failure.json').is_file()],
 branches_started=sum(q.is_dir() for q in runs),branches_complete=sum((q/'summary.json').is_file() for q in runs),
 failures=[str(q.relative_to(p)) for q in runs if (q/'failure.json').is_file()],result_exists=f.is_file())
if f.is_file():
 v=json.loads(f.read_text());out['result']=dict(status=v.get('status'),bytes=f.stat().st_size,
  complete=v.get('integrity',{}).get('complete_pairs'),errors=len(v.get('integrity',{}).get('errors',[])))
print(json.dumps(out))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace("__PREFLIGHT_PATHS__", repr(preflight)).replace(
        "__RUN_PATHS__", repr(runs)).replace("__RESULT_NAME__", repr(result_name()))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def poll_metadata():
    while True:
        try:
            return metadata()
        except RuntimeError as error:
            if "cmd failed (rc=255)" not in str(error):
                raise
            log("metadata_connection_interrupted", error=str(error), retry_seconds=60)
            time.sleep(60)


def main():
    log("continuation_started", preflight_directory=launch.protocol()["preflight_directory"])
    receipt = HERE / (launch.protocol()["preflight_directory"] + "_submission.json")
    if not receipt.is_file():
        raise RuntimeError("Submit both native preflight branches first")
    previous = None
    if not (HERE / "diagnostic_submission.json").exists():
        while True:
            value = poll_metadata()
            if value != previous:
                log("waiting_native_preflight", **value)
                previous = value
            if value["preflight_failures"]:
                raise RuntimeError("Native preflight failed; retain its resident failure record")
            if value["preflight_complete"] == 2:
                result = launch.submit_campaign()
                log("diagnostic_submitted", ids=[row["id"] for row in result["submitted"]])
                break
            time.sleep(60)
    if not (HERE / "diagnostic_finalize_submission.json").exists():
        result = launch.submit_finalizer()
        log("finalizer_submitted", ids=[row["id"] for row in result["submitted"]])
    while True:
        value = poll_metadata()
        if value != previous:
            log("waiting_fixed_forks", **value)
            previous = value
        if value["result_exists"]:
            result = value["result"]
            if result["bytes"] > 150000:
                raise RuntimeError("Reduce the resident summary before collection")
            if not (HERE / result_name()).exists():
                infra.collect_json(result_name())
            log("diagnostic_finished", **result, collected_bytes=(HERE / result_name()).stat().st_size,
                next_action="Analyze action-benefit truth before changing the response model or trust")
            return
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("stopped", error_type=type(error).__name__, error=str(error))
        raise
