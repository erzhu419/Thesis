"""Continue only this diagnostic batch after its native preflight succeeds."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch
from analysis import result_name

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
                     ensure_ascii=False, allow_nan=False), flush=True)


def metadata():
    p = launch.protocol()
    script = '''import json
from pathlib import Path
p=Path(__HERE__)
q=p/__PREFLIGHT_DIRECTORY__/'control/local_speed_drop/action_model/seed6201'
f=p/__RESULT_NAME__
out=dict(preflight_started=q.is_dir(),preflight_complete=(q/'summary.json').is_file(),
 preflight_failed=(q/'failure.json').is_file(),
 failures=[str(f.relative_to(p)) for f in (p/'runs/control').glob('*/*/seed*/failure.json')],
 result_exists=f.is_file())
if f.is_file():
 v=json.loads(f.read_text());out['result']=dict(status=v.get('status'),bytes=f.stat().st_size,
  complete=v.get('integrity',{}).get('complete_runs'),errors=len(v.get('integrity',{}).get('errors',[])))
print(json.dumps(out))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace(
        "__PREFLIGHT_DIRECTORY__", repr(p["preflight_directory"])).replace(
        "__RESULT_NAME__", repr(result_name()))
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
        raise RuntimeError("Submit the native preflight first")
    previous = None
    if not (HERE / "diagnostic_submission.json").exists():
        while True:
            value = poll_metadata()
            if value != previous:
                log("waiting_native_preflight", **value)
                previous = value
            if value["preflight_failed"]:
                raise RuntimeError("Native preflight failed; inspect resident failure record")
            if value["preflight_complete"]:
                result = launch.submit_campaign()
                log("diagnostic_submitted", ids=[row["id"] for row in result["submitted"]])
                break
            time.sleep(60)
    if not (HERE / "diagnostic_finalize_submission.json").exists():
        result = launch.submit_finalizer()
        log("finalizer_submitted", ids=[row["id"] for row in result["submitted"]])
    while True:
        value = poll_metadata()
        if value["failures"]:
            raise RuntimeError("Diagnosis failed: " + ", ".join(value["failures"]))
        if value["result_exists"]:
            result = value["result"]
            log("resident_diagnosis", **result)
            if result["status"] != "complete" or result["complete"] != 4 or result["errors"]:
                raise RuntimeError("Diagnosis integrity incomplete; retain resident evidence")
            if result["bytes"] > 500000:
                raise RuntimeError("Reduce the resident compact summary before collection")
            if not (HERE / result_name()).exists():
                infra.collect_json(result_name())
            log("diagnostic_complete", collected_bytes=(HERE / result_name()).stat().st_size,
                next_action="Inspect paired world/trust/shadow evidence before any further experiment")
            return
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("stopped", error_type=type(error).__name__, error=str(error))
        raise
