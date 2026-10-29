"""Continue native preflight -> four controlled calibration days -> one small summary."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
                     ensure_ascii=False, allow_nan=False), flush=True)


def metadata():
    paths = [str(Path("runs") / launch.cell_path(row)) for row in launch.cells()]
    script = '''import json
from pathlib import Path
p=Path(__HERE__)
runs=[p/q for q in __RUN_PATHS__]
f=p/__RESULT_NAME__
preflight=p/__PREFLIGHT_DIR__
out=dict(preflight_complete=(preflight/'gate.json').is_file(),
 preflight_failed=(preflight/'failure.json').is_file(),
 days_complete=sum((q/'summary.json').is_file() for q in runs),
 failures=[str(q.relative_to(p)) for q in runs if (q/'failure.json').is_file()],
 result_exists=f.is_file())
if f.is_file():
 v=json.loads(f.read_text());out['result']=dict(status=v.get('status'),bytes=f.stat().st_size,
  integrity=v.get('integrity',{}))
print(json.dumps(out))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace("__RUN_PATHS__", repr(paths)).replace(
        "__RESULT_NAME__", repr(launch.RESULT_NAME)).replace(
        "__PREFLIGHT_DIR__", repr(str(launch.preflight_directory())))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def main():
    log("continuation_started", priority="high", development_days=len(launch.cells()))
    if not (HERE / (launch.preflight_batch() + "_submission.json")).is_file():
        raise RuntimeError("Submit the single native preflight first")
    previous = None
    if not (HERE / "diagnostic_submission.json").exists():
        while True:
            value = metadata()
            if value != previous:
                log("waiting_native_preflight", **value)
                previous = value
            if value["preflight_failed"]:
                raise RuntimeError("Native preflight failed; retain the resident failure record")
            if value["preflight_complete"]:
                summary = infra.collect_json(str(launch.preflight_directory() / "gate.json"),
                                             max_bytes=launch.PREFLIGHT_MAX_BYTES)
                if summary.get("status") != "complete" or summary.get("integrity", {}).get("all_pass") is not True:
                    raise RuntimeError("Native preflight failed its status/integrity gate")
                receipt = launch.submit_campaign()
                log("diagnostic_submitted", ids=[row["id"] for row in receipt["submitted"]], priority="high")
                break
            time.sleep(60)
    if not (HERE / "diagnostic_finalize_submission.json").exists():
        receipt = launch.submit_finalizer()
        log("finalizer_submitted", ids=[row["id"] for row in receipt["submitted"]], priority="high")
    while True:
        value = metadata()
        if value != previous:
            log("waiting_controlled_calibration_days", **value)
            previous = value
        if value["result_exists"]:
            result = value["result"]
            if result["bytes"] > launch.SUMMARY_MAX_BYTES:
                raise RuntimeError("Reduce the resident summary before collection")
            if not (HERE / launch.RESULT_NAME).exists():
                infra.collect_json(launch.RESULT_NAME, max_bytes=launch.SUMMARY_MAX_BYTES)
            log("diagnostic_finished", **result, collected_bytes=(HERE / launch.RESULT_NAME).stat().st_size,
                next_action="Review controlled action coverage and endpoint predictions before independent confirmation")
            return
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("stopped", error_type=type(error).__name__, error=str(error))
        raise
