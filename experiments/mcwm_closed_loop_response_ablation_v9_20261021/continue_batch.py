"""Continue preflight -> 12 passive prefix ablations -> one small resident summary."""

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
f=p/'ablation_summary.json'
out=dict(preflight_complete=(p/'preflight/gate.json').is_file(),
 preflight_failed=(p/'preflight/failure.json').is_file(),
 states_complete=sum((q/'summary.json').is_file() for q in runs),
 failures=[str(q.relative_to(p)) for q in runs if (q/'failure.json').is_file()],
 result_exists=f.is_file())
if f.is_file():
 v=json.loads(f.read_text());out['result']=dict(status=v.get('status'),bytes=f.stat().st_size,
  integrity=v.get('integrity',{}))
print(json.dumps(out))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace("__RUN_PATHS__", repr(paths))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def main():
    log("continuation_started", priority="high", new_native_suffix_seconds=0)
    if not (HERE / "preflight_submission.json").is_file():
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
                summary = infra.collect_json("preflight/gate.json", max_bytes=12000)
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
            log("waiting_fixed_prefixes", **value)
            previous = value
        if value["result_exists"]:
            result = value["result"]
            if result["bytes"] > 150000:
                raise RuntimeError("Reduce the resident summary before collection")
            if not (HERE / launch.RESULT_NAME).exists():
                infra.collect_json(launch.RESULT_NAME)
            log("diagnostic_finished", **result, collected_bytes=(HERE / launch.RESULT_NAME).stat().st_size,
                next_action="Review separated response terms before independent confirmation or control tests")
            return
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("stopped", error_type=type(error).__name__, error=str(error))
        raise
