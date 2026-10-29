"""Native day -> native pair preflight -> four days -> twenty-four branches -> compact summary."""

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
    day_paths = [str(Path("runs") / launch.cell_path(row)) for row in launch.cells()]
    pair_paths = [str(Path("pairs") / launch.pair_path(row)) for row in launch.pair_cells()]
    script = '''import json
from pathlib import Path
p=Path(__HERE__)
pref=p/__PREFLIGHT_DIR__
def states(paths):
 out=[]
 for relative in paths:
  q=p/relative
  f=q/'summary.json'
  v=json.loads(f.read_text()) if f.is_file() else {}
  out.append(dict(path=relative,present=f.is_file(),status=v.get('status'),
   all_pass=v.get('integrity',{}).get('all_pass'),failed=(q/'failure.json').is_file()))
 return out
days=states(__DAY_PATHS__)
pairs=states(__PAIR_PATHS__)
out=dict(preflight_day_complete=(pref/'day/gate.json').is_file(),
 preflight_day_failed=(pref/'day/failure.json').is_file(),
 preflight_pairs_complete=sum((pref/'pair'/branch/'summary.json').is_file() for branch in ('low','high')),
 preflight_pairs_failed=[branch for branch in ('low','high') if (pref/'pair'/branch/'failure.json').is_file()],
 preflight_complete=(pref/'gate.json').is_file(),
 days_complete=sum(row['present'] for row in days),
 days_passed=sum(row['status']=='complete' and row['all_pass'] is True and not row['failed'] for row in days),
 pairs_complete=sum(row['present'] for row in pairs),
 failures=[row['path'] for row in days+pairs if row['failed'] or
  (row['present'] and (row['status']=='failed' or row['all_pass'] is False))],
 result_exists=(p/__RESULT_NAME__).is_file(),
 analysis_failures=[str(f.relative_to(p)) for f in (p/'summary_failure.json',pref/'summary_failure.json') if f.is_file()])
if out['result_exists']:
 f=p/__RESULT_NAME__;v=json.loads(f.read_text())
 out['result']=dict(status=v.get('status'),bytes=f.stat().st_size,integrity=v.get('integrity',{}))
print(json.dumps(out))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace(
        "__PREFLIGHT_DIR__", repr(str(launch.preflight_directory()))).replace(
        "__DAY_PATHS__", repr(day_paths)).replace("__PAIR_PATHS__", repr(pair_paths)).replace(
        "__RESULT_NAME__", repr(launch.RESULT_NAME))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]),
                                        timeout=45, check=True)
    return json.loads(output)


def receipt_exists(batch):
    return (HERE / (batch + "_submission.json")).is_file()


def submitted(stage, receipt):
    log(stage, ids=[row["id"] for row in receipt["submitted"]], priority="high")


def wait_until(stage, predicate, *, allow_recorded_run_failures=False):
    previous = None
    while True:
        value = metadata()
        if value != previous:
            log(stage, **value)
            previous = value
        if value["analysis_failures"]:
            raise RuntimeError("Native summary failed: "+str(value["analysis_failures"]))
        if (value["preflight_day_failed"] or value["preflight_pairs_failed"]
                or (value["failures"] and not allow_recorded_run_failures)):
            raise RuntimeError("A native run failed; retain its resident summary/failure record and review it")
        if predicate(value):
            return value
        time.sleep(60)


def main():
    log("continuation_started", priority="high", development_days=len(launch.cells()),
        diagnostic_branches=len(launch.pair_cells()), independent_confirmation="not_submitted")
    pref = launch.preflight_batch()
    if not receipt_exists(pref):
        raise RuntimeError("Submit the native preflight day first")
    if not receipt_exists(pref + "_pairs"):
        wait_until("waiting_native_preflight_day", lambda value: value["preflight_day_complete"])
        submitted("preflight_pairs_submitted", launch.submit_preflight_pairs())
    if not receipt_exists(pref + "_finalize"):
        submitted("preflight_finalizer_submitted", launch.submit_preflight_finalizer())
    if not receipt_exists("diagnostic"):
        wait_until("waiting_native_paired_preflight", lambda value: value["preflight_complete"])
        launch.require_gate(launch.preflight_directory() / "gate.json")
        submitted("diagnostic_days_submitted", launch.submit_campaign())
    if not receipt_exists("paired"):
        value = wait_until("waiting_four_common_prefix_days",
                           lambda value: value["days_complete"] == len(launch.cells()))
        if value["days_passed"] != len(launch.cells()):
            raise RuntimeError("Every new day must pass integrity before paired replay")
        submitted("paired_branches_submitted", launch.submit_pairs())
    if not receipt_exists("diagnostic_finalize"):
        submitted("diagnostic_finalizer_submitted", launch.submit_finalizer())
    value = wait_until("waiting_native_pairs_and_summary", lambda value: value["result_exists"],
                       allow_recorded_run_failures=True)
    result = value["result"]
    if result["bytes"] > launch.SUMMARY_MAX_BYTES:
        raise RuntimeError("Reduce the resident summary before collection")
    if not (HERE / launch.RESULT_NAME).is_file():
        infra.collect_json(launch.RESULT_NAME, max_bytes=launch.SUMMARY_MAX_BYTES)
    log("diagnostic_finished", **result, collected_bytes=(HERE / launch.RESULT_NAME).stat().st_size,
        next_action="Review common-prefix paired truth before deciding independent confirmation")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("stopped", error_type=type(error).__name__, error=str(error))
        raise
