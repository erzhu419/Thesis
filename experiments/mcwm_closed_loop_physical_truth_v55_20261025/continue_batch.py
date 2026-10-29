"""Resume from scheduler receipts without repeating prefixes or suffixes."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def state(stage, **fields):
    value = dict(stage=stage, **fields)
    (HERE/'continuation_state.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


def ids(receipt):
    return [row['id'] for row in receipt['submitted']]


def await_outputs(groups, task_ids, stage):
    paths = [[infra.remote(HERE/name) for name in group] for group in groups]
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps([[p for p in group if Path(p).exists()] for group in paths]))')
    deadline = time.monotonic()+launch.protocol().get('continuation_timeout_seconds', 86400)
    previous = None
    while time.monotonic() < deadline:
        found = json.loads(infra.scheduler.run_on('node001',
            shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=True)[1])
        if all(found):
            return found
        statuses = {row['id']: row['status'] for row in infra.status(task_ids)}
        # A worker may finish between output and status snapshots. One fresh
        # read distinguishes that completion from a terminal missing output.
        if any(not outputs and statuses[task_id] in ('done', 'failed', 'cancelled')
               for task_id, outputs in zip(task_ids, found)):
            found = json.loads(infra.scheduler.run_on('node001',
                shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=True)[1])
            if all(found):
                return found
        for task_id, outputs in zip(task_ids, found):
            if not outputs and statuses[task_id] in ('done', 'failed', 'cancelled'):
                raise RuntimeError('Terminal task without declared output: '+task_id)
        current = dict(ids=task_ids, outputs_present=[bool(row) for row in found], statuses=statuses)
        if current != previous:
            state(stage, **current)
            previous = current
        time.sleep(30)
    raise TimeoutError('Bounded continuation wait expired: '+stage)


def main():
    p = launch.protocol()
    native = launch.native()
    cases = p['scenarios']
    found = await_outputs([['runs/'+case+'/summary.json', 'runs/'+case+'/failure.json']
        for case in cases], ids(native), 'waiting_native_days')
    for case, outputs in zip(cases, found):
        if any(path.endswith('/failure.json') for path in outputs):
            failure = infra.collect('runs/'+case+'/failure.json')
            raise RuntimeError(case+' stopped: '+failure.get('message', str(failure)))
        summary = infra.collect('runs/'+case+'/summary.json')
        if summary['status'] != 'complete':
            raise RuntimeError(case+' summary is not complete')
    final = launch.finalize()
    found = await_outputs([['analysis/summary.json', 'analysis/failure.json']],
        ids(final), 'waiting_small_summary')
    if any(path.endswith('/failure.json') for path in found[0]):
        failure = infra.collect('analysis/failure.json')
        raise RuntimeError('Analysis failed: '+failure.get('message', str(failure)))
    summary = infra.collect('analysis/summary.json')
    result = dict(status=summary['status'], integrity=summary.get('integrity'),
        native_task_ids=ids(native), finalize_task_ids=ids(final),
        new_feedback_slots=0, new_training_updates=0, native_prefixes=2,
        diagnostic_suffix_budget=16, diagnostic_suffix_seconds=300)
    if not (HERE/'continuation_result.json').exists():
        infra.write(HERE/'continuation_result.json', result)
    state('collected', **result)


if __name__ == '__main__':
    with (HERE/'continuation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing continuation is alive; reuse its scheduler receipts.', flush=True)
            raise SystemExit(0)
        (HERE/'continuation_process.json').write_text(json.dumps(
            dict(pid=os.getpid(), started_unix=time.time()), indent=2)+'\n')
        try:
            main()
        except Exception as error:
            failure = dict(status='failed', type=type(error).__name__, message=str(error),
                new_feedback_slots=0, new_training_updates=0)
            if not (HERE/'continuation_failure.json').exists():
                infra.write(HERE/'continuation_failure.json', failure)
            state('failed', **failure)
            raise
