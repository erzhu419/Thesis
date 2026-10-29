"""Receipt-based continuation; stop on failure without repeating any updates."""
import json
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
        # A worker can write its receipt and become terminal between these two
        # snapshots. Read paths once more before diagnosing missing output.
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


def collect_workers(cells, found, maximum):
    for (case, arm, prefix), outputs in zip(cells, found):
        if any(path.endswith('/failure.json') for path in outputs):
            failure = infra.collect(prefix+'/failure.json', max_bytes=maximum)
            raise RuntimeError(case+'/'+arm+' stopped: '+failure.get('message', str(failure)))
        summary = infra.collect(prefix+'/summary.json', max_bytes=maximum)
        if summary['status'] != 'complete':
            raise RuntimeError(case+'/'+arm+' summary is not complete')


def main():
    p = launch.protocol()
    maximum = p.get('summary_max_bytes_per_worker', 64000)
    preflight = launch.preflight()
    cells = [(case, 'one_step', 'preflight/'+case) for case in p['scenarios']]
    found = await_outputs([[prefix+'/summary.json', prefix+'/failure.json']
        for _, _, prefix in cells], ids(preflight), 'waiting_first8_parity')
    collect_workers(cells, found, maximum)
    formal = launch.formal()
    cells = [(case, arm, 'runs/'+case+'/'+arm)
        for case in p['scenarios'] for arm in ('one_step', 'trace')]
    found = await_outputs([[prefix+'/summary.json', prefix+'/failure.json']
        for _, _, prefix in cells], ids(formal), 'waiting_parallel_arms')
    collect_workers(cells, found, maximum)
    final = launch.finalize()
    found = await_outputs([['analysis/summary.json', 'analysis/failure.json']],
        ids(final), 'waiting_small_summary')
    if any(path.endswith('/failure.json') for path in found[0]):
        failure = infra.collect('analysis/failure.json', max_bytes=64000)
        raise RuntimeError('Analysis failed: '+failure.get('message', str(failure)))
    summary = infra.collect('analysis/summary.json', max_bytes=p.get('summary_max_bytes', 64000))
    result = dict(status=summary['status'], integrity=summary.get('integrity'),
        preflight_task_ids=ids(preflight), formal_task_ids=ids(formal),
        finalize_task_ids=ids(final), new_feedback_slots=0, new_native_branches=0,
        preflight_updates_reused_per_control_arm=8)
    if not (HERE/'continuation_result.json').exists():
        infra.write(HERE/'continuation_result.json', result)
    state('collected', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        failure = dict(status='failed', type=type(error).__name__, message=str(error),
            new_feedback_slots=0, new_native_branches=0)
        if not (HERE/'continuation_failure.json').exists():
            infra.write(HERE/'continuation_failure.json', failure)
        state('failed', **failure)
        raise
