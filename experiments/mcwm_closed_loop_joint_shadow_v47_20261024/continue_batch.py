"""Reuse receipts: native M1 gate, concurrent arms, then one small summary."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def state(stage, **fields):
    value = dict(stage=stage, **fields)
    (HERE/'continuation_state.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


def await_outputs(groups, ids, stage):
    paths = [[infra.remote(HERE/name) for name in group] for group in groups]
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps([[p for p in group if Path(p).exists()] for group in paths]))')
    previous = None
    while True:
        try:
            found = json.loads(infra.scheduler.run_on('node001',
                shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=True)[1])
        except RuntimeError as error:
            if 'Broken pipe' not in str(error):
                raise
            state(stage, ids=ids, transport='retry_after_SSH_broken_pipe')
            time.sleep(15)
            continue
        if all(found):
            return found
        process = subprocess.run([sys.executable, str(infra.SCHEDULER), 'status',
            '--readonly', '--json', '--lock-timeout', '30', '--ids', ','.join(ids)],
            text=True, capture_output=True)
        statuses = None
        if process.returncode:
            if 'waiting for scheduler state lock' not in process.stderr:
                raise RuntimeError(process.stderr.strip() or process.stdout.strip())
        else:
            statuses = {row['id']: row['status'] for row in json.loads(process.stdout)['tasks']}
            for task_id, outputs in zip(ids, found):
                if not outputs and statuses[task_id] in ('done', 'failed', 'cancelled'):
                    raise RuntimeError('Terminal task without declared output: '+task_id)
        current = dict(ids=ids, outputs_present=[bool(row) for row in found], statuses=statuses)
        if current != previous:
            state(stage, **current)
            previous = current
        time.sleep(30)


def ids(receipt):
    return [row['id'] for row in receipt['submitted']]


def main():
    p = launch.protocol()
    receipt = launch.preflight()
    found = await_outputs([['preflight/stable/M1/summary.json', 'preflight/stable/M1/failure.json']],
                          ids(receipt), 'waiting_native_replay_gate')
    if not any(path.endswith('/summary.json') for path in found[0]):
        failed = infra.collect('preflight/stable/M1/failure.json',
                               max_bytes=p['summary_max_bytes_per_worker'])
        raise RuntimeError('Original M1 replay stopped: '+failed['message'])
    gate = infra.collect('preflight/stable/M1/summary.json',
                         max_bytes=p['summary_max_bytes_per_worker'])
    if gate['status'] != 'complete' or not gate['proof']['all_pass']:
        raise RuntimeError('Original M1 replay gate failed; formal learning was not submitted')
    state('preflight_passed', preflight_task_ids=ids(receipt))
    formal = launch.formal()
    cells = [(case, arm) for case in p['scenarios'] for arm in p['arms']]
    await_outputs([[f'runs/{case}/{arm}/summary.json', f'runs/{case}/{arm}/failure.json']
                   for case, arm in cells], ids(formal), 'waiting_parallel_arms')
    final = launch.finalize()
    await_outputs([['analysis/summary.json']], ids(final), 'waiting_small_summary')
    summary = infra.collect('analysis/summary.json', max_bytes=p['summary_max_bytes'])
    result = dict(status=summary['status'], integrity=summary['integrity'],
        preflight_task_ids=ids(receipt), formal_task_ids=ids(formal),
        finalize_task_ids=ids(final), new_feedback_slots=0, new_native_branches=0)
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
