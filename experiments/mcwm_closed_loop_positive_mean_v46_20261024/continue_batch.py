"""Await the four existing frozen shards; submit only their bounded analysis."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import infra
import launch


HERE = Path(__file__).resolve().parent
TERMINAL = {'done', 'failed', 'cancelled'}


def state(stage, **values):
    value = dict(stage=stage, **values)
    (HERE/'continuation_state.json').write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def lock_timeout(result):
    return 'waiting for scheduler state lock' in (result.stderr+result.stdout)


def task_states(ids):
    result = subprocess.run([sys.executable, str(infra.SCHEDULER), 'status',
        '--readonly', '--json', '--lock-timeout', '30', '--ids', ','.join(ids)],
        text=True, capture_output=True)
    if result.returncode:
        if lock_timeout(result):
            return None
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return {row['id']: row['status'] for row in json.loads(result.stdout)['tasks']}


def await_outputs(groups, ids, stage):
    if len(groups) != len(ids):
        raise ValueError('Each declared output group needs its own task ID')
    paths = [[infra.remote(HERE/name) for name in group] for group in groups]
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps([[p for p in group if Path(p).exists()] for group in paths]))')
    previous = None
    while True:
        found = json.loads(infra.scheduler.run_on('node001',
            shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=True)[1])
        if all(found):
            return found
        statuses = task_states(ids)
        if statuses is not None:
            missing = [dict(id=task_id, status=statuses[task_id], expected=group)
                for group, task_id, output in zip(groups, ids, found)
                if not output and statuses.get(task_id) in TERMINAL]
            if missing:
                raise RuntimeError('Terminal task has no declared output: '+json.dumps(missing))
        snapshot = dict(task_ids=ids, outputs_present=[bool(value) for value in found],
                        task_status=statuses)
        if snapshot != previous:
            state(stage, **snapshot)
            previous = snapshot
        time.sleep(30)


def finalize():
    """Reuse its receipt on restart; retry only a scheduler lock timeout."""
    receipt = HERE/'finalize_submission.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    jobs = HERE/'finalize_jobs.json'
    if not jobs.exists():
        infra.write(jobs, [infra.task(launch.protocol(), 'small-summary',
            runner='analyze.py', ram_mb=2000)])
    while True:
        result = subprocess.run([sys.executable, str(infra.SCHEDULER), 'submit-jsonl',
            '--file', str(jobs), '--trusted', '--json', '--lock-timeout', '30'],
            text=True, capture_output=True)
        if not result.returncode:
            value = json.loads(result.stdout)
            infra.write(receipt, value)
            return value
        if not lock_timeout(result):
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        state('waiting_finalize_scheduler_lock')
        time.sleep(30)


def main():
    receipt = HERE/'frozen_submission.json'
    if not receipt.exists():
        raise RuntimeError('Start the four frozen shards before this collector')
    submitted = json.loads(receipt.read_text())['submitted']
    p = launch.protocol()
    cells = [(case, checkpoint) for case in p['scenarios'] for checkpoint in p['checkpoints']]
    by_signature = {row['signature']: row['id'] for row in submitted}
    ids = [by_signature[p['name']+'/frozen/'+case+'/'+checkpoint]
           for case, checkpoint in cells]
    await_outputs([[f'runs/{case}/{checkpoint}/summary.json',
                    f'runs/{case}/{checkpoint}/failure.json']
                   for case, checkpoint in cells], ids, 'waiting_frozen_shards')
    result = finalize()
    final_ids = [row['id'] for row in result['submitted']]
    await_outputs([['analysis/summary.json']], final_ids, 'waiting_small_summary')
    summary = infra.collect('analysis/summary.json', max_bytes=p['summary_max_bytes'])
    value = dict(status=summary['status'], integrity=summary['integrity'],
        decision=summary['decision'], frozen_task_ids=ids, finalize_task_ids=final_ids,
        new_training_updates=0, new_feedback_slots=0, new_native_branches=0)
    path = HERE/'continuation_result.json'
    if not path.exists():
        infra.write(path, value)
    state('collected', **value)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = dict(status='failed', type=type(error).__name__, message=str(error),
            new_training_updates=0, new_feedback_slots=0, new_native_branches=0)
        path = HERE/'continuation_failure.json'
        if not path.exists():
            infra.write(path, value)
        state('failed', **value)
        raise
