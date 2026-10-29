"""Finish the fixed two native observations and one readout, then exit."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.supervisor import (
    read_only, ReadUnavailable)
from experiments.mcwm_closed_loop_native_context_v79_20261027 import infra


def save(value):
    raw = json.dumps(dict(updated_unix=time.time(), **value), ensure_ascii=False, indent=2)+'\n'
    temporary = HERE/'watch_state.pending'
    temporary.write_text(raw)
    temporary.replace(HERE/'watch_state.json')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def present(paths):
    code = ('from pathlib import Path;import json;print(json.dumps({p:Path(p).exists() for p in '
        +repr([infra.remote(HERE/p) for p in paths])+'}))')
    import shlex
    raw = infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
        timeout=45, check=True)[1]
    values = json.loads(raw)
    return {path: values[infra.remote(HERE/path)] for path in paths}


def advance():
    ready = HERE/'execution_ready.json'
    if not ready.exists() or json.loads(ready.read_text()).get('ready') is not True:
        return dict(stage='preparing', done=False)
    rows = [infra.task(f'seed{seed}', 'run.py', ['--seed', seed, '--out',
        infra.remote(HERE/'runs'/f'seed{seed}')]) for seed in (6208,6207)]
    receipt = infra.submit(rows, 'native')
    native = read_only(infra.status, 'native')
    if len(native) != 2:
        raise RuntimeError('Native receipt requires the two fixed tasks; no substitutes')
    if any(row['status'] not in ('done','failed','cancelled') for row in native):
        return dict(stage='native_observation', done=False, native_tasks=native)
    paths = [f'runs/seed{seed}/{name}.json' for seed in (6208,6207)
             for name in ('summary','failure')]
    found = read_only(present, paths)
    for path in paths:
        if found[path]:
            read_only(infra.collect, path)
    rows = [infra.task('summary', 'analyze.py', ['--out',
        infra.remote(HERE/'results'/'summary.json')])]
    summary_receipt = infra.submit(rows, 'summary')
    summary_tasks = read_only(infra.status, 'summary')
    if len(summary_tasks) != 1:
        raise RuntimeError('A single fixed summary task is required')
    task = summary_tasks[0]
    if task['status'] not in ('done','failed','cancelled'):
        return dict(stage='summary', done=False, native_tasks=native, summary_task=task)
    if task['status'] != 'done':
        return dict(stage='summary_attention', done=True, native_tasks=native, summary_task=task)
    result = read_only(infra.collect, 'results/summary.json')
    return dict(stage='complete', done=True, native_tasks=native, summary_task=task,
        result='results/summary.json', native_coverage=result.get('decision'),
        collection_started=False, response_training_started=False)


def main():
    with (HERE/'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing v79 watcher owns these receipts', flush=True)
            return
        (HERE/'watch_process.json').write_text(json.dumps(dict(pid=os.getpid(), started_unix=time.time())))
        while True:
            try:
                value = advance()
            except ReadUnavailable as error:
                value = dict(stage='transient_read_wait', done=False, message=str(error))
            except Exception as error:
                save(dict(stage='attention', done=True, message=str(error)))
                raise
            save(value)
            if value['done']:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
