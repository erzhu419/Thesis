"""Reuse diagnostic receipts, await outputs, then collect one small summary."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def await_outputs(groups, ids):
    paths = [[infra.remote(HERE/name) for name in group] for group in groups]
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
            'print(json.dumps([[p for p in group if Path(p).exists()] for group in paths]))')
    while True:
        found = json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
                                                timeout=45, check=True)[1])
        if all(found):
            return
        result = subprocess.run([sys.executable, str(infra.SCHEDULER), 'status', '--readonly',
            '--json', '--lock-timeout', '30', '--ids', ','.join(ids)], text=True, capture_output=True)
        if result.returncode:
            if 'waiting for scheduler state lock' not in result.stderr:
                raise RuntimeError(result.stderr.strip())
        elif any(row['status'] in ('failed', 'cancelled') for row in json.loads(result.stdout)['tasks']):
            raise RuntimeError('A diagnostic task ended before writing all declared outputs')
        time.sleep(30)


def main():
    receipt = HERE/'diagnostic_submission.json'
    if not receipt.exists():
        launch.stage()
        launch.cases()
    ids = [row['id'] for row in json.loads(receipt.read_text())['submitted']]
    await_outputs([[f'runs/{case}/summary.json', f'runs/{case}/failure.json']
                   for case in launch.protocol()['scenarios']], ids)
    path = HERE/'finalize_submission.json'
    result = json.loads(path.read_text()) if path.exists() else launch.finalize()
    await_outputs([['analysis/summary.json']], [row['id'] for row in result['submitted']])
    summary = infra.collect('analysis/summary.json', max_bytes=launch.protocol()['summary_max_bytes'])
    value = dict(status=summary['status'], integrity=summary['integrity'],
        decision=summary['decision'], new_training_updates=0, new_native_branches=0)
    infra.write(HERE/'continuation_result.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        infra.write(HERE/'continuation_failure.json', dict(type=type(error).__name__, message=str(error),
            new_training_updates=0, new_native_branches=0))
        raise
