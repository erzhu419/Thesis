"""Wait for both v39 prechecks, including failures; collect one small summary."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values)), flush=True)


def resident(code):
    while True:
        try:
            rc, out, error = infra.scheduler.run_on('node001',
                shlex.join([infra.PYTHON, '-c', code]), timeout=45, check=False)
        except subprocess.TimeoutExpired:
            log('transport_retry', reason='SSH read timeout')
            time.sleep(30)
            continue
        if rc == 0:
            return json.loads(out)
        if any(text in error for text in ('kex_exchange_identification', 'Connection closed',
                'Connection timed out', 'Connection reset')):
            log('transport_retry', reason=error[-600:])
            time.sleep(30)
            continue
        raise RuntimeError(error[-1800:])


def submitted(batch, callback):
    path = HERE/(batch+'_submission.json')
    value = json.loads(path.read_text()) if path.exists() else callback()
    log('submitted', batch=batch, ids=[row['id'] for row in value['submitted']], priority='high')


def wait_days():
    paths = {scenario: infra.remote(HERE/'runs'/scenario) for scenario in launch.protocol()['scenarios']}
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps({k:json.loads((Path(p)/"summary.json").read_text())["status"] '
        'if (Path(p)/"summary.json").exists() else "failed" if (Path(p)/"failure.json").exists() '
        'else None for k,p in paths.items()}))')
    previous = None
    while True:
        states = resident(code)
        if states != previous:
            log('pipelines', states=states)
            previous = states
        if all(value is not None for value in states.values()):
            infra.write(HERE/'precheck_completion.json', states)
            return states
        time.sleep(45)


def wait_summary():
    path = infra.remote(HERE/'analysis')
    limit = launch.protocol()['summary_max_bytes']
    code = ('from pathlib import Path;import json;p=Path('+repr(path)+');'
        's=p/"summary.json";f=p/"failure.json";'
        'assert not s.exists() or s.stat().st_size<='+str(limit)+';'
        'print(s.read_text() if s.exists() else json.dumps({"status":"analysis_failed",'
        '"failure":json.loads(f.read_text())}) if f.exists() else "null")')
    while True:
        result = resident(code)
        if result is not None:
            name = 'analysis/failure.json' if result['status'] == 'analysis_failed' else 'analysis/summary.json'
            infra.write(HERE/name, result)
            return result
        time.sleep(30)


def main():
    submitted('precheck', launch.submit_days)
    states = wait_days()
    # A failed case must not suppress the other case or erase its consumed budget.
    submitted('finalize', launch.submit_finalize)
    summary = wait_summary()
    result = dict(status='complete_precheck' if summary.get('integrity', {}).get('all_pass')
        else 'partial_precheck', pipeline_states=states, summary_status=summary['status'],
        priority='high', new_training_slots=0, new_optimizer_updates=0,
        decision=summary.get('decision'))
    infra.write(HERE/'continuation_result.json', result)
    log('finished', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = dict(status='failed', type=type(error).__name__, message=str(error))
        infra.write(HERE/'continuation_failure.json', value)
        log('failed', **value)
        raise
