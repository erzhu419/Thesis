"""Dispatch four days together; collect only bounded server-side summaries."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time
import infra
import launch

HERE = Path(__file__).resolve().parent


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,**values)),flush=True)


def submitted(batch, callback):
    path = HERE/(batch+'_submission.json')
    value = json.loads(path.read_text()) if path.exists() else callback()
    log('submitted',batch=batch,ids=[row['id'] for row in value['submitted']],priority='high')


def resident(code):
    rc, output, error = infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=False)
    if rc: raise RuntimeError(error[-1800:])
    return json.loads(output)


def wait_pairs():
    p = launch.protocol()
    seed, = p['training_seeds']
    paths = {scenario:infra.remote(HERE/'runs'/scenario/f'seed{seed}'/'pair_summary.json') for scenario in p['scenarios']}
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps({k:json.loads(Path(p).read_text())["status"] if Path(p).is_file() else None for k,p in paths.items()}))')
    log('waiting',files=list(paths.values()))
    while True:
        value = resident(code)
        if all(status is not None for status in value.values()):
            infra.write(HERE/'pairs_completion.json',value)
            log('pairs_finished',statuses=value)
            return value
        time.sleep(30)


def wait_summary():
    path = HERE/'pilot_summary.json'
    code = ('from pathlib import Path;p=Path('+repr(infra.remote(path))+');'
        'assert not p.is_file() or p.stat().st_size<=30000;'
        'print(p.read_text() if p.is_file() else "null")')
    log('waiting',file=path.name)
    while True:
        if path.exists(): return json.loads(path.read_text())
        value = resident(code)
        if value is not None:
            infra.write(path,value)
            return value
        time.sleep(30)


def main():
    submitted('four_day_pairs',launch.submit_pairs)
    wait_pairs()  # The small finalizer reserves no node while native days run.
    submitted('pilot_finalize',launch.submit_finalize)
    gate = wait_summary()
    if gate.get('integrity',{}).get('all_pass'):
        infra.collect('issue_audit_summary.json',max_bytes=30000)
        infra.collect('clock_pilot.json',max_bytes=10000)
    result = dict(status='pilot_complete' if gate.get('integrity',{}).get('all_pass') else 'stopped_integrity',
        priority='high',seed=6202,new_native_days=4,reused_native_days=0,
        further_seeds='not_submitted',independent_confirmation='not_submitted')
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__=='__main__':
    try: main()
    except Exception as error:
        value = dict(status='failed',type=type(error).__name__,message=str(error))
        infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
