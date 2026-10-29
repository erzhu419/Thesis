"""Resume the original four-day pilot without repeating its stable days."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import time
import infra
import launch

HERE=Path(__file__).resolve().parent


def log(stage,**values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,**values)),flush=True)


def submitted(batch,callback):
    path=HERE/(batch+'_submission.json')
    value=json.loads(path.read_text()) if path.exists() else callback()
    log('submitted',batch=batch,ids=[row['id'] for row in value['submitted']],priority='high')


def wait(name):
    log('waiting',file=name)
    path=HERE/name
    while True:
        if path.exists():return json.loads(path.read_text())
        code=('from pathlib import Path;p=Path('+repr(infra.remote(path))+');'
            'assert not p.is_file() or p.stat().st_size<=30000;'
            'print(p.read_text() if p.is_file() else "null")')
        rc,output,error=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=False)
        if rc:raise RuntimeError(error[-1800:])
        value=json.loads(output)
        if value is not None:
            infra.write(path,value)
            return value
        time.sleep(30)


def main():
    if not (HERE/'stable_gate.json').exists():
        submitted('stable_finalize',lambda:launch.submit_finalize('stable'))
    gate=wait('stable_gate.json')
    if gate.get('integrity',{}).get('all_pass') is not True:
        result=dict(status='stopped_stable_integrity',new_native_days=0)
    else:
        infra.collect('clock_stable.json',max_bytes=10000)
        submitted('shock_pair',launch.submit_shock)
        submitted('pilot_finalize',lambda:launch.submit_finalize('pilot'))
        gate=wait('pilot_summary.json')
        if gate.get('integrity',{}).get('all_pass'):
            infra.collect('issue_audit_summary.json',max_bytes=30000)
            infra.collect('clock_pilot.json',max_bytes=10000)
        result=dict(status='pilot_complete' if gate.get('integrity',{}).get('all_pass')
            else 'stopped_pilot_integrity',new_native_days=2)
    result.update(reused_stable_days=2,priority='high',independent_confirmation='not_submitted')
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        value=dict(status='failed',type=type(error).__name__,message=str(error))
        infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
