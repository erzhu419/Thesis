"""Advance the bounded execution probes only after each exact-equivalence gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time
import infra
import launch

HERE=Path(__file__).resolve().parent


def log(stage,**values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,**values)),flush=True)


def wait_file(name):
    log('waiting',file=name)
    target=HERE/name
    while True:
        if target.is_file():return json.loads(target.read_text())
        code='from pathlib import Path;import json;p=Path('+repr(infra.remote(target))+');assert not p.is_file() or p.stat().st_size<=20000;print(p.read_text() if p.is_file() else "null")'
        text=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=True)[1]
        value=json.loads(text)
        if value is not None:
            infra.write(target,value)
            return value
        time.sleep(30)


def submitted(batch,callback):
    receipt=HERE/(batch+'_submission.json')
    value=json.loads(receipt.read_text()) if receipt.is_file() else callback()
    log('submitted',batch=batch,ids=[r['id'] for r in value['submitted']],priority='high')


def main():
    if (HERE/'continuation_result.json').is_file():return
    native=wait_file(launch.NATIVE_RESULT)
    if native.get('all_pass') is not True:
        result=dict(status='stopped_native_gate',stage_b='closed')
    else:
        submitted('bus_probe',launch.submit_bus)
        submitted('equivalence_finalize',launch.submit_compare)
        gate=wait_file('equivalence_gate.json')
        if gate.get('integrity',{}).get('all_pass') is not True:
            result=dict(status='stopped_bus_equivalence',integrity=gate.get('integrity'),stage_b='closed')
        else:
            submitted('day_benchmark',launch.submit_day)
            submitted('day_equivalence_finalize',launch.submit_day_compare)
            gate=wait_file('day_equivalence_gate.json')
            result=dict(status='complete' if gate.get('integrity',{}).get('all_pass') else 'stopped_day_equivalence',
                integrity=gate.get('integrity'),decision=gate.get('decision'),stage_b='closed')
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        value=dict(status='failed',type=type(error).__name__,message=str(error))
        if not (HERE/'continuation_failure.json').exists():infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
