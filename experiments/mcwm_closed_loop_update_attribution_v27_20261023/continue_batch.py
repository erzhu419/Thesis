"""Reuse first anchor preflight within a full day, then parallelize three days."""
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


def resident(code):
    rc,out,error=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=False)
    if rc:raise RuntimeError(error[-1800:])
    return json.loads(out)


def wait_preflight():
    p=launch.protocol();cell=p['preflight_cell']
    failure=HERE/'runs'/f"seed{cell['seed']}"/cell['scenario']/'failure.json'
    code=('from pathlib import Path;import json;gate=Path('+repr(infra.remote(HERE/'preflight_gate.json'))+');'
        'failure=Path('+repr(infra.remote(failure))+');'
        'print(gate.read_text() if gate.is_file() else json.dumps({"status":"failed","failure":json.loads(failure.read_text())}) if failure.is_file() else "null")')
    log('waiting',file='preflight_gate.json')
    while True:
        value=resident(code)
        if value is not None:
            infra.write(HERE/'preflight_gate.json' if value['status']=='complete' else HERE/'preflight_failure.json',value)
            return value
        time.sleep(30)


def wait_days():
    p=launch.protocol()
    paths={f'{seed}/{scenario}':infra.remote(HERE/'runs'/f'seed{seed}'/scenario)
        for seed in p['training_seeds'] for scenario in p['scenarios']}
    code=('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps({k:json.loads((Path(p)/"summary.json").read_text())["status"] if (Path(p)/"summary.json").is_file() '
        'else "failed" if (Path(p)/"failure.json").is_file() else None for k,p in paths.items()}))')
    log('waiting',stage_files='four replay summaries')
    while True:
        value=resident(code)
        if all(v is not None for v in value.values()):
            infra.write(HERE/'replay_completion.json',value)
            return value
        time.sleep(30)


def wait_summary():
    path=HERE/'shadow_summary.json'
    code=('from pathlib import Path;p=Path('+repr(infra.remote(path))+');'
        'assert not p.is_file() or p.stat().st_size<=60000;print(p.read_text() if p.is_file() else "null")')
    log('waiting',file=path.name)
    while True:
        value=resident(code)
        if value is not None:
            infra.write(path,value)
            return value
        time.sleep(30)


def main():
    submitted('preflight',lambda:launch.submit_replays(True))
    gate=wait_preflight()
    if gate.get('integrity',{}).get('all_pass') is not True:
        result=dict(status='stopped_preflight_integrity',real_control_branches=0)
    else:
        submitted('remaining',lambda:launch.submit_replays(False))
        wait_days()
        submitted('finalize',launch.submit_finalize)
        gate=wait_summary()
        result=dict(status='diagnosis_complete' if gate.get('integrity',{}).get('all_pass') else 'stopped_integrity',
            real_control_branches=0,native_replay_seconds=96000,main_updates=1824,shadow_updates=256)
    result['priority']='high'
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        value=dict(status='failed',type=type(error).__name__,message=str(error))
        infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
