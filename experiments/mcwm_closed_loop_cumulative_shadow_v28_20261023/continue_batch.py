"""Reuse the first full day as native preflight, then parallelize three days."""
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
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),stage=stage,**values)),flush=True)


def resident(code):
    while True:
        try:
            rc,out,error = infra.scheduler.run_on('node001',
                shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=False)
        except subprocess.TimeoutExpired:
            log('transport_retry',reason='SSH read timeout')
            time.sleep(30)
            continue
        if rc == 0:
            return json.loads(out)
        if any(value in error for value in ('kex_exchange_identification','Connection closed',
                'Connection timed out','Connection reset')):
            log('transport_retry',reason=error[-600:])
            time.sleep(30)
            continue
        raise RuntimeError(error[-1800:])


def submitted(batch, callback):
    path = HERE/(batch+'_submission.json')
    value = json.loads(path.read_text()) if path.exists() else callback()
    log('submitted',batch=batch,ids=[row['id'] for row in value['submitted']],priority='high')


def wait_preflight():
    cell = launch.protocol()['preflight_cell']
    failure = HERE/'runs'/f"seed{cell['seed']}"/cell['scenario']/'failure.json'
    code = ('from pathlib import Path;import json;gate=Path('+repr(infra.remote(HERE/'preflight_gate.json'))+');'
        'failure=Path('+repr(infra.remote(failure))+');'
        'print(gate.read_text() if gate.is_file() else json.dumps({"status":"failed",'
        '"failure":json.loads(failure.read_text())}) if failure.is_file() else "null")')
    while True:
        value = resident(code)
        if value is not None:
            infra.write(HERE/('preflight_gate.json' if value['status']=='complete' else 'preflight_failure.json'),value)
            return value
        time.sleep(45)


def wait_days():
    p = launch.protocol()
    paths = {f'{seed}/{scenario}':infra.remote(HERE/'runs'/f'seed{seed}'/scenario)
        for seed in p['training_seeds'] for scenario in p['scenarios']}
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps({k:json.loads((Path(p)/"summary.json").read_text())["status"] '
        'if (Path(p)/"summary.json").is_file() else "failed" if (Path(p)/"failure.json").is_file() '
        'else None for k,p in paths.items()}))')
    previous = None
    while True:
        value = resident(code)
        if value != previous:
            log('replays',states=value)
            previous = value
        if all(state is not None for state in value.values()):
            infra.write(HERE/'replay_completion.json',value)
            return value
        time.sleep(45)


def wait_summary():
    code = ('from pathlib import Path;import json;root=Path('+repr(infra.remote(HERE))+');'
        'p=root/"shadow_summary.json";f=root/"analysis_failure.json";'
        'assert not p.exists() or p.stat().st_size<=50000;'
        'print(p.read_text() if p.exists() else json.dumps({"status":"analysis_failed",'
        '"failure":json.loads(f.read_text())}) if f.exists() else "null")')
    while True:
        value = resident(code)
        if value is not None:
            name = 'analysis_failure.json' if value['status']=='analysis_failed' else 'shadow_summary.json'
            infra.write(HERE/name,value)
            return value
        time.sleep(30)


def main():
    submitted('preflight',lambda:launch.submit_replays(True))
    gate = wait_preflight()
    if not gate.get('integrity',{}).get('all_pass'):
        result = dict(status='stopped_preflight_integrity')
    else:
        submitted('remaining',lambda:launch.submit_replays(False))
        states = wait_days()
        if any(state!='complete' for state in states.values()):
            result = dict(status='stopped_native_integrity',states=states)
        else:
            submitted('finalize',launch.submit_finalize)
            summary = wait_summary()
            result = dict(status='diagnosis_complete' if summary.get('integrity',{}).get('all_pass')
                else 'stopped_analysis_integrity',summary_status=summary['status'])
    result.update(priority='high',real_control_branches=0,extra_target_labels=0,
        main_native_replay_seconds=96000,main_policy_updates=1824,diagnostic_optimizer_updates=1984)
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        value = dict(status='failed',type=type(error).__name__,message=str(error))
        infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
