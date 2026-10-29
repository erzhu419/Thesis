"""Submit three-way shadow replays, then collect one small summary."""
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


def wait_days():
    p = launch.protocol()
    paths = {scenario:infra.remote(HERE/'runs'/scenario) for scenario in p['scenarios']}
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps({k:json.loads((Path(p)/"summary.json").read_text())["status"] '
        'if (Path(p)/"summary.json").is_file() else "failed" if (Path(p)/"failure.json").is_file() '
        'else None for k,p in paths.items()}))')
    previous = None
    while True:
        value = resident(code)
        if value != previous:
            log('shadow_triplets',states=value)
            previous = value
        if all(state is not None for state in value.values()):
            infra.write(HERE/'diagnostic_completion.json',value)
            return value
        time.sleep(45)


def wait_summary():
    limit = launch.protocol()['summary_max_bytes']
    code = ('from pathlib import Path;import json;root=Path('+repr(infra.remote(HERE))+');'
        'p=root/"shared_level_summary.json";f=root/"shared_level_summary_failure.json";'
        'assert not p.exists() or p.stat().st_size<='+str(limit)+';'
        'print(p.read_text() if p.exists() else json.dumps({"status":"analysis_failed",'
        '"failure":json.loads(f.read_text())}) if f.exists() else "null")')
    while True:
        value = resident(code)
        if value is not None:
            name = 'shared_level_summary_failure.json' if value['status'] == 'analysis_failed' else 'shared_level_summary.json'
            infra.write(HERE/name,value)
            return value
        time.sleep(30)


def main():
    submitted('diagnostic',launch.submit_days)
    states = wait_days()
    if any(state != 'complete' for state in states.values()):
        result = dict(status='stopped_diagnostic_integrity',states=states)
    else:
        submitted('finalize',launch.submit_finalize)
        summary = wait_summary()
        result = dict(status='diagnosis_complete' if summary.get('integrity',{}).get('all_pass')
            else 'stopped_analysis_integrity',summary_status=summary['status'])
    result.update(priority='high',planned_budget=launch.protocol()['compute_budget'])
    infra.write(HERE/'continuation_result.json',result)
    log('finished',**result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        value = dict(status='failed',type=type(error).__name__,message=str(error))
        infra.write(HERE/'continuation_failure.json',value)
        log('failed',**value)
        raise
