"""One high, two-CPU preflight; only its small JSON returns locally."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def stage():
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    value = dict(files=len(paths), bytes=sum(path.stat().st_size for path in paths),
        destination=remote(HERE), old_paths_written=[])
    write(HERE/'staging.json', value)
    return value


def resident_gate():
    source = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'
    prefix = source/'runs/executed_cql/validation/seed6207/stable/fresh'
    paths = [source/'inputs/h2oplus_seed6207.pt',
             *[prefix/name for name in ('summary.json', 'prefix_trace.jsonl', 'updates.jsonl')]]
    code = ('import sys,json;from pathlib import Path;sys.path.insert(0,'
        +repr(remote(HERE.parents[1]))+');'
        'import experiments.mcwm_closed_loop_supervision_plan_v84_20261027.run as run;'
        'checks={p:Path(p).is_file() for p in '+repr([remote(p) for p in paths])+'};'
        'print(json.dumps(dict(all_pass=all(checks.values()),resident_files=checks,'
        'runtime_import=True,new_native_seconds=0,new_training_updates=0)))')
    command = shlex.join(['env', *[key+'='+value for key,value in environment().items()],
                          PYTHON, '-c', code])
    result = json.loads(scheduler.run_on('node001', command, timeout=60, check=True)[1])
    write(HERE/'resident_gate.json', result)
    if not result['all_pass']:
        raise RuntimeError('Resident source inputs missing; no native task submitted')
    return result


def submit_preflight():
    receipt = HERE/'preflight_submission.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    if (HERE/'preflight_submission_attempt.json').exists():
        raise RuntimeError('Inspect existing submission before another attempt')
    p = json.loads((HERE/'protocol.json').read_text())
    task = dict(project='Thesis', description=p['name']+' one native remaining-day pair',
        signature=p['name']+'/preflight/seed6207/stable/102X',
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/'run.py'),
            '--out', remote(HERE/'runs/preflight')]),
        priority='high', allowed_nodes=CPUS, cpu=2, ram_mb=24000, vram=0, env=environment(),
        resource_family=p['name']+'/native', ram_resource_family=p['name']+'/native',
        allow_cpu_training=True,
        cpu_training_justification='Restore existing v75 fresh packet37 prefix only:185covariates,'
            '185outcomes,296updates per optimizer; no new50k initializer. Freeze at12000, '
            'one predeclared102X root, parallel two physical F-endpoint suffixes to24000. '
            'At most36000native seconds; zero suffix training or label feedback. '
            'No automatic collection or new response-model training.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)
    jobs = HERE/'preflight_jobs.json'
    write(jobs, [task])
    write(HERE/'preflight_submission_attempt.json', dict(signature=task['signature']))
    result = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(jobs),
        '--trusted', '--json', '--lock-timeout', '300'], text=True, capture_output=True, timeout=360)
    if result.returncode:
        write(HERE/'submission_failure.json', dict(returncode=result.returncode,
            stdout=result.stdout[-4000:], stderr=result.stderr[-4000:]))
        raise RuntimeError('Submission rejected; receipt retained')
    value = json.loads(result.stdout)
    write(receipt, value)
    ids = [row['id'] for row in value['submitted']]
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[arg for identifier in ids for arg in ('--task-id', identifier)]],
        capture_output=True, text=True, timeout=360)
    write(HERE/'preflight_dispatch.json', dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-6000:], stderr=dispatch.stderr[-3000:]))
    if dispatch.returncode:
        raise RuntimeError('Task submitted; inspect its existing ID before redispatch')
    return value


def status():
    ids = [row['id'] for row in json.loads((HERE/'preflight_submission.json').read_text())['submitted']]
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly', '--json',
        '--lock-timeout', '30', '--ids', ','.join(ids)], capture_output=True, text=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return [{key:row.get(key) for key in ('id','status','priority','node','signature')}
            for row in json.loads(result.stdout)['tasks']]


def collect():
    path = HERE/'runs/preflight/preflight.json'
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path;import json;p=Path('+repr(remote(path))+');'
        's=json.dumps(json.loads(p.read_text()),separators=(",",":"),allow_nan=False);'
        'assert len(s.encode())<=48000;print(s)')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    value = json.loads(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)
    return value
