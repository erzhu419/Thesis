"""High-priority frozen context parity; stage new files and pull small JSON."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, remote, scheduler, environment)


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def stage():
    files = [HERE/name for name in ('critic.py', 'context.py', 'audit.py', 'probe.py', 'run.py', 'protocol.json')]
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, files), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    return dict(files=[p.name for p in files], uploaded_bytes=sum(p.stat().st_size for p in files))


def submit(batch='readout'):
    receipt = HERE/(batch+'_submission.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    attempt = HERE/(batch+'_submission_attempt.json')
    if attempt.exists():
        raise RuntimeError('Resolve the recorded submission before another task is created')
    protocol = json.loads((HERE/'protocol.json').read_text())
    job = dict(project='Thesis', description='v78 frozen critic context parity readout',
        signature=protocol['name']+'/'+batch, cwd=str(HERE),
        cmd=shlex.join([PYTHON, '-u', remote(HERE/'run.py'), '--out', remote(HERE/'results')]),
        priority='high', allowed_nodes=CPUS, cpu=1, ram_mb=6000, vram=0,
        env=environment(),
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, [job])
    write(attempt, dict(signature=job['signature']))
    result = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(jobs),
        '--trusted', '--json', '--lock-timeout', '300'], text=True, capture_output=True, timeout=360)
    if result.returncode:
        write(HERE/'submission_failure.json', dict(returncode=result.returncode,
            stdout=result.stdout[-3000:], stderr=result.stderr[-3000:]))
        raise RuntimeError('See submission_failure.json; do not submit a duplicate')
    value = json.loads(result.stdout)
    write(receipt, value)
    task_id = value['submitted'][0]['id']
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', '--task-id', task_id], text=True, capture_output=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-10000:], stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:
        raise RuntimeError('Known task dispatch needs review; do not resubmit')
    return value


def status(batch='readout'):
    value = json.loads((HERE/(batch+'_submission.json')).read_text())
    task_id = value['submitted'][0]['id']
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly', '--json',
        '--lock-timeout', '30', '--ids', task_id], text=True, capture_output=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [{key: row.get(key) for key in ('id', 'status', 'priority', 'node')}
        for row in json.loads(result.stdout)['tasks']]


def collect(name):
    if name not in ('visibility.json', 'model_parity.json', 'decision.json'):
        raise ValueError('Only the three planned small summaries are collected')
    path = HERE/'results'/name
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path; p=Path('+repr(remote(path))+'); '
        'assert p.stat().st_size<=48000; print(p.read_text(),end="")')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    if len(raw.encode()) > 48000:
        raise ValueError('Small JSON exceeded48KB')
    value = json.loads(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(raw)
    return value
