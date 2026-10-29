"""Stage only v59 files; high tasks and bounded resident JSON summaries."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler,
)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def stage(files):
    paths = [Path(path).resolve() for path in files]
    for path in paths:
        path.relative_to(HERE)
        if path.parent != HERE or not (path.suffix == '.py' or path.name == 'protocol.json'):
            raise ValueError('Stage only v59 source and protocol: '+str(path))
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e',
        scheduler._ssh_rsync_shell_for_node('node001'), *map(str, paths),
        scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        check=True, timeout=90)
    return dict(files=len(paths), file_names=[path.name for path in paths],
        uploaded_bytes=sum(path.stat().st_size for path in paths), destination=remote(HERE))


def task(protocol, label, argv=(), *, runner='run.py', cpu=1, ram_mb=16000):
    prefix = protocol['name']
    return dict(project='Thesis', description=prefix+' '+label, signature=prefix+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=CPUS, cpu=cpu, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=prefix+'/cpu', ram_resource_family=prefix+'/cpu',
        vram_resource_family=prefix+'/cpu', allow_cpu_training=True,
        cpu_training_justification='New matched mu14 common prefixes with first actual '
            'pair preflight reused. Each native action pair runs in two CPU processes. '
            'Fresh and no_fresh learning are separate parallel workers; each retains '
            'parameters and Adam across285feedback slots/456updates. No learned policy '
            'deployment; heldout truth excluded and native suffixes never train.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(rows, batch):
    receipt = HERE/(batch+'_submission.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    jobs = HERE/(batch+'_jobs.json')
    if jobs.exists():
        if json.loads(jobs.read_text()) != rows:
            raise RuntimeError('Existing submission definition differs: '+batch)
    else:
        write(jobs, rows)
    attempt = HERE/(batch+'_submission_attempt.json')
    if attempt.exists():
        raise RuntimeError('Unresolved submission attempt; inspect scheduler receipt before '
            'resubmitting: '+batch)
    write(attempt, dict(batch=batch, signatures=[row['signature'] for row in rows]))
    try:
        result = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
            str(jobs), '--trusted', '--json', '--lock-timeout', '90'],
            text=True, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        write(HERE/(batch+'_submission_failure.json'),
            dict(type=type(error).__name__, message='Submission timed out; no automatic retry'))
        raise
    if result.returncode:
        write(HERE/(batch+'_submission_failure.json'),
            dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value = json.loads(result.stdout)
    write(receipt, value)
    return value


def status(ids):
    if not ids:
        return []
    result = subprocess.run([sys.executable, str(SCHEDULER), 'status', '--readonly',
        '--json', '--lock-timeout', '30', '--ids', ','.join(ids)],
        text=True, capture_output=True, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    fields = ('id', 'status', 'priority', 'node', 'signature', 'log_path')
    return [{key: row.get(key) for key in fields} for row in json.loads(result.stdout)['tasks']]


def _json_path(relative):
    path = (HERE/relative).resolve()
    path.relative_to(HERE)
    if path.suffix != '.json':
        raise ValueError('Only small JSON results may be collected')
    return path


def presence(relatives):
    paths = {str(name): remote(_json_path(name)) for name in relatives}
    code = ('from pathlib import Path;import json;paths='+repr(paths)+';'
        'print(json.dumps({name:Path(path).exists() for name,path in paths.items()}))')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]),
        timeout=45, check=True)[1]
    return json.loads(raw)


def collect(relative, *, max_bytes=48000):
    if not 0 < max_bytes <= 48000:
        raise ValueError('Only JSON summaries at most48000 bytes may be collected')
    path = _json_path(relative)
    if path.exists():
        if path.stat().st_size > max_bytes:
            raise ValueError('Local summary exceeds the small-result limit')
        return json.loads(path.read_text())
    code = ('from pathlib import Path;p=Path('+repr(remote(path))+');'
        'assert p.stat().st_size<='+str(max_bytes)+';print(p.read_text(),end="")')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]),
        timeout=45, check=True)[1]
    value = json.loads(raw)
    if len(raw.encode('utf-8')) > max_bytes:
        raise ValueError('Received summary exceeds the small-result limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(raw)
    return value


def publish_state(value):
    """Publish compact lifecycle outcomes for analysis, including missing failed outputs."""
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(raw.encode('utf-8')) > 48000:
        raise ValueError('Lifecycle state exceeds small-result limit')
    path = HERE/'continuation_inputs.json'
    path.write_text(raw+'\n')
    code = ('from pathlib import Path;p=Path('+repr(remote(path))+');'
        'p.write_text('+repr(raw+'\n')+')')
    scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)
