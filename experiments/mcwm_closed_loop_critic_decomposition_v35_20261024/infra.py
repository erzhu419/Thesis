"""Explicit v35 code staging and bounded resident summaries."""
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
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def stage(files):
    files = [Path(p).resolve() for p in files]
    groups = {}
    for path in files:
        path.relative_to(HERE)
        groups.setdefault(path.parent, []).append(path)
    if not files:
        return dict(files=0, uploaded_bytes=0)
    scheduler.run_on('node001', 'mkdir -p '+shlex.join([remote(p) for p in groups]), timeout=45, check=True)
    for parent, paths in groups.items():
        subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
            *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(parent)+'/'],
            check=True, timeout=90)
    return dict(files=len(files), uploaded_bytes=sum(p.stat().st_size for p in files), destination=remote(HERE))


def task(protocol, label, args, *, runner='run.py', ram_mb=8000):
    prefix = protocol['name']
    return dict(project='Thesis', description=prefix+' '+label, signature=prefix+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, args)]),
        priority='high', allowed_nodes=CPUS, cpu=1, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=prefix+'/cpu', ram_resource_family=prefix+'/cpu', vram_resource_family=prefix+'/cpu',
        allow_cpu_training=True,
        cpu_training_justification='Read-only CPU critic decomposition and factual action-coverage diagnostics from existing server checkpoints; zero training or native simulation.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(rows, batch):
    receipt = HERE/(batch+'_submission.json')
    if receipt.exists():
        raise FileExistsError(receipt)
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, rows)
    result = json.loads(subprocess.check_output([sys.executable, str(SCHEDULER), 'submit-jsonl',
        '--file', str(jobs), '--trusted', '--json'], text=True))
    write(receipt, result)
    return result


def collect(relative, *, max_bytes):
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    code = 'from pathlib import Path; p=Path('+repr(remote(path))+'); assert p.stat().st_size<='+str(max_bytes)+'; print(p.read_text(),end="")'
    value = json.loads(scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1])
    write(path, value)
    return value
