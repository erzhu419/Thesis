"""Stage only v47 code; reuse resident factual data without native branches."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

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
    groups = {}
    for path in map(lambda p: Path(p).resolve(), files):
        path.relative_to(HERE)
        groups.setdefault(path.parent, []).append(path)
    scheduler.run_on('node001', 'mkdir -p '+shlex.join([remote(p) for p in groups]),
                     timeout=45, check=True)
    for parent, paths in groups.items():
        subprocess.run(['rsync', '-a', '--timeout=30', '-e',
            scheduler._ssh_rsync_shell_for_node('node001'), *map(str, paths),
            scheduler._ssh_target_for_node('node001')+':'+remote(parent)+'/'],
            check=True, timeout=90)
    return dict(files=sum(map(len, groups.values())),
        uploaded_bytes=sum(p.stat().st_size for paths in groups.values() for p in paths),
        destination=remote(HERE))


def task(protocol, label, argv=(), *, runner='run.py', ram_mb=8000):
    prefix = protocol['name']
    return dict(project='Thesis', description=prefix+' '+label, signature=prefix+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=CPUS, cpu=1, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=prefix+'/cpu', ram_resource_family=prefix+'/cpu',
        vram_resource_family=prefix+'/cpu', allow_cpu_training=True,
        cpu_training_justification='Original285 factual slots and456 updates per arm, single-step M1/M8 cumulative shadow learning on CPU. First stable M1 source replay audit is separate456 diagnostic updates. Formal four independent workers run both arms concurrently; parameters and Adam persist. Extra target Q forward computation is separate, no new simulator branch, feedback or control rollout; checkpoints remain resident.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
        no_auto_requeue=True)


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
    while True:
        result = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl',
            '--file', str(jobs), '--trusted', '--json', '--lock-timeout', '30'],
            text=True, capture_output=True)
        if not result.returncode:
            value = json.loads(result.stdout)
            write(receipt, value)
            return value
        if 'waiting for scheduler state lock' not in result.stderr:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        print('Submission waiting for scheduler state lock: '+batch, flush=True)
        time.sleep(15)


def collect(relative, *, max_bytes):
    path = HERE/relative
    if path.exists():
        return json.loads(path.read_text())
    code = ('from pathlib import Path;p=Path('+repr(remote(path))+');'
        'assert p.stat().st_size<='+str(max_bytes)+';print(p.read_text(),end="")')
    value = json.loads(scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]),
        timeout=45, check=True)[1])
    write(path, value)
    return value
