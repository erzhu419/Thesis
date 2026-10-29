"""One scheduler-managed read-only probe; transfer only its small JSON result."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_full_state_recovery_v105_20261029.infra import (
    scheduler, remote, PYTHON, SCHEDULER, environment, write)


def stage():
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/'staging.json', dict(files=[p.name for p in paths], uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit():
    p = json.loads((HERE/'protocol.json').read_text())
    if not json.loads((HERE/'verification.json').read_text())['all_pass']:
        raise RuntimeError('The first-packet reconstruction and probe checks must pass')
    row = dict(project='Thesis', description=p['name']+' zero-update initial critic probe',
        signature=p['name']+'/initial6201', cwd=str(HERE),
        cmd=shlex.join([PYTHON, '-B', '-u', remote(HERE/'run.py'), '--thesis-root', remote(ROOT),
                       '--out', remote(HERE/'run')]),
        priority='high', allowed_nodes=p['runtime']['allowed_nodes'], cpu=1, ram_mb=16000,
        vram=0, env=environment(), resource_family=p['name']+'/readout',
        ram_resource_family=p['name']+'/readout', skip_launch_staging=True,
        no_auto_requeue=True, allow_no_resume=True, allow_no_ckpt=True,
        allow_cpu_training=True, cpu_training_justification='No optimizer updates: resident frozen70000 endpoint '
        'forward/autograd sensitivity from the five existing first-packet rows. No environment, collection or model fitting.')
    write(HERE/'jobs.json', [row])
    write(HERE/'submission_attempt.json', dict(signature=row['signature']))
    done = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(HERE/'jobs.json'),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if done.returncode:
        write(HERE/'submission_error.json', dict(stdout=done.stdout, stderr=done.stderr))
        raise RuntimeError('Submission failed; inspect receipt before any retry')
    result = json.loads(done.stdout)
    write(HERE/'submission.json', result)
    done = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', '--task-id', result['submitted'][0]['id']],
        capture_output=True, text=True, timeout=360)
    write(HERE/'dispatch.json', dict(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr))
    print(json.dumps(result), flush=True)


def collect():
    code = ('from pathlib import Path\n'
        'p=Path('+repr(remote(HERE/'run/summary.json'))+')\n'
        'print(p.read_text() if p.exists() else \'{"status":"pending"}\')\n')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-B', '-c', code]), timeout=45, check=True)[1]
    value = json.loads(raw)
    if value['status'] != 'pending':
        write(HERE/'summary.json', value)
        write(HERE/'collection_receipt.json', dict(downloaded_bytes=len(raw.encode()), checkpoint_downloaded=False))
    return value


if __name__ == '__main__':
    {'stage': stage, 'submit': submit, 'collect': lambda: print(json.dumps(collect()))}[sys.argv[1]]()
