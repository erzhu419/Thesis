"""Submit one fixed stable native fork; download only its bounded summary."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_logged_action_observation_v116_20261029.infra import (
    scheduler, remote, PYTHON, SCHEDULER, environment, write)


def stage():
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json', HERE/'anchor_binding.json'])
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/'staging.json', dict(files=[p.name for p in paths], uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit():
    p = json.loads((HERE/'protocol.json').read_text())
    if not json.loads((HERE/'implementation_checks.json').read_text())['all_pass']:
        raise RuntimeError('Resolve implementation checks before native execution')
    row = dict(project='Thesis', description=p['name']+' stable live fork',
        signature=p['name']+'/stable', cwd=str(HERE),
        cmd=shlex.join([PYTHON, '-B', '-u', remote(HERE/'control.py'),
            '--thesis-root', remote(ROOT), '--out', remote(HERE/'runs'/'stable')]),
        priority='high', allowed_nodes=p['runtime']['allowed_nodes'], cpu=2, ram_mb=8000, vram=0,
        env=environment(), resource_family=p['name']+'/live-fork', ram_resource_family=p['name']+'/live-fork',
        skip_launch_staging=True, no_auto_requeue=True, allow_no_resume=True, allow_no_ckpt=True)
    write(HERE/'jobs.json', [row])
    write(HERE/'submission_attempt.json', dict(signature=row['signature']))
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(HERE/'jobs.json'),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'submission_error.json', dict(stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect failed submission before considering any further action')
    value = json.loads(proc.stdout)
    write(HERE/'submission.json', value)
    print(json.dumps(value), flush=True)
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', '--task-id', value['submitted'][0]['id']],
        capture_output=True, text=True, timeout=360)
    write(HERE/'dispatch.json', dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))


def poll():
    code = ('from pathlib import Path\nimport json\n'
        'p=Path('+repr(remote(HERE/'runs'/'stable'/'summary.json'))+')\n'
        'x=json.loads(p.read_text()) if p.exists() else None\n'
        'raw=json.dumps(dict(complete=x is not None,summary=x),separators=(",",":"),allow_nan=False)\n'
        'assert len(raw.encode())<=48000\nprint(raw)\n')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-B', '-c', code]), timeout=45, check=True)[1]
    return json.loads(raw), len(raw.encode())


if __name__ == '__main__':
    {'stage': stage, 'submit': submit, 'poll': lambda: print(json.dumps(poll()[0]))}[sys.argv[1]]()
