"""Stage two fixed zero-learning replay jobs and collect bounded receipts."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_connection_full_day_v118_20261029.infra import (
    scheduler, remote, PYTHON, SCHEDULER, environment, write)


def stage():
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json', HERE/'reference_summary.json'])
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    proc = subprocess.run(['rsync', '-a', '--timeout=30', '-e',
        scheduler._ssh_rsync_shell_for_node('node001'), *map(str, paths),
        scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        capture_output=True, text=True, timeout=90)
    write(HERE/'staging.json', dict(returncode=proc.returncode,
        files=[p.name for p in paths], uploaded_bytes=sum(p.stat().st_size for p in paths), stderr=proc.stderr))
    proc.check_returncode()


def submit():
    if not json.loads((HERE/'implementation_checks.json').read_text())['all_pass']:
        raise RuntimeError('Resolve implementation checks before native execution')
    p = json.loads((HERE/'protocol.json').read_text())
    rows = []
    for scene in p['scenarios']:
        rows.append(dict(project='Thesis', description=p['name']+' '+scene+' zero-learning replay',
            signature=p['name']+'/'+scene, cwd=str(HERE),
            cmd=shlex.join([PYTHON, '-B', '-u', remote(HERE/'control.py'),
                '--thesis-root', remote(ROOT), '--scenario', scene, '--out', remote(HERE/'runs'/scene)]),
            priority='high', require_node='node001', allowed_nodes=['node001'], cpu=1,
            ram_mb=8000, vram=0, env=environment(), resource_family=p['name']+'/recorded-replay',
            ram_resource_family=p['name']+'/recorded-replay', skip_launch_staging=True,
            no_auto_requeue=True, allow_no_resume=True, allow_no_ckpt=True))
    write(HERE/'jobs.json', rows)
    write(HERE/'submission_attempt.json', dict(signatures=[r['signature'] for r in rows]))
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(HERE/'jobs.json'),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'submission_error.json', dict(stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect submission failure before any further action')
    result = json.loads(proc.stdout)
    write(HERE/'submission.json', result)
    print(json.dumps(result), flush=True)
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[v for r in result['submitted'] for v in ('--task-id', r['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/'dispatch.json', dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))


def poll():
    code = ('from pathlib import Path\nimport json\n'
        'p=Path('+repr(remote(HERE/'runs'))+')\n'
        'files=list(p.glob("*/summary.json"))\n'
        'cases={f.parent.name:json.loads(f.read_text()) for f in files} if len(files)==2 else {}\n'
        'raw=json.dumps(dict(finished=len(files),cases=cases),separators=(",",":"),allow_nan=False)\n'
        'assert len(raw.encode())<=48000\nprint(raw)\n')
    result = scheduler.run_on('node001', shlex.join([PYTHON, '-B', '-c', code]), timeout=45, check=True)
    return json.loads(result[1]), len(result[1].encode())


if __name__ == '__main__':
    {'stage': stage, 'submit': submit, 'poll': lambda: print(json.dumps(poll()[0]))}[sys.argv[1]]()
