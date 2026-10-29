"""Stage two fixed zero-learning prefixes; fetch bounded terminal summaries."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_same_state_holding_v114_20261029.infra import (
    scheduler, remote, PYTHON, SCHEDULER, environment, write)


def stage():
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json'])
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/'staging.json', dict(files=[p.name for p in paths],
        uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit():
    p = json.loads((HERE/'protocol.json').read_text())
    if not json.loads((HERE/'implementation_checks.json').read_text())['all_pass']:
        raise RuntimeError('Resolve replay and observation checks before native work')
    rows = []
    for case in p['cases']:
        scene = case['scenario']
        argv = [PYTHON, '-B', '-u', remote(HERE/'control.py'), '--thesis-root', remote(ROOT),
            '--scenario', scene, '--out', remote(HERE/'runs'/scene)]
        rows.append(dict(project='Thesis', description=p['name']+' '+scene,
            signature=p['name']+'/'+scene, cwd=str(HERE), cmd=shlex.join(argv),
            priority='high', allowed_nodes=p['runtime']['allowed_nodes'], cpu=1,
            ram_mb=p['runtime']['ram_mb'], vram=0, env=environment(),
            resource_family=p['name']+'/recorded-action-prefix',
            ram_resource_family=p['name']+'/recorded-action-prefix',
            skip_launch_staging=True, no_auto_requeue=True, allow_no_resume=True,
            allow_no_ckpt=True))
    write(HERE/'jobs.json', rows)
    write(HERE/'submission_attempt.json', dict(signatures=[row['signature'] for row in rows]))
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
        str(HERE/'jobs.json'), '--trusted', '--json', '--lock-timeout', '300'],
        capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'submission_error.json', dict(stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect submission evidence without automatic resubmission')
    result = json.loads(proc.stdout)
    write(HERE/'submission.json', result)
    print(json.dumps(result), flush=True)
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[item for row in result['submitted'] for item in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/'dispatch.json', dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))


def poll():
    code = ('from pathlib import Path\nimport json\n'
        'here=Path('+repr(remote(HERE))+')\n'
        'paths=list((here/"runs").glob("*/summary.json"))\n'
        'rows={p.parent.name:json.loads(p.read_text()) for p in paths} if len(paths)==2 else {}\n'
        'raw=json.dumps(dict(finished=len(paths),cases=rows),separators=(",",":"),allow_nan=False)\n'
        'assert len(raw.encode())<=40000\nprint(raw)\n')
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-B', '-c', code]), timeout=45, check=True)[1]
    return json.loads(raw), len(raw.encode())


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stage': stage()
    elif mode == 'submit': submit()
    elif mode == 'poll': print(json.dumps(poll()[0]), flush=True)
    else: raise SystemExit('Use stage, submit or poll')
