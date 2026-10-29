"""Stage the two fixed passive prefixes; return only bounded terminal receipts."""
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
    groups = [
        (HERE, sorted([*HERE.glob('*.py'), HERE/'protocol.json',
                       HERE/'implementation_checks.json'])),
    ]
    # Resolve the complete upload list before creating either remote directory.
    sizes = {str(path.relative_to(ROOT)): path.stat().st_size
             for _, paths in groups for path in paths}
    receipts = []
    for directory, paths in groups:
        scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(directory)),
                         timeout=45, check=True)
        proc = subprocess.run([
            'rsync', '-a', '--timeout=30', '-e',
            scheduler._ssh_rsync_shell_for_node('node001'), *map(str, paths),
            scheduler._ssh_target_for_node('node001')+':'+remote(directory)+'/',
        ], capture_output=True, text=True, timeout=90)
        receipts.append(dict(directory=str(directory.relative_to(ROOT)),
                             files=[path.name for path in paths],
                             returncode=proc.returncode, stderr=proc.stderr))
        if proc.returncode or len(receipts) == len(groups):
            write(HERE/'staging.json', dict(
                all_pass=len(receipts) == len(groups) and proc.returncode == 0,
                confirmed_uploaded_bytes=sum(sizes[str(path.relative_to(ROOT))]
                    for (_, done), receipt in zip(groups, receipts)
                    if receipt['returncode'] == 0 for path in done),
                failed_transfer_partial_bytes_unknown=proc.returncode != 0,
                file_bytes=sizes, transfers=receipts))
        proc.check_returncode()


def submit():
    if json.loads((HERE/'implementation_checks.json').read_text())['all_pass'] is not True:
        raise RuntimeError('Resolve implementation checks before native execution')
    p = json.loads((HERE/'protocol.json').read_text())
    scenes = [case['scenario'] for case in p['cases']]
    if len(scenes) != 2 or set(scenes) != {'stable', 'local_speed_drop'}:
        raise ValueError('This collection has exactly the two declared scenarios')
    rows = []
    for scene in scenes:
        argv = [PYTHON, '-B', '-u', remote(HERE/'control.py'),
                '--thesis-root', remote(ROOT), '--scenario', scene,
                '--out', remote(HERE/'runs'/scene)]
        rows.append(dict(
            project='Thesis', description=p['name']+' '+scene+' passive receiver audit',
            signature=p['name']+'/'+scene, cwd=str(HERE), cmd=shlex.join(argv),
            priority='high', require_node='node001', allowed_nodes=['node001'],
            cpu=1, ram_mb=8000, vram=0, env=environment(),
            resource_family=p['name']+'/passive-prefix',
            ram_resource_family=p['name']+'/passive-prefix',
            skip_launch_staging=True, no_auto_requeue=True,
            allow_no_resume=True, allow_no_ckpt=True))
    write(HERE/'jobs.json', rows)
    write(HERE/'submission_attempt.json', dict(signatures=[r['signature'] for r in rows]))
    proc = subprocess.run([
        sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(HERE/'jobs.json'),
        '--trusted', '--json', '--lock-timeout', '300',
    ], capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'submission_error.json', dict(
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect the submission failure; no automatic retry')
    result = json.loads(proc.stdout)
    write(HERE/'submission.json', result)
    print(json.dumps(result), flush=True)
    proc = subprocess.run([
        sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300',
        *[item for row in result['submitted'] for item in ('--task-id', row['id'])],
    ], capture_output=True, text=True, timeout=360)
    write(HERE/'dispatch.json', dict(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
    proc.check_returncode()


def poll():
    rc,raw,err=scheduler.run_on('node001', shlex.join([PYTHON, '-B',
        remote(HERE/'aggregate.py')]), timeout=45, check=False)
    if rc:
        raise RuntimeError('Remote summary failed: '+err)
    return json.loads(raw), len(raw.encode())


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stage':
        stage()
    elif mode == 'submit':
        submit()
    elif mode == 'poll':
        print(json.dumps(poll()[0], separators=(',', ':')), flush=True)
    else:
        raise SystemExit('Use stage, submit or poll')
