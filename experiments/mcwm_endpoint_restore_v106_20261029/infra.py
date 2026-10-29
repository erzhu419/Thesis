"""One bounded endpoint check; only small results leave node005."""
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
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json'])
    scheduler.run_on('node005', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node005'),
        *map(str, paths), scheduler._ssh_target_for_node('node005')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/'staging.json', dict(files=[p.name for p in paths],
        uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit():
    p = json.loads((HERE/'protocol.json').read_text())
    argv = [PYTHON, '-B', '-u', remote(HERE/'run.py'),
        '--thesis-root', remote(ROOT), '--out', remote(HERE/'run/seed6201')]
    row = dict(project='Thesis', description=p['name']+' saved endpoint restore and next update',
        signature=p['name']+'/seed6201', cwd=str(HERE), cmd=shlex.join(argv), priority='high',
        require_node='node005', cpu=1, ram_mb=p['runtime']['ram_mb'], vram=0, env=environment(),
        resource_family=p['name']+'/verification', ram_resource_family=p['name']+'/verification',
        skip_launch_staging=True, no_auto_requeue=True, allow_no_resume=True,
        allow_no_ckpt=True, allow_cpu_training=True,
        cpu_training_justification='Only two throwaway next-update checks from the saved v105 '
            '70000 endpoint using the original CPU runtime. No environment collection, '
            'source retraining, fresh labels, or model refitting. Saved endpoint stays70000.')
    write(HERE/'verification_jobs.json', [row])
    write(HERE/'submission_attempt.json', dict(signature=row['signature']))
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
        str(HERE/'verification_jobs.json'), '--trusted', '--json', '--lock-timeout', '300'],
        capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'submission_error.json', dict(stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect failed submission; no automatic retry')
    result = json.loads(proc.stdout)
    write(HERE/'verification_submission.json', result)
    print(json.dumps(result), flush=True)
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', '--task-id', result['submitted'][0]['id']],
        capture_output=True, text=True, timeout=360)
    write(HERE/'verification_dispatch.json', dict(returncode=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr))
    print(proc.stdout, flush=True)


def collect():
    p = json.loads((HERE/'protocol.json').read_text())
    target = remote(HERE/'run/seed6201/summary.json')
    code = ('from pathlib import Path\nimport json\np=Path('+repr(target)+')\n'
        'if not p.exists(): print(json.dumps(dict(status="pending")))\n'
        'else:\n'
        ' assert p.stat().st_size<='+str(p['small_summary_limit_bytes'])+'\n'
        ' print(json.dumps(dict(status="finished",summary=json.loads(p.read_text())),separators=(",",":")))\n')
    raw = scheduler.run_on('node005', shlex.join([PYTHON,'-B','-c',code]), timeout=45, check=True)[1]
    item = json.loads(raw)
    if item['status'] == 'finished':
        write(HERE/'summary.json', item['summary'])
        write(HERE/'collection_receipt.json', dict(downloaded_bytes=len(raw.encode()),
            checkpoint_downloaded=False))
        print(json.dumps(item['summary']), flush=True)
    else:
        print(raw, flush=True)


if __name__ == '__main__':
    {'stage':stage, 'submit':submit, 'collect':collect}[sys.argv[1]]()
