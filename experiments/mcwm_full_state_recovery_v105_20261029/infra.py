"""One fixed original CPU continuation; only small receipts leave the server."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    scheduler, remote, PYTHON, SCHEDULER)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def environment():
    p = protocol()
    return dict(THESIS_H2O_ROOT=remote(ROOT/p['h2o_root']),
        THESIS_MCWM_ROOT=remote(ROOT/p['mcwm_root']),
        THESIS_DATASET=remote(ROOT/p['dataset']), THESIS_SOURCE_ENV=remote(ROOT/p['source_env']),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MPLBACKEND='Agg', SDL_VIDEODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1')


def stage(label):
    paths = sorted([*HERE.glob('*.py'), HERE/'protocol.json'])
    scheduler.run_on('node005', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node005'),
        *map(str, paths), scheduler._ssh_target_for_node('node005')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    write(HERE/(label+'_staging.json'), dict(files=[p.name for p in paths],
        uploaded_bytes=sum(p.stat().st_size for p in paths)))


def submit():
    p = protocol()
    test = json.loads((HERE/'runtime_tests.json').read_text())
    if not test['all_pass']:
        raise RuntimeError('Original-runtime synthetic continuation checks failed')
    callback = json.loads((HERE/'callback_tests.json').read_text())
    if not callback['all_pass']:
        raise RuntimeError('Terminal export and verification callback failed')
    argv = [PYTHON, '-B', '-u', remote(HERE/'train.py'), '--seed', '6201',
        '--out', remote(HERE/'run/seed6201'), '--initializer', remote(ROOT/p['initializer']),
        '--h2o-root', remote(ROOT/p['h2o_root']), '--dataset', remote(ROOT/p['dataset']),
        '--source-env', remote(ROOT/p['source_env']), '--mcwm-root', remote(ROOT/p['mcwm_root']),
        '--thesis-root', remote(ROOT), '--device', 'cpu']
    name = p['name']
    row = dict(project='Thesis', description=name+' fixed6201 terminal state recovery',
        signature=name+'/seed6201', cwd=str(HERE), cmd=shlex.join(argv), priority='high',
        require_node='node005', cpu=1, ram_mb=13000, vram=0, env=environment(),
        resource_family=name+'/recovery', ram_resource_family=name+'/recovery',
        skip_launch_staging=True, no_auto_requeue=True, allow_no_resume=True,
        allow_no_ckpt=True, allow_cpu_training=True,
        cpu_training_justification='Reproduce the original node005 single-CPU6201 training exactly: '
            'reuse50000 IQL,20000 continuation updates and original25000 source-event cap. '
            'Export missing final learner/model/trust state; at most2 separately accounted '
            'throwaway diagnostic updates. Zero target interaction or fresh training.')
    write(HERE/'recovery_jobs.json', [row])
    write(HERE/'recovery_submission_attempt.json', dict(signature=row['signature']))
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
        str(HERE/'recovery_jobs.json'), '--trusted', '--json', '--lock-timeout', '300'],
        capture_output=True, text=True, timeout=360)
    if proc.returncode:
        write(HERE/'recovery_submission_error.json', dict(stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('Inspect failed submission receipt; do not resubmit blindly')
    result = json.loads(proc.stdout)
    write(HERE/'recovery_submission.json', result)
    proc = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', '--task-id', result['submitted'][0]['id']],
        capture_output=True, text=True, timeout=360)
    write(HERE/'recovery_dispatch.json', dict(returncode=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] != ['submit']:
        raise SystemExit('Use submit after original-runtime tests')
    submit()
