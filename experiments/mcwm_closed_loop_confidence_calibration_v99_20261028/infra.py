"""Submit eight fixed calibration collectors and relay only compact receipts locally."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS, PYTHON, SCHEDULER, environment, remote, scheduler)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def cells():
    p = protocol()
    return [(seed, scene) for seed in p['external_seeds'] for scene in p['scenarios']]


def key(seed, scene):
    return str(seed)+'/'+scene


def stage():
    paths = [*sorted(HERE.glob('*.py')), HERE/'protocol.json',
             HERE/'source_protocol.json', HERE/'source_model.json', HERE/'source_fit.json', HERE/'source_fit_summary.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, paths), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        timeout=90, check=True)
    result = dict(files=len(paths), uploaded_bytes=sum(path.stat().st_size for path in paths),
                  old_paths_written=[])
    write(HERE/'staging.json', result)
    return result


def resident_gate():
    command = shlex.join(['env', *[name+'='+value for name, value in environment().items()],
                         PYTHON, '-u', remote(HERE/'freeze_audit.py'),
                         '--out', remote(HERE/'freeze_audit.json')])
    rc, raw, error = scheduler.run_on('node001', command, timeout=90, check=False)
    if rc:
        raise RuntimeError('Frozen source audit failed: '+error[-1800:])
    if len(raw.encode()) > protocol()['retrieval']['audit_bytes']:
        raise ValueError('Frozen audit exceeds the compact receipt budget')
    audit = json.loads(raw)
    write(HERE/'freeze_audit.json', audit)
    if not audit['all_pass']:
        raise RuntimeError('Frozen actors, F or source snapshots failed the source audit')
    return audit


def task(label, runner, argv, cpu, ram_mb):
    name = protocol()['name']
    return dict(project='Thesis', description=name+' '+label, signature=name+'/'+label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, '-u', remote(HERE/runner), *map(str, argv)]),
        priority='high', allowed_nodes=CPUS, preferred_node='node004',
        cpu=cpu, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=name+'/'+runner, ram_resource_family=name+'/'+runner,
        allow_cpu_training=True,
        cpu_training_justification='Frozen native response collection only: two new external '
            'calibration seeds, two scenes and three first-line roots per scene;24 pairs with '
            'actually parallel reference/candidate suffixes, cap676800 simulation seconds. '
            'No neural update, bias refit, validation or old truth admission; fixed calibration follows complete collection.',
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True, no_auto_requeue=True)


def submit(batch, rows):
    path = HERE/(batch+'_submission.json')
    if path.exists():
        return json.loads(path.read_text())
    attempt = HERE/(batch+'_submission_attempt.json')
    if attempt.exists():
        raise RuntimeError('Inspect existing signatures before submitting another attempt')
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, rows)
    write(attempt, dict(signatures=[row['signature'] for row in rows]))
    process = subprocess.run([sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(jobs),
        '--trusted', '--json', '--lock-timeout', '300'], capture_output=True, text=True, timeout=360)
    if process.returncode:
        write(HERE/(batch+'_failure.json'), dict(returncode=process.returncode,
            stdout=process.stdout[-5000:], stderr=process.stderr[-2500:]))
        raise RuntimeError('Submission failed; inspect the retained attempt before retrying')
    result = json.loads(process.stdout)
    write(path, result)
    dispatch = subprocess.run([sys.executable, str(SCHEDULER), 'dispatch', '--no-bulk-window',
        '--lock-timeout', '300', *[part for row in result['submitted'] for part in ('--task-id', row['id'])]],
        capture_output=True, text=True, timeout=360)
    write(HERE/(batch+'_dispatch.json'), dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-9000:], stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:
        raise RuntimeError('Task IDs recorded; inspect them before redispatching')
    return result


def submit_collection():
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    if not (audit['all_pass'] and all(audit['checks'].values())):
        raise RuntimeError('Frozen source audit has not passed')
    rows = []
    for index, (seed, scene) in enumerate(cells()):
        argv = ['--external-seed', seed, '--scenario', scene,
                '--out', remote(HERE/'runs'/str(seed)/scene)]
        if index:
            argv += ['--gate-path', remote(HERE/'first_pair_gate.json')]
        row = task('collection/'+key(seed, scene), 'run.py', argv, 2, 8000)
        if index:
            row['wait_for_files'] = [str(HERE/'first_pair_gate.json')]
        rows.append(row)
    return submit('collection', rows)


def submit_summary():
    if not (HERE/'collection_submission.json').exists():
        raise RuntimeError('Submit the calibration collectors first')
    row = task('summary', 'infra.py', ['local-summary'], 1, 1000)
    row.update(cmd=shlex.join([sys.executable, '-u', str(HERE/'infra.py'), 'local-summary']),
        allowed_nodes=['local'], require_node='local', preferred_node='local',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
        cpu_training_justification='Relay the compact first-pair gate and eight native collector '
            'summaries; pure forward/error/budget analysis, no model fitting or simulation.')
    return submit('summary', [row])


def _pull(relative, limit):
    allowed = {'first_pair_gate.json', *['runs/'+key(seed, scene)+'/collector_summary.json'
                                      for seed, scene in cells()]}
    if relative not in allowed:
        raise ValueError('Only the declared compact gate and collector summaries may be pulled')
    source = remote(HERE/relative)
    code = '''import json
from pathlib import Path
p=Path(SOURCE)
if p.exists():
    body=json.loads(p.read_text())
    raw=json.dumps(body,separators=(',',':'),allow_nan=False)
    if len(raw.encode())>LIMIT: raise ValueError('Compact receipt exceeded its budget')
    result=dict(ready=True,body=body,body_bytes=len(raw.encode()))
else:
    failure=p.parent/'failure.json'
    body=json.loads(failure.read_text()) if failure.exists() else None
    result=dict(ready=False,failure=body)
print(json.dumps(result,separators=(',',':'),allow_nan=False))
'''.replace('SOURCE', repr(source)).replace('LIMIT', str(limit))
    raw = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)[1]
    return json.loads(raw)


def local_summary():
    from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.analysis import analyze
    p = protocol()
    audit = json.loads((HERE/'freeze_audit.json').read_text())
    model = json.loads((HERE/'source_model.json').read_text())
    inputs = HERE/'local_inputs'
    inputs.mkdir(exist_ok=True)
    deadline = time.monotonic()+21600
    received, failures, gate = {}, {}, None
    transferred = len(json.dumps(audit, separators=(',', ':')).encode())
    while time.monotonic() < deadline:
        if gate is None:
            value = _pull('first_pair_gate.json', p['retrieval']['gate_bytes'])
            if value['ready']:
                gate = value['body']
                transferred += value['body_bytes']
                write(HERE/'first_pair_gate.json', gate)
        for seed, scene in cells():
            name = key(seed, scene)
            if name in received or name in failures:
                continue
            value = _pull('runs/'+name+'/collector_summary.json', p['retrieval']['per_collector_bytes'])
            if value['ready']:
                received[name] = value['body']
                transferred += value['body_bytes']
                write(inputs/(str(seed)+'_'+scene+'.json'), value['body'])
            elif value['failure'] is not None:
                failure = value['failure']
                failures[name] = dict(external_seed=seed, scenario=scene, status='failed',
                    reason=failure.get('message', failure.get('type', 'collector failed')),
                    failure=failure, native_seconds=None)
                transferred += len(json.dumps(failure, separators=(',', ':')).encode())
                write(inputs/(str(seed)+'_'+scene+'_failure.json'), failure)
        if transferred > p['retrieval']['aggregate_input_cap_bytes']:
            raise ValueError('Compact aggregate transfer exceeds the fixed cap')
        if len(received)+len(failures) == len(cells()):
            result = analyze([received.get(key(seed, scene), failures.get(key(seed, scene)))
                              for seed, scene in cells()], p, audit, model)
            encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            if len(encoded.encode()) > p['retrieval']['final_summary_bytes']:
                raise ValueError('Final compact analysis exceeds the declared small-summary limit')
            write(HERE/'summary.json', result)
            write(HERE/'local_summary_receipt.json', dict(transferred_bytes=transferred,
                summary_bytes=len(encoded.encode()), source_node='node001', execution_node='local',
                native_seconds=0, fits=0, optimizer_updates=0, compact_inputs_only=True))
            print(json.dumps(dict(status=result['status'], summary=str(HERE/'summary.json'),
                                  transferred_bytes=transferred)), flush=True)
            return result
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='waiting_for_native_gate_and_collectors',
            current=len(received)+len(failures), total=len(cells()), unit='compact_collectors')), flush=True)
        time.sleep(30)
    raise TimeoutError('The fixed collection did not finish within the six-hour readout wait')


if __name__ == '__main__':
    if sys.argv[1:] != ['local-summary']:
        raise SystemExit('Use local-summary for the existing scheduled readout')
    local_summary()
