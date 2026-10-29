"""Finish only the five unrun v19 branches after auditing the retained first arm."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'mcwm_closed_loop_execution_phase_v19_20261022'
sys.path.insert(0, str(OLD))
import infra


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    infra.write(path, value)


def protocol():
    return json.loads((OLD/'protocol.json').read_text())


def stage():
    files = [HERE/name for name in ('clock_policy.py', 'run_remaining.py', 'audit_first.py',
                                   'finalize.py', 'protocol.json')]
    infra.scheduler.run_on('node001', 'mkdir -p '+shlex.quote(infra.remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', infra.scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, files), infra.scheduler._ssh_target_for_node('node001')+':'+infra.remote(HERE)+'/'],
        check=True, timeout=90)
    return dict(files=len(files), uploaded_bytes=sum(p.stat().st_size for p in files))


def source_root():
    return HERE.parent/protocol()['source_version']


def audit():
    argv = [infra.PYTHON, infra.remote(HERE/'audit_first.py'), '--root', infra.remote(OLD),
            '--source', infra.remote(source_root()), '--out', infra.remote(HERE/'first_arm_clock_audit.json')]
    result = infra.scheduler.run_on('node001', shlex.join(argv), timeout=45, check=True)
    return result[1]


def collect(name, limit):
    argv = [infra.PYTHON, '-c', 'from pathlib import Path; p=Path('+repr(infra.remote(HERE/name))+
        '); assert p.stat().st_size<='+str(limit)+'; print(p.read_text(),end="")']
    text = infra.scheduler.run_on('node001', shlex.join(argv), timeout=45, check=True)[1]
    value = json.loads(text)
    write(HERE/name, value)
    return value


def submit(rows, batch):
    output = HERE/(batch+'_submission.json')
    if output.exists():
        raise FileExistsError(output)
    jobs = HERE/(batch+'_jobs.json')
    write(jobs, rows)
    value = json.loads(subprocess.check_output([sys.executable, str(infra.SCHEDULER), 'submit-jsonl',
        '--file', str(jobs), '--trusted', '--json'], text=True))
    write(output, value)
    return value


def task(label, runner, args, ram):
    row = infra.task(protocol(), 'clock_audit/'+label, (), ram_mb=ram)
    row.update(cwd=str(HERE), cmd=shlex.join([infra.PYTHON, '-u', infra.remote(HERE/runner), *args]))
    return row


def remaining():
    evidence = json.loads((HERE/'first_arm_clock_audit.json').read_text())
    if evidence.get('reusable') is not True:
        raise RuntimeError('Native-event clock audit has not passed')
    rows = []
    for line in ('7X', '102X', '705X'):
        for arm in ('low', 'high'):
            if (line, arm) == ('7X', 'low'):
                continue
            args = ['--mode', 'branch', '--seed', '6201', '--scenario', 'stable',
                '--source', infra.remote(source_root()/'runs/stable/seed6201'),
                '--target-id', 'train/'+line+'/late', '--branch', arm,
                '--out', infra.remote(HERE/'branches'/line/arm)]
            rows.append(task(line+'/'+arm, 'run_remaining.py', args, 8000))
    return submit(rows, 'remaining_five')


def finalize():
    json.loads((HERE/'remaining_five_submission.json').read_text())
    args = ['--source', infra.remote(source_root()), '--root', infra.remote(HERE),
        '--original', infra.remote(OLD), '--out', infra.remote(HERE/'intervention_summary.json'), '--wait']
    return submit([task('finalize', 'finalize.py', args, 1024)], 'finalize')


def metadata():
    script = '''import json
from pathlib import Path
r=Path(ROOT)
out={}
for line in ('7X','102X','705X'):
 for arm in ('low','high'):
  if (line,arm)==('7X','low'):continue
  p=r/'branches'/line/arm
  q=p/'summary.json'
  if q.exists():
   d=json.loads(q.read_text());out[line+'/'+arm]={'status':d['status'],'failed_checks':[k for k,v in d['integrity']['checks'].items() if not v],'budget':d['budget']}
  elif (p/'failure.json').exists():out[line+'/'+arm]=json.loads((p/'failure.json').read_text())
  else:out[line+'/'+arm]={'status':'pending'}
q=r/'intervention_summary.json'
out['summary']={'exists':q.exists(),'bytes':q.stat().st_size if q.exists() else 0}
print(json.dumps(out))
'''.replace('ROOT', repr(infra.remote(HERE)))
    return json.loads(infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', script]),
                                           timeout=45, check=True)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'audit', 'collect-audit', 'remaining', 'finalize', 'metadata', 'collect'))
    operation = parser.parse_args().operation
    if operation == 'stage':
        p = protocol()
        p['intervention']['first_branch']['output_relative'] = infra.remote(OLD/'branches_retest/7X/low')
        write(HERE/'protocol.json', p)
        value = stage()
    elif operation == 'audit': value = audit()
    elif operation == 'collect-audit': value = collect('first_arm_clock_audit.json', 10000)
    elif operation == 'remaining': value = remaining()
    elif operation == 'finalize': value = finalize()
    elif operation == 'metadata': value = metadata()
    else: value = collect('intervention_summary.json', 30000)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
