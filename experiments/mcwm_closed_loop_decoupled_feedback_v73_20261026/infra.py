"""Receipted high v73 decoupled-feedback trial; small JSON only."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_recent30_mixed_cql_v70_20261026'
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS,PYTHON,SCHEDULER,environment,remote,scheduler)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def stage():
    paths=[*sorted(HERE.glob('*.py')),HERE/'protocol.json']
    scheduler.run_on('node001','mkdir -p '+shlex.quote(remote(HERE)),timeout=45,check=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str,paths),scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],timeout=90,check=True)
    return dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths),
        reference_implementation_unchanged=remote(SOURCE/'control.py'),
        native_implementation=remote(HERE/'control.py'))


def task(label,runner,argv):
    p=json.loads((HERE/'protocol.json').read_text())
    runner=Path(runner)
    if not runner.is_absolute():runner=HERE/runner
    return dict(project='Thesis',description=p['name']+' '+label,signature=p['name']+'/'+label,
        cwd=str(HERE),cmd=shlex.join([PYTHON,'-u',remote(runner),*map(str,argv)]),
        priority='high',allowed_nodes=CPUS,cpu=1,ram_mb=12000,vram=0,env=environment(),
        resource_family=p['name']+'/control',ram_resource_family=p['name']+'/control',
        allow_cpu_training=True,cpu_training_justification='Separate latest5 state/action covariates from unchanged risk5 complete outcomes. '
            'Fixed recent30 outcomes, original mixed .1V+.9SAC targets and executed-action CQL0.2. '
            'Four fresh full days plus one exact original no_fresh full-day regression; four cached v70 no_fresh references. '
            'Unchanged50000 initializer,285 complete outcomes and285 covariates with union at most342,456critic/V/actor/alpha updates,F/trust. '
            'First fresh/regression pair reuses full days and runs in parallel; after full baseline reproduction the remaining three fresh days may run in parallel. No branches.',
        allow_no_ckpt=True,allow_no_resume=True,skip_launch_staging=True,no_auto_requeue=True)


def submit(rows,batch):
    receipt=HERE/(batch+'_submission.json')
    if receipt.exists():return json.loads(receipt.read_text())
    jobs=HERE/(batch+'_jobs.json');attempt=HERE/(batch+'_submission_attempt.json')
    if attempt.exists():raise RuntimeError('Unresolved submission; inspect existing task signatures: '+batch)
    if jobs.exists():
        if json.loads(jobs.read_text())!=rows:raise RuntimeError('Fixed batch changed: '+batch)
    else:write(jobs,rows)
    write(attempt,dict(batch=batch,signatures=[r['signature'] for r in rows]))
    result=subprocess.run([sys.executable,str(SCHEDULER),'submit-jsonl','--file',str(jobs),
        '--trusted','--json','--lock-timeout','300'],capture_output=True,text=True,timeout=360)
    if result.returncode:
        write(HERE/(batch+'_submission_failure.json'),dict(returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr))
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value=json.loads(result.stdout);write(receipt,value)
    dispatch=subprocess.run([sys.executable,str(SCHEDULER),'dispatch','--no-bulk-window','--lock-timeout','300',
        *[arg for row in value['submitted'] for arg in ('--task-id',row['id'])]],
        capture_output=True,text=True,timeout=360)
    write(HERE/(batch+'_dispatch.json'),dict(returncode=dispatch.returncode,
        stdout=dispatch.stdout[-16000:],stderr=dispatch.stderr[-2000:]))
    if dispatch.returncode:raise RuntimeError('Known submitted tasks need dispatch review; do not resubmit')
    return value


def presence(relative_paths):
    paths=[(HERE/p).resolve() for p in relative_paths]
    for p in paths:p.relative_to(HERE)
    code='from pathlib import Path;import json;print(json.dumps({p:Path(p).exists() for p in '+repr([remote(p) for p in paths])+'}))'
    value=json.loads(scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1])
    return {name:value[remote(path)] for name,path in zip(relative_paths,paths)}


def collect(relative):
    path=(HERE/relative).resolve();path.relative_to(HERE)
    if path.suffix!='.json':raise ValueError('Small JSON results only')
    if path.exists():return json.loads(path.read_text())
    code='from pathlib import Path;import json;p=Path('+repr(remote(path))+');raw=json.dumps(json.loads(p.read_text()),ensure_ascii=False,separators=(",",":"),allow_nan=False);assert len(raw.encode())<=48000;print(raw,end="")'
    raw=scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1]
    if len(raw.encode())>48000:raise ValueError('Received result exceeds48KB')
    value=json.loads(raw);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(raw)
    return value


def common_prefix(rows):
    """Compare the captured pre-update rows on the server; return no trace data."""
    cached=HERE/'native_prefix_gate.json'
    if cached.exists():return json.loads(cached.read_text())
    prefixes=[row['prefix_before_first_update'] for row in rows]
    code='''from pathlib import Path
import json
prefixes = PREFIXES
checks = {"same_rows": prefixes[0]["rows"] == prefixes[1]["rows"],
    "same_clock": prefixes[0]["clock"] == prefixes[1]["clock"],
    "same_initial_updates": prefixes[0]["initial_updates"] == prefixes[1]["initial_updates"] == 50000}
compared = 0
difference = None
if all(checks.values()):
    with Path(prefixes[0]["trace_path"]).open() as left, Path(prefixes[1]["trace_path"]).open() as right:
        for index in range(prefixes[0]["rows"]):
            a, b = left.readline(), right.readline()
            if not a or not b:
                difference = {"row": index, "reason": "captured_prefix_missing"}
                break
            a, b = json.loads(a), json.loads(b)
            if a != b:
                difference = {"row": index, "fields": sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}
                break
            compared += 1
checks["full_pre_update_prefix_exact"] = compared == prefixes[0]["rows"] and compared > 0 and difference is None
print(json.dumps({"all_pass": all(checks.values()), "checks": checks,
    "rows_compared": compared, "cut_clock": prefixes[0]["clock"], "difference": difference}, allow_nan=False))
'''.replace('PREFIXES',repr(prefixes))
    value=json.loads(scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1])
    write(cached,value)
    return value


def status(ids):
    result=subprocess.run([sys.executable,str(SCHEDULER),'status','--readonly','--json',
        '--lock-timeout','30','--ids',','.join(ids)],text=True,capture_output=True,timeout=50)
    if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [{k:r.get(k) for k in ('id','status','priority','node','signature')}
        for r in json.loads(result.stdout)['tasks']]
