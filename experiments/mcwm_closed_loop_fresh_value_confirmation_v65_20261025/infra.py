"""Receipted high v65 independent confirmation; reuse frozen v64 native code."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_fresh_value_v64_20261025'
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
        native_implementation=remote(SOURCE/'control.py'))


def task(label,runner,argv):
    p=json.loads((HERE/'protocol.json').read_text())
    runner=Path(runner)
    if not runner.is_absolute():runner=HERE/runner
    return dict(project='Thesis',description=p['name']+' '+label,signature=p['name']+'/'+label,
        cwd=str(HERE),cmd=shlex.join([PYTHON,'-u',remote(runner),*map(str,argv)]),
        priority='high',allowed_nodes=CPUS,cpu=1,ram_mb=12000,vram=0,env=environment(),
        resource_family=p['name']+'/control',ram_resource_family=p['name']+'/control',
        allow_cpu_training=True,cpu_training_justification='Fixed v64 fresh-quarter V ordinary single-trajectory '
            'dripping-data actual full-day control versus matched no_fresh MC-WM. '
            'Unchanged50000 initializer,285labels456updates,F/trust and v64 target; '
            'predeclared three new v64 seeds, actual paired parallel native days. No counterfactual branches.',
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
        '--trusted','--json','--lock-timeout','90'],capture_output=True,text=True,timeout=180)
    if result.returncode:
        write(HERE/(batch+'_submission_failure.json'),dict(returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr))
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value=json.loads(result.stdout);write(receipt,value);return value


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


def status(ids):
    result=subprocess.run([sys.executable,str(SCHEDULER),'status','--readonly','--json',
        '--lock-timeout','30','--ids',','.join(ids)],text=True,capture_output=True,timeout=50)
    if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [{k:r.get(k) for k in ('id','status','priority','node','signature')}
        for r in json.loads(result.stdout)['tasks']]
