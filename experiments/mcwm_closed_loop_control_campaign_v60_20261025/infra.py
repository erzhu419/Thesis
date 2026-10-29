"""High-priority actual-control jobs and small resident results only."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    CPUS,PYTHON,SCHEDULER,environment,remote,scheduler)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write('\n')


def stage():
    paths=[*sorted(HERE.glob('*.py')),HERE/'protocol.json']
    scheduler.run_on('node001','mkdir -p '+shlex.quote(remote(HERE)),timeout=45,check=True)
    subprocess.run(['rsync','-a','--timeout=30','-e',scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str,paths),scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],check=True,timeout=90)
    return dict(files=len(paths),uploaded_bytes=sum(path.stat().st_size for path in paths))


def task(label,runner,argv):
    p=json.loads((HERE/'protocol.json').read_text())
    return dict(project='Thesis',description=p['name']+' '+label,signature=p['name']+'/'+label,
        cwd=str(HERE),cmd=shlex.join([PYTHON,'-u',remote(HERE/runner),*map(str,argv)]),
        priority='high',allowed_nodes=CPUS,cpu=1,ram_mb=12000,vram=0,env=environment(),
        allow_cpu_training=True,cpu_training_justification='Actual matched native control '
            'against no_fresh MC-WM. Original285feedback slots/456updates or frozen '
            'budgeted readout deployment. No new trace/entropy/memory diagnostic training.',
        resource_family=p['name']+'/control',ram_resource_family=p['name']+'/control',
        allow_no_ckpt=True,allow_no_resume=True,skip_launch_staging=True,no_auto_requeue=True)


def submit(rows,batch):
    receipt=HERE/(batch+'_submission.json')
    if receipt.exists():return json.loads(receipt.read_text())
    jobs=HERE/(batch+'_jobs.json');attempt=HERE/(batch+'_submission_attempt.json')
    if attempt.exists():raise RuntimeError('Unknown prior submission; inspect receipts before retry: '+batch)
    if jobs.exists():
        if json.loads(jobs.read_text())!=rows:raise RuntimeError('Batch definition changed: '+batch)
    else:write(jobs,rows)
    write(attempt,dict(batch=batch,signatures=[row['signature'] for row in rows]))
    result=subprocess.run([sys.executable,str(SCHEDULER),'submit-jsonl','--file',str(jobs),
        '--trusted','--json','--lock-timeout','90'],capture_output=True,text=True,timeout=180)
    if result.returncode:
        write(HERE/(batch+'_submission_failure.json'),dict(returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr))
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    value=json.loads(result.stdout);write(receipt,value);return value


def presence(relative_paths):
    paths=[(HERE/relative).resolve() for relative in relative_paths]
    for path in paths:path.relative_to(HERE)
    code='from pathlib import Path;import json;print(json.dumps({p:Path(p).exists() for p in '+repr([remote(path) for path in paths])+'}))'
    raw=scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1]
    found=json.loads(raw)
    return {relative:found[remote(path)] for relative,path in zip(relative_paths,paths)}


def collect(relative):
    path=(HERE/relative).resolve();path.relative_to(HERE)
    if path.suffix!='.json':raise ValueError('Only small JSON output may be downloaded')
    if path.exists():return json.loads(path.read_text())
    code='from pathlib import Path;p=Path('+repr(remote(path))+');assert p.stat().st_size<=48000;print(p.read_text(),end="")'
    raw=scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)[1]
    if len(raw.encode())>48000:raise ValueError('Summary exceeds48000bytes')
    value=json.loads(raw);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:handle.write(raw)
    return value


def status(ids):
    result=subprocess.run([sys.executable,str(SCHEDULER),'status','--readonly','--json',
        '--lock-timeout','30','--ids',','.join(ids)],text=True,capture_output=True,timeout=50)
    if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return [{k:row.get(k) for k in ('id','status','node','priority','signature')}
        for row in json.loads(result.stdout)['tasks']]
