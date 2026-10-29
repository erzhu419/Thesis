"""Receipted high v75 frozen-v73 validation; small JSON only."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_decoupled_feedback_v73_20261026'
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


def stage_gpu():
    execution=json.loads((HERE/'protocol.json').read_text())['initialization_execution']
    node,root=execution['node'],execution['worker_root']
    scheduler.run_on(node,'mkdir -p '+shlex.quote(root),timeout=45,check=True)
    files=[HERE/'initialize.py',HERE/'protocol.json']
    subprocess.run(['rsync','-a','--timeout=30','-e',scheduler._ssh_rsync_shell_for_node(node),
        *map(str,files),scheduler._ssh_target_for_node(node)+':'+root+'/'],timeout=90,check=True)
    return dict(node=node,files=len(files),bytes=sum(p.stat().st_size for p in files),destination=root)


def initializer_jobs():
    p=json.loads((HERE/'protocol.json').read_text());e=p['initialization_execution']
    result=[]
    for seed in p['development_seeds']:
        root=Path(e['worker_root'])
        env=dict(THESIS_H2O_ROOT=e['h2o_root'],THESIS_DATASET=e['dataset'],
            OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',
            PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
            CC='/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc',
            CXX='/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc',
            CPATH='/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include',
            TORCHINDUCTOR_CACHE_DIR=str(root/f'compile_cache/seed{seed}'))
        result.append(dict(project='Thesis',description=p['name']+f' original IQL initializer seed{seed}',
            signature=p['name']+f'/initializer/seed{seed}',cwd=str(HERE),
            cmd=shlex.join([e['python'],'-u',str(root/'initialize.py'),'--seed',str(seed),
                '--device','cuda:0','--inputs-dir',str(root/'inputs'),
                '--dataset',e['dataset'],'--template',e['template']]),
            priority='high',allowed_nodes=[e['node']],cpu=e['cpu'],ram_mb=e['ram_mb'],
            vram=e['vram_mb'],env=env,allow_gpu_over_one_third=True,
            resource_family=p['name']+'/initializer',ram_resource_family=p['name']+'/initializer',
            vram_resource_family=p['name']+'/initializer',
            ckpt_dir=str(root/f'inputs/seed{seed}'),allow_no_resume=True,
            skip_launch_staging=True,no_auto_requeue=True))
    return result


def dispatch_pending_initializers(queued):
    """Three full-cache initializers share two cards; reuse queued IDs after capacity frees."""
    import time
    ids=[row['id'] for row in queued if row['status']=='queued']
    if not ids:return dict(dispatched=[],reason='no_queued_initializers')
    if not any((HERE/f'inputs/seed{seed}/initializer_gate.json').exists() for seed in (6206,6207,6208)):
        return dict(dispatched=[],pending=ids,reason='waiting_first_gpu_initializer_completion')
    record=HERE/'initializers_queued_dispatch.json'
    previous=record if record.exists() else HERE/'initializers_dispatch.json'
    if previous.exists() and time.time()-previous.stat().st_mtime<120:
        return dict(dispatched=[],pending=ids,reason='waiting_dispatch_interval')
    value=dict(ids=ids,started_unix=time.time())
    record.write_text(json.dumps(value)+'\n')
    result=subprocess.run([sys.executable,str(SCHEDULER),'dispatch','--no-bulk-window',
        '--lock-timeout','300',*[arg for key in ids for arg in ('--task-id',key)]],
        capture_output=True,text=True,timeout=360)
    value.update(returncode=result.returncode,finished_unix=time.time(),
        stdout=result.stdout[-4000:],stderr=result.stderr[-2000:])
    record.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    if result.returncode:raise RuntimeError('Known queued initializer needs dispatch review; no resubmission')
    return dict(dispatched=ids,returncode=0)


def promote_initializers(seeds):
    """Move completed artifacts directly GPU->HPC; publish the small ready gate last."""
    p=json.loads((HERE/'protocol.json').read_text());e=p['initialization_execution']
    markers=[f'inputs/seed{seed}/initializer_gate.json' for seed in seeds]
    on_hpc=presence(markers)
    needed=[seed for seed,path in zip(seeds,markers) if not on_hpc[path]]
    if not needed:return dict(published=list(seeds),pending=[])
    roots={str(seed):str(Path(e['worker_root'])/f'inputs/seed{seed}/initializer_gate.json') for seed in needed}
    code='''from pathlib import Path
import json
result={}
for seed,name in ROOTS.items():
 p=Path(name)
 if p.exists():
  if p.stat().st_size>48000:raise ValueError('Initializer gate exceeds export budget')
  result[seed]=json.loads(p.read_text())
print(json.dumps(result))
'''.replace('ROOTS',repr(roots))
    gates=json.loads(scheduler.run_on(e['node'],shlex.join([e['python'],'-c',code]),timeout=45,check=True)[1])
    published=[]
    for seed_text,original in gates.items():
        seed=int(seed_text);gate=dict(original)
        destination=Path(remote(HERE/'inputs'))
        if original.get('all_pass') is True:
            if original.get('seed')!=seed or original.get('offline_updates')!=50000:
                raise ValueError('Completed GPU initializer differs from the fixed seed/update budget')
            scheduler.run_on('node001','mkdir -p '+shlex.quote(str(destination/f'seed{seed}')),timeout=45,check=True)
            target=scheduler._ssh_target_for_node('node001')
            for source,folder in [(original['initializer_path'],destination),
                    (original['recipe_path'],destination/f'seed{seed}')]:
                command=['rsync','-a','--timeout=60','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
                    source,target+':'+str(folder)+'/']
                scheduler.run_on(e['node'],shlex.join(command),timeout=180,check=True)
            gate.update(training_artifact_path=original['initializer_path'],
                initializer_path=str(destination/f'h2oplus_seed{seed}.pt'),
                training_recipe_path=original['recipe_path'],recipe_path=str(destination/f'seed{seed}/recipe.json'),
                publication='GPU directly to HPC; no local checkpoint bytes')
        out=destination/f'seed{seed}/initializer_gate.json'
        code='from pathlib import Path;import json;p=Path('+repr(str(out))+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps('+repr(gate)+',allow_nan=False)+"\\n")'
        scheduler.run_on('node001',shlex.join([PYTHON,'-c',code]),timeout=45,check=True)
        marker=HERE/f'initializer_seed{seed}_publication.json'
        if not marker.exists():write(marker,dict(seed=seed,all_pass=gate['all_pass'],
            source_node=e['node'],destination=str(out),local_checkpoint_bytes=0))
        published.append(seed)
    return dict(published=published,pending=[seed for seed in needed if str(seed) not in gates])


def task(label,runner,argv):
    p=json.loads((HERE/'protocol.json').read_text())
    runner=Path(runner)
    if not runner.is_absolute():runner=HERE/runner
    return dict(project='Thesis',description=p['name']+' '+label,signature=p['name']+'/'+label,
        cwd=str(HERE),cmd=shlex.join([PYTHON,'-u',remote(runner),*map(str,argv)]),
        priority='high',allowed_nodes=CPUS,cpu=1,ram_mb=12000,vram=0,env=environment(),
        resource_family=p['name']+'/control',ram_resource_family=p['name']+'/control',
        allow_cpu_training=True,cpu_training_justification='Frozen v73 native validation on three new training seeds. '
            'Twelve matched full days, paired arms execute concurrently; original single-thread CPU SUMO and456updates. '
            'Same seed initializer and F/trust;285covariates,285complete labels only in fresh,actual union<=342. '
            'All150000offline initializer steps are separate GPU tasks, not repeated here; no extra native branches.',
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


def common_prefix(rows, cache_key='native_prefix_gate.json'):
    """Compare the captured pre-update rows on the server; return no trace data."""
    cached=HERE/cache_key
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
