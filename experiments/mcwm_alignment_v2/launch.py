"""Stage code, reuse original initializers, and dispatch independent training/test jobs."""

from alignment_common import HERE, EXPERIMENTS, P, write, label
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import deploy as infra

SCHEDULER = infra.SCHEDULER
PREFIX = P["name"]


def root(node):
    return str(HERE).replace("/home/erzhu419/mine_code", infra.HPC) if node.startswith("node") else str(HERE)


def relay(source_node, directory, files, target_node, destination):
    infra.scheduler.run_on(target_node, "mkdir -p "+shlex.quote(destination), timeout=45, check=True)
    sender = subprocess.Popen(infra.ssh(source_node, "tar -C "+shlex.quote(directory)+" -cf - "+
                                         " ".join(map(shlex.quote,files))), stdout=subprocess.PIPE)
    try:
        receiver = subprocess.run(infra.ssh(target_node,"tar -C "+shlex.quote(destination)+" -xf -"),
                                  stdin=sender.stdout, timeout=180)
    finally:
        sender.stdout.close()
    if sender.wait(timeout=30) or receiver.returncode:
        raise RuntimeError("Server-to-server transfer failed")


def stage():
    original = Path('/home/erzhu419/mine_code/sumo-rl/H2Oplus')
    plan = json.loads((original / 'reproducibility/h2oplus_v14_terminal_taper_training_plan_v1.json').read_text())
    inputs = {j['seed']: j['offline_initializer']['path'] for j in plan['jobs']
              if j['seed'] in P['training_seeds']}
    directory = str(Path(next(iter(inputs.values()))).parent)
    files = [Path(inputs[s]).name for s in P['training_seeds']]
    for node in infra.GPUS:
        infra.sync(node, str(HERE)+'/', root(node), ['--include=*.py','--include=protocol.json','--exclude=*'])
        relay('jtl110gpu', directory, files, node, root(node)+'/inputs')
        mcroot = infra.environment(node)['THESIS_MCWM_ROOT']
        infra.sync(node, original.parents[1] / 'MC-WM/mc_wm/residual/policy_density.py', mcroot+'/mc_wm/residual')
        print('staged '+node, flush=True)


def gate(item):
    return HERE / 'gates' / (label(item['arm'],item['scenario'],item['seed']).replace('/','_')+'.json')


def job(item, *, case=None):
    node = item['node'] if case is None else infra.CPUS[(item['index']*10+case) % len(infra.CPUS)]
    suffix = label(item['arm'],item['scenario'],item['seed'])
    folder = ('smoke/' if item.get('smoke') else 'runs/')+suffix
    cmd = (f"{infra.executable(node)} -u {root(node)}/run.py --arm {item['arm']} "
           f"--scenario {item['scenario']} --seed {item['seed']} --out {root(node)}/{folder}")
    if item.get('smoke'):
        cmd += ' --smoke'
    if case is not None:
        cmd += f' --evaluate --case {case}'
    phase = f'eval{case}' if case is not None else 'train'
    resource_phase = 'eval' if case is not None else 'smoke' if item.get('smoke') else 'full-log-train'
    vram = 2200 if node == 'jtl110gpu' else 1024
    task = dict(project='Thesis', description=f'MC-WM alignment {folder} {phase}',
                signature=f'{PREFIX}/{folder}/{phase}', cwd=str(HERE), cmd=cmd, require_node=node,
                cpu=1, ram_mb=2500 if case is not None else 16000,
                vram=0 if case is not None else vram,
                resource_family=f'{PREFIX}/{item["arm"]}/{resource_phase}',
                ram_resource_family=f'{PREFIX}/{item["arm"]}/{resource_phase}',
                vram_resource_family=f'{PREFIX}/{item["arm"]}/{node}/{resource_phase}',
                env=infra.environment(node, case is not None), skip_launch_staging=True,
                stage_excludes=['runs','smoke','inputs','gates'],
                allow_no_ckpt=True, allow_no_resume=True, allow_gpu_over_one_third=True)
    if case is not None:
        task['wait_for_files'] = [str(gate(item))]
    return task


def submit(tasks, name):
    path = HERE / (name+'_jobs.json')
    write(path,tasks)
    result = json.loads(subprocess.check_output(['python3',SCHEDULER,'submit-jsonl','--file',str(path),
                                                  '--trusted','--json'],text=True))
    write(HERE/(name+'_submission.json'),result)
    ids = [t['id'] for t in result['submitted']]
    print(json.dumps(dict(name=name,submitted=len(ids),ids=ids)),flush=True)
    return ids


def placements():
    group_nodes = ['node007','jtl110gpu','jtl110gpu2','node007','jtl311linux',
                   'node007','jtl110gpu','jtl110gpu2','node007']
    items = []
    for i,seed in enumerate(P['training_seeds']):
        for j,scene in enumerate(P['scenarios']):
            for arm in P['arms']:
                items.append(dict(arm=arm, scenario=scene['name'], seed=seed,
                                  node=group_nodes[i*3+j], index=len(items)))
    return items


def sync_small():
    for node in ['node001','jtl110gpu','jtl110gpu2','jtl311linux']:
        result = subprocess.run(['rsync','-az','--include=*/','--include=*.json','--exclude=*',
                    '-e',infra.scheduler._ssh_rsync_shell_for_node(node),
                    infra.scheduler._ssh_target_for_node(node)+':'+root(node)+'/runs/',str(HERE/'runs')+'/'],
                    capture_output=True,text=True)
        if result.returncode not in (0,23):
            print(json.dumps(dict(collection_error=node, detail=result.stderr[-500:])),flush=True)


def watch():
    items = json.loads((HERE/'locations.json').read_text())
    deadline = time.monotonic()+48*3600
    while time.monotonic()<deadline:
        sync_small()
        for item in items:
            if gate(item).exists():
                continue
            relative = 'runs/'+label(item['arm'],item['scenario'],item['seed'])
            if not (HERE/relative/'results/ready.json').exists():
                continue
            try:
                if not item['node'].startswith('node'):
                    relay(item['node'],root(item['node'])+'/'+relative,['policy.pt'],
                          'node001',root('node001')+'/'+relative)
                else:
                    infra.scheduler.run_on('node001','test -s '+shlex.quote(root('node001')+'/'+relative+'/policy.pt'),
                                           timeout=40,check=True)
            except (RuntimeError,subprocess.SubprocessError) as error:
                print(json.dumps(dict(handoff_pending=relative, detail=str(error)[-500:])),flush=True)
                continue
            write(gate(item),item)
        from collect_results import summarize
        summary = summarize()
        write(HERE/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('complete','training_complete','evaluation_complete')}),flush=True)
        if summary['complete']:
            return
        time.sleep(60)
    raise TimeoutError('MC-WM alignment campaign is incomplete after 48 hours')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=['stage','smoke','submit','watch','collect'])
    args = parser.parse_args()
    if args.operation == 'stage':
        stage()
    elif args.operation == 'smoke':
        items = [dict(arm=arm,scenario=scene,seed=6201,node=node,index=i,smoke=True)
                 for i,(node,arm,scene) in enumerate([
                     ('node007','mcwm','calibrated'),('jtl110gpu','mcwm','joint25'),
                     ('jtl110gpu2','mcwm','joint50'),('jtl311linux','h2o_matched','calibrated')])]
        write(HERE/'smoke_locations.json',items)
        submit([job(item) for item in items],'smoke')
    elif args.operation == 'submit':
        items = placements()
        write(HERE/'locations.json',items)
        tasks = []
        for item in items:
            tasks.append(job(item))
            tasks.extend(job(item,case=case) for case in range(10))
        submit(tasks,'campaign')
        submit([dict(project='Thesis',description='MC-WM alignment policy handoff and compact result collection',
                     signature=PREFIX+'/watcher',cwd=str(HERE),cmd=f'python3 -u {HERE}/launch.py watch',
                     require_node='local',cpu=1,ram_mb=512,vram=0,allow_no_ckpt=True,allow_no_resume=True,
                     skip_launch_staging=True)],'watcher')
    elif args.operation == 'watch':
        watch()
    else:
        sync_small()
        from collect_results import summarize
        summary=summarize();write(HERE/'summary.json',summary);print(json.dumps(summary,indent=2))
