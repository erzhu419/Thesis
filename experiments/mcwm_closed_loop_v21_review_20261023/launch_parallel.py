"""Use two reserved CPUs for new main flows; preserve old native evidence."""
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'mcwm_closed_loop_service_shift_v21_20261022'
CELLS=((6202,'stable'),(6201,'local_speed_drop'),(6202,'local_speed_drop'))


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def day_job(seed,scenario):
    out=HERE/'stage_b/runs'/scenario/f'seed{seed}'
    job=infra.task(protocol(),'parallel/'+str(out.relative_to(HERE)),
        ['--mode','day','--seed',seed,'--scenario',scenario,'--out',infra.remote(out)],
        runner='parallel_run.py',ram_mb=24000)
    job['cpu']=2
    job['cpu_training_justification']='Parent clock pauses; two isolated single-thread native action siblings execute simultaneously, then deterministic low/high feedback admission.'
    return job


def branch_job(seed,scenario,source,target,out,*,preflight=False):
    args=['--mode','branch','--seed',seed,'--scenario',scenario,'--source',infra.remote(source),
        '--target-id',target,'--out',infra.remote(out), '--branch',out.name]
    if preflight:args.append('--preflight')
    return infra.task(protocol(),'parallel/'+str(out.relative_to(HERE)),args,
        runner='parallel_run.py',ram_mb=8000)


def submit_parity():
    source=ORIGINAL/'stage_a/runs/stable/seed6201'
    return infra.submit([branch_job(6201,'stable',source,'train/7X/early',HERE/'parity'/arm,
        preflight=True) for arm in ('low','high')],'parity')


def submit_response_days():
    gate=json.loads((HERE/'service_gate.json').read_text())
    parity=json.loads((HERE/'parity_gate.json').read_text())
    if not (gate['integrity']['all_pass'] and gate['scientific_gate']['passed'] and parity['integrity']['all_pass']):
        raise RuntimeError('Service scientific admission and exact parallel reproduction must pass')
    return infra.submit([day_job(seed,scene) for seed,scene in CELLS],'response_days')
