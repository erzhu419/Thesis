"""Isolated native-fork and exact bus equivalence probes, all high priority."""
import argparse
import json
from pathlib import Path
import shlex
import infra

HERE=Path(__file__).resolve().parent
NATIVE_RESULT='native_probe_retest.json'


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def native_job():
    cfg=Path(infra.environment()['THESIS_SUMO_ROOT'])/'online_control/control_sim_traci_period.sumocfg'
    row=infra.task(protocol(),'native_fork_probe',['--sumo-cfg',str(cfg),
        '--out',infra.remote(HERE/NATIVE_RESULT)],runner='native_fork_probe.py',ram_mb=8000)
    row['cpu']=2
    return row


def prepare():
    old=HERE.parent/'mcwm_closed_loop_stratified_pairs_v23_20261023/stage_a/runs/stable/seed6201'
    args=[infra.PYTHON,infra.remote(HERE/'prepare_source.py'),'--source',infra.remote(old),
          '--out',infra.remote(HERE/'source_anchor')]
    rc,out,err=infra.scheduler.run_on('node001',shlex.join(args),timeout=45,check=False)
    if rc:raise RuntimeError(err[-2500:])
    return json.loads(out)


def submit_bus():
    gate=infra.collect(NATIVE_RESULT,max_bytes=10000)
    if gate.get('all_pass') is not True:
        raise RuntimeError('Native state/torch fork probe must pass first')
    row=infra.task(protocol(),'bus_fork_equivalence',['--mode','probe','--seed',6201,
        '--scenario','stable','--source',infra.remote(HERE/'source_anchor'),
        '--target-id','train/7X/bootstrap','--branch','low','--preflight',
        '--out',infra.remote(HERE/'bus_probe')],runner='probe_run.py',ram_mb=24000)
    row['cpu']=2
    return infra.submit([row],'bus_probe')


def submit_compare():
    row=infra.task(protocol(),'bus_fork_equivalence_finalize',['--probe',infra.remote(HERE/'bus_probe'),
        '--source',infra.remote(HERE/'source_anchor'),'--target-id','train/7X/bootstrap',
        '--out',infra.remote(HERE/'equivalence_gate.json'),'--wait'],runner='compare.py',ram_mb=2048)
    return infra.submit([row],'equivalence_finalize')


def submit_day():
    gate=infra.collect('equivalence_gate.json',max_bytes=20000)
    if gate.get('integrity',{}).get('all_pass') is not True:
        raise RuntimeError('Exact bus fork equivalence must pass before whole-day execution benchmark')
    row=infra.task(protocol(),'whole_day_execution_benchmark',['--mode','day','--seed',6201,
        '--scenario','stable','--preflight','--out',infra.remote(HERE/'benchmark_day')],
        runner='fast_day.py',ram_mb=24000)
    row['cpu']=2
    row['allowed_nodes']=[node for node in row['allowed_nodes'] if node!='node004']
    return infra.submit([row],'day_benchmark')


def submit_day_compare():
    original=HERE.parent/'mcwm_closed_loop_stratified_pairs_v23_20261023/stage_a/runs/stable/seed6201'
    row=infra.task(protocol(),'whole_day_execution_equivalence',['--fast',infra.remote(HERE/'benchmark_day'),
        '--source',infra.remote(original),'--out',infra.remote(HERE/'day_equivalence_gate.json'),'--wait'],
        runner='day_compare.py',ram_mb=2048)
    return infra.submit([row],'day_equivalence_finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','native','prepare','bus','finalize','day',
        'day-finalize','collect-native','collect','collect-day'))
    args=parser.parse_args()
    if args.operation=='stage':
        names=('bootstrap.py','infra.py','history.py','candidate.py','shift_model.py','sampling.py',
            'fork_state.py','native_fork_probe.py','probe_run.py','prepare_source.py','compare.py',
            'fast_day.py','day_compare.py','launch.py','protocol.json')
        result=infra.stage([HERE/name for name in names])
    elif args.operation=='native':result=infra.submit([native_job()],'native_probe_retest')
    elif args.operation=='prepare':result=prepare()
    elif args.operation=='bus':result=submit_bus()
    elif args.operation=='finalize':result=submit_compare()
    elif args.operation=='day':result=submit_day()
    elif args.operation=='day-finalize':result=submit_day_compare()
    elif args.operation=='collect-native':result=infra.collect(NATIVE_RESULT,max_bytes=10000)
    elif args.operation=='collect-day':result=infra.collect('day_equivalence_gate.json',max_bytes=20000)
    else:result=infra.collect('equivalence_gate.json',max_bytes=20000)
    print(json.dumps(result,ensure_ascii=False,indent=2))
