"""Reuse completed v25 stable days; resume its unchanged paired shock runner."""
import argparse
import json
from pathlib import Path
import shlex
import infra

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/'mcwm_closed_loop_fresh_attribution_v25_20261023'


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    files = ['infra.py', 'analyze.py', 'clock_diagnostic.py', 'launch.py', 'protocol.json']
    result = infra.stage([HERE/name for name in files])
    # A new reference, not a copy or modification of the completed stable days.
    code = ('from pathlib import Path;p=Path('+repr(infra.remote(HERE/'runs/stable'))+');'
        'target=Path('+repr(infra.remote(ORIGINAL/'runs/stable'))+');'
        'assert target.is_dir();p.parent.mkdir(parents=True,exist_ok=True);'
        'assert not p.is_symlink() or p.readlink()==target;'
        'p.symlink_to(target,target_is_directory=True) if not p.exists() else None;'
        'print(str(p)+" -> "+str(p.resolve()))')
    rc, output, error = infra.scheduler.run_on('node001', shlex.join([infra.PYTHON,'-c',code]), timeout=45,check=False)
    if rc: raise RuntimeError(error[-1800:])
    result['stable_reuse'] = output.strip()
    return result


def submit_shock():
    gate = infra.collect('stable_gate.json', max_bytes=30000)
    if gate.get('integrity',{}).get('all_pass') is not True:
        raise RuntimeError('Revised stable implementation gate must pass; cost sign is not a gate')
    args = ['--seed',6201,'--scenario','local_speed_drop','--out',
        infra.remote(HERE/'runs/local_speed_drop/seed6201')]
    row = infra.task(protocol(),'factual_pair/local_speed_drop',args,
        runner=ORIGINAL/'run_pair.py',ram_mb=16000)
    row['cpu'] = 2
    row['cpu_training_justification'] = 'Two unchanged v25 single-thread factual arms run concurrently; only output directory is new.'
    return infra.submit([row],'shock_pair')


def submit_finalize(stage):
    output = 'stable_gate.json' if stage=='stable' else 'pilot_summary.json'
    args = ['--root',infra.remote(HERE),'--stage',stage,'--out',infra.remote(HERE/output),'--wait']
    return infra.submit([infra.task(protocol(),stage+'/finalize',args,
        runner='analyze.py',ram_mb=2048)],stage+'_finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','stable-finalize','shock','pilot-finalize'))
    op=parser.parse_args().operation
    value=stage() if op=='stage' else submit_shock() if op=='shock' else submit_finalize('stable' if op=='stable-finalize' else 'pilot')
    print(json.dumps(value,ensure_ascii=False,indent=2))
