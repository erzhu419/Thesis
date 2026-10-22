"""Reuse the full native preflight cell; then add three days and36 branches."""
import argparse
from itertools import product
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
RESULT_NAME = 'learning_summary.json'
SUMMARY_MAX_BYTES = 150000
GATE_MAX_BYTES = 20000


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def cells(*, preflight=False):
    p=protocol()
    if preflight:
        return [dict(seed=p['preflight']['seed'],scenario=p['preflight']['scenario'])]
    return [dict(seed=seed,scenario=scene) for seed,scene in product(p['training_seeds'],p['scenarios'])]


def day_path(cell, *, preflight=False):
    return Path('runs')/cell['scenario']/f"seed{cell['seed']}"


def remaining_cells():
    first=cells(preflight=True)[0]
    return [cell for cell in cells() if cell!=first]


def evaluation_cells(*, preflight=False):
    p=protocol()
    return [dict(**cell,line=line,window=window['name']) for cell in cells(preflight=preflight)
            for line,window in product(p['target_lines'],p['heldout_windows'])]


def remaining_evaluation_cells():
    first=cells(preflight=True)[0]
    return [cell for cell in evaluation_cells() if any(cell[k]!=first[k] for k in first)]


def evaluation_path(cell,arm,*,preflight=False):
    return Path('pairs')/cell['scenario']/f"seed{cell['seed']}"/cell['line']/cell['window']/arm


def source_ready():
    p=protocol()
    keys=('initializer_updates','history_rows','source_rows','horizon_seconds','target_record_budget',
          'policy_update_budget','model_init_updates','model_refit_updates','feedback_delay_seconds')
    reference=HERE.parent/'mcwm_closed_loop_fresh_difference_v17_20261021/protocol.json'
    script='''import json
from pathlib import Path
p=json.loads(Path(__REF__).read_text())
for key,value in __EXPECTED__.items():
 assert p[key]==value,key
q=Path(__INPUTS__)
for seed in __SEEDS__:
 assert (q/('h2oplus_seed%d.pt'%seed)).is_file()
for name in ('history.npz','history_metadata.json','source_9101.npz'):
 assert (q/name).is_file()
assert len(json.loads(Path(__PHYSICAL__).read_text()))==102
assert p['target_seed_offset'] != __NEW_OFFSET__, 'New target prefixes must use new external seeds'
assert p['training_seeds'] == __SEEDS__, 'Initializer seeds unchanged'
print(json.dumps(dict(initializers=len(__SEEDS__),source_intervals=102,initialization_and_budget_equal=True,target_seeds=[s+__NEW_OFFSET__ for s in __SEEDS__])))
'''.replace('__REF__',repr(infra.remote(reference))).replace('__EXPECTED__',repr({k:p[k] for k in keys})).replace('__INPUTS__',repr(str(infra.RESIDENT_INPUTS))).replace('__SEEDS__',repr(p['training_seeds'])).replace('__NEW_OFFSET__',repr(p['target_seed_offset'])).replace('__PHYSICAL__',repr(str(infra.RESIDENT_SOURCE/'source_intervals.json')))
    return json.loads(infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',script]),timeout=45,check=True)[1])


def day_job(cell,*,preflight=False):
    path=day_path(cell,preflight=preflight)
    args=['--mode','day','--seed',str(cell['seed']),'--scenario',cell['scenario'],'--out',infra.remote(HERE/path)]
    if preflight:
        args.append('--preflight')
    return infra.task(protocol(),str(path),args,ram_mb=16000)


def evaluation_job(cell,arm,*,preflight=False):
    path=evaluation_path(cell,arm,preflight=preflight)
    source=day_path(cell,preflight=preflight)
    args=['--mode','branch','--seed',str(cell['seed']),'--scenario',cell['scenario'],
          '--source',infra.remote(HERE/source),'--target-id',f"heldout/{cell['line']}/{cell['window']}",
          '--branch',arm,'--out',infra.remote(HERE/path)]
    if preflight:
        args.append('--preflight')
    return infra.task(protocol(),str(path),args,ram_mb=8000)


def require_gate(relative,*,coverage=False):
    path=HERE/relative
    v=json.loads(path.read_text()) if path.is_file() else infra.collect_json(str(relative),max_bytes=GATE_MAX_BYTES)
    if v.get('status')!='complete' or v.get('integrity',{}).get('all_pass') is not True:
        raise RuntimeError('Native budgeted paired gate must pass: '+str(relative))
    if coverage and v.get('design_coverage',{}).get('all_pass') is not True:
        raise RuntimeError('Three-line training and two-window coverage must pass: '+str(relative))
    return v


def submit_preflight():
    source_ready()
    return infra.submit([day_job(cells(preflight=True)[0],preflight=True)],'preflight')


def days_status(*,preflight=False):
    paths=[str(day_path(cell,preflight=preflight)) for cell in cells(preflight=preflight)]
    script='''import json
from pathlib import Path
root=Path(__HERE__)
out=[]
for relative in __PATHS__:
 q=root/relative
 if (q/'failure.json').is_file():out.append(dict(path=relative,status='failed'));continue
 f=q/'summary.json'
 if not f.is_file():out.append(dict(path=relative,status='pending'));continue
 v=json.loads(f.read_text())
 out.append(dict(path=relative,status=v.get('status'),all_pass=v.get('integrity',{}).get('all_pass'),coverage=v.get('design_coverage',{}).get('all_pass')))
print(json.dumps(out))
'''.replace('__HERE__',repr(infra.remote(HERE))).replace('__PATHS__',repr(paths))
    return json.loads(infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',script]),timeout=45,check=True)[1])


def submit_evaluations(*,preflight=False):
    if preflight:
        require_gate(day_path(cells(preflight=True)[0])/'gate.json',coverage=True)
        targets=evaluation_cells(preflight=True)
    else:
        require_gate(Path('preflight_gate.json'))
        rows=days_status()
        if len(rows)!=len(cells()) or not all(row.get('status')=='complete' and row.get('all_pass') is True and row.get('coverage') is True for row in rows):
            raise RuntimeError('All four native days must pass integrity and coverage before remaining validation')
        targets=remaining_evaluation_cells()
    jobs=[evaluation_job(cell,arm,preflight=preflight) for cell in targets for arm in ('low','high')]
    return infra.submit(jobs,'preflight_pairs' if preflight else 'paired')


def submit_finalizer(*,preflight=False):
    batch='preflight_pairs' if preflight else 'paired'
    json.loads((HERE/(batch+'_submission.json')).read_text())
    root=HERE
    out=root/'preflight_gate.json' if preflight else root/RESULT_NAME
    args=['--root',infra.remote(root),'--out',infra.remote(out),'--wait']
    if preflight:args.append('--preflight')
    return infra.submit([infra.task(protocol(),'preflight/finalize' if preflight else 'diagnostic/finalize',
                                    args,runner='analyze.py',ram_mb=1024)],
                        'preflight_finalize' if preflight else 'diagnostic_finalize')


def submit_days():
    require_gate(Path('preflight_gate.json'))
    return infra.submit([day_job(cell) for cell in remaining_cells()],'diagnostic')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','preflight','preflight-pairs','preflight-finalize',
                                           'days','pairs','finalize','status','collect'))
    parser.add_argument('--batch',default='diagnostic')
    args=parser.parse_args()
    if args.operation=='stage':
        names=('bootstrap.py','budget.py','difference.py','candidate.py','run.py','analyze.py',
               'infra.py','launch.py','continue_batch.py','protocol.json')
        value=infra.stage([HERE/name for name in names])
    elif args.operation=='preflight':value=submit_preflight()
    elif args.operation=='preflight-pairs':value=submit_evaluations(preflight=True)
    elif args.operation=='preflight-finalize':value=submit_finalizer(preflight=True)
    elif args.operation=='days':value=submit_days()
    elif args.operation=='pairs':value=submit_evaluations()
    elif args.operation=='finalize':value=submit_finalizer()
    elif args.operation=='status':
        rows=json.loads((HERE/(args.batch+'_submission.json')).read_text())['submitted']
        value=infra.status([row['id'] for row in rows])
    else:
        v=infra.collect_json(RESULT_NAME,max_bytes=SUMMARY_MAX_BYTES)
        value=dict(status=v['status'],integrity=v['integrity'],bytes=(HERE/RESULT_NAME).stat().st_size)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
