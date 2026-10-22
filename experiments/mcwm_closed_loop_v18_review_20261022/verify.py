"""Verify distinct simulation submissions and count resident execution warnings."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_state_response_v18_20261022'
sys.path.insert(0,str(SOURCE))
import infra


def main():
    native=[];all_ids=[]
    batches=('preflight','preflight_pairs','preflight_finalize','diagnostic','paired','diagnostic_finalize')
    for name in batches:
        receipt=json.loads((SOURCE/(name+'_submission.json')).read_text())['submitted']
        jobs=json.loads((SOURCE/(name+'_jobs.json')).read_text())
        all_ids.extend(row['id'] for row in receipt)
        if name.endswith('finalize'):continue
        for row,job in zip(receipt,jobs):
            args=shlex.split(job['cmd'])
            arg=lambda key:args[args.index(key)+1] if key in args else None
            native.append(dict(id=row['id'],signature=job['signature'],priority=job['priority'],
                key=[arg(k) for k in ('--mode','--seed','--scenario','--target-id','--branch')],
                out=arg('--out'),source=arg('--source')))
    native_ids=[row['id'] for row in native]
    def logs(node):
        script='''import json
from pathlib import Path
out=[]
for task in __IDS__:
 p=Path('/tmp')/('sched_'+task+'.log')
 if p.is_file():out.append(dict(id=task,holding_warnings=sum('Failed to apply holding' in line for line in p.open(errors='replace'))))
print(json.dumps(out))
'''.replace('__IDS__',repr(native_ids))
        response=infra.scheduler.run_on(node,shlex.join([infra.PYTHON,'-c',script]),timeout=45,check=True)
        return dict(node=node,logs=json.loads(response[1]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        log_rows=list(pool.map(logs,infra.CPUS))
    seen=[row['id'] for group in log_rows for row in group['logs']]
    status=subprocess.run([sys.executable,str(infra.SCHEDULER),'status','--readonly','--json',
        '--lock-timeout','5','--ids',','.join(all_ids)],capture_output=True,text=True)
    if status.returncode:
        scheduler=dict(available=False,message=status.stderr.strip())
    else:
        rows=json.loads(status.stdout)['tasks']
        scheduler=dict(available=True,count=len(rows),status_counts=dict(Counter(row['status'] for row in rows)),
            priority_counts=dict(Counter(row['priority'] for row in rows)),
            not_done=[{k:row.get(k) for k in ('id','status','node','log_path')} for row in rows if row['status']!='done'])
    result=dict(native_simulations=len(native),unique_signatures=len({r['signature'] for r in native}),
        unique_operations=len({tuple(r['key']) for r in native}),unique_outputs=len({r['out'] for r in native}),
        modes=dict(Counter(r['key'][0] for r in native)),priorities=dict(Counter(r['priority'] for r in native)),
        preflight_main_ids=[row['id'] for row in native if row['key'][0]=='day' and row['key'][1:3]==['6201','stable']],
        first_day_heldout_ids=[row['id'] for row in native if row['key'][0]=='branch' and row['key'][1:3]==['6201','stable']],
        task_logs=len(seen),missing_log_ids=sorted(set(native_ids)-set(seen)),
        repeated_log_ids=[key for key,count in Counter(seen).items() if count>1],
        holding_warning_total=sum(r['holding_warnings'] for g in log_rows for r in g['logs']),
        holding_warning_rows=[dict(node=g['node'],**r) for g in log_rows for r in g['logs'] if r['holding_warnings']],
        log_counts_by_node={g['node']:len(g['logs']) for g in log_rows},scheduler=scheduler)
    script='''import json
from collections import Counter
from pathlib import Path
p=Path(__ROOT__)
protocol=json.loads((p/'protocol.json').read_text())
checks=('full_native_prefix_exact','acquisition_predictions_exact','selected_evidence_exact',
 'source_receipts_exact','source_counters_exact','suffix_zero_training','native_suffix_exact',
 'actual_action_exact','fixed300_truth','no_future_calibration','global_transition_observed',
 'no_heldout_fit','readouts_admitted_pairs_only')
def audit(paths,training=False):
 out=dict(expected=len(paths),found=0,status_counts={},check_pass={k:0 for k in checks},
  check_fail={k:0 for k in checks},prefix_matched_issues=0,missing=[],failed_paths=[],
  execution_counts={},suffix_nonzero=0,suffix_delta_found=0,suffix_delta_missing=0,truth_labels_enter_training=0,logs=0,holding_warnings=0)
 statuses=Counter();executions=Counter()
 for f in paths:
  if not f.is_file():out['missing'].append(str(f.relative_to(p)));continue
  v=json.loads(f.read_text());out['found']+=1;statuses[v.get('status')]+=1
  for k in checks:
   z=v.get('integrity',{}).get('checks',{}).get(k)
   out['check_pass'][k]+=z is True;out['check_fail'][k]+=z is False
  out['prefix_matched_issues']+=v.get('native',{}).get('prefix_matched_issues',0)
  if v.get('integrity',{}).get('all_pass') is not True:out['failed_paths'].append(str(f.relative_to(p)))
  e=v.get('execution',{});executions[str([e.get('requested_hold_seconds'),e.get('executed_hold_seconds')])]+=1
  delta=v.get('suffix',{}).get('training_delta')
  if isinstance(delta,dict) and delta:
   out['suffix_delta_found']+=1;out['suffix_nonzero']+=any(x!=0 for x in delta.values())
  else:out['suffix_delta_missing']+=1
  out['truth_labels_enter_training']+=v.get('budget',{}).get('truth_labels_enter_training',0)
  if training:
   log=f.parent.parent/(f.parent.name+'.log')
   if log.is_file():
    out['logs']+=1;out['holding_warnings']+=sum('Failed to apply holding' in line for line in log.open(errors='replace'))
 out['status_counts']=dict(statuses);out['execution_counts']=dict(executions)
 return out
training=[];heldout=[]
for seed in protocol['training_seeds']:
 for scene in protocol['scenarios']:
  day=p/'runs'/scene/('seed%d'%seed)
  for index in range(protocol['training_pairs_per_day_maximum']):
   for arm in ('low','high'):training.append(day/'training_branches'/str(index)/arm/'summary.json')
  for line in protocol['target_lines']:
   for window in protocol['heldout_windows']:
    for arm in ('low','high'):heldout.append(p/'pairs'/scene/('seed%d'%seed)/line/window['name']/arm/'summary.json')
print(json.dumps(dict(training=audit(training,True),heldout=audit(heldout))))
'''.replace('__ROOT__',repr(infra.remote(SOURCE)))
    response=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',script]),timeout=60,check=True)
    result['native_raw_audit']=json.loads(response[1])
    path=HERE/'live_verification.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
