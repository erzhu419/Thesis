"""Read resident v16 evidence; return only counts needed for this review."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_budgeted_difference_v16_20261021'
sys.path.insert(0, str(SOURCE))
import infra


def resident(script, node='node001'):
    result = infra.scheduler.run_on(node, shlex.join([infra.PYTHON, '-c', script]),
                                    timeout=90, check=True)
    return json.loads(result[1])


def main():
    receipts = [json.loads((SOURCE/(name+'_submission.json')).read_text())
                for name in ('diagnostic', 'paired', 'diagnostic_finalize')]
    tasks = infra.status([row['id'] for receipt in receipts for row in receipt['submitted']])
    script = '''import json,sys
from pathlib import Path
from collections import Counter
root=Path(__ROOT__)
sys.path.insert(0,str(root.parents[1]))
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration
p=json.loads((root/'protocol.json').read_text())
contract=json.loads((root/p['reference_inputs']/'history_metadata.json').read_text())['contract']
cal=ControlledCalibration.__new__(ControlledCalibration);cal.transform=ExecutionAction(contract)
out=[];children=[];child_logs=[]
for seed in p['training_seeds']:
 for scene in p['scenarios']:
  day=root/'runs'/scene/('seed%d'%seed)
  summary=json.loads((day/'summary.json').read_text())
  stop=summary['budget']['training_stopped_at']
  byline={line:dict(all_issues=0,last_issue=None,station_times=[],joint_times=[]) for line in p['target_lines']}
  with (day/'prefix_trace.jsonl').open() as f:
   for text in f:
    row=json.loads(text);line=row['line_id']
    if line not in byline:continue
    b=byline[line];t=row['started_at'];b['all_issues']+=1;b['last_issue']=t
    if row['station_id']!='7X13_102X26_705X27':continue
    b['station_times'].append(t)
    if all(cal._action_for_hold(row['s'],hold) is not None for hold in (0.,7.5)):
     b['joint_times'].append(t)
  coverage=[]
  for line,b in byline.items():
   times=b.pop('station_times');joint=b.pop('joint_times')
   windows=[dict(name='after_training',start=stop,end=23700),*p['heldout_windows']]
   coverage.append(dict(line=line,**b,station_issues=len(times),station_last=max(times) if times else None,
    joint_issues=len(joint),joint_last=max(joint) if joint else None,
    windows=[dict(name=w['name'],issues=sum(w['start']<=t<w['end'] for t in times),
     joint=sum(w['start']<=t<w['end'] for t in joint),times=[t for t in times if w['start']<=t<w['end']]) for w in windows]))
  out.append(dict(seed=seed,scenario=scene,training_stop=stop,coverage=coverage))
  children.extend(day.glob('training_branches/*/*/summary.json'))
  child_logs.extend(day.glob('training_branches/*/*.log'))
children.extend((root/'pairs').rglob('summary.json'))
complete=[];checks=Counter();matched=0
for path in children:
 v=json.loads(path.read_text())
 if v['status']!='complete':continue
 complete.append(path)
 for key,value in v['integrity']['checks'].items():
  if not value:checks[key]+=1
 matched+=v['native']['prefix_matched_issues']
warnings=[]
for path in child_logs:
 n=sum('Failed to apply holding' in line for line in path.open(errors='replace'))
 if n:warnings.append(dict(path=str(path.relative_to(root)),count=n))
result=dict(coverage=out,complete_children=len(complete),exact_prefix_issues=matched,
 failed_child_checks=dict(checks),training_child_logs=len(child_logs),holding_warning_logs=warnings)
print(json.dumps(result))
'''.replace('__ROOT__', repr(infra.remote(SOURCE)))
    resident_value = resident(script)
    grouped = {}
    for task in tasks:
        if task.get('log_path') and task.get('node'):
            grouped.setdefault(task['node'], []).append((task['id'], task['log_path']))
    def logs(item):
        node, rows = item
        code = '''import json
from pathlib import Path
out=[]
for task,path in __ROWS__:
 p=Path(path)
 out.append(dict(id=task,exists=p.is_file(),holding_warnings=sum('Failed to apply holding' in line for line in p.open(errors='replace')) if p.is_file() else None))
print(json.dumps(out))
'''.replace('__ROWS__', repr(rows))
        return dict(node=node, logs=resident(code, node))
    with ThreadPoolExecutor(max_workers=4) as pool:
        log_values = list(pool.map(logs, grouped.items()))
    value = dict(scheduler=dict(tasks=tasks, status_counts=dict(Counter(t['status'] for t in tasks)),
                    priority_counts=dict(Counter(t['priority'] for t in tasks))),
                 resident=resident_value, task_logs=log_values)
    path = HERE/'live_evidence.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(bytes=path.stat().st_size,resident=resident_value,
        statuses=value['scheduler']['status_counts'],priorities=value['scheduler']['priority_counts'],
        task_logs=sum(len(row['logs']) for row in log_values),
        missing_logs=[x['id'] for row in log_values for x in row['logs'] if not x['exists']],
        holding_warnings=sum(x['holding_warnings'] or 0 for row in log_values for x in row['logs'])),ensure_ascii=False))


if __name__=='__main__':main()
