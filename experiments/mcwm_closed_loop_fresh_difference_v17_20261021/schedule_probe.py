"""Zero-label read of resident issue clocks for the v17 schedule."""
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent/'mcwm_closed_loop_budgeted_difference_v16_20261021'


def simulate(value, start=16700, end=17670):
    """Replay the original single-pair queue using clocks, never responses."""
    output=[]
    for day in value['late_candidates']:
        selected=[]
        for index,line in enumerate(day['lines']):
            issue=next((t for t in line['training_candidates'] if start<=t<end),None)
            if issue is not None:
                selected.append(dict(line=line['line'],index=3*index+2,issue=issue,available=issue+330))
        pending=list(selected)
        for deadline in range(1200,18001,300):
            ready=sorted((row for row in pending if row['available']<=deadline),key=lambda row:(row['available'],row['index']))
            if ready:
                ready[0]['admitted_deadline']=deadline
                pending.remove(ready[0])
        output.append(dict(seed=day['seed'],scenario=day['scenario'],selected=selected,
                           all_three_admitted=len(selected)==3 and not pending))
    return dict(window=dict(name='late',start=start,end=end),days=output,
                assumptions='Earlier pairs admitted before this window; native global endpoint finishes within300s. New prefix must confirm both.')


def main():
    code = '''import json,sys
from pathlib import Path
root=Path(__ROOT__)
sys.path.insert(0,str(root.parents[1]))
from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import ControlledCalibration
p=json.loads((root/'protocol.json').read_text())
contract=json.loads((root/p['reference_inputs']/'history_metadata.json').read_text())['contract']
cal=ControlledCalibration.__new__(ControlledCalibration);cal.transform=ExecutionAction(contract)
rows=[]
for seed in p['training_seeds']:
 for scene in p['scenarios']:
  source=root/'runs'/scene/('seed%d'%seed)/'prefix_trace.jsonl'
  times={line:[] for line in p['target_lines']}
  unsupported={line:0 for line in p['target_lines']}
  with source.open() as stream:
   for text in stream:
    row=json.loads(text);line=row['line_id'];t=row['started_at']
    if line not in times or row['station_id']!='7X13_102X26_705X27' or not 15000<=t<20000:continue
    if all(cal._action_for_hold(row['s'],hold) is not None for hold in (0.,7.5)):
     times[line].append(t)
    else:unsupported[line]+=1
  rows.append(dict(seed=seed,scenario=scene,lines=[dict(line=line,
   training_candidates=[t for t in ts if t<=17670],unsupported=unsupported[line],
   heldout=[dict(start=start,end=end,count=sum(start<=t<end for t in ts),
    first=next((t for t in ts if start<=t<end),None))
    for start,end in ((18010,18700),(18700,20000))]) for line,ts in times.items()]))
value=dict(source='Resident v16 issue clocks and observable action support only; no response truth read.',
 feedback_delay=30,endpoint_seconds=300,deadline=18000,late_candidates=rows)
payload=json.dumps(value,separators=(',',':'))
assert len(payload.encode())<=10000
print(payload)
'''.replace('__ROOT__', repr(infra.remote(REFERENCE)))
    response = infra.scheduler.run_on('node001', shlex.join([infra.PYTHON, '-c', code]),
                                      timeout=90, check=True)[1]
    value = json.loads(response)
    value['proposed_late_queue']=simulate(value)
    path = HERE/'schedule_evidence.json'
    if path.exists():raise FileExistsError(path)
    payload=json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'
    assert len(payload.encode())<=10000
    path.write_text(payload)
    print(payload)


if __name__=='__main__':main()
