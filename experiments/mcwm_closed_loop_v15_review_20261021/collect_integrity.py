"""Small read-only resident budget, reproduction and backend command audit."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
V15 = HERE.parent/'mcwm_closed_loop_training_pairs_v15_20261021'
sys.path.insert(0,str(V15))
import infra

RESIDENT = r'''
import json,pathlib,collections
root=pathlib.Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_training_pairs_v15_20261021')
paths=[*sorted(root.glob('pairs/*/*/*/*/*/summary.json')),*sorted(root.glob('preflight/pair/*/summary.json'))]
rows=[];passes=collections.Counter();bad=[]
for path in paths:
 v=json.loads(path.read_text()); c=v['integrity']['checks'];passes.update(k for k,x in c.items() if x)
 bad.extend([str(path.relative_to(root)),k] for k,x in c.items() if not x)
 rows.append(dict(path=str(path.parent.relative_to(root)),status=v['status'],state_id=v['state_id'],
  budget=v['budget'],external_seed=v['prefix']['external_seed'],
  native=v['native'],execution={k:v['execution'][k] for k in ('requested_hold_seconds','executed_hold_seconds')},
  suffix_seconds=v['suffix']['seconds'],nonzero_suffix={k:x for k,x in v['suffix']['training_delta'].items() if x},
  source_prefix_packets=len(v['source']['prefix_receipts']),
  source_full_day_budget={k:v['source']['full_day_budget'][k] for k in ('target_labels','policy_updates','native_seconds')},
  background_scope={name:dict(fit_rows=x['fit_rows'],max_available_at=x['max_available_at'],max_sequence=x['max_sequence']) for name,x in v['context']['innovation_background']['endpoints'].items()},
  issue_time=v['selected']['started_at'],issue_sequence=v['selected']['sequence']))
print(json.dumps(dict(branches=rows,check_pass_counts=dict(passes),false_checks=bad),separators=(',',':')))
'''


def main():
    summary=json.loads((V15/'training_pair_summary.json').read_text())
    pref=json.loads((V15/'preflight/gate.json').read_text())
    ids=[r['id'] for p in sorted(V15.glob('*_submission.json')) for r in json.loads(p.read_text())['submitted']]
    tasks=infra.status(ids)
    (HERE/'scheduler_receipt.json').write_text(json.dumps(tasks,separators=(',',':'))+'\n')
    _,out,_=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',RESIDENT]),timeout=45,check=True)
    receipt=json.loads(out)
    by_node=defaultdict(list)
    for task in tasks:
        if '/pairs/' in task['signature'] or '/pair/' in task['signature']:
            by_node[task['node']].append(dict(id=task['id'],path=task['log_path']))
    def audit(item):
        node,entries=item
        code="import json,pathlib; entries="+repr(entries)+"; print(json.dumps([dict(id=e['id'],exists=pathlib.Path(e['path']).is_file(),holding_failures=pathlib.Path(e['path']).read_text(errors='replace').count('Failed to apply holding') if pathlib.Path(e['path']).is_file() else None) for e in entries]))"
        _,out,_=infra.scheduler.run_on(node,shlex.join([infra.PYTHON,'-c',code]),timeout=45,check=True)
        return dict(node=node,logs=json.loads(out))
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipt['native_logs']=list(pool.map(audit,by_node.items()))
    receipt['scheduler']=dict(tasks=len(tasks),status=dict(Counter(t['status'] for t in tasks)),priority=dict(Counter(t['priority'] for t in tasks)))
    receipt['main_summary']=dict(integrity=summary['integrity'],budget=summary['budget'],state_counts=summary['state_counts'],duplicate_prefixes=summary['duplicate_prefixes'],missing=summary['missing'])
    receipt['preflight_budget']=pref['budget']
    main=[x for x in receipt['branches'] if x['path'].startswith('pairs/')]
    receipt['totals']=dict(main_branches=len(main),preflight_branches=len(receipt['branches'])-len(main),
      main_prefix_matched_issues=sum(x['native']['prefix_matched_issues'] for x in main),
      main_old_prediction_matches=sum(x['native']['v14_matched_predictions'] for x in main),
      holding_failures=sum(x['holding_failures'] or 0 for n in receipt['native_logs'] for x in n['logs']),
      missing_logs=[x['id'] for n in receipt['native_logs'] for x in n['logs'] if not x['exists']],
      nonzero_suffix_counts=[x['path'] for x in receipt['branches'] if x['nonzero_suffix']])
    raw=json.dumps(receipt,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'integrity_receipt.json').write_text(raw)
    print(json.dumps(dict(bytes=len(raw.encode()),scheduler=receipt['scheduler'],totals=receipt['totals'],false_checks=receipt['false_checks'],budget=summary['budget'],preflight_budget=pref['budget'])))


if __name__=='__main__':main()
