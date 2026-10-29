"""Read-only resident receipt and node-local warning counts; no trajectory transfer."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
V13 = HERE.parent / 'mcwm_closed_loop_separated_contrast_v13_20261021'
sys.path.insert(0, str(V13))
import infra

RESIDENT = r'''
import json,pathlib,collections
root=pathlib.Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_separated_contrast_v13_20261021')
days=[]; pairs=[];checks=collections.Counter();false=[]
for path in sorted(root.glob('runs/*/*/summary.json')):
 d=json.loads(path.read_text());b=d['budget'];c=d['integrity']['checks'];checks.update(k for k,v in c.items() if v)
 false.extend([str(path.relative_to(root)),k] for k,v in c.items() if not v)
 days.append(dict(cell=d['cell'],status=d['status'],budget={k:v for k,v in b.items() if k!='pending_sequence_ids'},native=d['native'],heldout={l:{k:v for k,v in x.items() if k in ('n','low','high','missing_actions')} for l,x in d['coverage']['heldout']['coverage'].items()}))
for path in sorted(root.glob('pairs/*/*/*/*/*/summary.json')):
 d=json.loads(path.read_text());c=d['integrity']['checks'];checks.update(k for k,v in c.items() if v)
 false.extend([str(path.relative_to(root)),k] for k,v in c.items() if not v)
 pairs.append(dict(cell=d['cell'],status=d['status'],sequence=d['selected']['sequence'],started_at=d['selected']['started_at'],external_seed=d['prefix']['external_seed'],execution=d['execution'],suffix=d['suffix'],budget=d['budget'],prefix_matched_issues=d['native']['prefix_matched_issues']))
print(json.dumps(dict(days=days,pairs=pairs,check_pass_counts=dict(checks),false_checks=false),separators=(',',':')))
'''


def main():
    rows = json.loads((HERE / 'scheduler_receipt.json').read_text())
    _, stdout, _ = infra.scheduler.run_on('node001', infra.PYTHON+' -c '+shlex.quote(RESIDENT), timeout=45, check=True)
    receipt = json.loads(stdout)
    by_node = defaultdict(list)
    for row in rows:
        if '/runs/' in row['signature'] or '/pairs/' in row['signature']:
            by_node[row['node']].append(dict(id=row['id'],path=row['log_path']))
    def logs(item):
        node, entries=item
        source = "import json,pathlib; entries="+repr(entries)+"; print(json.dumps([dict(id=e['id'],exists=pathlib.Path(e['path']).is_file(),holding_failures=pathlib.Path(e['path']).read_text(errors='replace').count('Failed to apply holding') if pathlib.Path(e['path']).is_file() else None) for e in entries]))"
        _, output, _ = infra.scheduler.run_on(node, infra.PYTHON+' -c '+shlex.quote(source),timeout=45,check=True)
        return dict(node=node,logs=json.loads(output))
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipt['native_log_audit']=list(pool.map(logs,by_node.items()))
    receipt['scheduler']=dict(tasks=len(rows),status=dict(Counter(r['status'] for r in rows)),priority=dict(Counter(r['priority'] for r in rows)),failed=[dict(id=r['id'],signature=r['signature'],reason='resident summary exceeded 200000-byte transfer budget') for r in rows if r['status']=='failed'])
    raw=json.dumps(receipt,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'integrity_receipt.json').write_text(raw)
    print(json.dumps(dict(bytes=len(raw.encode()),days=len(receipt['days']),pairs=len(receipt['pairs']),false_checks=receipt['false_checks'],logs=sum(len(x['logs']) for x in receipt['native_log_audit']),holding_failures=sum(x['holding_failures'] or 0 for n in receipt['native_log_audit'] for x in n['logs']),missing_logs=[x['id'] for n in receipt['native_log_audit'] for x in n['logs'] if not x['exists']])))


if __name__=='__main__':
    main()
