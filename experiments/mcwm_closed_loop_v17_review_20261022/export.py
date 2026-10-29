"""Recompute a small review summary without modifying the v17 experiment."""
import inspect
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mcwm_closed_loop_fresh_difference_v17_20261021'
sys.path.insert(0,str(SOURCE))
import infra


def export(root):
    import sys
    from collections import Counter
    sys.path.insert(0,str(root))
    import analyze as original
    value=original.analyze(root)
    original_bytes=len((json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    def checks(row,order=None):
        raw=row.get('checks',{})
        items=list(zip(order,raw)) if isinstance(raw,list) else list(raw.items())
        return dict(all_pass=row.get('all_pass'),check_count=sum(v is not None for _,v in items),
                    failed_checks=[k for k,v in items if v is False])
    for day in value['days']:
        day['integrity']=checks(day['integrity'])
        audit=day['budget_audit'];audit['checks']=checks(audit)
        day['budget'].pop('semantics',None)
        for model in day.get('readouts',{}).values():
            for key in ('semantics','pair_ids','model'):
                model.pop(key,None)
    # Keep observed truths and all action effects. Per-branch replay budgets
    # remain in the aggregate; their repeated training-row copies are omitted.
    for row in value['training_pairs']+value['heldout_pairs']:
        row['integrity']=checks(row['integrity'],value['schema']['pairs']['pair_check_order'])
    for row in value['training_pairs']:
        for key in ('budget','suffix_training_delta','execution','branch_status'):
            row.pop(key,None)
    def strip_duplicate_ids(obj):
        if isinstance(obj,dict):
            obj.pop('common_state_ids',None)
            for item in obj.values():strip_duplicate_ids(item)
        elif isinstance(obj,list):
            for item in obj:strip_duplicate_ids(item)
    strip_duplicate_ids(value)
    value.pop('interpretation',None)
    children=[];logs=[]
    for _,day in original.expected_days(root,original.read_json(root/'protocol.json')):
        children.extend(day.glob('training_branches/*/*/summary.json'))
        logs.extend(day.glob('training_branches/*/*.log'))
    children.extend((root/'pairs').rglob('summary.json'))
    status=Counter();failed=Counter();matched=0
    for path in children:
        child=json.loads(path.read_text());status[child['status']]+=1
        matched+=child.get('native',{}).get('prefix_matched_issues',0)
        for key,passed in child.get('integrity',{}).get('checks',{}).items():
            if passed is False:failed[key]+=1
    warning_files=[]
    for path in logs:
        count=sum('Failed to apply holding' in line for line in path.open(errors='replace'))
        if count:warning_files.append(dict(path=str(path.relative_to(root)),count=count))
    value['resident_audit']=dict(child_count=len(children),child_statuses=dict(status),
        exact_prefix_issues=matched,failed_child_checks=dict(failed),
        training_logs=len(logs),holding_warning_files=warning_files)
    failure=root/'summary_failure.json'
    value['export']=dict(original_summary_bytes=original_bytes,
        original_failure=json.loads(failure.read_text()) if failure.is_file() else None,
        source_results_unchanged=True,method='Original analyze() in memory; remove repeated metadata only.',
        train_rows_omitted_fields=['budget','suffix_training_delta','execution','branch_status'],
        integrity_checks='All-pass plus failed names and check count; original raw evidence remains resident.')
    return value


def main():
    script='import json\nfrom pathlib import Path\n'+inspect.getsource(export)+'''
value=export(Path(__ROOT__))
payload=json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\\n'
if len(payload.encode())>150000:raise RuntimeError('Review summary must stay small before transfer')
print(payload,end='')
'''.replace('__ROOT__',repr(infra.remote(SOURCE)))
    result=infra.scheduler.run_on('node001',shlex.join([infra.PYTHON,'-c',script]),timeout=90,check=True)
    value=json.loads(result[1])
    path=HERE/'review_summary.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(result[1])
    print(json.dumps(dict(bytes=path.stat().st_size,status=value['status'],integrity=value['integrity'],
        missing=value['missing'],budget=value['budget'],overall=value['heldout_overall'],
        label_age=value['heldout_label_age'],export=value['export'],resident_audit=value['resident_audit']),ensure_ascii=False))


if __name__=='__main__':main()
