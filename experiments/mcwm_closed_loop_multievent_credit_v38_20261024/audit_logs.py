"""Resident audit of existing v37 suffixes; never reconstruct missing states."""
from collections import Counter
import json
from pathlib import Path

SOURCE=Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_paired_coverage_v37_20261024')


def read(path):return json.loads(path.read_text())


def main():
    result=dict(cases={},checkpoint_read=False,truth_used_for_selection=False,new_training=0)
    for scenario in ('stable','local_speed_drop'):
        root=SOURCE/'runs/collection'/scenario
        records=read(root/'pair_truth.json')
        counts=Counter();sizes=[];examples=[];missing=[]
        for record in records:
            parent=root/'branches'/record['id'].replace('/','_')
            for arm in ('low','high'):
                path=parent/arm
                summary=path/'summary.json'
                if not summary.exists():missing.append([record['id'],arm,'summary']);continue
                value=read(summary);counts['suffixes']+=1
                trace=path/'suffix_trace.jsonl'
                if trace.exists():
                    counts['trace_files']+=1;sizes.append(trace.stat().st_size)
                    with trace.open() as h:rows=sum(bool(line.strip()) for line in h)
                    counts['trace_rows']+=rows;counts['nonempty_traces']+=bool(rows)
                else:missing.append([record['id'],arm,'trace'])
                counts['event_chain_files']+=int((path/'events.jsonl').exists())
                counts['tick_ledger_files']+=int((path/'ticks.jsonl').exists())
                counts['first_transition']+=value.get('global_transition') is not None
                counts['fixed300_complete']+=value.get('truth',{}).get('fixed300',{}).get('dt')==300.
                counts['boundary_present']+='boundary' in value
                if len(examples)<1:
                    examples.append(dict(slot=record['id'],arm=arm,summary_keys=list(value),
                        boundary_keys=list(value.get('boundary',{})),physical_keys=list(value.get('physical',{}))))
        result['cases'][scenario]=dict(counts=dict(counts),trace_bytes_min=min(sizes),trace_bytes_max=max(sizes),
            missing=missing,examples=examples,recoverable='first transition and exact300 ledger only; no complete event/memory chain',
            needs_new_native_event_chain=True)
    text=json.dumps(result,ensure_ascii=False,separators=(',',':'))
    assert len(text.encode())<=8000
    print(text)


if __name__=='__main__':main()
