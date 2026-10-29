"""Freeze only the already-completed v23 first issue and its two baselines."""
import argparse
import json
from pathlib import Path
import shutil

FIELDS = ('id','role','index','status','line_id','window_name','selected','nominal_a',
          'actions','features','prediction','prefix','service_history','service_prediction','sampling')


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    original = source/'training_branches/0'
    low = json.loads((original/'low/summary.json').read_text())
    assert low['id'] == 'train/7X/bootstrap' and low['integrity']['all_pass']
    assert low['prefix']['counters']['service_shared_pairs'] == 0
    row = {key: low[key] for key in FIELDS}
    row['status'] = 'selected'
    (out/'acquisitions.json').write_text(json.dumps(dict(records=[row])))
    count = row['selected']['sequence']+1
    with (source/'prefix_trace.jsonl').open() as src, (out/'prefix_trace.jsonl').open('x') as dst:
        for _ in range(count):
            line = src.readline()
            if not line:
                raise ValueError('Original completed prefix is missing a native issue')
            dst.write(line)
    for arm in ('low','high'):
        target = out/'training_branches/0'/arm
        target.mkdir(parents=True)
        shutil.copyfile(original/arm/'summary.json',target/'summary.json')
    receipt = dict(source=str(source),target_id=row['id'],issue_seconds=row['selected']['started_at'],
        prefix_issues=count,source_writes=0,new_training_labels=0,
        scope='Resident first completed prefix and immutable paired summaries only; no CSV/checkpoint copy.')
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.out)))
