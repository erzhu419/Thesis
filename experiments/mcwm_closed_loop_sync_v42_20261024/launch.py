"""High-priority existing-log and cached-distribution precheck."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def stage():
    return infra.stage([HERE/name for name in ('protocol.json','release_plan.json','README.md',
        'sync_plan.json','infra.py','journal.py','controls.py','run.py','analyze.py','launch.py','continue_batch.py')])


def submit():
    p=json.loads((HERE/'protocol.json').read_text())
    row=infra.task(p,'issue-visible-sync',[],ram_mb=8000)
    row['cpu']=2
    return infra.submit([row],'precheck')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','submit','collect'))
    op=parser.parse_args().operation
    result=stage() if op=='stage' else submit() if op=='submit' else infra.collect('analysis/summary.json',max_bytes=24000)
    print(json.dumps(result,indent=2))
