"""One high-priority task: fork two read-only scenario workers and summarize."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def stage():
    return infra.stage([HERE/name for name in ('protocol.json','release_plan.json','README.md',
        'bootstrap.py','infra.py','overlap.py','attribution.py','run.py','analyze.py','launch.py','continue_batch.py')])


def submit():
    p=json.loads((HERE/'protocol.json').read_text())
    row=infra.task(p,'conditional-overlap',[],ram_mb=16000)
    row['cpu']=2
    # v40 used node004; retain its CPU path for bit-exact initializer replay.
    row['require_node']='node004'
    return infra.submit([row],'precheck')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','submit','collect'))
    op=parser.parse_args().operation
    result=stage() if op=='stage' else submit() if op=='submit' else infra.collect('analysis/summary.json',max_bytes=24000)
    print(json.dumps(result,indent=2))
