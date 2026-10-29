"""High-priority three-worker cumulative shadow comparison."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([HERE/name for name in ('bootstrap.py','infra.py','learning.py',
        'run.py','analyze.py','launch.py','protocol.json','README.md')])


def submit_days():
    p=protocol();rows=[]
    for scenario in p['scenarios']:
        row=infra.task(p,'triplet/6202/'+scenario,
            ['--seed',6202,'--scenario',scenario,'--out',infra.remote(HERE/'runs'/scenario)],ram_mb=24000)
        row['cpu']=3;rows.append(row)
    return infra.submit(rows,'diagnostic')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'shared-level/finalize',
        ['--root',infra.remote(HERE/'runs'),'--out',infra.remote(HERE/'shared_level_summary.json')],
        runner='analyze.py',ram_mb=2048)],'finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','days','finalize','collect'))
    op=parser.parse_args().operation
    value=stage() if op=='stage' else submit_days() if op=='days' else submit_finalize() if op=='finalize' \
        else infra.collect('shared_level_summary.json',max_bytes=20000)
    print(json.dumps(value,indent=2))
