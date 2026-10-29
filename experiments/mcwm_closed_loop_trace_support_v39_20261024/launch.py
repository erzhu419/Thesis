"""Explicit v39-only staging and high-priority resident precheck tasks."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def protocol(): return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([HERE/name for name in ('bootstrap.py','infra.py','protocol.json','README.md',
        'support.py','trace.py','budget.py','context.py','probe.py','run.py','analyze.py','launch.py','continue_batch.py')])


def submit_days():
    rows=[]
    for scenario in protocol()['scenarios']:
        row=infra.task(protocol(),'precheck/6202/'+scenario,
            ['--scenario',scenario,'--out',infra.remote(HERE/'runs'/scenario)],ram_mb=24000)
        row['cpu']=3;rows.append(row)
    return infra.submit(rows,'precheck')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'precheck/finalize',
        ['--runs',infra.remote(HERE/'runs'),'--out',infra.remote(HERE/'analysis')],
        runner='analyze.py',ram_mb=4096)],'finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','days','finalize','collect'))
    op=parser.parse_args().operation
    result=stage() if op=='stage' else submit_days() if op=='days' else submit_finalize() if op=='finalize' else infra.collect('analysis/summary.json',max_bytes=24000)
    print(json.dumps(result,indent=2))
