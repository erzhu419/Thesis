"""Submit resident inference diagnostics with high priority."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([HERE/name for name in ('bootstrap.py','infra.py','decomposition.py',
        'coverage.py','run.py','analyze.py','launch.py','protocol.json','README.md')])


def submit_days():
    p=protocol()
    return infra.submit([infra.task(p,'decompose/'+scenario,
        ['--scenario',scenario,'--out',infra.remote(HERE/'runs'/scenario)],ram_mb=8000)
        for scenario in p['scenarios']],'diagnostic')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'decompose/finalize',
        ['--root',infra.remote(HERE/'runs'),'--out',infra.remote(HERE/'decomposition_summary.json')],
        runner='analyze.py',ram_mb=2048)],'finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','days','finalize','collect'))
    op=parser.parse_args().operation
    value=stage() if op=='stage' else submit_days() if op=='days' else submit_finalize() if op=='finalize' \
        else infra.collect('decomposition_summary.json',max_bytes=20000)
    print(json.dumps(value,indent=2))
