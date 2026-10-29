"""High-priority four-day fixed-algorithm replication at seed6202."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names = ('bootstrap.py','run.py','run_pair.py','analyze.py','audit_summary.py',
             'clock_diagnostic.py','infra.py','launch.py','protocol.json')
    return infra.stage([HERE/name for name in names])


def submit_pairs():
    p = protocol()
    seed, = p['training_seeds']
    rows = []
    for scenario in p['scenarios']:
        args = ['--seed',seed,'--scenario',scenario,'--out',
                infra.remote(HERE/'runs'/scenario/f'seed{seed}')]
        row = infra.task(p,'factual_pair/'+scenario,args,runner='run_pair.py',ram_mb=16000)
        row['cpu'] = 2
        row['cpu_training_justification'] = ('Two independent single-thread factual arms launch before waiting; '
            'same v25 algorithm, matched initialization and285/456 budgets; no world or target branches.')
        rows.append(row)
    return infra.submit(rows,'four_day_pairs')


def submit_finalize():
    args = ['--root',infra.remote(HERE),'--stage','pilot',
            '--out',infra.remote(HERE/'pilot_summary.json')]
    return infra.submit([infra.task(protocol(),'four_day/finalize',args,
        runner='analyze.py',ram_mb=2048)],'pilot_finalize')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','pairs','finalize','collect'))
    op = parser.parse_args().operation
    value = stage() if op=='stage' else submit_pairs() if op=='pairs' else submit_finalize() if op=='finalize' \
        else infra.collect('pilot_summary.json',max_bytes=30000)
    print(json.dumps(value,ensure_ascii=False,indent=2))
