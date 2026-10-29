"""Stage v38 only, submit paired native pipelines, then resident accounting."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names = ('bootstrap.py', 'infra.py', 'protocol.json', 'README.md', 'audit_logs.py',
        'existing_log_audit.json', 'run.py', 'collect.py', 'chain_stream.py', 'frozen.py',
        'analyze_chain.py', 'analyze.py', 'launch.py', 'continue_batch.py')
    return infra.stage([HERE/name for name in names])


def submit_days():
    rows = []
    for scenario in protocol()['scenarios']:
        row = infra.task(protocol(), 'native/6202/'+scenario,
            ['--seed', 6202, '--scenario', scenario, '--out', infra.remote(HERE/'runs'/scenario)],
            ram_mb=24000)
        row['cpu'] = 2
        rows.append(row)
    return infra.submit(rows, 'native')


def submit_finalize():
    row = infra.task(protocol(), 'credit/finalize',
        ['--runs', infra.remote(HERE/'runs'), '--out', infra.remote(HERE/'analysis')],
        runner='analyze.py', ram_mb=8000)
    return infra.submit([row], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'days', 'finalize', 'collect'))
    op = parser.parse_args().operation
    result = (stage() if op == 'stage' else submit_days() if op == 'days' else
        submit_finalize() if op == 'finalize' else
        infra.collect('analysis/summary.json', max_bytes=protocol()['summary_max_bytes']))
    print(json.dumps(result, indent=2))
