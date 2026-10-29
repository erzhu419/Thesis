"""Two preflight gates, four parallel shadow arms, then a resident summary."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    files = sorted(HERE.glob('*.py'))
    files.extend(HERE/name for name in ('protocol.json', 'README.md') if (HERE/name).exists())
    return infra.stage(files)


def preflight():
    p = protocol()
    return infra.submit([infra.task(p, 'preflight/'+case,
        ['--scenario', case, '--arm', 'one_step', '--phase', 'preflight'])
        for case in p['scenarios']], 'preflight')


def formal():
    p = protocol()
    for case in p['scenarios']:
        gate = json.loads((HERE/'preflight'/case/'summary.json').read_text())
        if gate['status'] != 'complete' or not gate['proof']['all_pass']:
            raise RuntimeError('First8 response-update parity must pass in both scenarios')
    return infra.submit([infra.task(p, 'formal/'+case+'/'+arm,
        ['--scenario', case, '--arm', arm, '--phase', 'formal'])
        for case in p['scenarios'] for arm in ('one_step', 'trace')], 'formal')


def finalize():
    p = protocol()
    for case in p['scenarios']:
        for arm in ('one_step', 'trace'):
            summary = json.loads((HERE/'runs'/case/arm/'summary.json').read_text())
            if summary['status'] != 'complete':
                raise RuntimeError('All four formal workers must complete before analysis')
    return infra.submit([infra.task(p, 'small-summary', runner='analyze.py', ram_mb=4000)], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'preflight', 'formal', 'finalize', 'collect'))
    operation = parser.parse_args().operation
    value = (stage() if operation == 'stage' else preflight() if operation == 'preflight' else
        formal() if operation == 'formal' else finalize() if operation == 'finalize' else
        infra.collect('analysis/summary.json', max_bytes=protocol().get('summary_max_bytes', 64000)))
    print(json.dumps(value, ensure_ascii=False, indent=2))
