"""Two high native day workers, then one resident small-summary task."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([*sorted(HERE.glob('*.py')), HERE/'protocol.json'])


def native():
    p = protocol()
    return infra.submit([infra.task(p, 'native/'+case,
        ['--scenario', case, '--seed', p.get('training_seed', 6202),
         '--out', 'runs/'+case])
        for case in p['scenarios']], 'native')


def finalize():
    p = protocol()
    for case in p['scenarios']:
        summary = json.loads((HERE/'runs'/case/'summary.json').read_text())
        if summary['status'] != 'complete':
            raise RuntimeError('Both native day workers must complete before analysis')
    return infra.submit([infra.task(p, 'small-summary', runner='analyze.py',
        cpu=1, ram_mb=4000)], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'native', 'finalize', 'collect'))
    operation = parser.parse_args().operation
    value = (stage() if operation == 'stage' else native() if operation == 'native' else
        finalize() if operation == 'finalize' else
        infra.collect('analysis/summary.json', max_bytes=48000))
    print(json.dumps(value, ensure_ascii=False, indent=2))
