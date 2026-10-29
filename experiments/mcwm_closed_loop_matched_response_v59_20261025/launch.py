"""Two parallel native collectors, parallel matched learners, one small summary."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent
ARMS = ('fresh', 'no_fresh')


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([*sorted(HERE.glob('*.py')), HERE/'protocol.json'])


def native():
    p = protocol()
    return infra.submit([infra.task(p, 'collection/'+case,
        ['--scenario', case, '--seed', p.get('training_seed', 6202),
         '--out', infra.remote(HERE/'runs'/'collection'/case)],
        runner='collect.py', cpu=2) for case in p['scenarios']], 'native')


def learning(case):
    p = protocol()
    if case not in p['scenarios']:
        raise ValueError('Unknown scenario: '+case)
    summary = json.loads((HERE/'runs'/'collection'/case/'summary.json').read_text())
    if summary.get('status') != 'complete':
        raise RuntimeError('Complete native collection required before learning: '+case)
    return infra.submit([infra.task(p, 'learning/'+case+'/'+arm,
        ['--scenario', case, '--arm', arm, '--seed', p.get('training_seed', 6202),
         '--out', infra.remote(HERE/'runs'/'learning'/case/arm)], cpu=1)
        for arm in ARMS], 'learning_'+case)


def finalize():
    value = json.loads((HERE/'continuation_inputs.json').read_text())
    if not value.get('all_terminal'):
        raise RuntimeError('All collectors and admitted learners must be terminal before summary')
    return infra.submit([infra.task(protocol(), 'small-summary', runner='analyze.py',
        argv=['--root', infra.remote(HERE), '--out', infra.remote(HERE/'analysis'/'summary.json')],
        cpu=1, ram_mb=4000)], 'finalize')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'native', 'start', 'learning', 'finalize', 'collect'))
    parser.add_argument('--scenario')
    args = parser.parse_args()
    if args.operation == 'start':
        value = dict(staging=stage(), native=native())
    elif args.operation == 'learning':
        if args.scenario is None:
            parser.error('--scenario is required for learning')
        value = learning(args.scenario)
    else:
        value = dict(stage=stage, native=native, finalize=finalize,
            collect=lambda: infra.collect('analysis/summary.json'))[args.operation]()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
