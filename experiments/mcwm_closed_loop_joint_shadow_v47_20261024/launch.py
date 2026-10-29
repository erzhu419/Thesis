"""Launch the audited single-step experiment; original runs stay read-only."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names = ('infra.py', 'launch.py', 'continue_batch.py', 'protocol.json', 'README.md',
        'mean_target.py', 'replay.py', 'run.py', 'score.py', 'summary.py', 'analyze.py',
        'policy_summary.py', 'policy_summary_spec.json', 'coverage_summary.py')
    return infra.stage([HERE/name for name in names])


def preflight():
    p = protocol()
    return infra.submit([infra.task(p, 'preflight/stable/M1',
        ['--scenario', 'stable', '--arm', 'M1', '--phase', 'preflight'])], 'preflight')


def formal():
    p = protocol()
    gate = json.loads((HERE/'preflight/stable/M1/summary.json').read_text())
    if gate['status'] != 'complete' or not gate['proof']['all_pass']:
        raise RuntimeError('The complete native57-pack replay must pass before M8 learning')
    return infra.submit([infra.task(p, 'formal/'+case+'/'+arm,
        ['--scenario', case, '--arm', arm, '--phase', 'formal'])
        for case in p['scenarios'] for arm in p['arms']], 'formal')


def finalize():
    return infra.submit([infra.task(protocol(), 'small-summary', runner='analyze.py',
        ram_mb=2000)], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'preflight', 'formal', 'finalize', 'collect'))
    op = parser.parse_args().operation
    value = (stage() if op == 'stage' else preflight() if op == 'preflight' else
        formal() if op == 'formal' else finalize() if op == 'finalize' else
        infra.collect('analysis/summary.json', max_bytes=protocol()['summary_max_bytes']))
    print(json.dumps(value, ensure_ascii=False, indent=2))
