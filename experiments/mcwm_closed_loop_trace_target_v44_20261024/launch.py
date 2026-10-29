"""Launch the two read-only resident scenarios in one high-priority batch."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([HERE/name for name in ('protocol.json', 'README.md',
        'infra.py', 'trace.py', 'probe.py', 'summary.py', 'run.py', 'analyze.py', 'launch.py',
        'continue_batch.py')])


def cases():
    rows = [infra.task(protocol(), 'precheck/'+case, ['--scenario', case])
            for case in protocol()['scenarios']]
    return infra.submit(rows, 'precheck')


def finalize():
    return infra.submit([infra.task(protocol(), 'small-summary', runner='analyze.py',
        ram_mb=2000)], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'cases', 'finalize', 'collect'))
    op = parser.parse_args().operation
    value = (stage() if op == 'stage' else cases() if op == 'cases' else
        finalize() if op == 'finalize' else
        infra.collect('analysis/summary.json', max_bytes=protocol()['summary_max_bytes']))
    print(json.dumps(value, ensure_ascii=False, indent=2))
