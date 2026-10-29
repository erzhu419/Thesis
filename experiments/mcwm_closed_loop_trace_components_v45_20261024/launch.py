"""High-priority resident component and crossed-factor diagnosis."""
import argparse
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([HERE/name for name in ('protocol.json', 'README.md',
        'infra.py', 'components.py', 'factor.py', 'run.py', 'analyze.py',
        'summary.py', 'launch.py', 'continue_batch.py', 'repack.py')])


def cases():
    return infra.submit([infra.task(protocol(), 'diagnostic/'+case,
        ['--scenario', case]) for case in protocol()['scenarios']], 'diagnostic')


def finalize():
    return infra.submit([infra.task(protocol(), 'small-summary', runner='analyze.py',
        ram_mb=2000)], 'finalize')


def repack():
    row = infra.task(protocol(), 'summary-repack', runner='repack.py', ram_mb=2000)
    row['cpu_training_justification'] = ('Read completed JSON arrays after the summary-size failure; '
        'no checkpoint load, rescoring, optimizer step or native simulation.')
    return infra.submit([row], 'repack')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'cases', 'finalize', 'repack', 'collect'))
    op = parser.parse_args().operation
    value = (stage() if op == 'stage' else cases() if op == 'cases' else
        finalize() if op == 'finalize' else repack() if op == 'repack' else
        infra.collect('analysis_repacked/summary.json', max_bytes=protocol()['summary_max_bytes']))
    print(json.dumps(value, ensure_ascii=False, indent=2))
