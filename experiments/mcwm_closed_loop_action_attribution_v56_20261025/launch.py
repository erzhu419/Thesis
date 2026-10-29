"""Two high read-only attribution workers, then one bounded summary task."""
import argparse
import json
import shlex
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    return infra.stage([*sorted(HERE.glob('*.py')), HERE/'protocol.json'])


def formal():
    p = protocol()
    return infra.submit([infra.task(p, 'frozen-attribution/'+case,
        ['--scenario', case]) for case in p['scenarios']], 'formal')


def finalize():
    p = protocol()
    code = ('from pathlib import Path;import json;root=Path('+repr(infra.remote(HERE))+');'
        'rows=[json.loads((root/"runs"/scene/"summary.json").read_text()) for scene in '+
        repr(p['scenarios'])+'];'
        'assert all(row["status"]=="complete" and row["integrity"]["all_pass"] for row in rows);'
        'print("ready")')
    infra.scheduler.run_on('node001', shlex.join([infra.PYTHON,'-c',code]), timeout=45, check=True)
    return infra.submit([infra.task(p, 'small-summary', runner='analyze.py',
        cpu=1, ram_mb=4000)], 'finalize')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage', 'formal', 'finalize', 'collect'))
    operation = parser.parse_args().operation
    value = (stage() if operation == 'stage' else formal() if operation == 'formal' else
        finalize() if operation == 'finalize' else
        infra.collect('analysis/summary.json', max_bytes=48000))
    print(json.dumps(value, ensure_ascii=False, indent=2))
