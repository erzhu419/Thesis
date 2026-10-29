"""Launch the fixed two-scenario pilot after the resident runtime gate."""
import json
from pathlib import Path
import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    files = sorted(HERE.glob('*.py'))+[HERE/'protocol.json', HERE/'release_plan_reference.json', HERE/'README.md']
    return infra.stage(files)


def runtime():
    row = infra.task(protocol(), 'resident-preflight', [], runner='resident_preflight.py', ram_mb=8000)
    row.update(cpu=1, cpu_training_justification='Load actual resident initializer and inspect unchanged critic parity, stochastic actor/F measures and action RNG isolation; no SGD, new labels or native prefix.')
    return infra.submit([row], 'resident_preflight')


def days():
    rows = [infra.task(protocol(), 'native/'+case,
        ['--scenario', case, '--seed', protocol()['training_seed'], '--out', infra.remote(HERE/'runs'/case)])
        for case in protocol()['scenarios']]
    return infra.submit(rows, 'native')


def finalize():
    row = infra.task(protocol(), 'small-summary', [], runner='analyze.py', ram_mb=2000)
    row.update(cpu=1, cpu_training_justification='Read completed native logs and credit summaries; no model, SGD or native collection.')
    return infra.submit([row], 'finalize')


if __name__ == '__main__':
    print(json.dumps(stage()))
    print(json.dumps(runtime()))
