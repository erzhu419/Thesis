"""Stage isolated v34 files and submit two high-priority native pairs."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names=('bootstrap.py','infra.py','frozen.py','selection.py','fork_state.py','suffix.py',
           'run.py','analyze.py','launch.py','protocol.json','README.md')
    return infra.stage([HERE/name for name in names])


def submit_days():
    p=protocol();tasks=[]
    for seed in p['training_seeds']:
        for scenario in p['scenarios']:
            row=infra.task(p,f'pair/{seed}/{scenario}', ['--seed',seed,'--scenario',scenario,
                '--out',infra.remote(HERE/'runs'/f'seed{seed}'/scenario)],ram_mb=24000)
            row['cpu']=2
            tasks.append(row)
    return infra.submit(tasks,'paired')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'truth/finalize',
        ['--root',infra.remote(HERE/'runs'/'seed6202'),
         '--out',infra.remote(HERE/'truth_summary.json')],
        runner='analyze.py',ram_mb=2048)],'finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','days','finalize','collect'))
    op=parser.parse_args().operation
    value=stage() if op=='stage' else submit_days() if op=='days' else submit_finalize() if op=='finalize' \
        else infra.collect('truth_summary.json',max_bytes=20000)
    print(json.dumps(value,indent=2))
