"""High-priority v27 exact replays and memory-only shadow diagnosis."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def stage():
    names=('bootstrap.py','infra.py','run.py','shadow.py','analyze.py','launch.py','protocol.json')
    return infra.stage([HERE/name for name in names])


def submit_replays(preflight):
    p=protocol()
    first=p['preflight_cell']
    rows=[]
    for seed in p['training_seeds']:
        for scenario in p['scenarios']:
            if (dict(seed=seed,scenario=scenario)==first) != preflight:
                continue
            args=['--seed',seed,'--scenario',scenario,'--out',
                infra.remote(HERE/'runs'/f'seed{seed}'/scenario)]
            row=infra.task(p,f'replay/{seed}/{scenario}',args,runner='run.py',ram_mb=12000)
            row['cpu_training_justification']='Exact native fresh-origin replay; four memory-only eight-step paired optimizer probes, no new target control branches.'
            rows.append(row)
    return infra.submit(rows,'preflight' if preflight else 'remaining')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'shadow/finalize',
        ['--root',infra.remote(HERE),'--out',infra.remote(HERE/'shadow_summary.json')],
        runner='analyze.py',ram_mb=2048)],'finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','preflight','remaining','finalize','collect'))
    op=parser.parse_args().operation
    value=stage() if op=='stage' else submit_replays(True) if op=='preflight' \
        else submit_replays(False) if op=='remaining' else submit_finalize() if op=='finalize' \
        else infra.collect('shadow_summary.json',max_bytes=60000)
    print(json.dumps(value,ensure_ascii=False,indent=2))
