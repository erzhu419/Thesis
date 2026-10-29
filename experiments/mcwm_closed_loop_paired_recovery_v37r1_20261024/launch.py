"""Resume the original matched shadow budget without native recollection."""
import argparse
import json
from pathlib import Path
import infra

HERE=Path(__file__).resolve().parent


def protocol():return json.loads((HERE/'protocol.json').read_text())


def source():return HERE.parent/protocol()['source_v37_directory']


def stage():
    return infra.stage([HERE/name for name in ('bootstrap.py','infra.py','run.py','analyze.py','protocol.json','README.md','launch.py')])


def submit_days():
    rows=[]
    for scenario in protocol()['scenarios']:
        original=source()/'runs'/'collection'/scenario
        row=infra.task(protocol(),'recover/6202/'+scenario,['--seed',6202,'--scenario',scenario,
            '--dataset',infra.remote(original/'dataset.json'),'--heldout',infra.remote(original/'heldout_prediction_inputs.json'),
            '--out',infra.remote(HERE/'runs'/scenario)],ram_mb=24000)
        row['cpu']=3
        rows.append(row)
    return infra.submit(rows,'recovery')


def submit_finalize():
    return infra.submit([infra.task(protocol(),'partial/finalize',['--root',infra.remote(HERE/'runs'),
        '--source',infra.remote(source()),'--out',infra.remote(HERE/'recovery_summary.json')],
        runner='analyze.py',ram_mb=2048)],'finalize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','days','finalize','collect'))
    op=parser.parse_args().operation
    result=stage() if op=='stage' else submit_days() if op=='days' else submit_finalize() if op=='finalize' else infra.collect('recovery_summary.json',max_bytes=24000)
    print(json.dumps(result,indent=2))
