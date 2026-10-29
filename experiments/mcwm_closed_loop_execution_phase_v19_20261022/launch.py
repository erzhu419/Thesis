"""Read existing traces, then gate one plus five frozen execution-rule branches."""
import argparse
import json
from pathlib import Path
import shlex

import infra

HERE = Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/'protocol.json').read_text())


def source_root():
    return HERE.parent/protocol()['source_version']


def source_day():
    p=protocol()
    return source_root()/'runs'/p['fixed_scenario']/f"seed{p['fixed_seed']}"


def branches():
    return [(line,arm) for line in protocol()['fixed_lines'] for arm in ('low','high')]


def first_branch():
    p=protocol()['intervention']['first_branch']
    return p['line'],p['arm']


def branch_path(line,arm):
    if (line,arm)==first_branch():
        return Path(protocol()['intervention']['first_branch']['output_relative'])
    return Path('branches')/line/arm


def gate(relative):
    path=HERE/relative
    value=json.loads(path.read_text()) if path.is_file() else infra.collect_json(str(relative),max_bytes=10000)
    if value.get('status')!='complete' or value.get('integrity',{}).get('all_pass') is not True:
        raise RuntimeError('Execution-phase diagnostic gate failed: '+str(relative))
    return value


def submit_shadow():
    args=['--source',infra.remote(source_root()),'--out',infra.remote(HERE/'shadow_summary.json')]
    return infra.submit([infra.task(protocol(),'shadow',args,runner='shadow.py',ram_mb=4096)],'shadow')


def branch_job(line,arm):
    p=protocol()
    path=branch_path(line,arm)
    args=['--mode','branch','--seed',str(p['fixed_seed']),'--scenario',p['fixed_scenario'],
          '--source',infra.remote(source_day()),'--target-id',f"train/{line}/{p['fixed_window']}",
          '--branch',arm,'--out',infra.remote(HERE/path)]
    return infra.task(p,str(path),args,runner='run.py',ram_mb=8000)


def submit_preflight():
    decision=gate(Path('shadow_decision.json'))
    if decision.get('material_difference') is not True:
        raise RuntimeError('Material shadow difference required before new suffix queries')
    return infra.submit([branch_job(*first_branch())],'preflight_retest')


def submit_remaining():
    gate(Path('preflight_gate.json'))
    return infra.submit([branch_job(line,arm) for line,arm in branches() if (line,arm)!=first_branch()], 'intervention')


def submit_finalizer(*,preflight=False):
    batch='preflight_retest' if preflight else 'intervention'
    json.loads((HERE/(batch+'_submission.json')).read_text())
    out=HERE/('preflight_gate.json' if preflight else 'intervention_summary.json')
    args=['--source',infra.remote(source_root()),'--root',infra.remote(HERE),'--out',infra.remote(out),'--wait']
    if preflight:args.append('--preflight')
    return infra.submit([infra.task(protocol(),'preflight/finalize' if preflight else 'intervention/finalize',
        args,runner='analyze.py',ram_mb=1024)],'preflight_finalize' if preflight else 'intervention_finalize')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('stage','shadow','collect-shadow','preflight','remaining','finalize','collect'))
    args=parser.parse_args()
    if args.operation=='stage':
        names=('bootstrap.py','phase_stream.py','run.py','analyze.py','shadow.py','infra.py','launch.py','continue_batch.py','protocol.json')
        value=infra.stage([HERE/name for name in names])
    elif args.operation=='shadow':value=submit_shadow()
    elif args.operation=='collect-shadow':value=infra.collect_json('shadow_summary.json',max_bytes=80000)
    elif args.operation=='preflight':value=submit_preflight()
    elif args.operation=='remaining':value=submit_remaining()
    elif args.operation=='finalize':value=submit_finalizer()
    else:value=infra.collect_json('intervention_summary.json',max_bytes=30000)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
