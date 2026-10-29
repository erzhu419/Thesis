"""One2CPU seed task restores two packet51 prefixes and four frozen suffixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def run(seed,out):
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    children={};logs={};started=last=time.monotonic()
    try:
        for arm in ('fresh','no_fresh'):
            logs[arm]=(out/(arm+'_parent.log')).open('x')
            command=[sys.executable,'-u',str(HERE/'run.py'),'--seed',str(seed),
                '--scenario','stable','--arm',arm,'--out',str(out/arm)]
            children[arm]=subprocess.Popen(command,stdout=logs[arm],stderr=subprocess.STDOUT)
        while any(child.poll() is None for child in children.values()):
            now=time.monotonic()
            if now-last>=45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='paired_prefixes_and_tails',
                    seed=seed,elapsed_seconds=now-started,
                    children={arm:dict(pid=child.pid,returncode=child.poll()) for arm,child in children.items()})),flush=True)
                last=now
            time.sleep(1)
        rows={arm:json.loads((out/arm/'prefix_summary.json').read_text())
              for arm in children if (out/arm/'prefix_summary.json').exists()}
        checks=dict(both_prefixes_complete=len(rows)==2,
            processes_successful=all(child.returncode==0 for child in children.values()),
            prefix_integrity=len(rows)==2 and all(row['status']=='complete' and row['integrity']['all_pass'] for row in rows.values()),
            four_suffixes_present=all((out/arm/'suffixes'/branch/'summary.json').exists()
                for arm in children for branch in ('low','high')))
        result=dict(status='complete' if all(checks.values()) else 'failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),seed=seed,
            children={arm:dict(pid=child.pid,returncode=child.returncode) for arm,child in children.items()},
            wall_seconds=time.monotonic()-started,
            prefix_summaries={arm:str(out/arm/'prefix_summary.json') for arm in rows})
        with (out/'seed_summary.json').open('x') as handle: json.dump(result,handle,indent=2)
        if not all(checks.values()): raise RuntimeError('Fixed seed restoration/tails failed; no rerun')
        return result
    finally:
        for handle in logs.values(): handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(6208,6207),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.seed,args.out)
