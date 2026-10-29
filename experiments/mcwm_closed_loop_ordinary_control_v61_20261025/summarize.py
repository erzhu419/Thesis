"""Reuse the frozen full-day cost calculation on the ordinary-control outputs."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_control_campaign_v60_20261025.control_analyze import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('development','confirmation'),required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();value=analyze(HERE,'recent30',args.stage)
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode())>48000:raise ValueError('Ordinary-control result exceeds48KB')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as output:output.write(raw)
    print(json.dumps({k:value[k] for k in ('status','candidate','stage','decision')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
