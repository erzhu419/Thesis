"""One v53 native day:24 warmup updates and gated same-prefix acquisition."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--seed',type=int,default=6202)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text())
    started=time.time()
    try:
        from experiments.mcwm_closed_loop_sync_collection_v53_20261025.collect import collect
        result=collect(args,protocol)
        print(json.dumps(dict(status=result['status'],integrity=result['integrity'])),flush=True)
    except BaseException as error:
        args.out.mkdir(parents=True,exist_ok=True)
        path=args.out/'run_failure.json'
        if not path.exists():
            path.write_text(json.dumps(dict(type=type(error).__name__,message=str(error),
                started=started,finished=time.time()),separators=(',',':'))+'\n')
        raise


if __name__=='__main__': main()
