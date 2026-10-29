"""Continue only the fixed v83 eight-day campaign, then stop."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.continue_control import (
    Continuation, write_json)
from experiments.mcwm_closed_loop_support_td_v83_20261027.launch import advance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=45, choices=range(1,61))
    args = parser.parse_args()
    with (HERE/'continue_control.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('An existing v83 watcher owns the receipts; no duplicate started.', flush=True)
            return
        write_json(HERE/'continue_control_process.json', dict(pid=os.getpid(),started_unix=time.time()))
        Continuation(HERE, advance=advance).run(watch=args.watch, interval=args.interval)


if __name__ == '__main__':
    main()
