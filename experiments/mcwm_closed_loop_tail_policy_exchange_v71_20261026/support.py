"""Fixed actor handoff and original-diagonal checks; all artifacts stay resident."""
from pathlib import Path
import json
import time

import torch

ARMS = {'low':'fresh', 'high':'no_fresh'}
COUNTS = dict(critic=456,value=456,actor=456,alpha=456,policy=456,world=0,trust=0)


class PrefixComplete(Exception):
    def __init__(self, receipt):
        self.receipt = receipt


def publish_actor(path, learner, *, seed, arm, counts):
    if counts != COUNTS:
        raise RuntimeError('Final policy published before all456 updates')
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary=path.with_suffix('.pending')
    torch.save(dict(seed=seed,arm=arm,counts=counts,
        actor={key:value.detach().cpu().clone() for key,value in learner.actor.state_dict().items()}),temporary)
    temporary.replace(path)


def opposite_actor(path, *, seed, arm, timeout=3600):
    path=Path(path);started=last=time.monotonic()
    while not path.exists():
        failed=path.parent/'failure.json'
        if failed.exists():
            raise RuntimeError('Opposite prefix failed; no replacement actor used: '+str(failed))
        now=time.monotonic()
        if now-started>timeout:
            raise TimeoutError('Opposite final actor not ready; prefix retained without rerun')
        if now-last>=45:
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='wait_same_seed_final_actor',seed=seed,arm=arm)),flush=True)
            last=now
        time.sleep(.2)
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['seed']!=seed or value['arm']!=arm or value['counts']!=COUNTS:
        raise RuntimeError('Opposite actor has wrong initialization or update budget')
    return value['actor']


def trace_child(old, path, *, reference_path, reference_cookie, diagonal):
    from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
    observed,matched=old.observed_issues,old.matched_issues
    old.close()
    result=NativePrefixTrace(path,reference_path if diagonal else None)
    if diagonal:
        result.reference.seek(reference_cookie)
    result.observed_issues,result.matched_issues=observed,matched
    return result


def cost_windows(metrics, denominator):
    if denominator!=15079:
        raise RuntimeError('Departed passenger cohort changed')
    points={str(int(row['time_seconds'])):row for row in metrics if row['time_seconds'] in (18300,20100,24000)}
    if set(points)!={'18300','20100','24000'}:
        raise RuntimeError('Missing actual native cost endpoint')
    result={}
    for key,start in (('tail','18300'),('post20100','20100')):
        lo,hi=points[start],points['24000']
        wait=hi['accrued_wait_seconds']-lo['accrued_wait_seconds']
        ride=hi['accrued_in_vehicle_seconds']-lo['accrued_in_vehicle_seconds']
        result[key]=dict(wait=wait,ride=ride,cost=2*wait+ride,
            denominator=denominator,cost_per_passenger=(2*wait+ride)/denominator)
    return points,result


def diagonal_checks(summary, reference, metrics, reference_metrics, trace, diagonal):
    if not diagonal:
        return dict(ledger_exact=None,metrics_exact=None,full_remaining_trace_exact=None)
    return dict(ledger_exact=summary['cost']==reference['cost'],metrics_exact=metrics==reference_metrics,
        full_remaining_trace_exact=trace.comparison_exact and trace.matched_issues==
            trace.observed_issues==reference['factual_trace_rows'] and trace.reference.readline()=='')


def acquire_tail_slot(seed_dir, arm, clock, timeout=3600):
    """Pause both prefixes, then run one entry's two suffixes on the reserved2CPU.

    Return the locked file to the parent. Forked suffixes close their inherited
    descriptor without LOCK_UN; the parent unlocks after both suffixes finish.
    """
    import fcntl
    import os

    seed_dir=Path(seed_dir)
    peer='no_fresh' if arm=='fresh' else 'fresh'
    ready=seed_dir/arm/'entry_ready.json'
    pending=ready.with_suffix('.pending')
    with pending.open('x') as output:
        json.dump(dict(arm=arm,clock=float(clock),pid=os.getpid()),output,allow_nan=False)
        output.write('\n')
    pending.replace(ready)
    started=last=time.monotonic()
    lock=None
    try:
        while True:
            failure=seed_dir/peer/'failure.json'
            if failure.exists():
                raise RuntimeError('Opposite prefix failed before tail execution: '+str(failure))
            both_ready=(seed_dir/peer/'entry_ready.json').exists()
            if both_ready:
                if lock is None:
                    lock=(seed_dir/'tail_execution.lock').open('a')
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    return lock
                except BlockingIOError:
                    pass
            now=time.monotonic()
            if now-started>timeout:
                raise TimeoutError('Same-seed tail entry or execution slot not ready; prefixes retained')
            if now-last>=45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(
                    phase='wait_tail_execution_slot' if both_ready else 'wait_same_seed_tail_entry',
                    arm=arm,current=float(clock),unit='sim_seconds')),flush=True)
                last=now
            time.sleep(.2)
    except BaseException:
        if lock is not None:
            lock.close()
        raise
