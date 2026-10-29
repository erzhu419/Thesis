"""Two live-state Linux children; the caller owns their native suffix and exit."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _same_rng(before):
    py, array, tensor = _rng_state()
    old_py, old_array, old_tensor = before
    return (py == old_py and array[0] == old_array[0]
            and np.array_equal(array[1], old_array[1])
            and array[2:] == old_array[2:] and torch.equal(tensor, old_tensor))


def _trace_cookie(trace):
    return dict(writer=trace.writer.tell(), reference=None if trace.reference is None
                else trace.reference.tell())


def _write(path, value):
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def fork_children(out, trace, native_clock, counters, prepare_child, *, timeout_seconds=180):
    """Return an arm in each child, or a checked receipt in the paused parent.

    ``prepare_child(arm)`` must detach trace/log descriptors and set the caller's
    child output to ``out/arm``. Children later write ``summary.json`` there and
    exit with ``os._exit``; they must not return to the parent's normal cleanup.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trace.writer.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    saved_rng = _rng_state()
    native_before = float(native_clock())
    counters_before = copy.deepcopy(counters())
    cookie_before = _trace_cookie(trace)
    children, exitcodes, values = {}, {}, {}
    failure = None
    in_child = False
    started = last_progress = time.monotonic()
    try:
        for arm in ('low', 'high'):
            (out/arm).mkdir(parents=True, exist_ok=True)
            pid = os.fork()
            if pid == 0:
                in_child = True
                try:
                    # CPython reseeds random's singleton after fork. Restore it
                    # before the caller executes any child preparation.
                    random.setstate(saved_rng[0])
                    np.random.set_state(saved_rng[1])
                    torch.set_rng_state(saved_rng[2])
                    prepare_child(arm)
                    return arm
                except BaseException:
                    (out/arm/'fork_prepare_failure.txt').write_text(traceback.format_exc())
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(1)
            children[arm] = pid
        pending = set(children)
        while pending:
            for arm in tuple(pending):
                found, status = os.waitpid(children[arm], os.WNOHANG)
                if not found:
                    continue
                exitcodes[arm] = os.waitstatus_to_exitcode(status)
                pending.remove(arm)
                if exitcodes[arm] != 0:
                    raise RuntimeError(f'Live fork {arm} exited {exitcodes[arm]}; logs retained')
                value = json.loads((out/arm/'summary.json').read_text())
                if value.get('status') != 'complete' or not value.get('integrity', {}).get('all_pass'):
                    raise RuntimeError(f'Live fork {arm} failed native integrity; logs retained')
                values[arm] = value
            now = time.monotonic()
            if pending and now-started >= timeout_seconds:
                raise TimeoutError(f'Live fork timed out after {timeout_seconds}s: {sorted(pending)}')
            if pending and now-last_progress >= 45:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='live_fork_wait',
                    current=native_before, unit='sim_seconds', running=sorted(pending),
                    child_pids={arm:children[arm] for arm in sorted(pending)})), flush=True)
                last_progress = now
            if pending:
                time.sleep(.02)
    except BaseException as error:
        failure = error
    finally:
        # Reap both children even if a sibling failed or preparation deadlocked.
        if not in_child:
            for arm, pid in children.items():
                if arm in exitcodes:
                    continue
                found, status = os.waitpid(pid, os.WNOHANG)
                if not found:
                    os.kill(pid, signal.SIGKILL)
                    _, status = os.waitpid(pid, 0)
                exitcodes[arm] = os.waitstatus_to_exitcode(status)
    clocks = {arm:value['execution_wall'] for arm,value in values.items()}
    overlap = (min(value['native_finished_wall'] for value in clocks.values())
               -max(value['native_started_wall'] for value in clocks.values())) if len(clocks) == 2 else 0.
    receipt = dict(native_before=native_before, native_after=float(native_clock()),
        counters_before=counters_before, counters_after=counters(),
        parent_rng_unchanged=_same_rng(saved_rng),
        parent_trace_cookie_unchanged=cookie_before == _trace_cookie(trace),
        children=clocks, child_pids=children, child_exitcodes=exitcodes,
        overlap_seconds=max(0., overlap))
    receipt['all_pass'] = (failure is None and len(clocks) == 2 and overlap > 0
        and receipt['native_before'] == receipt['native_after']
        and receipt['counters_before'] == receipt['counters_after']
        and receipt['parent_rng_unchanged'] and receipt['parent_trace_cookie_unchanged'])
    if failure is not None:
        receipt['error'] = dict(type=type(failure).__name__, message=str(failure))
    _write(out/'live_fork.json', receipt)
    if failure is not None:
        raise failure
    if not receipt['all_pass']:
        raise RuntimeError('Live children overlap or paused-parent state check failed')
    return receipt
