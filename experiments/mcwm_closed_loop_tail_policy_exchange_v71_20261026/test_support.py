"""Actor handoff, independent trace positions and fixed native cost endpoints."""
import json
import fcntl
import multiprocessing
import os
import queue
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026 import support as v
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def rng_exact(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_final_actor_handoff_preserves_native_rng_and_exact_parameters(tmp_path):
    source = torch.nn.Linear(3, 1)
    target = torch.nn.Linear(3, 1)
    path = tmp_path/'actor_final.pt'
    before = rng_state()
    v.publish_actor(path, SimpleNamespace(actor=source), seed=6201, arm='fresh', counts=v.COUNTS)
    assert path.exists() and not path.with_suffix('.pending').exists()
    target.load_state_dict(v.opposite_actor(path, seed=6201, arm='fresh', timeout=0))
    rng_exact(before, rng_state())
    assert all(torch.equal(value, target.state_dict()[key])
               for key, value in source.state_dict().items())
    with pytest.raises(FileExistsError):
        v.publish_actor(path, SimpleNamespace(actor=source), seed=6201, arm='fresh', counts=v.COUNTS)


def test_actor_handoff_rejects_incomplete_budget_or_wrong_seed_arm(tmp_path):
    learner = SimpleNamespace(actor=torch.nn.Linear(2, 1))
    path = tmp_path/'actor_final.pt'
    with pytest.raises(RuntimeError, match='before all456'):
        v.publish_actor(path, learner, seed=6201, arm='fresh',
                        counts=dict(v.COUNTS, actor=455))
    assert not path.exists()
    v.publish_actor(path, learner, seed=6201, arm='fresh', counts=v.COUNTS)
    for seed, arm in ((6203, 'fresh'), (6201, 'no_fresh')):
        with pytest.raises(RuntimeError, match='wrong initialization'):
            v.opposite_actor(path, seed=seed, arm=arm, timeout=0)


def test_trace_child_reopens_diagonal_at_saved_cookie_without_moving_other_reader(tmp_path):
    reference = tmp_path/'reference.jsonl'
    reference.write_text('\n'.join(json.dumps({'sequence': i}) for i in range(3))+'\n')
    old = NativePrefixTrace(tmp_path/'parent.jsonl', reference)
    independent_parent_reader = reference.open()
    try:
        old.reference.readline()
        independent_parent_reader.readline()
        cookie = old.reference.tell()
        old.observed_issues = old.matched_issues = 1
        child = v.trace_child(old, tmp_path/'child.jsonl', reference_path=reference,
                              reference_cookie=cookie, diagonal=True)
        try:
            assert old.writer.closed and old.reference.closed
            assert child.observed_issues == child.matched_issues == 1
            assert json.loads(child.reference.readline()) == {'sequence': 1}
            assert independent_parent_reader.tell() == cookie
            assert json.loads(independent_parent_reader.readline()) == {'sequence': 1}
        finally:
            child.close()
        crossing = NativePrefixTrace(tmp_path/'other_parent.jsonl', reference)
        cross = v.trace_child(crossing, tmp_path/'cross_child.jsonl',
                              reference_path=reference, reference_cookie=cookie, diagonal=False)
        assert cross.reference is None
        cross.close()
    finally:
        independent_parent_reader.close()


def metrics():
    return [dict(time_seconds=clock, accrued_wait_seconds=wait,
                 accrued_in_vehicle_seconds=ride)
            for clock, wait, ride in ((18300., 100., 200.), (20100., 130., 240.),
                                      (24000., 250., 500.))]


def test_window_costs_use_actual_18300_20100_endpoints_and_full_day_cohort():
    points, windows = v.cost_windows(metrics(), 15079)
    assert set(points) == {'18300', '20100', '24000'}
    assert windows['tail'] == dict(wait=150., ride=300., cost=600., denominator=15079,
                                   cost_per_passenger=600./15079)
    assert windows['post20100']['cost_per_passenger'] == 500./15079
    with pytest.raises(RuntimeError, match='cohort changed'):
        v.cost_windows(metrics(), 100)
    with pytest.raises(RuntimeError, match='Missing actual native'):
        v.cost_windows(metrics()[1:], 15079)


def test_diagonal_requires_exact_metrics_ledger_and_exhausted_full_trace(tmp_path):
    path = tmp_path/'reference.jsonl'
    path.write_text('')
    trace = SimpleNamespace(comparison_exact=True, matched_issues=3, observed_issues=3,
                            reference=path.open())
    summary = dict(cost={'value': 100.})
    reference = dict(cost={'value': 100.}, factual_trace_rows=3)
    try:
        assert all(v.diagonal_checks(summary, reference, metrics(), metrics(), trace, True).values())
        changed = metrics()
        changed[-1]['accrued_wait_seconds'] += 1
        assert v.diagonal_checks(summary, reference, changed, metrics(), trace, True)['metrics_exact'] is False
        trace.matched_issues = 2
        assert v.diagonal_checks(summary, reference, metrics(), metrics(), trace, True)['full_remaining_trace_exact'] is False
        assert set(v.diagonal_checks(summary, reference, changed, metrics(), trace, False).values()) == {None}
    finally:
        trace.reference.close()


def tail_slot_worker(seed_dir, arm, release, events):
    lock = v.acquire_tail_slot(seed_dir, arm, 18300., timeout=10)
    # Match the native fork contract: both suffixes close the inherited handle,
    # while the paused parent continues to own its exclusive open description.
    children = []
    for _ in range(2):
        child = os.fork()
        if child == 0:
            lock.close()
            os._exit(0)
        children.append(child)
    for child in children:
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    ready = all((seed_dir/name/'entry_ready.json').exists() for name in ('fresh', 'no_fresh'))
    events.put(('acquired', arm, time.monotonic(), ready))
    if not release.wait(10):
        raise RuntimeError('Test parent did not release tail slot')
    released = time.monotonic()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    events.put(('released', arm, released, True))


def test_two_prefix_barrier_and_exclusive_tail_slot_survive_both_child_closes(tmp_path):
    for arm in ('fresh', 'no_fresh'):
        (tmp_path/arm).mkdir()
    ctx = multiprocessing.get_context('fork')
    events = ctx.Queue()
    releases = {arm: ctx.Event() for arm in ('fresh', 'no_fresh')}
    processes = {arm: ctx.Process(target=tail_slot_worker,
        args=(tmp_path, arm, releases[arm], events)) for arm in releases}
    try:
        processes['fresh'].start()
        deadline = time.monotonic()+5
        while not (tmp_path/'fresh/entry_ready.json').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert (tmp_path/'fresh/entry_ready.json').exists()
        assert not (tmp_path/'fresh/entry_ready.pending').exists()
        with pytest.raises(queue.Empty):
            events.get(timeout=.25)
        assert not (tmp_path/'tail_execution.lock').exists()
        processes['no_fresh'].start()
        first = events.get(timeout=5)
        assert first[0] == 'acquired' and first[3]
        # The other entry stays paused even after both forked children closed
        # their descriptors. Its parent must not run a third native suffix.
        with pytest.raises(queue.Empty):
            events.get(timeout=.25)
        releases[first[1]].set()
        second, released_first = None, None
        for _ in range(2):
            event = events.get(timeout=5)
            if event[0] == 'acquired':
                second = event
            else:
                released_first = event
        assert second and released_first and second[3]
        assert second[1] != first[1]
        assert second[2] >= released_first[2]
        releases[second[1]].set()
        assert events.get(timeout=5)[:2] == ('released', second[1])
        for process in processes.values():
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        for release in releases.values():
            release.set()
        for process in processes.values():
            if process.pid is not None:
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
        events.close()


def test_tail_slot_propagates_peer_failure_without_starting_suffixes(tmp_path):
    for arm in ('fresh', 'no_fresh'):
        (tmp_path/arm).mkdir()
    (tmp_path/'no_fresh/failure.json').write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError, match='Opposite prefix failed'):
        v.acquire_tail_slot(tmp_path, 'fresh', 18301., timeout=0)
    assert json.loads((tmp_path/'fresh/entry_ready.json').read_text())['clock'] == 18301.
    assert not (tmp_path/'tail_execution.lock').exists()
