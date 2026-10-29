"""An equal fork anchor must not conceal unequal earlier commands or states."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace


def issue(sequence=0):
    return dict(agent='bus0', line_id='7X', vehicle_id='fleet0', trip_id='trip0',
        station_id='diagnostic', domain='target', episode_id=0, sequence=sequence,
        started_at=18100.+sequence, s=np.arange(229, dtype=np.float32),
        raw_s=np.arange(500, dtype=np.float32), event=np.arange(40, dtype=np.float32),
        a=np.asarray([-.75], np.float32), proposed_a=np.asarray([.2], np.float32))


def snapshot(sequence=0):
    return SimpleNamespace(time_seconds=18100.+sequence, waiting_passengers=12,
        in_vehicle_passengers=7, accrued_wait_seconds=12345., accrued_in_vehicle_seconds=67890.)


def reference(tmp_path, n=3):
    path = tmp_path/'main.jsonl'
    trace = NativePrefixTrace(path)
    try:
        for sequence in range(n):
            trace.observe(issue(sequence), snapshot(sequence))
    finally:
        trace.close()
    return path


def test_every_prefix_issue_is_compared_without_requiring_main_suffix(tmp_path):
    trace = NativePrefixTrace(tmp_path/'pair.jsonl', reference(tmp_path))
    try:
        for sequence in range(2):
            trace.observe(issue(sequence), snapshot(sequence))
        assert trace.matched_issues == trace.observed_issues == 2
        assert trace.comparison_exact
    finally:
        trace.close()


@pytest.mark.parametrize('field', ['s', 'raw_s', 'event', 'a', 'proposed_a',
                                  'line_id', 'trip_id', 'station_id', 'started_at'])
def test_early_prefix_differences_fail_even_when_later_anchor_would_match(tmp_path, field):
    trace = NativePrefixTrace(tmp_path/'pair.jsonl', reference(tmp_path))
    event = issue()
    if isinstance(event[field], np.ndarray):
        event[field][0] += 1.
    elif isinstance(event[field], str):
        event[field] += '_changed'
    else:
        event[field] += 1.
    try:
        with pytest.raises(RuntimeError, match=field):
            trace.observe(event, snapshot())
        assert trace.matched_issues == 0
        assert not trace.comparison_exact
    finally:
        trace.close()


def test_native_ledger_mismatch_is_detected(tmp_path):
    trace = NativePrefixTrace(tmp_path/'pair.jsonl', reference(tmp_path))
    ledger = snapshot()
    ledger.accrued_wait_seconds += 1.
    try:
        with pytest.raises(RuntimeError, match='issue_ledger'):
            trace.observe(issue(), ledger)
    finally:
        trace.close()


def test_fork_compares_nominal_command_but_keeps_actual_branch_command_in_trace(tmp_path):
    path = tmp_path/'pair.jsonl'
    trace = NativePrefixTrace(path, reference(tmp_path))
    event, nominal = issue(), np.asarray([-.75], np.float32)
    event['a'] = np.asarray([-1.], np.float32)
    try:
        trace.observe(event, snapshot(), nominal_action=nominal)
        assert trace.matched_issues == 1 and trace.comparison_exact
    finally:
        trace.close()
    assert json.loads(path.read_text())['a'] == [-1.]


def test_extra_issue_beyond_reference_is_not_claimed_as_match(tmp_path):
    trace = NativePrefixTrace(tmp_path/'pair.jsonl', reference(tmp_path, n=1))
    try:
        trace.observe(issue(), snapshot())
        with pytest.raises(RuntimeError, match='reference ended'):
            trace.observe(issue(1), snapshot(1))
        assert trace.matched_issues == 1 and not trace.comparison_exact
    finally:
        trace.close()
