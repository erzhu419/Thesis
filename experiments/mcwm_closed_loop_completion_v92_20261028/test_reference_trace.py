from copy import deepcopy
from io import StringIO
import json
from unittest.mock import patch

import pytest

from experiments.mcwm_closed_loop_completion_v92_20261028.reference_trace import JoinedReference


def row(sequence):
    return dict(sequence=sequence, started_at=float(sequence), a=[-.9],
                s=[float(sequence), .3], issue_ledger=dict(time_seconds=float(sequence)))


def reader(parent=None, arm=None):
    parent = [row(index) for index in range(4)] if parent is None else parent
    arm = [row(index) for index in range(2, 8)] if arm is None else arm
    streams = [StringIO(''.join(json.dumps(value)+'\n' for value in rows)) for rows in (parent, arm)]
    with patch.object(__import__('pathlib').Path, 'open', side_effect=streams):
        value = JoinedReference('/fake/parent.jsonl', '/fake/reference.jsonl', stop_at=6.)
    return value


def read_all(value):
    rows = []
    while True:
        raw = value.readline()
        if not raw:
            return rows
        rows.append(json.loads(raw))


def test_overlap_is_verified_and_emitted_once_then_stops_before_boundary():
    value = reader()
    assert read_all(value) == [row(index) for index in range(6)]
    assert value.readline() == ''
    assert value.summary() == dict(parent_trace_path='/fake/parent.jsonl',
        reference_arm_trace_path='/fake/reference.jsonl', stop_at=6., parent_rows=4,
        overlap_rows=2, continuation_rows=2, last_sequence=5, boundary_reached=True,
        merged_file_created=False)


def test_tell_does_not_consume_and_inherited_child_close_does_not_move_parent():
    value = reader()
    assert json.loads(value.readline())['sequence'] == 0
    cookie = value.tell()
    assert value.tell() == cookie
    inherited = deepcopy(value)
    inherited.close()
    assert value.tell() == cookie
    assert json.loads(value.readline())['sequence'] == 1
    assert value.tell() != cookie
    value.close()
    value.close()
    with pytest.raises(ValueError):
        value.readline()


def test_parent_gap_or_repeat_cannot_be_hidden_by_the_reference_tail():
    for indices in ([0, 1, 3], [0, 1, 1, 2, 3]):
        with pytest.raises(RuntimeError, match='skips or repeats|does not align'):
            read_all(reader(parent=[row(index) for index in indices]))


def test_arm_gap_or_repeat_is_rejected_even_inside_the_overlap():
    for indices in ([2, 4, 5, 6], [2, 2, 3, 4, 5, 6]):
        with pytest.raises(RuntimeError, match='skips or repeats'):
            read_all(reader(arm=[row(index) for index in indices]))


def test_full_shared_row_is_compared_not_only_sequence_and_clock():
    arm = [row(index) for index in range(2, 8)]
    arm[0]['a'] = [-.7]
    with pytest.raises(RuntimeError, match='differ in their overlap'):
        read_all(reader(arm=arm))


def test_no_verified_overlap_or_truncated_continuation_cannot_claim_completion():
    with pytest.raises(RuntimeError, match='nonempty verified'):
        read_all(reader(arm=[row(index) for index in range(4, 8)]))
    with pytest.raises(RuntimeError, match='ends before|ends inside'):
        read_all(reader(arm=[row(index) for index in range(2, 5)]))
