"""Exact factual endpoints, original budget boundaries and replay placement."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.facts import load_facts, iter_issues


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def fixture(root):
    root.mkdir()
    times = [10., 15., 20., 25., 25., 30., 35., 40., 45., 50.]
    traces = []
    for sequence, clock in enumerate(times):
        traces.append(dict(sequence=sequence, started_at=clock, vehicle_id=f'fleet{sequence%3}',
            line_id='L', trip_id=f'trip{sequence%3}', station_id=f'stop{sequence}',
            episode_id=0, domain='target', s=[sequence, .5], raw_s=[clock-1., sequence],
            a=[-.75], proposed_a=[-.5], issue_ledger=dict(time_seconds=clock,
                waiting_passengers=2., in_vehicle_passengers=3.,
                accrued_wait_seconds=2.*clock, accrued_in_vehicle_seconds=3.*clock)))

    def context(clock, value):
        return dict(observed_at=clock, vector=[value], feature_names=['L:feature'], line_ids=['L'],
            metadata=dict(latest_native_observation_at=clock-1., latest_recent_apply_at=clock-2.,
                latest_recent_physical_observation_at=clock-1., command_is_feature=False,
                selected_diagnostic_vehicle=False))

    selected = [0, 1, 4, 5]
    contexts = [dict(sequence=sequence, line_id='L', issue=times[sequence],
        completed=times[sequence+1], available=times[sequence+1]+3.,
        dt=times[sequence+1]-times[sequence], memory_s=context(times[sequence], sequence+.1),
        memory_ns=context(times[sequence+1], sequence+.2)) for sequence in selected]
    receipts = [dict(deadline=25, processed_at=25., records=2, sequence_ids=[0, 1], max_available_at=23.),
        dict(deadline=40, processed_at=40., records=2, sequence_ids=[4, 5], max_available_at=38.)]
    updates = [dict(packet=i+1, processed_at=row['processed_at'], marker=f'exact{i}')
        for i, row in enumerate(receipts)]
    protocol = dict(feedback_first_deadline=25, feedback_last_deadline=40,
        feedback_interval_seconds=15, feedback_records_per_packet=2,
        target_record_budget=4, feedback_delay_seconds=3.)
    summary = dict(status='complete', integrity=dict(all_pass=True), receipts=receipts,
        factual_trace_rows=len(traces))
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'protocol.json').write_text(json.dumps(dict(protocol=protocol)))
    write_lines(root/'prefix_trace.jsonl', traces)
    write_lines(root/'memory_training_rows.jsonl', contexts)
    write_lines(root/'updates.jsonl', updates)
    return traces, contexts, summary


def test_reconstruction_uses_exact_next_global_endpoint_and_native_ledger_reward(tmp_path):
    root = tmp_path/'facts'
    traces, contexts, _ = fixture(root)
    facts = load_facts(root, TransitMDPConfigV7())
    assert facts['proof']['all_pass']
    rows = [row for packet in facts['packets'] for row in packet['rows']]
    assert [row.sequence for row in rows] == [0, 1, 4, 5]
    for row in rows:
        assert row.r == -35. and row.dt == 5. and row.terminal is False
        assert row.next_vehicle_id == traces[row.sequence+1]['vehicle_id']
        assert row.started_at == traces[row.sequence]['issue_ledger']['time_seconds']
        assert row.raw_s[0] == row.started_at-1.
        np.testing.assert_array_equal(row.s, np.asarray(traces[row.sequence]['s'], np.float32))
        np.testing.assert_array_equal(row.ns, np.asarray(traces[row.sequence+1]['s'], np.float32))
        assert row.memory_s == next(ctx['memory_s'] for ctx in contexts if ctx['sequence'] == row.sequence)
    assert facts['metadata']['admitted_labels'] == 4
    assert facts['metadata']['new_training_labels'] == 0
    assert list(iter_issues(root)) == traces
    assert len(traces) > len(rows)  # Ordinary endpoint rows do not become labels.


def test_updates_precede_the_first_issue_of_a_whole_same_clock_batch(tmp_path):
    root = tmp_path/'facts'
    _, _, _ = fixture(root)
    facts = load_facts(root, TransitMDPConfigV7())
    first, second = facts['packets']
    # Sequence3 and4 are both issued at25; updating after3 would split a batch.
    assert first['processed_before_sequence'] == 3
    assert first['processed_after_sequence'] == 2
    assert second['processed_before_sequence'] == 7
    assert second['reference_update'] == dict(packet=2, processed_at=40., marker='exact1')


@pytest.mark.parametrize('change, message', [
    ('duplicate', 'extra or duplicate'), ('extra', 'extra or duplicate'), ('missing', 'missing'),
    ('duration', 'duration'), ('available', 'availability'), ('clock', 'endpoint clocks'),
    ('future', 'future'), ('oracle', 'diagnostic vehicle'),
])
def test_invalid_budget_rows_and_context_are_rejected(tmp_path, change, message):
    root = tmp_path/'facts'
    _, contexts, _ = fixture(root)
    if change == 'duplicate':
        contexts.append(copy.deepcopy(contexts[0]))
    elif change == 'extra':
        contexts.append(dict(contexts[0], sequence=8))
    elif change == 'missing':
        contexts.pop()
    elif change == 'duration':
        contexts[0]['dt'] = 0.
    elif change == 'available':
        contexts[0]['available'] += 1.
    elif change == 'clock':
        contexts[0]['completed'] += 1.
    elif change == 'future':
        contexts[0]['memory_ns']['metadata']['latest_native_observation_at'] = 100.
    elif change == 'oracle':
        contexts[0]['memory_s']['metadata']['selected_diagnostic_vehicle'] = True
    write_lines(root/'memory_training_rows.jsonl', contexts)
    with pytest.raises(ValueError, match=message):
        load_facts(root, TransitMDPConfigV7())


def test_missing_final_endpoint_is_not_fabricated_as_terminal(tmp_path):
    root = tmp_path/'facts'
    traces, _, summary = fixture(root)
    write_lines(root/'prefix_trace.jsonl', traces[:6])
    summary['factual_trace_rows'] = 6
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='no next global decision endpoint'):
        load_facts(root, TransitMDPConfigV7())


def test_original_receipt_windows_and_unique_budget_are_enforced(tmp_path):
    root = tmp_path/'facts'
    _, _, summary = fixture(root)
    summary['receipts'][0]['sequence_ids'] = [0, 0]
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='duplicated'):
        load_facts(root, TransitMDPConfigV7())

    _, _, summary = fixture(tmp_path/'other')
    # Moving a receipt's processing earlier would permit a label before its
    # prescribed deadline despite retaining the same factual endpoint.
    summary['receipts'][0]['processed_at'] = 24.
    (tmp_path/'other'/'summary.json').write_text(json.dumps(summary))
    write_lines(tmp_path/'other'/'updates.jsonl', [dict(packet=1, processed_at=24.), dict(packet=2, processed_at=40.)])
    with pytest.raises(ValueError, match='outside its original feedback window'):
        load_facts(tmp_path/'other', TransitMDPConfigV7())
