"""Reconstruct only v32's admitted factual labels from its resident trace.

The full trace supplies exact adjacent global decision endpoints and optional
actor replay checks. It never supplies additional training labels. No native
simulation, diagnostic branch truth, or checkpoint is read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.mcwm_closed_loop_v1.stream import Transition, transition_reward_v7


def iter_issues(run_root):
    """Stream ordinary factual controls in their original issue order."""
    with (Path(run_root)/'prefix_trace.jsonl').open() as source:
        for line in source:
            yield json.loads(line)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _context(context, clock, name):
    _require(float(context['observed_at']) == clock, f'{name} is not aligned with its ledger clock')
    metadata = context['metadata']
    for key in ('latest_native_observation_at', 'latest_recent_apply_at',
                'latest_recent_physical_observation_at'):
        value = metadata.get(key)
        _require(value is None or float(value) <= clock, f'{name} includes future {key}')
    _require(not metadata['command_is_feature'], f'{name} contains a command feature')
    _require(not metadata['selected_diagnostic_vehicle'], f'{name} selects a diagnostic vehicle')
    vector = np.asarray(context['vector'], dtype=np.float32)
    _require(vector.ndim == 1 and bool(np.isfinite(vector).all()), f'{name} has an invalid context vector')
    _require(len(vector) == len(context['feature_names']), f'{name} feature names are misaligned')
    return len(vector)


def _transition(start, end, context, contract, delay):
    sequence = int(start['sequence'])
    _require(int(end['sequence']) == sequence+1, f'No adjacent global endpoint for sequence {sequence}')
    issue, completed = float(start['started_at']), float(end['started_at'])
    _require(float(start['issue_ledger']['time_seconds']) == issue, 'Issue ledger clock differs from issue')
    _require(float(end['issue_ledger']['time_seconds']) == completed, 'Next ledger clock differs from completion')
    _require(context['line_id'] == start['line_id'], f'Context line differs at sequence {sequence}')
    _require(float(context['issue']) == issue and float(context['completed']) == completed,
        f'Context endpoint clocks differ at sequence {sequence}')
    dt, available = completed-issue, completed+delay
    _require(dt > 0 and dt == float(context['dt']), f'Nonpositive or different duration at sequence {sequence}')
    _require(float(context['available']) == available, f'Different feedback availability at sequence {sequence}')
    s_dim = _context(context['memory_s'], issue, 'memory_s')
    ns_dim = _context(context['memory_ns'], completed, 'memory_ns')
    _require(s_dim == ns_dim, 'Context dimension changed within a transition')
    _require(context['memory_s']['feature_names'] == context['memory_ns']['feature_names'],
        'Context feature order changed within a transition')
    row = Transition(s=np.asarray(start['s'], np.float32), a=np.asarray(start['a'], np.float32),
        ns=np.asarray(end['s'], np.float32),
        r=float(transition_reward_v7(start['issue_ledger'], end['issue_ledger'], contract)),
        dt=dt, terminal=False, episode_id=int(start['episode_id']), domain=start['domain'],
        started_at=issue, completed_at=completed, available_at=available,
        vehicle_id=str(start['vehicle_id']), line_id=str(start['line_id']),
        station_id=str(start['station_id']), next_vehicle_id=str(end['vehicle_id']),
        sequence=sequence, raw_s=np.asarray(start['raw_s'], np.float32),
        raw_ns=np.asarray(end['raw_s'], np.float32), trip_id=str(start['trip_id']))
    row.memory_s, row.memory_ns = context['memory_s'], context['memory_ns']
    return row


def load_facts(run_root, contract, protocol=None):
    """Return original packets, exact update references and reconstruction proof.

    Packet ``processed_before_sequence`` identifies where updates precede the
    next issue. All updates at one processed clock occur before its first issue,
    including a simultaneous batch; they must not split that batch.
    """
    root = Path(run_root)
    summary = json.loads((root/'summary.json').read_text())
    _require(summary['status'] == 'complete' and summary['integrity']['all_pass'],
        'Only a completed, verified v32 factual day is supported')
    if protocol is None:
        protocol = json.loads((root/'protocol.json').read_text())['protocol']
    receipts = summary['receipts']
    deadlines = list(range(protocol['feedback_first_deadline'], protocol['feedback_last_deadline']+1,
        protocol['feedback_interval_seconds']))
    _require([row['deadline'] for row in receipts] == deadlines, 'Packet deadline schedule differs')
    _require(all(row['records'] == protocol['feedback_records_per_packet'] == len(row['sequence_ids'])
        for row in receipts), 'Packet label counts differ')
    selected = [int(sequence) for receipt in receipts for sequence in receipt['sequence_ids']]
    _require(len(selected) == len(set(selected)) == protocol['target_record_budget'],
        'Feedback sequence IDs are duplicated or do not match the original budget')
    selected_set = set(selected)
    contexts = {}
    with (root/'memory_training_rows.jsonl').open() as source:
        for line in source:
            context = json.loads(line)
            sequence = int(context['sequence'])
            _require(sequence in selected_set and sequence not in contexts,
                'Context rows include extra or duplicate training labels')
            contexts[sequence] = context
    _require(set(contexts) == selected_set, 'An admitted context row is missing')
    with (root/'updates.jsonl').open() as source:
        updates = [json.loads(line) for line in source]
    _require(len(updates) == len(receipts), 'First-step update reference count differs')

    transitions, before_sequences, after_sequences = {}, {}, {}
    previous, trace_count, last_clock = None, 0, None
    insertion_index = 0
    for current in iter_issues(root):
        sequence, clock = int(current['sequence']), float(current['started_at'])
        _require(sequence == trace_count, 'Factual trace sequence is not complete and consecutive')
        _require(last_clock is None or clock >= last_clock, 'Factual issue clocks moved backwards')
        while insertion_index < len(receipts) and clock >= float(receipts[insertion_index]['processed_at']):
            before_sequences[insertion_index] = sequence
            after_sequences[insertion_index] = sequence-1 if previous is not None else None
            insertion_index += 1
        if previous is not None and int(previous['sequence']) in selected_set:
            transition = _transition(previous, current, contexts[int(previous['sequence'])],
                contract, float(protocol['feedback_delay_seconds']))
            transitions[transition.sequence] = transition
        previous, last_clock = current, clock
        trace_count += 1
    _require(set(transitions) == selected_set, 'An admitted label has no next global decision endpoint')
    _require(trace_count == summary['factual_trace_rows'], 'Complete factual trace row count differs')
    _require(insertion_index == len(receipts), 'An update has no subsequent factual issue for replay placement')

    packets = []
    feature_names = None
    for index, (receipt, update) in enumerate(zip(receipts, updates)):
        rows = [transitions[int(sequence)] for sequence in receipt['sequence_ids']]
        lower = float(receipt['deadline'])-protocol['feedback_interval_seconds']
        _require(all(lower < row.available_at <= receipt['deadline'] <= receipt['processed_at'] for row in rows),
            'An admitted label is outside its original feedback window')
        _require(receipt['max_available_at'] == max(row.available_at for row in rows),
            'Packet availability receipt differs')
        _require(update['packet'] == index+1 and update['processed_at'] == receipt['processed_at'],
            'Packet update reference does not match the original receipt')
        _require([row.sequence for row in rows] == [row.sequence for row in sorted(rows,
            key=lambda row: (row.available_at, row.sequence))], 'Original packet order changed')
        for row in rows:
            names = row.memory_s['feature_names']
            feature_names = names if feature_names is None else feature_names
            _require(names == feature_names, 'Context feature order changed across admitted labels')
        packets.append(dict(index=index+1, receipt=receipt, rows=rows, reference_update=update,
            processed_before_sequence=before_sequences[index], processed_after_sequence=after_sequences[index]))
    checks = dict(original_budget_only=True, admitted_sequences_unique=True,
        adjacent_global_endpoints=True, ledger_clocks_exact=True, reward_native_function=True,
        delay_and_positive_duration_exact=True, context_causal=True, original_packet_order=True,
        complete_factual_trace=True, no_native_simulation=True, no_diagnostic_truth=True)
    proof = dict(all_pass=all(checks.values()), checks=checks)
    return dict(packets=packets, reference_updates=updates,
        metadata=dict(run_root=str(root), admitted_labels=len(selected), packets=len(packets),
            factual_trace_rows=trace_count, feature_dimension=len(feature_names),
            feature_names=feature_names, first_sequence=min(selected), last_sequence=max(selected),
            new_training_labels=0, native_simulation_seconds=0,
            reference_budget=summary.get('budget'), reference_cell=summary.get('cell')),
        proof=proof)
