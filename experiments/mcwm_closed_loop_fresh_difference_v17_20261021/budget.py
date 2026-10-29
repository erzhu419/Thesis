"""Two counterfactual action outcomes replace two of the original five slots."""
from __future__ import annotations

from copy import copy
import numpy as np

ARMS = ('low', 'high')
ARRAY_FIELDS = ('s', 'a', 'ns', 'raw_s', 'raw_ns')
TRANSITION_FIELDS = ('s', 'a', 'ns', 'r', 'dt', 'terminal', 'episode_id', 'domain',
    'started_at', 'completed_at', 'available_at', 'vehicle_id', 'line_id', 'station_id',
    'next_vehicle_id', 'sequence', 'raw_s', 'raw_ns', 'trip_id')


def paired_label_ids(index):
    """Native issue sequences are nonnegative; each paired action has its own ID."""
    index = int(index)
    if index < 0:
        raise ValueError('Pair index must be nonnegative')
    return dict(low=-(2*index+1), high=-(2*index+2))


def transition_to_dict(row):
    result = {}
    for key in TRANSITION_FIELDS:
        value = getattr(row, key)
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value.item() \
            if isinstance(value, np.generic) else value
    return result


def transition_from_dict(value):
    from experiments.mcwm_closed_loop_v1.stream import Transition
    result = dict(value)
    for key in ARRAY_FIELDS:
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], np.float32)
    return Transition(**result)


def pair_to_dict(record):
    result = dict(record)
    result['features'] = np.asarray(record['features'], dtype=float).tolist()
    result['outcomes'] = {arm: dict(value, transition=transition_to_dict(value['transition']))
        for arm, value in record['outcomes'].items()}
    return result


def pair_from_dict(value):
    result = dict(value)
    result['outcomes'] = {arm: dict(row, transition=transition_from_dict(row['transition']))
        for arm, row in value['outcomes'].items()}
    return result


class PairPacketBudget:
    """Admit one ready pair plus three ordinary rows, or five ordinary rows.

    The main trajectory's factual issue is excluded once it reserves a pair.
    Its two frozen-branch global transitions own the two replacement slots;
    their300-second waiting/ride fields supply a single contrast fit row.
    """

    def __init__(self, protocol):
        self.p = protocol
        self.deadlines = list(range(protocol['feedback_first_deadline'],
            protocol['feedback_last_deadline']+1, protocol['feedback_interval_seconds']))
        if len(self.deadlines)*protocol['feedback_records_per_packet'] != protocol['target_record_budget']:
            raise ValueError('Packet schedule differs from the original target label budget')
        self.rows, self.pending_pairs, self.admitted, self.receipts = [], [], [], []
        self.reserved, self.seen_pair_ids = {}, set()
        self.selected_sequences, self.factual_exclusions_seen = set(), set()
        self.last_pairs, self.index = [], 0

    def reserve(self, source_sequence, pair_id):
        sequence = int(source_sequence)
        if sequence < 0 or sequence in self.reserved or pair_id in self.reserved.values():
            raise ValueError('Reserve one previously unused native source issue per pair')
        if sequence in self.selected_sequences or any(row.sequence == sequence for row in self.rows):
            raise ValueError('Reserve the pair at issue before its factual outcome is ingested')
        self.reserved[sequence] = pair_id

    def ingest(self, rows):
        for row in rows:
            if row.sequence in self.reserved:
                self.factual_exclusions_seen.add(int(row.sequence))
                continue
            if row.sequence < 0:
                raise ValueError('Paired action rows must enter together through add_pair')
            self.rows.append(row)

    def add_pair(self, record):
        if record['role'] != 'train':
            raise ValueError('Heldout branch truth cannot enter training feedback')
        source = int(record['source_sequence'])
        if self.reserved.get(source) != record['id'] or record['id'] in self.seen_pair_ids:
            raise ValueError('A pair needs exactly one matching reserved source issue')
        labels = paired_label_ids(record['index'])
        in_use = self.selected_sequences | {row['outcomes'][arm]['transition'].sequence
            for row in self.pending_pairs for arm in ARMS}
        if set(labels.values()) & in_use:
            raise ValueError('Paired action identities cannot consume another training slot')
        for arm in ARMS:
            truth = record['outcomes'][arm]['truth']
            if truth.get('status') != 'complete' or truth.get('dt') != 300.:
                raise ValueError('Both exact300-second outcomes are required for a training pair')
        available = max(float(record['issued_at'])+300., *(
            float(record['outcomes'][arm]['transition'].completed_at) for arm in ARMS)) \
            + float(self.p['feedback_delay_seconds'])
        result = dict(record, available_at=available, selected_for_training=False)
        result['features'] = np.asarray(record['features'], dtype=float).tolist()
        result['outcomes'] = {}
        for arm in ARMS:
            row = copy(record['outcomes'][arm]['transition'])
            row.sequence, row.available_at = labels[arm], available
            result['outcomes'][arm] = dict(record['outcomes'][arm], transition=row)
        self.pending_pairs.append(result)
        self.seen_pair_ids.add(result['id'])
        return result

    def due(self, now):
        return self.index < len(self.deadlines) and float(now) >= self.deadlines[self.index]

    def take(self, now):
        deadline = self.deadlines[self.index]
        if float(now) < deadline:
            raise ValueError('Feedback deadline has not arrived')
        lower = deadline-self.p['feedback_interval_seconds']
        ready = [row for row in self.rows if row.available_at <= deadline]
        ordinary = [row for row in ready if lower < row.available_at and row.dt > 0 and not row.terminal]
        pairs = sorted((record for record in self.pending_pairs if record['available_at'] <= deadline),
            key=lambda record: (record['available_at'], record['index']))[:1]
        n = self.p['feedback_records_per_packet']
        ordinary_n = n-2*len(pairs)
        if len(ordinary) < ordinary_n:
            raise RuntimeError(f'Insufficient ordinary feedback at {deadline}: {len(ordinary)}, need {ordinary_n}')
        ordinary_rows = sorted(ordinary, key=lambda row: (row.available_at, row.sequence))[-ordinary_n:]
        paired_rows = [record['outcomes'][arm]['transition'] for record in pairs for arm in ARMS]
        selected = sorted(ordinary_rows+paired_rows, key=lambda row: (row.available_at, row.sequence))
        ids = {int(row.sequence) for row in selected}
        if len(ids) != n or ids & self.selected_sequences or ids & self.reserved.keys():
            raise ValueError('Each packet must contain five distinct nonfactual action labels')
        discarded = [int(row.sequence) for row in ready if row.sequence not in ids]
        self.rows = [row for row in self.rows if row.available_at > deadline]
        used = {record['id'] for record in pairs}
        self.pending_pairs = [record for record in self.pending_pairs if record['id'] not in used]
        for record in pairs:
            record.update(selected_for_training=True, admitted_at=float(now), admitted_deadline=deadline)
        self.last_pairs = pairs
        self.admitted.extend(pairs)
        self.selected_sequences.update(ids)
        self.receipts.append(dict(deadline=deadline, processed_at=float(now), records=n,
            sequence_ids=[int(row.sequence) for row in selected],
            ordinary_records=ordinary_n, paired_records=2*len(pairs),
            ordinary_sequence_ids=[int(row.sequence) for row in ordinary_rows],
            pair_ids=[record['id'] for record in pairs],
            pair_source_sequences=[record['source_sequence'] for record in pairs],
            pair_outcome_ids=[int(row.sequence) for row in paired_rows],
            pair_available_at=[record['available_at'] for record in pairs],
            max_available_at=max(float(row.available_at) for row in selected),
            positive_global_records=sum(row.dt > 0 and not row.terminal for row in selected),
            learned_global_records=sum(row.dt > 0 and not row.terminal for row in selected),
            zero_global_records=sum(row.dt == 0 for row in selected)))
        self.index += 1
        return selected, discarded

    def summary(self):
        lines = ('7X', '102X', '705X')
        return dict(target_labels=len(self.selected_sequences), packets=len(self.receipts),
            ordinary_labels=sum(row['ordinary_records'] for row in self.receipts),
            paired_action_labels=2*len(self.admitted), training_pairs=len(self.admitted),
            pairs_by_line={line: sum(record['line_id'] == line for record in self.admitted) for line in lines},
            zero_global_admitted=sum(row['zero_global_records'] for row in self.receipts),
            pending_pairs=len(self.pending_pairs), pending_pair_ids=[row['id'] for row in self.pending_pairs],
            reserved_pairs=len(self.reserved), reserved_source_sequences=sorted(self.reserved),
            excluded_factual_sequences=sorted(self.factual_exclusions_seen),
            selected_sequence_ids=sorted(self.selected_sequences),
            admitted_pair_ids=[row['id'] for row in self.admitted],
            paired_outcomes_per_pair=2,
            semantics='Two branch action outcomes replace two ordinary labels in the same five-slot packet; factual source outcomes never add a third label.')
