"""Exact provenance for the existing285 slots; no trace label is admitted here."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy


# Paired rows receive a negative slot ID and the joint300s release clock.
# These are the only differences between their native and admitted TD records.
PAIR_REWRITTEN_FIELDS = frozenset(('sequence', 'available_at'))


def _transition(row):
    return row.get('transition', row)


def _differences(native, admitted, *, paired=False):
    omitted = PAIR_REWRITTEN_FIELDS if paired else frozenset()
    return sorted(key for key in native.keys() | admitted.keys()
                  if key not in omitted and native.get(key) != admitted.get(key))


def build_budget_index(dataset, label_origins=None, pair_truth=None, *, prefix_id):
    """Index resident dataset rows by source provenance, never by sequence alone.

    ``prefix_id`` is the original v37 collection directory, not a replay branch.
    Optional ``label_origins`` is a sequence-to-provenance mapping; the actual
    v37 receipts already contain this information under ``receipt.slots``.
    Returns a plain dict suitable for ``audit_rows(rows, origin, index)``.
    """
    metadata = dataset['metadata']
    scenario = metadata['scenario']
    slots, by_source, missing, ids = [], {}, [], set()
    origins = label_origins or {}
    records = {record['id']: record for record in (pair_truth or [])}
    for packet in dataset['packets']:
        receipt = packet['receipt']
        provenance = {record['sequence']: record for record in receipt.get('slots', [])}
        for row in packet['rows']:
            slot_id = row['sequence']
            if slot_id in ids:
                raise ValueError('A dataset slot is repeated')
            ids.add(slot_id)
            source = provenance.get(slot_id, origins.get(slot_id))
            if source is None:
                missing.append(dict(slot_id=slot_id, reason='missing_receipt_provenance'))
                continue
            kind = source['origin']
            if kind not in ('ordinary', 'paired'):
                raise ValueError('Heldout truth is present in the admitted dataset')
            source_sequence = source['source_sequence']
            pair_id, arm = source.get('pair_id'), source.get('arm')
            if kind == 'paired' and (pair_id is None or arm not in ('low', 'high')):
                missing.append(dict(slot_id=slot_id, reason='missing_paired_provenance'))
                continue
            if kind == 'ordinary' and (slot_id < 0 or slot_id != source_sequence):
                raise ValueError('An ordinary slot has a rewritten source identity')
            if kind == 'paired' and slot_id >= 0:
                raise ValueError('A paired slot lost its negative admitted identity')
            if kind == 'paired' and pair_id in records:
                pair = records[pair_id]
                if pair['role'] != 'train' or pair['source_sequence'] != source_sequence:
                    raise ValueError('Paired receipt does not belong to a training source')
                reference = pair['outcomes'][arm]['transition']
                if reference is None or _differences(reference, row, paired=True):
                    raise ValueError('Paired admitted row differs from its recorded physical outcome')
            key = (kind, source_sequence, pair_id, arm)
            if key in by_source:
                raise ValueError('One physical action source owns more than one slot')
            entry = dict(slot_id=slot_id, packet=packet['index'], scenario=scenario,
                prefix_id=str(prefix_id), kind=kind, source_sequence=source_sequence,
                branch_id=pair_id, arm=arm, row=row, admitted_at=receipt['processed_at'])
            by_source[key] = entry
            slots.append(entry)
    counts = Counter(slot['kind'] for slot in slots)
    pairs = [record for record in records.values() if record['role'] == 'train']
    return dict(scenario=scenario, prefix_id=str(prefix_id), slots=slots,
        by_source=by_source, slot_ids=ids, metadata=metadata,
        summary=dict(total_slots=len(slots), ordinary_slots=counts['ordinary'],
            paired_slots=counts['paired'], training_pairs=len(pairs),
            complete_provenance=not missing, missing=missing,
            paired_td_rows_per_arm=1,
            paired_first_reward_zero=sum(slot['row']['r'] == 0 for slot in slots
                                         if slot['kind'] == 'paired'),
            paired_first_zero_duration=sum(slot['row']['dt'] == 0 for slot in slots
                                          if slot['kind'] == 'paired'),
            paired_fixed300_wait_ride_aggregate_in_original_two_slots=True,
            paired_later_event_chain_in_original_slots=False,
            slot_ids=sorted(ids)))


def audit_rows(rows, origin, index):
    """Read-only audit against an explicit original-prefix and branch origin.

    ``origin`` requires scenario, prefix_id and kind.  For native branches it
    also records branch_id and arm.  Paired rows can match only the original
    training branch's FIRST transition, with its positive source_sequence;
    the later rows never inherit that first row's authorization.
    """
    output, counts = [], Counter()
    required = ('scenario', 'prefix_id', 'kind')
    for ordinal, wrapped in enumerate(rows):
        row = _transition(wrapped)
        sequence = row.get('sequence')
        value = dict(row=ordinal, sequence=sequence, slot_id=None,
            holding_matches_slot=None, authorized=False,
            integer_sequence_collision=sequence in index['slot_ids'])
        missing = [key for key in required if origin.get(key) is None]
        if origin.get('kind') == 'paired':
            missing.extend(key for key in ('branch_id', 'arm', 'source_sequence')
                           if origin.get(key) is None)
        if missing:
            value.update(status='missing_provenance', reason='missing_origin:'+','.join(missing))
        elif origin['scenario'] != index['scenario']:
            value.update(status='not_in_budget', reason='different_scenario')
        elif str(origin['prefix_id']) != index['prefix_id']:
            value.update(status='not_in_budget', reason='different_physical_prefix')
        elif origin['kind'] == 'heldout':
            value.update(status='not_in_budget', reason='heldout_branch_never_admitted')
        elif origin['kind'] not in ('ordinary', 'paired'):
            value.update(status='not_in_budget', reason='nontraining_branch_source')
        else:
            paired = origin['kind'] == 'paired'
            source_sequence = origin['source_sequence'] if paired else sequence
            key = (origin['kind'], source_sequence,
                   origin.get('branch_id') if paired else None,
                   origin.get('arm') if paired else None)
            slot = index['by_source'].get(key)
            if slot is None:
                value.update(status='not_in_budget', reason='physical_action_source_not_admitted')
            elif paired and ordinal != 0:
                value.update(status='not_in_budget', reason='later_paired_event_not_authorized')
            else:
                differences = _differences(row, slot['row'], paired=paired)
                value.update(slot_id=slot['slot_id'], packet=slot['packet'],
                    holding_matches_slot=row.get('a') == slot['row'].get('a'))
                if differences:
                    value.update(status='not_in_budget', reason='physical_transition_mismatch',
                        different_fields=differences)
                else:
                    value.update(status='authorized', reason='exact_physical_source_and_td',
                        authorized=True, admitted_at=slot['admitted_at'],
                        admitted_sequence_rewrite=paired,
                        admitted_release_clock_rewrite=paired)
        counts[value['status']] += 1
        counts['integer_sequence_collisions'] += int(value['integer_sequence_collision'])
        output.append(value)
    authorized = [value for value in output if value['authorized']]
    return dict(origin=deepcopy(origin), rows=output,
        summary=dict(rows=len(output), authorized_rows=len(authorized),
            unauthorized_rows=len(output)-len(authorized),
            slot_ids=[value['slot_id'] for value in authorized],
            holding_fact_matches=(all(value['holding_matches_slot'] for value in authorized)
                                  if authorized else None),
            counts=dict(counts), feedback_added=0, truth_admitted=False))
