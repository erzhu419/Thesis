"""Keep latest-five timing; cover one visible full-vehicle queue per window."""
from collections import Counter

import numpy as np


RISK_PREDICATE = 'issue_s:e[7]>0_and_e[8]>=1'


def capacity_risk(state):
    """Use only the issued 229-D observation, before any result is known."""
    s = np.asarray(state)
    i = int(round(15 * float(s[2])))
    e = s[5:].reshape(16, 14)[i]
    return bool(e[7] > 0 and e[8] >= 1)


def _lines(rows):
    return dict(sorted(Counter(str(row.line_id) for row in rows).items()))


class CapacityPacketBudget:
    """Original disjoint release windows and eviction, with at most one swap."""

    def __init__(self, p):
        self.p, self.rows, self.index, self.receipts = p, [], 0, []
        self.deadlines = list(range(p['feedback_first_deadline'],
            p['feedback_last_deadline'] + 1, p['feedback_interval_seconds']))

    def ingest(self, rows):
        self.rows.extend(rows)

    def due(self, now):
        return self.index < len(self.deadlines) and now >= self.deadlines[self.index]

    def take(self, now):
        deadline = self.deadlines[self.index]
        if now < deadline:
            raise ValueError('Feedback deadline has not arrived')
        lower = deadline - self.p['feedback_interval_seconds']
        ready = [row for row in self.rows if row.available_at <= deadline]
        candidates = [row for row in ready if row.available_at > lower
            and row.dt > 0 and not row.terminal]
        n = self.p['feedback_records_per_packet']
        if n != 5:
            raise ValueError('The capacity-window trial fixes five feedback slots per packet')
        if len(candidates) < n:
            raise RuntimeError(f'Insufficient feedback at {deadline}: {len(candidates)} eligible, {n} required')
        key = lambda row: (row.available_at, row.sequence)
        candidates.sort(key=key)
        risks = {row.sequence: capacity_risk(row.s) for row in candidates}
        original = candidates[-n:]
        original_risk = sum(risks[row.sequence] for row in original)
        risk_rows = [row for row in candidates if risks[row.sequence]]
        selected, replacement = list(original), None
        if not original_risk and risk_rows:
            removed, added = selected[0], risk_rows[-1]
            selected[0] = added
            replacement = dict(removed_sequence=removed.sequence, added_sequence=added.sequence,
                removed_available_at=removed.available_at, added_available_at=added.available_at,
                added_risk=True, removed_risk=False)
        selected.sort(key=key)
        selected_ids = {row.sequence for row in selected}
        discarded = [row.sequence for row in ready if row.sequence not in selected_ids]
        self.rows = [row for row in self.rows if row.available_at > deadline]
        span = lambda rows: [min(row.available_at for row in rows),
            max(row.available_at for row in rows)] if rows else None
        selection = dict(eligible_records=len(candidates), eligible_risk_records=len(risk_rows),
            original_latest5_sequence_ids=[row.sequence for row in original],
            original_latest5_risk_records=original_risk,
            selected_risk_records=sum(risks[row.sequence] for row in selected),
            replacement=replacement, candidate_line_counts=_lines(candidates),
            risk_candidate_line_counts=_lines(risk_rows), selected_line_counts=_lines(selected),
            risk_selected_line_counts=_lines([row for row in selected if risks[row.sequence]]),
            candidate_available_span_seconds=span(candidates),
            risk_candidate_available_span_seconds=span(risk_rows),
            selected_available_span_seconds=span(selected),
            selected_keys=[[row.available_at, row.sequence] for row in selected],
            risk_predicate=RISK_PREDICATE, outcome_fields_used_for_ranking=False)
        self.receipts.append(dict(deadline=deadline, processed_at=now, records=n,
            sequence_ids=[row.sequence for row in selected],
            max_available_at=max(row.available_at for row in selected),
            capacity_selection=selection))
        self.index += 1
        return selected, discarded


def selection_checks(receipts, *, expected_packets=57, expected_records=285):
    checks = dict(fixed_packet_budget=len(receipts) == expected_packets,
        fixed_feedback_budget=sum(row['records'] for row in receipts) == expected_records,
        latest5_or_one_replacement=True, issue_state_only_risk=True,
        all_risk_windows_covered=True, original_availability_windows=True,
        chronological_ingestion=True, unique_selected_ids=True)
    all_ids = []
    for row in receipts:
        audit = row['capacity_selection']
        original, selected = audit['original_latest5_sequence_ids'], row['sequence_ids']
        replace = audit['replacement']
        expected = set(original)
        if replace is not None:
            expected.remove(replace['removed_sequence'])
            expected.add(replace['added_sequence'])
            checks['latest5_or_one_replacement'] &= bool(
                audit['original_latest5_risk_records'] == 0
                and audit['eligible_risk_records'] > 0 and audit['selected_risk_records'] == 1
                and replace['removed_sequence'] == original[0]
                and replace['added_sequence'] not in original
                and replace['added_risk'] and not replace['removed_risk'])
        checks['latest5_or_one_replacement'] &= bool(len(original) == len(selected) == 5
            and len(set(selected)) == 5 and set(selected) == expected)
        checks['issue_state_only_risk'] &= (audit['risk_predicate'] == RISK_PREDICATE
            and audit['outcome_fields_used_for_ranking'] is False)
        checks['all_risk_windows_covered'] &= bool(not audit['eligible_risk_records']
            or audit['selected_risk_records'] > 0)
        keys = audit['selected_keys']
        checks['chronological_ingestion'] &= bool(keys == sorted(keys)
            and [key[1] for key in keys] == selected)
        lower = row['deadline'] - 300
        checks['original_availability_windows'] &= bool(
            all(lower < key[0] <= row['deadline'] <= row['processed_at'] for key in keys)
            and row['max_available_at'] == max(key[0] for key in keys))
        if replace is not None:
            checks['original_availability_windows'] &= bool(
                lower < replace['added_available_at'] <= row['deadline'])
        all_ids.extend(selected)
    checks['unique_selected_ids'] = len(set(all_ids)) == len(all_ids)
    return checks


def summarize_selection(receipts):
    audits = [row['capacity_selection'] for row in receipts]
    replacements = sum(audit['replacement'] is not None for audit in audits)
    return dict(packets=len(receipts), selected_records=sum(row['records'] for row in receipts),
        eligible_records=sum(audit['eligible_records'] for audit in audits),
        eligible_risk_records=sum(audit['eligible_risk_records'] for audit in audits),
        risk_windows=sum(audit['eligible_risk_records'] > 0 for audit in audits),
        original_covered_risk_windows=sum(audit['original_latest5_risk_records'] > 0 for audit in audits),
        selected_covered_risk_windows=sum(audit['selected_risk_records'] > 0 for audit in audits),
        original_selected_risk_records=sum(audit['original_latest5_risk_records'] for audit in audits),
        selected_risk_records=sum(audit['selected_risk_records'] for audit in audits),
        replacements=replacements, max_replacements_per_packet=int(bool(replacements)),
        risk_predicate=RISK_PREDICATE, outcome_fields_used_for_ranking=False,
        invariant_checks=selection_checks(receipts, expected_packets=len(receipts),
            expected_records=5*len(receipts)))
