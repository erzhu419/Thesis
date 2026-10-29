"""Catch wrong-window, future-result and order changes in the only new mechanism."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.selection import (
    CapacityPacketBudget, capacity_risk, selection_checks, summarize_selection)


def config(**updates):
    return dict(feedback_first_deadline=1200, feedback_last_deadline=18000,
        feedback_interval_seconds=300, feedback_records_per_packet=5, **updates)


class Row:
    def __init__(self, sequence, available, *, risk=False, dt=1., terminal=False,
                 line='7X', index=3):
        self.sequence, self.available_at = sequence, available
        self.dt, self.terminal, self.line_id = dt, terminal, line
        self.s = np.zeros(229, np.float32)
        self.s[2] = index/15
        self.s[5:].reshape(16, 14)[index, 7:9] = (.2, 1.) if risk else (.2, .98)

    @property
    def r(self):
        raise AssertionError('Outcome reward read while selecting feedback')

    @property
    def ns(self):
        raise AssertionError('Successor state read while selecting feedback')


def test_latest_risk_replaces_oldest_latest5_and_is_ingested_chronologically():
    rows = [Row(i, 1000+i, risk=i in (1, 3), line='102X' if i == 3 else '7X')
        for i in range(10)]
    budget = CapacityPacketBudget(config())
    budget.ingest(rows[::-1])
    selected, discarded = budget.take(1203.)
    assert [row.sequence for row in selected] == [3, 6, 7, 8, 9]
    assert set(discarded) == {0, 1, 2, 4, 5}
    audit = budget.receipts[0]['capacity_selection']
    assert audit['original_latest5_sequence_ids'] == [5, 6, 7, 8, 9]
    assert audit['replacement']['added_sequence'] == 3
    assert audit['replacement']['removed_sequence'] == 5
    assert audit['selected_line_counts'] == {'102X': 1, '7X': 4}
    assert all(selection_checks(budget.receipts, expected_packets=1, expected_records=5).values())


@pytest.mark.parametrize('risk_ids', [(), (1, 9), (7, 8, 9)])
def test_no_swap_if_no_risk_or_latest5_already_cover_it(risk_ids):
    budget = CapacityPacketBudget(config())
    budget.ingest([Row(i, 1000+i, risk=i in risk_ids) for i in range(10)])
    selected, _ = budget.take(1200.)
    assert [row.sequence for row in selected] == [5, 6, 7, 8, 9]
    assert budget.receipts[0]['capacity_selection']['replacement'] is None


def test_original_window_terminal_dt_and_availability_rules_are_unchanged():
    budget = CapacityPacketBudget(config())
    budget.ingest([Row(i, 1000+i) for i in range(5)] + [
        Row(10, 900, risk=True),  # Lower boundary is excluded.
        Row(11, 1100, risk=True, dt=0),
        Row(12, 1101, risk=True, terminal=True),
        Row(13, 1200, risk=True),  # Upper boundary is eligible.
        Row(14, 1201, risk=True)])
    selected, discarded = budget.take(1210.)
    assert [row.sequence for row in selected] == [1, 2, 3, 4, 13]
    assert {row.sequence for row in budget.rows} == {14}
    assert set(discarded) == {0, 10, 11, 12}
    assert budget.receipts[0]['capacity_selection']['eligible_records'] == 6


def test_risk_uses_indexed_issue_entity_and_strict_full_capacity():
    row = Row(0, 1000, risk=True, index=15)
    assert capacity_risk(row.s)
    entity = row.s[5:].reshape(16, 14)[15]
    entity[7] = 0
    assert not capacity_risk(row.s)
    entity[7], entity[8] = .2, .9999
    assert not capacity_risk(row.s)
    row.s[5:].reshape(16, 14)[0, 7:9] = (1., 1.)
    assert not capacity_risk(row.s)  # Another entity cannot make this issue risky.


def test_equal_availability_uses_sequence_for_latest_and_output_order():
    budget = CapacityPacketBudget(config())
    budget.ingest([Row(i, 1000, risk=i == 1) for i in (8, 3, 7, 1, 6, 5)])
    selected, _ = budget.take(1200.)
    assert [row.sequence for row in selected] == [1, 5, 6, 7, 8]


def test_all57_packets_keep285_unique_slots_without_reusing_discarded_rows():
    budget = CapacityPacketBudget(config())
    for packet, deadline in enumerate(budget.deadlines):
        assert not budget.due(deadline-1)
        budget.ingest([Row(packet*10+i, deadline-100+i, risk=i == 1)
            for i in range(10)])
        assert budget.due(deadline)
        selected, _ = budget.take(deadline+2.)
        assert len(selected) == 5 and not budget.rows
    assert not budget.due(24000.)
    assert all(selection_checks(budget.receipts).values())
    summary = summarize_selection(budget.receipts)
    assert summary['replacements'] == summary['selected_risk_records'] == 57
    assert summary['selected_records'] == 285
    assert summary['risk_windows'] == summary['selected_covered_risk_windows'] == 57


def test_audit_rejects_wrong_order_and_increased_per_packet_swap():
    budget = CapacityPacketBudget(config())
    budget.ingest([Row(i, 1000+i, risk=i == 1) for i in range(10)])
    budget.take(1200.)
    receipts = deepcopy(budget.receipts)
    receipts[0]['sequence_ids'].reverse()
    assert not selection_checks(receipts, expected_packets=1, expected_records=5)['chronological_ingestion']
    receipts = deepcopy(budget.receipts)
    receipts[0]['capacity_selection']['original_latest5_risk_records'] = 1
    assert not selection_checks(receipts, expected_packets=1, expected_records=5)['latest5_or_one_replacement']
