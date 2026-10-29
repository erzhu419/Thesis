"""Detect selection leakage, altered latest-five state weighting and budget drift."""
from copy import deepcopy

from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.test_selection import Row, config
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.selection import (
    DecoupledPacketBudget, selection_checks, summarize_selection)
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.bootstrap import formal


def test_risk_outcome_replacement_never_changes_latest5_covariates_or_reads_results():
    budget, old = DecoupledPacketBudget(config()), formal.PacketBudget(config())
    packet = [Row(i, 1000+i, risk=i == 1) for i in range(10)]
    budget.ingest(packet[::-1])
    old.ingest(packet[::-1])
    cov, outcomes = budget.take(1200.)
    expected, _ = old.take(1200.)
    assert [row.sequence for row in cov] == [row.sequence for row in expected] == [5, 6, 7, 8, 9]
    assert [row.sequence for row in outcomes] == [1, 6, 7, 8, 9]
    assert all(selection_checks(budget.receipts, expected_packets=1, expected_records=5).values())
    assert summarize_selection(budget.receipts)['target_record_union'] == 6
    altered = deepcopy(budget.receipts)
    altered[0]['covariate_sequence_ids'] = altered[0]['outcome_sequence_ids']
    assert not selection_checks(altered, expected_packets=1, expected_records=5)['covariate_latest5']


def test57_packets_have285_each_and_actual_union_up_to342_without_repeated_records():
    budget, old = DecoupledPacketBudget(config()), formal.PacketBudget(config())
    for packet, deadline in enumerate(budget.deadlines):
        incoming = [Row(packet*10+i, deadline-100+i, risk=i == 1) for i in range(10)]
        budget.ingest(incoming)
        old.ingest(incoming)
        cov, outcomes = budget.take(deadline+2.)
        expected, _ = old.take(deadline+2.)
        assert [row.sequence for row in cov] == [row.sequence for row in expected]
        assert len(cov) == len(outcomes) == 5
    assert all(selection_checks(budget.receipts).values())
    summary = summarize_selection(budget.receipts)
    assert summary['covariate_unique_records'] == summary['outcome_unique_records'] == 285
    assert summary['target_record_union'] == summary['target_record_union_bound'] == 342
    assert summary['target_record_overlap'] == 228
    assert summary['replacements'] == 57
