"""Detect over-budget financing, premature suffix use and skipped zero-dt events."""
from copy import deepcopy
import json

from experiments.mcwm_closed_loop_trace_design_v40_20261024.budget_plan import (
    admitted_prefixes, audit_real_branch, build_plan, compact_plan,
)


def branch():
    rows, now = [], 1200.
    for index in range(45):
        duration = 0. if index == 1 else 1.
        end = now+duration
        rows.append(dict(sequence=100+index, started_at=now, completed_at=end,
            available_at=end+30., dt=duration, s=[index], ns=[index+1],
            raw_s=[index], raw_ns=[index+1],
            memory_s={'index': index}, memory_ns={'index': index+1}))
        now = end
    return rows


def test_exact_fixed_financing_and_release_schedule():
    plan = build_plan()
    assert plan['all_pass'] and all(plan['checks'].values())
    assert plan['financing']['ordinary_replaced']+plan['financing']['paired_root_entitlements_cancelled'] == 264
    assert [pair['first_deadline'] for pair in plan['pairs']] == [2100., 7500., 12900.]
    assert [pair['final_deadline'] for pair in plan['pairs']] == [7200., 12600., 18000.]
    assert [pair['release_margin_seconds'] for pair in plan['pairs']] == [270.]*3
    assert sum(packet['updates'] for packet in plan['packets']) == 456
    assert len(json.dumps(compact_plan(plan), separators=(',', ':'))) < 24000


def test_physical_readiness_does_not_authorize_future_prefix_slots():
    plan = build_plan()
    assert set(admitted_prefixes(plan, 3).values()) == {0}
    first = plan['pairs'][0]['id']
    assert admitted_prefixes(plan, 4)[first+'/low'] == 3
    assert admitted_prefixes(plan, 4)[first+'/high'] == 2
    assert admitted_prefixes(plan, 20)[first+'/low'] == 43
    assert admitted_prefixes(plan, 21)[first+'/low'] == 45
    assert admitted_prefixes(plan, 21)[first+'/high'] == 45
    assert set(admitted_prefixes(plan, 57).values()) == {45}


def test_zero_duration_is_a_real_slot_and_cannot_be_skipped():
    rows = branch()
    result = audit_real_branch(rows, issue=1200., first_deadline=2100.)
    assert result['all_pass'] and result['zero_duration_events'] == 1
    assert result['real_events'] == 45 and result['feedback_added'] == 0
    assert result['training_authorized'] is False
    dropped = rows[:1]+rows[2:]
    result = audit_real_branch(dropped, issue=1200., first_deadline=2100.)
    assert not result['all_pass']
    assert 'need_exactly45_real_closed_events' in result['reasons']
    assert any('native_event_was_skipped' in reason for reason in result['reasons'])


def test_late_or_unreleased_rows_fail_without_substitution():
    rows = branch()
    early = deepcopy(rows)
    early[-1]['available_at'] = early[-1]['completed_at']
    assert any('feedback_precedes_release' in reason for reason in
        audit_real_branch(early, issue=1200., first_deadline=2100.)['reasons'])
    late = deepcopy(rows)
    late[-1]['available_at'] = 2101.
    assert any('not_ready_for_first_fixed_packet' in reason for reason in
        audit_real_branch(late, issue=1200., first_deadline=2100.)['reasons'])
    cut = deepcopy(rows)
    cut[-1]['completed_at'] = 1501.
    cut[-1]['dt'] = 1501.-cut[-1]['started_at']
    cut[-1]['available_at'] = 1531.
    assert any('past_fixed_native_cap' in reason for reason in
        audit_real_branch(cut, issue=1200., first_deadline=2100.)['reasons'])


def test_cross_branch_state_or_memory_splice_is_not_a_contiguous_chain():
    rows = branch()
    rows[10]['memory_s'] = {'index': -1}
    result = audit_real_branch(rows, issue=1200., first_deadline=2100.)
    assert not result['all_pass']
    assert 'event10:memory_s_continuity_missing' in result['reasons']
    rows = branch()
    rows[10]['terminal'] = True
    assert 'event10:trace_continues_past_native_terminal' in audit_real_branch(
        rows, issue=1200., first_deadline=2100.)['reasons']
