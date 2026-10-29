"""Held-out coverage must not compete across lines or precede the last update."""

import numpy as np
import pytest

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.test_calibration import (
    ExampleTransform, protocol,
)
from experiments.mcwm_closed_loop_response_residual_v11_20261021.coverage import HeldoutCoverage


def make(*, transform=None, **kwargs):
    return HeldoutCoverage(protocol(), 6201, transform or ExampleTransform(),
        station_id='diagnostic', start_after=18005., **kwargs)


def event(sequence, line='7X', now=18100.):
    return dict(sequence=sequence, line_id=line, station_id='diagnostic',
        started_at=now, s=np.zeros(229, np.float32), a=np.asarray([-1.], np.float32))


def execute(coverage, issued):
    decision = coverage.propose(issued)
    assert decision is not None
    issued['a'] = decision['action'].copy()
    coverage.commit(issued, decision)
    return decision


def test_all_three_lines_get_both_actions_even_in_one_old_shared_window():
    coverage = make()
    for index, line in enumerate(('7X', '102X', '705X')):
        execute(coverage, event(index, line, now=18100.))
        execute(coverage, event(index+3, line, now=18120.))
    summary = coverage.summary(24000.)
    assert summary['complete']
    assert all(row['low'] == row['high'] == 1 for row in summary['coverage'].values())
    assert summary['missing_by_line'] == {}
    assert coverage.propose(event(99)) is None


def test_stops_strictly_after_actual_last_update_and_preserves_endpoint_budget():
    coverage = make()
    assert coverage.propose(event(0, now=18005.)) is None
    assert coverage.propose(event(1, now=18004.)) is None
    assert coverage.propose(event(2, now=23670.)) is not None
    assert coverage.propose(event(3, now=23671.)) is None
    assert not coverage.summary(23669.)['observation_window_closed']
    assert coverage.summary(24000.)['missing_by_line']['7X'] == [0., 7.5]


def test_proposing_or_failed_execution_does_not_consume_a_line_slot():
    coverage = make()
    issued = event(0)
    decision = coverage.propose(issued)
    assert coverage.line_successes['7X'] == 0
    wrong = dict(issued, a=np.asarray([-.5], np.float32))
    with pytest.raises(ValueError, match='actual 0 / 7.5'):
        coverage.commit(wrong, decision)
    assert coverage.line_successes['7X'] == 0
    next_decision = coverage.propose(event(1))
    assert next_decision['assignment'] == decision['assignment']
    issued['a'] = decision['action'].copy()
    coverage.commit(issued, decision)
    with pytest.raises(ValueError, match='same held-out issue'):
        coverage.commit(issued, decision)


def test_ineligible_event_retains_assignment_and_is_not_zero_filled():
    coverage = make(transform=ExampleTransform(floor=60.))
    assert coverage.propose(event(0, line='705X')) is None
    summary = coverage.summary(24000.)
    assert summary['statuses'] == {'collapsed': 1}
    assert summary['coverage']['705X']['n'] == 0
    assert summary['coverage']['705X']['missing_actions'] == [0., 7.5]


def test_order_is_seed_fixed_and_independent_of_global_rng_or_other_line_traffic():
    np.random.seed(991)
    before = np.random.get_state()[1].copy()
    left, right = make(), make()
    np.testing.assert_array_equal(before, np.random.get_state()[1])
    execute(left, event(0, '7X'))
    execute(left, event(1, '7X'))
    a = execute(left, event(2, '705X'))
    b = execute(right, event(2, '705X'))
    assert a['assignment'] == b['assignment']
    assert left.action_sequences == right.action_sequences


def test_explicit_existing_f_inversion_callback_is_used_for_both_actions():
    calls = []
    transform = ExampleTransform()
    source = make(transform=transform)

    def reachable(state, hold):
        calls.append(hold)
        return source._action_for_hold(state, hold)

    coverage = make(transform=transform, action_for_hold=reachable)
    execute(coverage, event(0))
    assert calls == [0., 7.5]


def test_branch_override_reports_actual_hold_and_keeps_common_plan_visible():
    coverage = make()
    issued = event(0)
    nominal = coverage.propose(issued)
    actual = 7.5-nominal['planned_hold_seconds']
    issued['a'] = np.asarray([actual/30.-1.], np.float32)
    forced = dict(nominal, action=issued['a'].copy(), target_hold_seconds=actual,
                  executed_hold_seconds=actual)
    coverage.commit(issued, forced)
    record = coverage.summary()['assignments'][0]
    assert record['planned_hold_seconds'] == nominal['planned_hold_seconds']
    assert record['executed_hold_seconds'] == actual
    assert record['branch_override']
    assert coverage.line_successes['7X'] == 1
