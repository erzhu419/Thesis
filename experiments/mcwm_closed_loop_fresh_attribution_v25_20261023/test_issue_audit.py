"""Verify real F increments, persistent window census, and strict history timing."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.issue_audit import (
    IssueAudit, MAPPING_COLUMNS, headway_feature,
)


def audit():
    return IssueAudit(dict(horizon_seconds=24000, diagnostic_station_id='upstream'),
        ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.)))


def event(now, *, line='7X', floor=False, reserve=0.):
    value = np.zeros(229, np.float32)
    value[0] = np.float32(np.float32(now)/24000.)
    local = value[5:].reshape(16, 14)[0]
    local[[3, 4, 5, 6, 8, 9, 11, 12]] = [1., 1.+reserve/300., 1., 1.,
        .1 if floor else .5, .01, True, not floor]
    raw = np.zeros(16, np.float32)
    raw[[3, 4, 5, 6]] = [1200., 600., 600., 600.]
    return dict(started_at=float(now), line_id=line, station_id='upstream', s=value, event=raw)


def history(now, residuals):
    return dict(status='supported', mean_residual_scaled=float(np.mean(residuals))/60., segments=[
        dict(available=True, latest=dict(completed_at=now-2, observed_at=now-1, residual=value))
        for value in residuals])


def group(result, line, phase):
    row = next(row for row in result['mapping'] if row['line'] == line and row['phase'] == phase)
    return dict(zip(MAPPING_COLUMNS, row['values']))


def window(result, line, name):
    return next(row for row in result['windows'] if row['line'] == line and row['name'] == name)


def test_census_continues_after_first_and_uses_unclipped_model_covariate():
    observed = audit()
    for now, residuals in ((1500, [0., 0., 30.]), (1600, [0., 0., -30.]), (1700, [1., 1., 1.])):
        row = event(now)
        # Canonical forward/backward ratios coincide, raw headway difference is1.
        assert headway_feature(row) == 1.
        observed.observe(row, np.array([1.], np.float32), history(now, residuals))
    result = window(observed.summary(), '7X', 'bootstrap')
    assert result['issues'] == result['joint_eligible'] == 3
    assert len(result['third_joint_counts']) == len(result['mean_joint_counts']) == 3
    assert all(key.startswith('1|') for key in result['third_joint_counts'])
    assert result['raw_headway_range'] == [1., 1.]


def test_missing_equal_issue_history_and_action_support_remain_explicit():
    observed = audit()
    observed.observe(event(4000), [1.])
    future = history(4001, [1., 2., 3.])
    future['segments'][1]['latest']['observed_at'] = 4001
    observed.observe(event(4001), [1.], future)
    observed.observe(event(4002, floor=True), [-1.], history(4002, [1., 2., 3.]))
    observed.observe(event(4003), [1.], history(4003, [0., 0., 20.]))
    result = window(observed.summary(), '7X', 'response')
    assert result['issues'] == 4
    assert result['history_missing'] == result['history_noncausal'] == 1
    assert result['action_unsupported'] == 1
    assert result['joint_eligible'] == 1
    assert result['bins_agree'] == 0  # third20 versus mean6.667 seconds.


def test_floor_condition_actual_increment_and_shadow_reserve_are_distinct():
    observed = audit()
    observed.observe(event(1559, floor=True), [1.])
    observed.observe(event(3815, floor=True), [-1.])
    # With a sufficiently large reserve, no-floor already reaches60 seconds.
    saturated = event(3816, floor=True)
    saturated['s'][5+3] = .7
    observed.observe(saturated, [1.])
    observed.observe(event(18000, floor=True), [1.])
    observed.observe(event(19000, reserve=30.), [1.])
    result = observed.summary()
    before = group(result, '7X', 'pre_floor')
    middle = group(result, '7X', 'reserve_floor')
    tail = group(result, '7X', 'tail')
    assert before['floor_condition'] == before['floor_changed'] == 0
    assert middle['floor_condition'] == 2 and middle['floor_changed'] == 1
    assert middle['floor_increment_sum_seconds'] == pytest.approx(60.)
    assert middle['native_tail_absolute_sum_seconds'] == pytest.approx(112.5)
    assert middle['reserve_changed'] == 1
    assert tail['floor_condition'] == tail['native_tail_changed'] == 0


def test_observer_is_read_only_deterministic_and_summary_is_bounded():
    observed = audit()
    row, proposal = event(3815), np.array([1.], np.float32)
    snapshot = history(3815, [1., 2., 3.])
    before = deepcopy((row, proposal, snapshot))
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for line_index in range(12):
        for now in (1559, 3815, 19000):
            item = event(now, line=str(line_index))
            observed.observe(item, proposal)
    observed.observe(row, proposal, snapshot)
    assert np.array_equal(before[0]['s'], row['s']) and np.array_equal(before[0]['event'], row['event'])
    assert np.array_equal(before[1], proposal) and before[2] == snapshot
    assert random.getstate() == rng[0]
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), rng[1]))
    assert torch.equal(torch.get_rng_state(), rng[2])
    result = observed.summary()
    assert result == observed.summary()
    assert len(json.dumps(result).encode()) < 20000
    assert result['training_rows'] == result['environment_queries'] == result['retained_full_states'] == 0
