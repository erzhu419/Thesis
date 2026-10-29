"""Verify factual execution-map accounting without SUMO or resident data."""
import json
import numpy as np

from experiments.mcwm_closed_loop_execution_phase_v19_20261022.shadow import (
    STATISTICS, compact_export, decode_summary, mapping_diagnostics, phase, summarize, window_name,
)


CONTRACT = dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.)


def state(*, clock=.5, forward=True, backward=True, forward_ratio=1.,
          backward_ratio=1., load=.5, demand=0.):
    value = np.zeros(229, np.float32)
    value[0] = clock
    event = value[5:].reshape(16, 14)[0]
    event[[3, 4, 5, 6, 8, 9, 11, 12]] = [forward_ratio, backward_ratio,
        1., 1., load, demand, forward, backward]
    return value


def test_native_reserve_and_tail_share_the_real_state_and_proposal():
    states = np.stack([state(backward_ratio=1.1), state(clock=.8, backward_ratio=1.1)])
    before = states.copy()
    result = mapping_diagnostics(states, [[1.], [1.]], CONTRACT)
    assert np.array_equal(states, before)
    assert np.allclose(result['tail'], [7.5, 7.5])
    assert np.allclose(result['native'], [30., 7.5], atol=1.e-4)
    assert result['reserve_eligible'].tolist() == [True, False]
    assert result['reserve_changed'].tolist() == [True, False]
    assert result['floor_changed'].tolist() == [False, False]


def test_reserve_condition_need_not_change_the_action():
    result = mapping_diagnostics(np.stack([state(), state(backward_ratio=1.5)]),
        [[1.], [-1.]], CONTRACT)
    assert result['reserve_eligible'].tolist() == [True, False]
    assert result['reserve_positive'].tolist() == [False, True]
    assert result['reserve_changed'].tolist() == [False, False]
    assert np.allclose(result['native'], result['tail'])


def test_floor_condition_and_floor_actual_increment_are_distinct():
    states = np.stack([state(backward=False, load=.1, demand=.01),
        state(forward_ratio=.7, backward=False, load=.1, demand=.01),
        state(clock=.1, backward=False, load=.1, demand=.01),
        state(clock=.75, backward=False, load=.1, demand=.01)])
    result = mapping_diagnostics(states, np.ones((4, 1)), CONTRACT)
    assert result['floor_condition'].tolist() == [True, True, False, False]
    assert result['floor_changed'].tolist() == [True, False, False, False]
    assert np.allclose(result['native'], [60., 60., 7.5, 7.5], atol=1.e-4)


def test_summary_reports_controlled_recorded_overrides_without_using_them_as_native():
    # native,tail,recorded,reserve eligible/positive/changed,floor condition/changed,
    # selected controlled sequence,stored-native disagreement.
    rows = [[30., 7.5, 0., 1, 1, 1, 0, 0, 1, 1],
            [7.5, 7.5, 7.5, 0, 1, 0, 0, 0, 0, 0]]
    actual = dict(zip(STATISTICS, summarize(rows)))
    assert actual['n'] == 2
    assert actual['controlled_sequences'] == actual['recorded_native_mismatch'] == 1
    assert actual['holding_different'] == actual['holding_difference_ge1s'] == 1
    assert actual['absolute_difference_mean'] == 11.25
    assert actual['native_holding_mean'] == 18.75
    assert actual['recorded_holding_mean'] == 3.75


def test_phase_and_windows_follow_the_original_boundaries():
    p = dict(training_windows=[dict(name='late', start=16700, end=17670)],
        heldout_windows=[dict(name='heldout1', start=18010, end=18700)])
    assert [phase(t) for t in (.149, .15, .749, .75)] == ['pre_floor', 'reserve_floor', 'reserve_floor', 'tail']
    assert window_name(16700., p) == 'late'
    assert window_name(17670., p) == 'outside_windows'
    assert window_name(18010., p) == 'heldout1'


def test_compact_export_preserves_all_numbers_keys_and_explicit_missing_rows():
    stats = [1733, 1200, 1590, 855, 165, 163, 900, 892,
        19.347861, 20.234231, 52.5, 52.5, 26.847861, 7.5, 26.82521, 3, 3]
    days = [dict(seed=seed, scenario=scene, trace_issues=17000, selected_anchors=15,
        matched_anchors=15, original_budget=dict(target_labels=285, packets=57, ordinary_labels=267))
        for seed in (6201, 6202) for scene in ('stable', 'local_speed_drop')]
    groups = [dict(key=[d['seed'], d['scenario'], line, window, phase_name], statistics=list(stats))
        for d in days for line in ('7X', '102X', '705X')
        for window, phase_name in (('early', 'pre_floor'), ('early', 'reserve_floor'),
            ('post1', 'reserve_floor'), ('late', 'reserve_floor'), ('heldout1', 'tail'),
            ('heldout2', 'tail'), ('outside_windows', 'pre_floor'),
            ('outside_windows', 'reserve_floor'), ('outside_windows', 'tail'))]
    anchors = [dict(key=[d['seed'], d['scenario'], line, role, window], sequence=12345,
        at=17123.0, phase='reserve_floor', actor_updates=[50400, 400],
        observables=[613.234567, 1001.1234, 0., 47., 50., .0012345678, 7., 1., 0.],
        source_features=[1., 0., 0., .23456789, .94], canonical=[.71345833, 1.5, .75, .94, .23],
        mapping=[60., 7.5, 32.34512, 0.], source_exact=True, neighborhood=list(stats))
        for d in days for line in ('7X', '102X', '705X')
        for role, window in (('train', 'early'), ('train', 'post1'), ('train', 'late'),
            ('heldout', 'heldout1'), ('heldout', 'heldout2'))]
    anchors.append(dict(key=[6201, 'stable', '7X', 'train', 'late'], status='missing_trace_anchor'))
    value = dict(status='integrity_failed', schema=dict(statistics_order=list(STATISTICS)), days=days,
        groups=groups, anchors=anchors, coverage=[dict(key=['7X', 'train', 'late'], n=4,
            ranges=[[0., 613.234567], [0., 1001.1234]])], fixed_anchor_keys=[anchors[0]['key']],
        total=list(stats), integrity=dict(all_pass=False))
    original = json.dumps(value, sort_keys=True)
    encoded = compact_export(value)
    assert decode_summary(json.loads(json.dumps(encoded))) == value
    assert json.dumps(value, sort_keys=True) == original
    payload = json.dumps(encoded, separators=(',', ':'))
    raw = json.dumps(value, separators=(',', ':'))
    assert len(payload) < .7*len(raw)
    assert len(payload) < 60000
