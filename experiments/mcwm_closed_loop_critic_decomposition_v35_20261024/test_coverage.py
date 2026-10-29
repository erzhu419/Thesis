"""Executed-action support, empty windows and residual-design confounding."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.coverage import coverage


def memory(value):
    vector = np.zeros(84)
    vector[0], vector[1] = value, .5
    return dict(vector=vector.tolist(), feature_names=[f'feature{index}' for index in range(84)])


def fixture():
    values = [('7X', 1000., 0., 0.), ('7X', 9500., 1.875, .2),
        ('7X', 12500., 60., .4), ('7X', 16000., 60., .8),
        ('102X', 10000., 60., .1), ('102X', 17000., 60., .7),
        ('705X', 8000., 30., .3), ('705X', 14000., 30., .7)]
    rows = [SimpleNamespace(sequence=index, line_id=line, started_at=clock,
        a=np.asarray([2*hold/60.-1], np.float32), proposed_a=np.asarray([-1.], np.float32),
        memory_s=memory(value)) for index, (line, clock, hold, value) in enumerate(values)]
    context = memory(.3)
    context['vector'][1], context['vector'][80] = .75, .1
    selected = [dict(slot='7X/early', holds=dict(low=0., high=1.875),
        selected=dict(sequence=400, line_id='7X', started_at=9400., memory_s=context))]
    return dict(packets=[dict(rows=rows[:4]), dict(rows=rows[4:])]), selected


def unpack(result, row):
    return dict(zip(result['columns']['action'], row))


def support(result, window):
    return dict(zip(result['columns']['selected_support'], result['selected'][0]['supports'][window]))


def test_only_actual_executed_target_actions_define_line_and_window_support():
    facts, selected = fixture()
    result = coverage(facts, selected, TransitMDPConfigV7())
    total = unpack(result, result['all_actions'])
    assert total['count'] == 8
    assert total['exact_zero'] == 1 and total['near_zero'] == 1
    assert total['near_1.875'] == 1 and total['floor60'] == 4 and total['low_le_1.875'] == 2
    assert result['metadata']['source_history_rows_included'] == 0
    assert result['metadata']['neighborhood_seconds'] == .5
    assert result['metadata']['neighborhood_is_gate'] is False
    seven = unpack(result, result['by_line']['7X'])
    assert seven['min'] == 0. and seven['max'] == 60.
    assert seven['median'] == pytest.approx((1.875+60.)/2)
    assert unpack(result, result['by_line_window']['7X/9000-12000'])['count'] == 1
    assert unpack(result, result['by_line_window']['7X/12000-18000'])['count'] == 2
    assert unpack(result, result['v34_window_training']['7X/early'])['count'] == 1
    assert unpack(result, result['v34_window_training']['7X/tail'])['count'] == 0
    # All recorded proposals are zero hold, unlike these actually executed
    # actions; counting proposals or applying F again would fail these checks.
    assert total['mean'] > 30.


def test_selected_low_high_support_and_all84_context_extrapolation_are_separate():
    facts, selected = fixture()
    result = coverage(facts, selected, TransitMDPConfigV7())
    full, middle, late = [support(result, key) for key in ('full', '9000-12000', '12000-18000')]
    assert full['low_in_range'] and full['high_in_range']
    assert full['near_low'] == full['near_high'] == 1
    assert full['nearest_low_seconds'] == full['nearest_high_seconds'] == 0.
    assert middle['near_low'] == 0 and middle['near_high'] == 1
    assert middle['nearest_low_seconds'] == 1.875
    assert middle['low_in_range'] is False and middle['high_in_range'] is True
    assert late['near_low'] == late['near_high'] == 0
    assert late['nearest_high_seconds'] == 58.125
    assert late['low_in_range'] is late['high_in_range'] is False
    assert full['memory_outside_indices'] == [1, 80]
    assert middle['memory_outside_indices'] == late['memory_outside_indices'] == [0, 1, 80]
    assert full['memory_outside_count'] == 2
    assert len(result['memory']['min']) == len(result['memory']['max']) == 84
    assert result['memory']['min'][0] == 0. and result['memory']['max'][0] == .8


def test_constant_floor_actions_make_level_action_context_design_rank_deficient():
    facts, selected = fixture()
    result = coverage(facts, selected, TransitMDPConfigV7())
    floor = result['residual_design']['102X']
    assert floor['varying_features'] == 1 and floor['constant_nonzero_features'] == 1
    assert floor['columns'] == 4 and floor['rank'] == 2
    assert floor['full_rank_condition'] is None
    mixed = result['residual_design']['7X']
    assert mixed['rank'] == mixed['columns'] == 4
    assert mixed['full_rank_condition'] is not None
    assert unpack(result, result['by_line']['102X'])['std'] == 0.


def test_constant_presence_is_retained_as_a_real_global_action_slope():
    facts, selected = fixture()
    for packet in facts['packets']:
        for row in packet['rows']:
            row.memory_s['vector'] = [1.]+[0.]*83
    result = coverage(facts, selected, TransitMDPConfigV7())
    seven = result['residual_design']['7X']
    assert seven['varying_features'] == 0 and seven['constant_nonzero_features'] == 1
    assert seven['all_zero_features'] == 83
    assert seven['rank'] == seven['columns'] == 2
    assert seven['full_rank_condition'] is not None


def test_unobserved_training_windows_and_unselected_slots_remain_missing():
    facts, selected = fixture()
    selected[0]['selected']['line_id'] = '406X'
    selected.append(dict(slot='705X/tail', selected=None))
    result = coverage(facts, selected, TransitMDPConfigV7())
    empty = support(result, 'full')
    assert empty['count'] == 0
    assert empty['nearest_low_seconds'] is empty['nearest_high_seconds'] is None
    assert empty['low_in_range'] is empty['high_in_range'] is None
    assert empty['memory_outside_count'] is empty['memory_outside_indices'] is None
    assert result['selected'][1] == dict(slot='705X/tail', status='unselected')
