"""Verify the final slot contract and deny SGD before complete causal admission."""
import json
import pytest

from experiments.mcwm_closed_loop_trace_design_v40_20261024.release_plan import (
    build_release_plan, compact_release_plan, require_phase,
)


def ids():
    return [row['new_id'] for row in build_release_plan()['slots']]


def ready(**changes):
    slots = ids()
    return dict(native_clock=9630., admitted_slot_ids=slots, completed_updates=24,
        available_response_slot_ids=slots[15:], latest_response_available_at=9620.) | changes


def test_exact_manifest_financing_common_window_and_updates():
    plan = build_release_plan()
    assert plan['all_pass'] and all(plan['checks'].values())
    assert len(plan['packets']) == 57 and len(plan['slots']) == 285
    assert sum(packet['updates_after_packet'] for packet in plan['packets']) == 24
    assert sum(phase['updates'] for phase in plan['phases']) == 456
    assert {tuple(pair['issue_window']) for pair in plan['pairs']} == {(9000., 9300.)}
    assert {pair['label_age_upper_bound_seconds'] for pair in plan['pairs']} == {630.}
    assert plan['financing']['ordinary_replaced']+plan['financing']['root_entitlements_cancelled'] == 264
    assert len(json.dumps(compact_release_plan(plan), separators=(',', ':'))) < 24000


def test_warmup24_uses_only15_ordinary_and_preserves_behavior_reference():
    slots = ids()
    for index in range(1, 4):
        result = require_phase(f'warmup_{index}_update', native_clock=1200.+(index-1)*300.,
            admitted_slot_ids=slots[:5*index], completed_updates=8*(index-1))
        assert result['allowed_updates'] == 8 and result['updates_after'] == index*8
    with pytest.raises(ValueError, match='entered early'):
        require_phase('warmup_3_update', native_clock=1800.,
            admitted_slot_ids=slots[:20], completed_updates=16)
    assert build_release_plan()['source_contract']['warmup_learner_does_not_modify_behavior_reference']


def test_physically_ready270_does_not_allow_sgd_before_full_admission():
    slots = ids()
    for count in (15, 20, 280):
        response = require_phase('response_admit', **ready(admitted_slot_ids=slots[:count]))
        assert response['allowed_updates'] == 0
        with pytest.raises(ValueError, match='before all285'):
            require_phase('response_update', **ready(admitted_slot_ids=slots[:count]))
    response = require_phase('response_update', **ready())
    assert response == dict(phase='response_update', allowed_updates=432, updates_after=456)


def test_missing_late_or_duplicate_response_and_clock_advance_are_rejected():
    slots = ids()
    with pytest.raises(ValueError, match='All270'):
        require_phase('response_admit', **ready(available_response_slot_ids=slots[15:-1]))
    with pytest.raises(ValueError, match='release time'):
        require_phase('response_update', **ready(latest_response_available_at=9631.))
    with pytest.raises(ValueError, match='native9630'):
        require_phase('response_update', **ready(native_clock=9629.))
    with pytest.raises(ValueError, match='more than once'):
        require_phase('response_update', **ready(admitted_slot_ids=slots+slots[-1:]))


def test_update_order_and_postfreeze_zero_training():
    with pytest.raises(ValueError, match='exactly24'):
        require_phase('response_update', **ready(completed_updates=32))
    for clock in (9630., 18000., 24000.):
        result = require_phase('frozen', native_clock=clock,
            admitted_slot_ids=ids(), completed_updates=456)
        assert result['allowed_updates'] == 0
    with pytest.raises(ValueError, match='before285/456'):
        require_phase('frozen', native_clock=18000., admitted_slot_ids=ids(), completed_updates=455)
