"""Detect sequence collisions, cross-branch admission and missing feedback."""
from copy import deepcopy

from experiments.mcwm_closed_loop_trace_support_v39_20261024.budget import (
    audit_rows, build_budget_index,
)


def dataset():
    ordinary = dict(sequence=11, available_at=50., started_at=10., completed_at=20.,
        s=[1.], a=[-1.], r=-8., ns=[2.], dt=10., terminal=False,
        memory_s={'observed_at':10.}, memory_ns={'observed_at':20.})
    native = dict(ordinary, sequence=80, available_at=60., started_at=20.,
        completed_at=30., a=[-.9375], r=-9.)
    admitted = dict(native, sequence=-1, available_at=350.)
    payload = dict(metadata=dict(scenario='stable'), packets=[dict(index=1,
        receipt=dict(processed_at=400., slots=[
            dict(sequence=11, origin='ordinary', source_sequence=11),
            dict(sequence=-1, origin='paired', source_sequence=80,
                pair_id='train/7X/w0', arm='high')]), rows=[ordinary, admitted])])
    truth = [dict(id='train/7X/w0', role='train', source_sequence=80,
        outcomes={'high':dict(transition=native)})]
    return payload, truth, ordinary, native


def build():
    payload, truth, ordinary, native = dataset()
    return build_budget_index(payload, pair_truth=truth, prefix_id='/v37/stable'), ordinary, native


def test_native_heldout_integer_collision_is_not_an_admitted_ordinary_slot():
    index, ordinary, _ = build()
    proof = audit_rows([ordinary], dict(scenario='stable', prefix_id='/v37/stable',
        kind='heldout', branch_id='v38/7X/early', arm='low'), index)
    assert proof['rows'][0]['integer_sequence_collision'] is True
    assert proof['summary']['authorized_rows'] == 0
    assert proof['rows'][0]['reason'] == 'heldout_branch_never_admitted'


def test_exact_ordinary_source_and_action_required_not_just_same_sequence():
    index, ordinary, _ = build()
    origin = dict(scenario='stable', prefix_id='/v37/stable', kind='ordinary')
    assert audit_rows([ordinary], origin, index)['summary']['authorized_rows'] == 1
    changed = dict(ordinary, a=[-.9375])
    proof = audit_rows([changed], origin, index)
    assert proof['summary']['authorized_rows'] == 0
    assert proof['rows'][0]['holding_matches_slot'] is False
    assert proof['rows'][0]['different_fields'] == ['a']
    assert audit_rows([ordinary], dict(origin, scenario='local_speed_drop'), index)['rows'][0]['reason'] == 'different_scenario'
    assert audit_rows([ordinary], dict(origin, prefix_id='/v38/stable'), index)['rows'][0]['reason'] == 'different_physical_prefix'


def test_pair_rewritten_slot_and_release_are_allowed_only_on_first_actual_td():
    index, _, native = build()
    origin = dict(scenario='stable', prefix_id='/v37/stable', kind='paired',
        branch_id='train/7X/w0', arm='high', source_sequence=80)
    later = dict(native, sequence=81, started_at=30., completed_at=40.)
    proof = audit_rows([native, later], origin, index)
    assert proof['summary']['authorized_rows'] == 1
    assert proof['summary']['slot_ids'] == [-1]
    assert proof['rows'][0]['admitted_release_clock_rewrite'] is True
    assert proof['rows'][1]['reason'] == 'later_paired_event_not_authorized'
    changed_origin = dict(origin, arm='low')
    assert audit_rows([native], changed_origin, index)['summary']['authorized_rows'] == 0
    assert index['summary']['paired_fixed300_wait_ride_aggregate_in_original_two_slots'] is True
    assert index['summary']['paired_later_event_chain_in_original_slots'] is False


def test_missing_provenance_is_missing_not_an_authorized_or_zero_label():
    index, ordinary, native = build()
    proof = audit_rows([ordinary], dict(scenario='stable', prefix_id='/v37/stable'), index)
    assert proof['rows'][0]['status'] == 'missing_provenance'
    payload, truth, _, _ = dataset()
    payload = deepcopy(payload)
    payload['packets'][0]['receipt']['slots'].pop()
    index = build_budget_index(payload, pair_truth=truth, prefix_id='/v37/stable')
    assert index['summary']['complete_provenance'] is False
    assert index['summary']['missing'] == [dict(slot_id=-1, reason='missing_receipt_provenance')]
