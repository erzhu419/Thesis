"""State selection checks outcome blindness, executable actions and replay scope."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v15_selection_test', HERE/'select_states.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def state(*, completion=False):
    # Canonical current slot0; completion floor is a real supported rule.
    value = [0.2 if completion else 0.1, 1/16, 0., 0., 0.]+[0.]*224
    if completion:
        value[5+11], value[5+9], value[5+8] = 1., 1., 0.
    return value


def trace(sequence, now, *, line='7X', completion=False):
    return dict(sequence=sequence, started_at=now, line_id=line, vehicle_id='bus',
        trip_id='trip', station_id='station', s=state(completion=completion),
        raw_s=[1.], event=[2.], issue_ledger={'time_seconds': now},
        a=[-1.], proposed_a=[-0.4])


def evaluation(sequence, now, *, role='train', line='7X', truth=None):
    return dict(sequence=sequence, issued_at=now, role=role, line_id=line,
        hold_seconds=0., prediction={'key': sequence}, truth=truth or {'cost': sequence})


def receipt(processed_at, sequences):
    return dict(processed_at=processed_at, records=5,
        calibration_sequence_ids=sequences, sequence_ids=sequences)


@pytest.fixture
def transform():
    return selection.ExecutionAction(dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600.))


@pytest.fixture
def protocol():
    return dict(paired_seed=6201, target_seed_offset=77000, initializer_updates=50000,
        target_record_budget=285, policy_update_budget=456, policy_updates_per_packet=8,
        horizon_seconds=24000, reference_v14_root='../v14',
        reference_v14_preflight_directory='preflight', scenarios=['stable','local_speed_drop'],
        target_lines=['7X','102X','705X'],
        training_windows=[dict(name='pre_shock',start=6000,end=8700),
                          dict(name='post_shock',start=12000,end=15000)],
        preflight=dict(seed=6201, scenario='stable', target_line='7X',
            target_record_budget=50, policy_update_budget=80, preflight_stop_seconds=5000,
            training_windows=[dict(name='preflight',start=1200,end=3300)]))


def test_first_admitted_training_issue_wins_without_outcome_selection(transform, protocol):
    summary = dict(receipts=[receipt(2000,[1,2])], evaluations=[
        evaluation(2,1600,truth={'cost':-1e20}), evaluation(1,1500,truth={'cost':1e20}),
        evaluation(0,1400,role='heldout'), evaluation(3,1450)])
    cell, window = next(selection.cells(protocol, preflight=True))
    row = selection.choose_state(summary,{1:trace(1,1500),2:trace(2,1600)},cell,window,{},transform,protocol)
    assert row['selected']['sequence'] == 1
    assert row['source_truth']['cost'] == 1e20
    assert row['executable']['high']['hold_seconds'] == 7.5
    changed = copy.deepcopy(summary)
    changed['evaluations'][0]['truth'],changed['evaluations'][1]['truth'] = {'missing':True},{'cost':-1e100}
    other = selection.choose_state(changed,{1:trace(1,1500),2:trace(2,1600)},cell,window,{},transform,protocol)
    assert other['selected'] == row['selected']


def test_completion_floor_excludes_an_actually_ineligible_issue(transform, protocol):
    summary=dict(receipts=[receipt(2000,[1,2])], evaluations=[evaluation(1,1500),evaluation(2,1600)])
    cell,window=next(selection.cells(protocol,preflight=True))
    row=selection.choose_state(summary,{1:trace(1,1500,completion=True),2:trace(2,1600)},cell,window,{},transform,protocol)
    assert row['selected']['sequence']==2 and row['candidates_considered']==2
    assert selection.executable_actions(trace(0,0,completion=True),transform) is None


def test_same_clock_prior_receipt_is_included_and_future_is_excluded(transform, protocol):
    summary=dict(receipts=[receipt(1500,[5]),receipt(2000,[10])],evaluations=[evaluation(10,1500)])
    cell,window=next(selection.cells(protocol,preflight=True))
    row=selection.choose_state(summary,{10:trace(10,1500)},cell,window,{},transform,protocol)
    assert row['source']['prefix_receipts']==summary['receipts'][:1]
    assert row['source']['prefix_target_labels']==5
    assert row['source']['prefix_policy_updates']==8
    assert row['source']['selected_calibration_sequence_ids_before_issue']==[5]


def test_missing_preserves_all_twelve_design_cells(transform, protocol):
    cells=list(selection.cells(protocol))
    assert len(cells)==12
    rows=[selection.choose_state(dict(receipts=[],evaluations=[]),{},cell,window,{},transform,protocol)
          for cell,window in cells]
    assert len({row['state_id'] for row in rows})==12
    assert all(row['status']=='selection_missing' and row['selected'] is None for row in rows)


def test_compact_summary_contains_no_state_predictions_or_outcomes(transform, protocol):
    summary=dict(receipts=[receipt(2000,[1])],evaluations=[evaluation(1,1500)])
    cell,window=next(selection.cells(protocol,preflight=True))
    row=selection.choose_state(summary,{1:trace(1,1500)},cell,window,
        dict(external_seed=83201,day_relative='preflight/day'),transform,protocol)
    small=selection.compact_summary(dict(status='complete',mode='preflight',states=[row]))
    assert small['expected_branches']==2 and small['maximum_new_suffix_seconds']==600
    assert small['states'][0]['sequence']==1
    payload=json.dumps(small)
    assert len(payload)<20000
    assert 'prediction' not in payload and 'source_truth' not in payload and 'raw_s' not in payload


def test_source_budget_and_preflight_native_stop_match_existing_contract(protocol):
    cell,_=next(selection.cells(protocol,preflight=True))
    source=dict(status='complete',integrity={'all_pass':True},cell=cell,
        initializer={'initial_updates':50000},
        budget={'target_labels':50,'policy_updates':80,'native_seconds':5001.})
    value=selection.source_metadata(source,'preflight/day',cell,protocol,preflight=True)
    assert value['external_seed']==83201 and value['horizon_seconds']==24000
    bad=copy.deepcopy(source);bad['budget']['target_labels']=51
    with pytest.raises(AssertionError):
        selection.source_metadata(bad,'preflight/day',cell,protocol,preflight=True)
