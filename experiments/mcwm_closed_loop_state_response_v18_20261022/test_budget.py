"""Prove paired labels replace ordinary slots and retain delayed native identity."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v16_budget_test', HERE/'budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def protocol():
    return dict(feedback_first_deadline=600, feedback_last_deadline=1200,
        feedback_interval_seconds=300, feedback_records_per_packet=5,
        target_record_budget=15, feedback_delay_seconds=30)


def transition(seq, *, issued=100., available=550., dt=2., action=-1.):
    return SimpleNamespace(s=np.arange(229,dtype=np.float32), a=np.asarray([action],np.float32),
        ns=np.arange(229,dtype=np.float32)+1, r=-123., dt=dt, terminal=False,
        episode_id=0,domain='target',started_at=issued,completed_at=issued+dt,
        available_at=available,vehicle_id='bus',line_id='7X',station_id='stop',
        next_vehicle_id='nextbus',sequence=seq,raw_s=np.arange(229,dtype=np.float32),
        raw_ns=np.arange(229,dtype=np.float32)+2,trip_id='trip')


def pair(index=0, *, seq=7, issued=100., global_dt=2., role='train'):
    return dict(id=f'pair{index}',index=index,role=role,source_sequence=seq,
        issued_at=issued,line_id='7X',features=[1.,0.,0.,.2],outcomes={
            'low':dict(truth=dict(status='complete',dt=300.,wait=100.,ride=500.,cost=700.),
                       transition=transition(seq,issued=issued,available=issued+32,dt=global_dt)),
            'high':dict(truth=dict(status='complete',dt=300.,wait=120.,ride=470.,cost=710.),
                        transition=transition(seq,issued=issued,available=issued+32,dt=global_dt,action=-.75))})


def ordinary(first, *, available=550.):
    return [transition(i,available=available) for i in range(first,first+6)]


def test_two_action_rows_replace_two_ordinary_rows_and_exclude_factual():
    value=budget.PairPacketBudget(protocol());original=pair()
    value.reserve(7,'pair0')
    value.ingest([transition(7),*ordinary(10)])
    normalized=value.add_pair(original)
    packet,discarded=value.take(600.)
    assert len(packet)==5 and {row.sequence for row in packet}=={-1,-2,13,14,15}
    assert len(value.last_pairs)==1 and normalized['selected_for_training'] is True
    assert normalized['available_at']==430.
    assert original['outcomes']['low']['transition'].sequence==7  # Source outcome is not mutated.
    assert value.receipts[-1]['ordinary_records']==3 and value.receipts[-1]['paired_records']==2
    assert value.receipts[-1]['pair_source_sequences']==[7]
    assert value.summary()['excluded_factual_sequences']==[7]
    assert value.summary()['target_labels']==5 and 7 not in value.selected_sequences
    assert discarded==[10,11,12]


def test_later_pair_cannot_fit_the_earlier_deadline_and_each_packet_stays_five():
    value=budget.PairPacketBudget(protocol());value.reserve(7,'pair0')
    record=value.add_pair(pair(issued=400.))
    value.ingest(ordinary(10))
    first,_=value.take(600.)
    assert len(first)==5 and value.last_pairs==[] and record['selected_for_training'] is False
    value.ingest(ordinary(20,available=850.))
    second,_=value.take(900.)
    assert len(second)==5 and {row.sequence for row in second if row.sequence<0}=={-1,-2}
    assert record['admitted_deadline']==900 and record['available_at']==730.
    value.ingest(ordinary(30,available=1150.));value.take(1200.)
    assert value.summary()['target_labels']==15
    assert value.summary()['ordinary_labels']==13 and value.summary()['paired_action_labels']==2
    assert all(row['records']==5 for row in value.receipts)


def test_zero_global_pair_uses_two_slots_and_keeps300_truth_for_the_readout():
    value=budget.PairPacketBudget(protocol());value.reserve(7,'pair0')
    record=value.add_pair(pair(global_dt=0.));value.ingest(ordinary(10))
    packet,_=value.take(600.)
    assert len(packet)==5 and len([row for row in packet if row.dt>0])==3
    assert value.receipts[-1]['zero_global_records']==2
    assert len(value.last_pairs)==1 and record['outcomes']['low']['truth']['dt']==300.
    assert value.summary()['paired_action_labels']==2


def test_one_ready_pair_per_packet_and_no_extra_label_for_repeated_pair():
    value=budget.PairPacketBudget(protocol())
    for index,seq in ((0,7),(1,8)):
        value.reserve(seq,f'pair{index}');value.add_pair(pair(index,seq=seq))
    value.ingest(ordinary(10));packet,_=value.take(600.)
    assert len(packet)==5 and [row['id'] for row in value.last_pairs]==['pair0']
    assert value.summary()['pending_pair_ids']==['pair1']
    with pytest.raises(ValueError,match='reserved'):
        value.add_pair(pair())
    value.ingest(ordinary(20,available=850.));value.take(900.)
    assert value.summary()['paired_action_labels']==4
    assert value.summary()['ordinary_labels']==6
    assert budget.paired_label_ids(1)==dict(low=-3,high=-4)


def test_heldout_and_unreserved_pair_cannot_enter_budget():
    value=budget.PairPacketBudget(protocol())
    with pytest.raises(ValueError,match='Heldout'):
        value.add_pair(pair(role='heldout'))
    with pytest.raises(ValueError,match='reserved'):
        value.add_pair(pair())
    with pytest.raises(ValueError,match='together'):
        value.ingest([transition(-1)])


def test_transition_and_pair_codec_preserve_native_float32_arrays_and_negative_ids():
    value=budget.PairPacketBudget(protocol());value.reserve(7,'pair0')
    normalized=value.add_pair(pair())
    encoded=budget.pair_to_dict(normalized)
    restored=budget.pair_from_dict(encoded)
    assert restored['available_at']==430.
    for arm,seq in (('low',-1),('high',-2)):
        expected=normalized['outcomes'][arm]['transition'];actual=restored['outcomes'][arm]['transition']
        assert actual.sequence==seq and actual.available_at==430.
        for key in budget.ARRAY_FIELDS:
            assert getattr(actual,key).dtype==np.float32
            assert np.array_equal(getattr(actual,key),getattr(expected,key))
        assert actual.r==expected.r and actual.completed_at==expected.completed_at
