import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('v20_identity_mechanism',Path(__file__).with_name('service_mechanism.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def event(kind,t,ids=(),bus='bus',stop='shared',line='7X',**extra):
    return dict(kind=kind,clock_seconds=t,passenger_ids=list(ids),bus_id=bus,station_id=stop,
        line_id=line,person_sample_interval=[t-1,t],effective_capacity=5,eligible_waiting=2,**extra)


def test_same_person_matching_preserves_intervals_single_arm_and_changed_context():
    selected=dict(trip_id='bus',station_id='shared',line_id='7X')
    low=[event('board',10,['a','b']),event('board',13,['only_low'])]
    high=[event('board',12,['b']),event('board',20,['a'],bus='other'),event('board',14,['only_high'])]
    groups,_,_=m.pair_people(low,high,selected,'board')
    own=groups['controlled_trip']
    assert own['counts']==[2,1,1,1]
    assert own['delay']==[2,0,0,2,2,6.,10,6.,1,11,0]
    assert own['one_interval_example'][0][2]==[9,10] and own['one_interval_example'][1][2]==[19,20]
    assert m.delays([(event('board',10),dict(kind='board'))])==[0,0,0,0,None,None,None,None,None,None,1]


def fixture(root):
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    cases=[]
    for line in m.LINES:
        cases.append(dict(line_id=line,clock_decomposition=dict(stages=[[0,75,1,-1,1],[75,150,0,0,0],[150,225,0,0,0],[225,300,0,0,0]])))
        for arm in ('low','high'):
            shift=8 if arm=='high' else 0; own=line+'_44'; stop='7X13_102X26_705X27'
            selected=dict(trip_id=own,station_id=stop,line_id=line)
            path=root/'branches/native'/line/arm
            write(path/'summary.json',dict(status='complete',integrity=dict(all_pass=True,
                checks=dict(old_native_truth_exact=True)),suffix=dict(training_delta=dict(policy=0,world=0)),selected=selected))
            rows=[event('board',110+shift,['resident_person_B_both'],bus=own,stop=stop,line=line),
                  event('alight',180+shift,['resident_person_A_init'],bus=own,stop=stop,line=line),
                  event('passenger_sample_missing',190,bus='terminal',previous_onboard_count=3)]
            for i in range(5):
                s=stop if i==0 else 'later_stop_'+str(i)
                for kind in ('arrival','departure','service_ready','issue'):
                    rows.append(event(kind,100+i*30+shift,bus=own,stop=s,line=line,
                        action=[-.75] if arm=='high' else [-1.],native_detection_time=99+i*30+shift,observed_at=100+i*30+shift))
                rows.append(event('arrival',100+i*30+shift,bus='neighbour_'+str(i),stop=stop,line=line))
            write(path/'service_events.json',dict(events=rows))
    write(root/'service_summary.json',dict(status='complete',integrity=dict(all_pass=True),cases=cases))


def test_gated_identity_summary_keeps_initial_candidates_cutoff_missing_and_action_trace(tmp_path):
    fixture(tmp_path); value=m.analyze(tmp_path)
    assert len(value['cases'])==3
    row=value['cases'][0]
    assert row['initial_onboard_candidates']['controlled_trip']['n']==1
    assert row['initial_onboard_candidates']['controlled_trip']['delay'][6]==8
    assert row['missing']['low']['terminal_samples']==1 and row['missing']['low']['terminal_people']==3
    trace=row['controlled_trip']
    assert trace['stops']==5 and len(trace['first_four'])==4
    assert trace['first_four'][0][7:]==[[[100,0.]],[[108,7.5]]]
    assert row['fork_stop_order']['low']['arrival']['first_four'][0][1]==[['7X','7X_44'],['7X','neighbour_0']]
    data=json.dumps(m.compact(value),separators=(',',':'))
    assert len(data.encode())<12000 and 'resident_person_' not in data
    path=tmp_path/'branches/native/7X/high/summary.json'; bad=json.loads(path.read_text())
    bad['suffix']['training_delta']['policy']=1;path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError,match='zero-training'):m.analyze(tmp_path)


def test_repeat_transfer_patterns_are_separate_from_single_boarding_and_capacity_missing():
    rows=[event('board',1,['p']),event('alight',10,['p']),event('board',10,['p'],bus='next')]
    assert m.repeats(rows)==dict(repeat_board=1,repeat_alight=0,multiple_board_vehicles=1,
        observed_transfer_pattern=1,transfer_with_same_sample=1)
    rows[0]['effective_capacity']=0
    rows[1]['effective_capacity']=None;rows[1]['eligible_waiting']=None
    cap=m.capacities(rows,dict(trip_id='bus',station_id='shared',line_id='7X'))['controlled_trip']
    assert cap[:9]==[2,1,0,0,1,1,2,2,1]
