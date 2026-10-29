from types import SimpleNamespace
import pytest
from experiments.mcwm_closed_loop_service_shift_v21_20261022.history import (
    NativeRunHistory, future_services, native_running_reference, segment_key,
)


class Bridge:
    def __init__(self):
        line = SimpleNamespace(stop_id_l=list('ABCD'),edge_between_stop_d={
            'A':[], 'B':['ab'],'C':['bc'],'D':['cd']})
        self.line_obj_dic = {'7X':line,'102X':line}
        self.bus_obj_dic = {}
        self.passenger_time_ledger = SimpleNamespace(snapshot=lambda:SimpleNamespace(time_seconds=0.))

    def _synchronize_bus_passenger_transition(self,*args):
        return 17


def bus(line='7X'):
    return SimpleNamespace(belong_line_id_s=line,
        arriver_timetable_d=dict(A=100,B=110,C=120,D=130),
        arriver_stop_time_d={},depart_stop_time_d={})


def issue():
    return dict(line_id='7X',station_id='A',started_at=100.,trip_id='current')


def complete(history,bridge,bus_id,line='7X',delay=0):
    b=bus(line)
    b.depart_stop_time_d=dict(A=10,B=30,C=50)
    b.arriver_stop_time_d=dict(B=25+delay,C=45+delay,D=65+delay)
    bridge.bus_obj_dic[bus_id]=b
    history.observe_bus(bus_id,b,80.)
    return b


def test_native_reference_is_departure_arrival_static_increment_not_service_interval():
    b=bus()
    assert native_running_reference(b,'A','B')==10
    b.arriver_timetable_d.pop('B')
    assert native_running_reference(b,'A','B') is None


def test_same_physical_shared_line_history_and_strict_causal_missing():
    bridge=Bridge(); history=NativeRunHistory(bridge)
    complete(history,bridge,'prior','102X')
    snapshot=history.snapshot(issue())
    assert snapshot['status']=='supported'
    assert snapshot['mean_residual_scaled']==pytest.approx(5/60)
    assert all(s['latest']['contributing_lines']==['102X'] for s in snapshot['segments'])
    segment=snapshot['segments'][1]; rows=history.rows[segment_key(segment)]
    rows[0]['observed_at']=100.
    missing=history.snapshot(issue())
    assert missing['status']=='history_missing' and missing['mean_residual_scaled'] is None
    rows[0]['observed_at']=99.; rows[0]['completed_at']=100.
    assert history.snapshot(issue())['status']=='history_missing'
    history.close()


def test_same_stop_pair_different_edge_path_does_not_supply_history():
    bridge=Bridge()
    bridge.line_obj_dic['102X']=SimpleNamespace(stop_id_l=list('ABCD'),
        edge_between_stop_d={'A':[],'B':['different'],'C':['bc'],'D':['cd']})
    history=NativeRunHistory(bridge); complete(history,bridge,'prior','102X')
    snapshot=history.snapshot(issue())
    assert snapshot['segments'][0]['latest'] is None
    assert snapshot['status']=='history_missing'
    history.close()


def test_equal_clock_history_is_independent_of_native_bus_iteration_order():
    outputs=[]
    for order in (('z','a'),('a','z')):
        bridge=Bridge(); history=NativeRunHistory(bridge)
        for bus_id in order: complete(history,bridge,bus_id)
        outputs.append(history.snapshot(issue()))
        history.close()
    assert outputs[0]==outputs[1]
    assert outputs[0]['segments'][0]['latest']['tied_completed_intervals']==2


def test_hook_delegates_and_restores_and_future_cutoff_remains_censored():
    bridge=Bridge(); history=NativeRunHistory(bridge)
    b=complete(history,bridge,'current')
    assert bridge._synchronize_bus_passenger_transition('current','Edge','Stop',b,80.)==17
    b.arriver_stop_time_d=dict(B=120.,C=160.,D=410.)
    b.depart_stop_time_d=dict(B=125.,C=165.,D=415.)
    future=future_services(bridge,issue(),400.)
    assert future['arrival']==[20.,60.,None]
    assert future['departure']==[25.,65.,None]
    assert future['status']['arrival']==['complete','complete','censored']
    history.close()
    assert '_synchronize_bus_passenger_transition' not in bridge.__dict__
