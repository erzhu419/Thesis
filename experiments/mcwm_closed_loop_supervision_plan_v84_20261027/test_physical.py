"""The observer cannot change execution or keep polling a departed trip all day."""
from types import SimpleNamespace

from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.physical import install


class Bridge:
    def __init__(self):
        self.now, self.hold, self.calls = 12000., None, []
        self.bus_obj_dic = {'bus': SimpleNamespace(
            just_server_stop_data_d={'A': {}}, arriver_stop_time_d={},
            depart_stop_time_d={}, bus_state_s='stop', next_stop_id_s='B')}
        self.passenger_time_ledger = SimpleNamespace(snapshot=lambda: SimpleNamespace(
            time_seconds=self.now))

    def apply_action(self, event, command):
        self.calls.append((event, command))
        self.hold = event.assigned_action = command[0]
        event.metadata = {'applied_additional_hold_seconds': command[0]}
        return 'applied'

    def _advance_one_step(self):
        self.now += 1
        if self.now == 12301:
            self.bus_obj_dic['bus'].depart_stop_time_d['A'] = self.now
        if self.now == 12304:
            self.bus_obj_dic['bus'].arriver_stop_time_d['B'] = self.now
        return 'advanced'


def api(bridge):
    def at_stop(_):
        return bridge.now < 12301 or bridge.now >= 12304

    def stops(_, __):
        return [SimpleNamespace(stoppingPlaceID='A' if bridge.now < 12301 else 'B',
                                duration=bridge.hold)]

    vehicle = SimpleNamespace(
        isAtBusStop=at_stop, getStops=stops,
        getActionStepLength=lambda _: 1., getLastActionTime=lambda _: bridge.now,
        getStopState=lambda _: int(at_stop(_)), getSpeed=lambda _: 0.,
        getLaneID=lambda _: 'lane', getLanePosition=lambda _: 1.,
        getPosition=lambda _: (0., 0.))
    return SimpleNamespace(vehicle=vehicle, simulation=SimpleNamespace(
        getTime=lambda: bridge.now, getDeltaT=lambda: 1.))


def test_real_hooks_apply_once_and_stop_after_next_arrival_beyond_old300():
    bridge = Bridge()
    original_apply, original_advance = bridge.apply_action, bridge._advance_one_step
    probe = install(bridge, dict(trip_id='bus', station_id='A', started_at=12000.),
                    traci=api(bridge))
    event = SimpleNamespace(bus_id='bus', stop_id='A', sim_time=11999.)
    command = [7.5, 1.]
    assert bridge.apply_action(event, command) == 'applied'
    for _ in range(310):
        assert bridge._advance_one_step() == 'advanced'
    value = probe.summary(True)
    assert bridge.calls == [(event, command)]
    assert value['native_action']['applied_hold_seconds'] == 7.5
    assert value['native_action']['event_time'] == 11999.
    assert value['native_action']['real_clock_before'] == 12000.
    assert value['native_action']['real_clock_after'] == 12000.
    assert value['departure']['clock_seconds'] == 12301.
    assert value['next_same_vehicle_arrival']['clock_seconds'] == 12304.
    assert value['native_history_next_arrival']['clock_seconds'] == 12304.
    assert value['native_tick_count'] == 304
    assert value['sampling_stopped_after_next_arrival']
    # Later trip removal cannot manufacture missing observations after completion.
    del bridge.bus_obj_dic['bus']
    for _ in range(10):
        bridge._advance_one_step()
    assert probe.summary(True) == value
    probe.close()
    assert bridge.apply_action == original_apply
    assert bridge._advance_one_step == original_advance


def test_unobserved_departure_and_arrival_stay_missing():
    bridge = Bridge()
    probe = install(bridge, dict(trip_id='bus', station_id='A', started_at=12000.),
                    traci=api(bridge))
    bridge.apply_action(SimpleNamespace(bus_id='bus', stop_id='A', sim_time=12000.), [0., 1.])
    bridge._advance_one_step()
    value = probe.summary(True)
    assert value['departure']['clock_seconds'] is None
    assert value['next_same_vehicle_arrival']['clock_seconds'] is None
    assert not value['sampling_stopped_after_next_arrival']
    probe.close()
