"""Check causal endpoint alignment and unchanged original replay semantics."""
from dataclasses import fields
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.mcwm_closed_loop_context_wiring_v80_20261027.stream import (
    ContextEventStream, EventStream, EI)
from experiments.mcwm_closed_loop_v1.stream import Transition, TransitMDPConfigV7
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_DIM
from bus_h2o.passenger_time import PassengerTimeSnapshot


class Source:
    def __init__(self):
        self.line_map = {'L': SimpleNamespace(line_idx=0, stations=list(range(4)),
            bus_all=[], passenger_time_ledger=SimpleNamespace(snapshot=self.snapshot))}
        self.commands = []
        self.reset()

    def snapshot(self):
        return PassengerTimeSnapshot(float(self.time), 2, 3, 2.*self.time, 3.*self.time)

    def passenger_time_at_event(self, agent):
        return self.snapshot()

    def reset(self):
        self.time, self.position, self.state, self.done = 0, -1, {}, False

    def step_to_event(self, commands):
        self.commands.append(commands)
        batches = [(10, [(0, 0), (1, 0)]), (20, [(2, 0)]),
                   (30, [(0, 1)]), (40, [(2, 1)])]
        self.position += 1
        if self.position == len(batches):
            self.time, self.state, self.done = 60, {}, True
            return
        self.time, events = batches[self.position]
        self.state, buses = {}, []
        for bus, stop in events:
            event = np.zeros(RAW_EVENT_STATE_DIM, np.float32)
            for name, value in dict(station_index=stop, sim_time_seconds=self.time-1,
                forward_headway_seconds=300, backward_headway_seconds=360,
                target_forward_headway_seconds=360, target_backward_headway_seconds=360,
                bus_capacity=50, onboard_passengers=5, base_dwell_seconds=10).items():
                event[EI[name]] = value
            self.state['L', bus] = [[event]]
            buses.append(SimpleNamespace(bus_id=bus, trip_id=str(bus),
                last_station=SimpleNamespace(station_name=f'S{stop}')))
        self.line_map['L'].bus_all = buses


def reader(issue):
    now = issue['started_at']
    return dict(observed=issue['sequence'] != 2,
        z=None if issue['sequence'] == 2 else float(issue['sequence']),
        reasons=['synthetic_missing'] if issue['sequence'] == 2 else [],
        clocks=dict(issue=now, event=float(issue['event'][EI['sim_time_seconds']]),
            obs_start=now, obs_end=now, atomic_at_issue=True))


def run(stream):
    rows, physical = [], []
    while not stream.done or stream.episode_id < 0:
        rows.extend(stream.step(lambda state: [-1.]))
        physical.extend(stream.release_intervals())
    return rows, physical


@pytest.mark.parametrize('delay,thinning,execute', [(0,1,False), (30,1,False),
                                                 (15,2,False), (0,1,True)])
def test_original_fields_actions_delay_and_physical_intervals_are_exact(delay, thinning, execute):
    old_env, new_env = Source(), Source()
    config = TransitMDPConfigV7(horizon_seconds=60)
    kw = dict(delay_seconds=delay, record_every=thinning, execute_transform=execute)
    old = EventStream(old_env, [3], config, **kw)
    new = ContextEventStream(new_env, [3], config, context_reader=reader, **kw)
    rows0, physical0 = run(old)
    rows1, physical1 = run(new)
    # Include rows still delayed past the horizon; never release them early.
    for left, right in zip(rows0+old._queue, rows1+new._queue, strict=True):
        for field in fields(Transition):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b
        assert right.ctx_s['binding']['sequence'] == right.sequence
        if right.terminal:
            assert not right.ctx_ns['observed']
            assert right.ctx_ns['binding']['kind'] == 'absorbing_terminal'
        else:
            binding = right.ctx_ns['binding']
            assert binding['sequence'] == right.sequence+1
            assert binding['vehicle_id'] == right.next_vehicle_id
            assert binding['issue_clock'] == right.completed_at
    assert old_env.commands == new_env.commands
    assert [(p.sequence,p.dt) for p in physical0] == [(p.sequence,p.dt) for p in physical1]
    assert (old.completed_count,old.thinned_count) == (new.completed_count,new.thinned_count)


def test_same_tick_other_vehicle_missing_and_reset_bind_to_actual_issue():
    seen = []
    stream = ContextEventStream(Source(), [3], TransitMDPConfigV7(horizon_seconds=60),
        execute_transform=False, context_reader=reader, on_issue=seen.append)
    rows, _ = run(stream)
    assert rows[0].dt == 0 and rows[0].vehicle_id != rows[0].next_vehicle_id
    assert rows[0].ctx_s['z'] == 0 and rows[0].ctx_ns['z'] == 1
    assert rows[0].ctx_s['binding']['event_clock'] == 9
    assert rows[0].ctx_s['binding']['issue_clock'] == 10
    assert not rows[1].ctx_ns['observed'] and rows[1].ctx_ns['z'] is None
    assert all('ctx_s' in issue for issue in seen)
    stream.reset()
    rows, _ = run(stream)
    assert all(row.ctx_s['binding']['episode_id'] == 1 for row in rows)


def test_changed_read_clock_is_rejected_before_native_apply():
    def late(issue):
        context = reader(issue)
        context['clocks']['obs_end'] += 1
        return context
    env = Source()
    stream = ContextEventStream(env, [3], TransitMDPConfigV7(horizon_seconds=60),
        context_reader=late)
    with pytest.raises(RuntimeError, match='atomically'):
        stream.step(lambda state: [-1.])
    assert env.commands == [{}]  # reset only; no new command was applied.


def test_native_reader_resolves_bridge_after_lazy_reset(monkeypatch):
    import sys
    from experiments.mcwm_closed_loop_context_wiring_v80_20261027 import stream as module
    class Bridge:
        line_obj_dic = {'L': SimpleNamespace(stop_id_l=['S0','S1'])}
    env = SimpleNamespace(_bridge=None)
    read = module.NativeContextReader(env)
    env._bridge = Bridge()
    env._current_events = [SimpleNamespace(bus_id='trip',line_id='L',stop_id='S0',
        stop_idx=0,sim_time=9.)]
    api = object()
    monkeypatch.setattr(sys.modules[Bridge.__module__], 'traci', api, raising=False)
    def observe(actual_api, **kwargs):
        assert actual_api is api
        return kwargs
    monkeypatch.setattr(module, 'observe_context', observe)
    result = read(dict(trip_id='trip',line_id='L',station_id='S0',started_at=10.))
    assert result == dict(trip_id='trip',line_id='L',station_index=0,
        route_stops=['S0','S1'],issue_clock=10.,event_clock=9.)
