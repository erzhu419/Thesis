"""Fake interface/order tests; these are not native coverage evidence."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('native_context_observer_v79',
    Path(__file__).with_name('observer.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeBridge:
    max_hold_seconds = 60.

    def __init__(self, *, clock=12001., plan=('S0', 'S1'), stage=None):
        self.clock, self.applied, self.stage_calls, self.plan_calls = clock, [], [], 0
        self.stage = stage or SimpleNamespace(type=3, line='L', destStop='S1')
        self.line_obj_dic = {'L': SimpleNamespace(stop_id_l=['S0', 'S1', 'S2'])}
        self.ledger = SimpleNamespace(time_seconds=clock, waiting_passengers=2,
            in_vehicle_passengers=1, accrued_wait_seconds=44., accrued_in_vehicle_seconds=22.)
        self.passenger_time_ledger = SimpleNamespace(snapshot=lambda: self.ledger)
        self.rng = {'unchanged': (1, 2, 3)}

        def get_stage(person, index):
            self.stage_calls.append((person, index))
            if isinstance(self.stage, Exception):
                raise self.stage
            return self.stage

        def get_stops(trip, limit):
            self.plan_calls += 1
            return [SimpleNamespace(stoppingPlaceID=stop) for stop in plan][:limit]

        self.api = SimpleNamespace(simulation=SimpleNamespace(getTime=lambda: self.clock),
            vehicle=SimpleNamespace(getPersonIDList=lambda trip: ('private_onboard',),
                getPersonCapacity=lambda trip: 50., getLine=lambda trip: 'L', getStops=get_stops),
            busstop=SimpleNamespace(getPersonIDs=lambda stop: ('private_queue',)),
            person=SimpleNamespace(getStage=get_stage))

    def apply_action(self, event, command):
        self.applied.append((event, command))
        return 'native-result'


def make(tmp_path, bridge):
    return MODULE.NativeContextObserver(bridge, tmp_path/'trace.jsonl', api=bridge.api,
        state_probe=lambda: dict(bridge.rng), state_equal=lambda a, b: a == b)


def event(clock=12001., *, stop='S0', index=0):
    native = SimpleNamespace(bus_id='L_trip', line_id='L', stop_id=stop,
                             stop_idx=index, sim_time=clock-1.)
    issued = dict(trip_id='L_trip', line_id='L', station_id=stop,
        started_at=clock, a=np.asarray([-.75], np.float32), sequence=7)
    return issued, native


def test_sample_is_before_same_native_event_and_passes_command_unchanged(tmp_path):
    bridge = FakeBridge()
    observer = make(tmp_path, bridge)
    issued, native = event()
    capture = observer.capture(issued, native)
    command = [7.5, 1.]
    assert bridge.apply_action(native, command) == 'native-result'
    assert bridge.applied == [(native, command)] and bridge.applied[0][1] is command
    summary = observer.summary()
    assert summary['all_pass'] and summary['captures']['observed'] == 1
    assert summary['coverage']['subgroups']['release_positive']['observed'] == 1
    assert summary['stage_semantics']['queue']['nonempty_line_examples'][0]['line'] == 'L'
    assert summary['clocks']['issue_minus_event']['min'] == 1.
    assert capture['context']['clocks']['obs_start'] == 12001.
    observer.close()
    rows = [json.loads(row) for row in (tmp_path/'trace.jsonl').read_text().splitlines()]
    assert rows[0]['observation_end_order'] < rows[1]['apply_start_order']
    assert rows[1]['capture_seq'] == rows[0]['capture_seq']
    assert 'private_onboard' not in json.dumps(summary) and 'private_queue' not in json.dumps(rows)
    assert 'apply_action' not in bridge.__dict__


@pytest.mark.parametrize('clock', [11999., 16200.])
def test_window_is_right_open_and_does_not_read_context(tmp_path, clock):
    bridge = FakeBridge(clock=clock)
    observer = make(tmp_path, bridge)
    issued, native = event(clock)
    assert observer.capture(issued, native) is None
    bridge.apply_action(native, [7.5, 1.])
    assert not bridge.stage_calls and bridge.plan_calls == 0
    assert observer.summary()['captures']['total'] == 0
    observer.close()


@pytest.mark.parametrize('failure', ['absent', 'wrong_event', 'wrong_holding'])
def test_actual_apply_rejects_uncaptured_or_mismatched_control(tmp_path, failure):
    bridge = FakeBridge()
    observer = make(tmp_path, bridge)
    issued, native = event()
    if failure != 'absent':
        observer.capture(issued, native)
    if failure == 'wrong_event':
        native = SimpleNamespace(**vars(native))
    with pytest.raises(RuntimeError):
        bridge.apply_action(native, [1.875 if failure == 'wrong_holding' else 7.5, 1.])
    assert bridge.applied == []
    observer.close()


def test_native_current_only_plan_is_unscorable_not_invented_future(tmp_path):
    bridge = FakeBridge(plan=('S0',))
    observer = make(tmp_path, bridge)
    issued, native = event()
    row = observer.capture(issued, native)
    bridge.apply_action(native, [7.5, 1.])
    assert row['observed'] and row['next_stop_plan']['status'] == 'current_only'
    assert row['next_stop_plan']['matches'] is None
    assert observer.summary()['all_pass']
    observer.close()


def test_missing_stage_stays_missing_and_terminal_is_excluded(tmp_path):
    bridge = FakeBridge(stage=RuntimeError('native stage unavailable'))
    observer = make(tmp_path, bridge)
    issued, native = event()
    row = observer.capture(issued, native)
    bridge.apply_action(native, [7.5, 1.])
    assert row['context']['z'] is None and not row['observed']
    assert observer.summary()['coverage']['membership_unknown']['release_positive'] == 1
    _, terminal = event(stop='S2', index=2)
    bridge.apply_action(terminal, [0., 1.])
    assert observer.summary()['captures']['terminal'] == 1
    assert bridge.stage_calls == [('private_onboard', 0), ('private_queue', 0)]
    observer.close()


def test_sampling_rng_or_clock_mutation_is_detected_before_action(tmp_path):
    bridge = FakeBridge()
    observer = make(tmp_path, bridge)
    original_stage = bridge.api.person.getStage

    def changed_stage(person, index):
        bridge.rng['changed'] = True
        return original_stage(person, index)

    bridge.api.person.getStage = changed_stage
    issued, native = event()
    with pytest.raises(RuntimeError, match='sampling changed'):
        observer.capture(issued, native)
    assert bridge.applied == []
    assert not observer.summary()['checks']['sample_rng_unchanged']
    observer.close()
