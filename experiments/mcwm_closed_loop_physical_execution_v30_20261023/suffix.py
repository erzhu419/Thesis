"""Three parallel native suffixes copied from one live, paused factual prefix."""
from __future__ import annotations

import copy
from dataclasses import asdict
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from experiments.mcwm_closed_loop_physical_execution_v30_20261023.fork_state import fork_children
from experiments.mcwm_closed_loop_v1.stream import policy_action_to_hold_seconds_v7

HOLD_SECONDS = dict(low=0., mid=.9375, high=1.875)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def selected_state(event, snapshot):
    selected = {key: event[key] for key in ('sequence', 'started_at', 'line_id',
        'vehicle_id', 'trip_id', 'station_id', 's', 'raw_s', 'event')}
    selected['issue_ledger'] = {key: float(getattr(snapshot, key)) for key in (
        'time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
        'accrued_wait_seconds', 'accrued_in_vehicle_seconds')}
    return plain(selected)


def endpoint(start, stop, *, status='complete'):
    if stop is None:
        return dict(status=status, dt=None, wait=None, ride=None, cost=None)
    wait = float(stop.accrued_wait_seconds-start.accrued_wait_seconds)
    ride = float(stop.accrued_in_vehicle_seconds-start.accrued_in_vehicle_seconds)
    return dict(status=status, dt=float(stop.time_seconds-start.time_seconds),
        wait=wait, ride=ride, cost=2*wait+ride)


class NativePhysicalObserver:
    """Read only the selected physical bus, retaining short ticks and boundaries.

    SUMO presence changes and native arrival/departure histories are independent
    observations. Neither uses the action's predicted departure metadata. Bus
    logic runs before simulationStep, so its history may be discovered one tick
    after the TraCI change while preserving the actual history clock.
    """

    def __init__(self, bridge, *, traci=None):
        self.bridge = bridge
        self.api = traci if traci is not None else importlib.import_module(
            bridge.__class__.__module__).traci
        self.bus = self.station = self.start = self.cutoff = None
        self.apply_before = self.apply_after = self.previous = None
        self.tick_samples, self.transitions = [], []
        self.transition_rows_omitted = self.native_tick_count = 0
        self.missing = Counter()
        self.seen = set()
        self.departure = self.next_arrival = None
        self.history_departure = self.history_arrival = None

    def _get(self, field, getter):
        try:
            return plain(getter())
        except Exception:
            # SUMO legitimately removes a trip at its terminal; keep absent
            # observations absent instead of treating removal as a departure.
            self.missing[field] += 1
            return None

    def activate(self, selected):
        self.bus, self.station = selected['trip_id'], selected['station_id']
        self.start, self.cutoff = float(selected['started_at']), float(selected['started_at'])+300.
        obj = self.bridge.bus_obj_dic.get(self.bus)
        for kind, attr in (('arrival', 'arriver_stop_time_d'), ('departure', 'depart_stop_time_d')):
            for station, values in getattr(obj, attr, {}).items():
                values = values if isinstance(values, (list, tuple)) else (values,)
                self.seen.update((kind, str(station), occurrence, float(clock))
                    for occurrence, clock in enumerate(values))

    def sample(self):
        vehicle, bus = self.api.vehicle, self.bus
        obj = self.bridge.bus_obj_dic.get(bus)
        service = getattr(obj, 'just_server_stop_data_d', {})
        clock = self._get('native_clock_seconds', lambda: float(self.api.simulation.getTime()))
        at_stop = self._get('is_at_bus_stop', lambda: bool(vehicle.isAtBusStop(bus)))
        stops = self._get('current_stop', lambda: vehicle.getStops(bus, 1)) if at_stop else None
        current_stop = str(stops[0].stoppingPlaceID) if stops and stops[0].stoppingPlaceID else None
        return dict(native_clock_seconds=clock,
            ledger_clock_seconds=float(self.bridge.passenger_time_ledger.snapshot().time_seconds),
            simulation_step_length_seconds=self._get('simulation_step_length_seconds', lambda: float(self.api.simulation.getDeltaT())),
            action_step_length_seconds=self._get('action_step_length_seconds', lambda: float(vehicle.getActionStepLength(bus))),
            last_action_time_seconds=self._get('last_action_time_seconds', lambda: float(vehicle.getLastActionTime(bus))),
            is_at_bus_stop=at_stop, stop_state=self._get('stop_state', lambda: int(vehicle.getStopState(bus))),
            speed_mps=self._get('speed_mps', lambda: float(vehicle.getSpeed(bus))),
            lane_id=self._get('lane_id', lambda: str(vehicle.getLaneID(bus))),
            lane_position_m=self._get('lane_position_m', lambda: float(vehicle.getLanePosition(bus))),
            position_xy=self._get('position_xy', lambda: list(vehicle.getPosition(bus))),
            native_bus_state=getattr(obj, 'bus_state_s', None),
            current_service_station_id=current_stop,
            registered_stop_duration_seconds=self._get('registered_stop_duration_seconds',
                lambda: float(stops[0].duration)) if stops else None,
            native_current_or_just_served_station_id=str(next(iter(service))) if service else None,
            next_stop_id=getattr(obj, 'next_stop_id_s', None))

    def _event(self, station, clock, source, observed_at):
        return dict(status='observed', bus_id=self.bus, station_id=str(station),
            clock_seconds=float(clock), relative_seconds=float(clock)-self.start,
            observed_at=float(observed_at), time_source=source)

    def _observe_histories(self, observed_at):
        obj = self.bridge.bus_obj_dic.get(self.bus)
        for kind, attr in (('departure', 'depart_stop_time_d'), ('arrival', 'arriver_stop_time_d')):
            for station, values in getattr(obj, attr, {}).items():
                values = values if isinstance(values, (list, tuple)) else (values,)
                for occurrence, clock in enumerate(values):
                    key = kind, str(station), occurrence, float(clock)
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    if not self.start <= float(clock) <= self.cutoff:
                        continue
                    event = self._event(station, clock, 'native_bus_'+kind+'_history', observed_at)
                    if kind == 'departure' and str(station) == self.station and self.history_departure is None:
                        self.history_departure = event
                    departure = self.departure or self.history_departure
                    if (kind == 'arrival' and str(station) != self.station
                            and self.history_arrival is None and departure is not None
                            and float(clock) >= departure['clock_seconds']):
                        self.history_arrival = event

    def observe(self, row, *, tick=False):
        if tick:
            self.native_tick_count += 1
            if len(self.tick_samples) < 12:
                self.tick_samples.append(row)
        clock, previous = row['native_clock_seconds'], self.previous
        if clock is not None and previous is not None:
            was, now = previous['is_at_bus_stop'], row['is_at_bus_stop']
            previous_station = (previous['current_service_station_id'] or
                previous['native_current_or_just_served_station_id'])
            if (was is True and now is False and self.departure is None
                    and previous_station == self.station):
                self.departure = self._event(self.station, clock,
                    'traci_isAtBusStop_true_to_false', clock)
            if (was is False and now is True and self.departure is not None
                    and self.next_arrival is None and row['current_service_station_id']
                    and row['current_service_station_id'] != self.station):
                self.next_arrival = self._event(row['current_service_station_id'], clock,
                    'traci_isAtBusStop_false_to_true', clock)
            fields = ('is_at_bus_stop', 'stop_state', 'native_bus_state',
                'current_service_station_id', 'next_stop_id')
            if any(previous[field] != row[field] for field in fields):
                if len(self.transitions) < 32:
                    self.transitions.append(row)
                else:
                    self.transition_rows_omitted += 1
        if clock is not None:
            self._observe_histories(clock)
        self.previous = row

    def summary(self, cutoff_reached):
        def value(primary, secondary):
            return primary or secondary or dict(status='truncated' if cutoff_reached else 'missing',
                bus_id=self.bus, station_id=None, clock_seconds=None, relative_seconds=None,
                observed_at=None, time_source=None)
        departure = value(self.departure, self.history_departure)
        arrival = value(self.next_arrival, self.history_arrival)
        samples = [row for row in (self.apply_before, self.apply_after) if row is not None]
        return plain(dict(apply_before=self.apply_before, apply_after=self.apply_after,
            short_tick_samples=self.tick_samples, transitions=self.transitions,
            transition_rows_omitted=self.transition_rows_omitted, native_tick_count=self.native_tick_count,
            departure=departure, next_same_vehicle_arrival=arrival,
            native_history_departure=self.history_departure,
            native_history_next_arrival=self.history_arrival,
            coverage=dict(actual_clock_observed=bool(samples) and all(row['native_clock_seconds'] is not None for row in samples),
                action_step_length_observed=bool(samples) and all(row['action_step_length_seconds'] is not None for row in samples),
                issue_stop_state_observed=bool(samples) and all(row['is_at_bus_stop'] is not None and row['stop_state'] is not None for row in samples),
                departure_observed=departure['status'] == 'observed',
                next_arrival_observed=arrival['status'] == 'observed'), missing=dict(self.missing)))


class NativeTruthTriple:
    """Parent resumes its original action; children replace just the selected one.

    The caller records the common nominal issue once before ``fork`` and uses
    the existing ForkEventStream. Children never return to main-day training:
    call ``observe_issue`` for every child issue, ``observe_progress`` after
    each child step, and ``finish_child`` at the bounded native cutoff.
    """

    def __init__(self, out, expected, counts, external_seed):
        self.out, self.expected = Path(out), expected
        self.counts, self.external_seed = counts, int(external_seed)
        self.in_child, self.arm = False, None
        self.receipt, self.values = None, None
        self.selected = self.execution = self.next_snapshot = None
        self.native_actions = []
        self.before = None
        self.started_wall = None
        self.physical = None

    def fork(self, event, stream, trace, prepare_child=None):
        selected = self.expected['selected']
        if (event['sequence'] != selected['sequence']
                or event['started_at'] != selected['started_at']
                or not np.array_equal(event['s'], np.asarray(selected['s'], np.float32))):
            raise RuntimeError('The declared v29 tail state was not exactly reproduced')
        self.before = copy.deepcopy(self.counts())

        def prepare(arm):
            self.in_child, self.arm = True, arm
            self.started_wall = time.time()
            # Open route parsers were already detached by the approved helper.
            # Detach the Python trace/logs before the half-issued event resumes.
            trace.writer.close()
            if trace.reference is not None:
                trace.reference.close()
            trace.path = self.out/arm/'suffix_trace.jsonl'
            trace.writer = trace.path.open('x', encoding='utf-8')
            trace.reference = None
            self._install_native_observer(stream)
            if prepare_child is not None:
                prepare_child(arm)

        result = fork_children(self.out, trace, lambda: stream.now, self.counts, prepare)
        if isinstance(result, str):
            return np.asarray(self.expected['actions'][result], np.float32)
        self.receipt = result
        self.values = {arm: json.loads((self.out/arm/'summary.json').read_text())
            for arm in ('low', 'mid', 'high')}
        if any(self.values['low']['selected'] != self.values[arm]['selected']
                for arm in ('mid', 'high')):
            raise RuntimeError('Parallel suffixes changed the inherited selected state')
        if any(self.values['low']['native']['other_batch_actions'] !=
                self.values[arm]['native']['other_batch_actions'] for arm in ('mid', 'high')):
            raise RuntimeError('Nonselected controls in the fork batch differ')
        return None

    def _install_native_observer(self, stream, *, traci=None):
        bridge = stream.env._bridge
        apply_action = bridge.apply_action
        advance = bridge._advance_one_step
        self.physical = NativePhysicalObserver(bridge, traci=traci)

        def observed_advance(*args, **kwargs):
            result = advance(*args, **kwargs)
            if self.physical.start is not None:
                self.physical.observe(self.physical.sample(), tick=True)
            return result

        def observe_native(event, command):
            selected = self.selected
            # A queued decision's event clock can precede the current passenger
            # ledger clock. Identify this first application by the selected
            # trip/stop, preserving both clocks rather than equating them.
            selected_application = selected is not None and not self.native_actions and (
                    str(event.bus_id) == selected['trip_id']
                    and str(event.stop_id) == selected['station_id'])
            if selected_application:
                self.physical.apply_before = self.physical.sample()
                self.physical.observe(self.physical.apply_before)
            apply_action(event, command)
            if selected_application:
                self.physical.apply_after = self.physical.sample()
                self.physical.observe(self.physical.apply_after)
                self.native_actions.append(dict(bus_id=str(event.bus_id),
                    station_id=str(event.stop_id), time=float(event.sim_time),
                    issue_ledger_time=selected['started_at'],
                    real_clock_before=self.physical.apply_before['native_clock_seconds'],
                    real_clock_after=self.physical.apply_after['native_clock_seconds'],
                    assigned_action=float(event.assigned_action),
                    applied_hold_seconds=float(event.metadata['applied_additional_hold_seconds']),
                    predicted_departure_time_seconds=float(event.metadata['predicted_departure_time_seconds'])))

        # This assignment happens only after the child exists; the paused
        # parent's bridge and all native action semantics remain untouched.
        bridge.apply_action = observe_native
        bridge._advance_one_step = observed_advance

    def observe_issue(self, event, stream):
        if not self.in_child:
            return
        if event['sequence'] == self.expected['selected']['sequence']:
            self.selected = selected_state(event, stream.fork_snapshot)
            self.physical.activate(self.selected)
            action = np.asarray(event['a'], np.float32)
            actual = float(policy_action_to_hold_seconds_v7(action[0], stream.contract))
            requested = HOLD_SECONDS[self.arm]
            self.execution = dict(requested_hold_seconds=requested,
                executed_hold_seconds=actual, executed_action=action.tolist(),
                nominal_action=plain(stream.fork['nominal_a']))
            if (actual != requested or not np.array_equal(action,
                    np.asarray(self.expected['actions'][self.arm], np.float32))):
                raise RuntimeError('The native intervention did not execute the saved F candidate')
        self.observe_progress(stream)

    def observe_progress(self, stream):
        transition = stream.fork_transition
        if self.next_snapshot is None and transition is not None:
            snapshot = stream._snapshot()
            if float(snapshot.time_seconds) == transition.completed_at:
                self.next_snapshot = snapshot

    def build_summary(self, stream):
        self.observe_progress(stream)
        transition, start = stream.fork_transition, stream.fork_snapshot
        next_truth = endpoint(start, self.next_snapshot, status=(
            'complete' if self.next_snapshot is not None else 'missing'))
        next_truth.update(completed_at=transition.completed_at if transition is not None else None,
            next_vehicle_id=transition.next_vehicle_id if transition is not None else None,
            terminal=transition.terminal if transition is not None else None,
            zero_duration=transition.dt == 0 if transition is not None else None)
        fixed = endpoint(start, stream.cutoff_snapshot,
            status='complete' if stream.cutoff_reached else 'truncated')
        after = copy.deepcopy(self.counts())
        delta = {key: after[key]-self.before[key] for key in self.before}
        native_action = self.native_actions[0] if len(self.native_actions) == 1 else None
        physical = self.physical.summary(stream.cutoff_reached) if self.physical is not None else None
        if self.execution is not None:
            self.execution['native_applied_hold_seconds'] = (
                native_action['applied_hold_seconds'] if native_action is not None else None)
            self.execution['native_application'] = native_action
        checks = dict(live_prefix_inherited=self.in_child and self.selected is not None,
            actual_action_exact=self.execution is not None and
                self.execution['executed_hold_seconds'] == HOLD_SECONDS[self.arm],
            native_action_applied=native_action is not None and
                native_action['applied_hold_seconds'] == native_action['assigned_action'] ==
                HOLD_SECONDS[self.arm],
            physical_issue_observed=physical is not None and all(physical['coverage'][key] for key in
                ('actual_clock_observed', 'action_step_length_observed', 'issue_stop_state_observed')),
            physical_native_ticks_exact=physical is not None and physical['native_tick_count'] == stream.native_suffix_steps,
            suffix_zero_training=after == self.before and all(value == 0 for value in delta.values()),
            native_suffix_exact=stream.cutoff_reached and stream.native_suffix_steps == 300,
            fixed300_truth=fixed['status'] == 'complete' and fixed['dt'] == 300.,
            target_clock_unchanged=stream.contract.horizon_seconds == 24000.,
            no_terminal_fabrication=not stream.done and stream.final_snapshot is None)
        return plain(dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), arm=self.arm,
            selected=self.selected, external_seed=self.external_seed, execution=self.execution, physical=physical,
            truth=dict(next_event=next_truth, fixed300=fixed),
            global_transition=asdict(transition) if transition is not None else None,
            native=dict(other_batch_actions=stream.fork_batch_other_controls,
                suffix_steps=stream.native_suffix_steps),
            counters=dict(before=self.before, after=after, delta=delta),
            budget=dict(suffix_sim_seconds=stream.now-start.time_seconds,
                replay_target_labels=0, replay_policy_updates=0, new_truth_labels=1),
            execution_wall=dict(pid=os.getpid(), native_started_wall=self.started_wall,
                native_finished_wall=time.time())))

    def finish_child(self, stream):
        summary = self.build_summary(stream)
        with (self.out/self.arm/'summary.json').open('x') as output:
            json.dump(summary, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary['integrity']['all_pass'] else 1)
