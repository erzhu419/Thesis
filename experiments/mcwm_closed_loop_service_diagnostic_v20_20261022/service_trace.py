"""Passive native service observations; complete trajectories stay resident."""
from collections import Counter
import importlib
import json
from pathlib import Path


_ABSENT = object()


def plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class ServiceTrace:
    """Observe existing bridge methods without changing their arguments/results.

    Clock rows are post-native-step passenger-ledger snapshots. Service events
    retain native bus-history or bridge-calculated service-completion clocks
    and their observation clock. Analysis must difference cumulative ledgers, not integrate
    their displayed inventory by an assumed left/right rectangle convention.
    """

    def __init__(self, bridge, path, *, traci=None, duration_seconds=300.):
        self.bridge, self.path = bridge, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = self.path.open('x')
        self.api = traci if traci is not None else importlib.import_module(bridge.__class__.__module__).traci
        self.duration = float(duration_seconds)
        self.start = self.cutoff = None
        self.bus = self.line = self.station = None
        self.rows, self.events, self.missing = {}, [], Counter()
        self.onboard, self.person_clocks, self.seen = {}, {}, set()
        self.fork_baseline_visible = False
        self.event_counts, self.line_counts = Counter(), Counter()
        self.last_count_clock = None
        self.original = {}
        self.closed = False
        self._install()

    def _install(self):
        for name in ('_advance_one_step', '_synchronize_bus_passenger_transition', '_collect_new_events',
                     '_synchronize_passenger_time_counts'):
            self.original[name] = (getattr(self.bridge, name), self.bridge.__dict__.get(name, _ABSENT))
        advance = self.original['_advance_one_step'][0]
        transition = self.original['_synchronize_bus_passenger_transition'][0]
        collect = self.original['_collect_new_events'][0]
        counts = self.original['_synchronize_passenger_time_counts'][0]

        def observed_advance(*args, **kwargs):
            result = advance(*args, **kwargs)
            if self.start is not None:
                self._sample_clock()
            return result

        def observed_transition(bus_id, previous_state, current_state, bus_obj, native_time):
            result = transition(bus_id, previous_state, current_state, bus_obj, native_time)
            if self.start is not None:
                self._history_events(bus_id, bus_obj, native_time, kinds=('arrival', 'departure'))
            return result

        def observed_counts(native_time):
            result = counts(native_time)
            self.last_count_clock = float(native_time)
            if self.start is not None:
                self._observe_people(float(native_time))
            return result

        def observed_collect(native_time):
            result = collect(native_time)
            if self.start is not None:
                # Native collect iterates this same insertion-ordered mapping;
                # ties preserve its observation order, without claiming a
                # physical ordering of simultaneous services.
                for bus_id, bus_obj in self.bridge.bus_obj_dic.items():
                    self._history_events(bus_id, bus_obj, native_time, kinds=('service_ready',))
            return result

        self.bridge._advance_one_step = observed_advance
        self.bridge._synchronize_bus_passenger_transition = observed_transition
        self.bridge._collect_new_events = observed_collect
        self.bridge._synchronize_passenger_time_counts = observed_counts

    def _get(self, field, getter):
        try:
            return getter()
        except Exception:
            # Vehicles can disappear at their terminal in the supported native
            # window. Missing observations remain absent rather than zero.
            self.missing[field] += 1
            return None

    def _queue(self, station, line):
        if station is None:
            self.missing['station_identity'] += 1
            return dict(waiting=None, eligible_waiting=None)
        ids = self._get('waiting_person_ids', lambda: list(self.api.busstop.getPersonIDs(station)))
        if ids is None:
            return dict(waiting=None, eligible_waiting=None)
        passengers = self.bridge.passenger_obj_dic
        unknown = [p for p in ids if p not in passengers or not hasattr(passengers[p], 'passable_line_l')]
        if unknown:
            self.missing['waiting_passenger_eligibility'] += len(unknown)
        eligible = None if unknown else sum(line in passengers[p].passable_line_l for p in ids)
        return dict(waiting=len(ids), eligible_waiting=eligible)

    def _bus_state(self, bus_id, bus_obj, station):
        ids = self._get('onboard_person_ids', lambda: set(self.api.vehicle.getPersonIDList(bus_id)))
        capacity = self._get('vehicle_capacity', lambda: float(self.api.vehicle.getPersonCapacity(bus_id)))
        onboard = len(ids) if ids is not None else None
        free = max(0., capacity-onboard) if capacity is not None and onboard is not None else None
        queue = self._queue(station, str(bus_obj.belong_line_id_s))
        return ids, dict(onboard=onboard, capacity=capacity, free_capacity=free,
            effective_capacity=free, capacity_source='traci.vehicle.getPersonCapacity',
            configured_bus_capacity=getattr(self.bridge, 'BusCap', None), **queue)

    def _emit(self, kind, clock, bus_id, line, station, **fields):
        if not self.start <= float(clock) <= self.cutoff:
            return
        event = dict(kind=kind, clock_seconds=float(clock), relative_seconds=float(clock)-self.start,
            observed_at=float(self.bridge.passenger_time_ledger.snapshot().time_seconds),
            observation_order=len(self.events), bus_id=str(bus_id), line_id=str(line),
            station_id=str(station) if station is not None else None, **plain(fields))
        self.events.append(event)
        self.event_counts[kind] += 1
        self.line_counts[str(line)] += 1
        self.writer.write(json.dumps(dict(record='service', **event), allow_nan=False)+'\n')

    def _history_events(self, bus_id, bus_obj, observed_time, kinds):
        histories = dict(arrival=bus_obj.arriver_stop_time_d,
            departure=bus_obj.depart_stop_time_d, service_ready=bus_obj.service_completion_history)
        for kind in kinds:
            for station, values in histories[kind].items():
                times = values if isinstance(values, (list, tuple)) else (values,)
                for occurrence, clock in enumerate(times):
                    key = (str(bus_id), kind, str(station), occurrence)
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    if not self.start <= float(clock) <= self.cutoff:
                        continue
                    _, state = self._bus_state(bus_id, bus_obj, station)
                    self._emit(kind, clock, bus_id, bus_obj.belong_line_id_s, station,
                        native_detection_time=float(observed_time),
                        time_source='bridge.service_completion_history' if kind == 'service_ready'
                            else 'native_bus_'+kind+'_history', **state)

    def _observe_people(self, clock):
        if not self.start <= clock <= self.cutoff:
            return
        for bus_id in set(self.onboard)-set(self.bridge.active_bus_ids):
            prior = self.onboard.pop(bus_id)
            self.person_clocks.pop(bus_id, None)
            if prior:
                self.missing['vehicle_removed_before_person_sample'] += len(prior)
                obj = self.bridge.bus_obj_dic[bus_id]
                self._emit('passenger_sample_missing', clock, bus_id, obj.belong_line_id_s, None,
                    previous_onboard_count=len(prior), passenger_count=None,
                    reason='Vehicle removed before the next person-set observation; alighting time not fabricated.')
        for bus_id in self.bridge.active_bus_ids:
            obj = self.bridge.bus_obj_dic[bus_id]
            current = self._get('onboard_person_ids', lambda b=bus_id: set(self.api.vehicle.getPersonIDList(b)))
            previous = self.onboard.get(bus_id)
            if bus_id not in self.onboard:
                cached = self.bridge._bus_onboard_passenger_ids.get(bus_id)
                previous = None if cached is None else set(cached)
            interval_start = self.person_clocks.get(bus_id, clock)
            if current is None or previous is None:
                self.missing['passenger_sample_baseline'] += 1
            elif current != previous:
                station, station_source, at_end = self._exchange_station(bus_id, obj)
                _, state = self._bus_state(bus_id, obj, station)
                for kind, changed in (('board', current-previous), ('alight', previous-current)):
                    if changed:
                        self._emit(kind, clock, bus_id, obj.belong_line_id_s, station,
                            passenger_count=len(changed), passenger_ids=sorted(changed),
                            person_sample_interval=[interval_start, clock],
                            passenger_change_observation='discrete_native_person_samples',
                            station_source=station_source,
                            station_at_interval_end_confirmed=at_end,
                            station_interval_uncertainty=not at_end, **state)
            self.onboard[bus_id], self.person_clocks[bus_id] = current, clock

    def _exchange_station(self, bus_id, obj):
        # counts are sampled before bus_running. At the end of a dwell its
        # current/just-served stop is still here even if SUMO's next-stop list
        # has already advanced to the next station.
        service = getattr(obj, 'just_server_stop_data_d', {})
        if getattr(obj, 'bus_state_s', None) == 'Stop' and service:
            return str(next(iter(service))), 'native_current_or_just_served_stop', False
        at_stop = self._get('actual_busstop_presence', lambda: bool(self.api.vehicle.isAtBusStop(bus_id)))
        if at_stop is True:
            stops = self._get('passenger_change_current_stop', lambda: self.api.vehicle.getStops(bus_id, 1))
            if stops and stops[0].stoppingPlaceID:
                return str(stops[0].stoppingPlaceID), 'traci_current_stop_confirmed_at_busstop', True
        self.missing['passenger_change_station_identity'] += 1
        return None, 'not_observed', False

    def activate(self, *, t0, bus, line, station):
        if self.start is not None:
            raise RuntimeError('A native service suffix can be activated only once')
        self.start, self.cutoff = float(t0), float(t0)+self.duration
        self.bus, self.line, self.station = str(bus), str(line), str(station)
        for bus_id, obj in self.bridge.bus_obj_dic.items():
            for kind, history in (('arrival', obj.arriver_stop_time_d),
                ('departure', obj.depart_stop_time_d), ('service_ready', obj.service_completion_history)):
                for stop, values in history.items():
                    values = values if isinstance(values, (list, tuple)) else (values,)
                    self.seen.update((str(bus_id), kind, str(stop), i) for i in range(len(values)))
        for bus_id in self.bridge.active_bus_ids:
            self.onboard[bus_id] = self._get('initial_onboard_person_ids',
                lambda b=bus_id: set(self.api.vehicle.getPersonIDList(b)))
            self.person_clocks[bus_id] = self.start
        self.fork_baseline_visible = self.onboard.get(self.bus) is not None
        self._sample_clock()

    def _sample_clock(self):
        snapshot = self.bridge.passenger_time_ledger.snapshot()
        clock = float(snapshot.time_seconds)
        if not self.start <= clock <= self.cutoff:
            return
        if clock in self.rows:
            return
        row = dict(time_seconds=clock, relative_seconds=clock-self.start,
            waiting_passengers=snapshot.waiting_passengers,
            in_vehicle_passengers=snapshot.in_vehicle_passengers,
            accrued_wait_seconds=float(snapshot.accrued_wait_seconds),
            accrued_in_vehicle_seconds=float(snapshot.accrued_in_vehicle_seconds),
            sample_stage='fork_or_post_native_step', ledger_inventory_observed_at=self.last_count_clock,
            fork_queue=self._queue(self.station, self.line), fork_queue_observed_at=clock)
        self.rows[clock] = row
        self.writer.write(json.dumps(dict(record='clock', **row), allow_nan=False)+'\n')

    def at_issue(self, event):
        if self.start is None:
            return
        row = {key: plain(event[key]) for key in ('sequence', 'started_at', 'line_id', 'vehicle_id',
            'trip_id', 'station_id', 's', 'raw_s', 'event', 'a', 'proposed_a') if key in event}
        self.writer.write(json.dumps(dict(record='issue', **row), allow_nan=False)+'\n')
        self._emit('issue', event['started_at'], event['trip_id'], event['line_id'], event['station_id'],
            sequence=event['sequence'], event_detection_time=float(event['raw_s'][0]), action=plain(event['a']))

    def finish(self, cutoff_snapshot):
        if self.start is None:
            raise RuntimeError('No service suffix was activated')
        # The final delivered tick is not followed by another native bus update.
        # Read its actual person sets so an exchange in (299,300] is not lost.
        self._observe_people(float(cutoff_snapshot.time_seconds))
        self._sample_clock()
        rows = [self.rows[k] for k in sorted(self.rows)]
        expected = [self.start+i for i in range(int(self.duration)+1)]
        valid_capacity = [e for e in self.events if e.get('capacity') is not None and e.get('free_capacity') is not None]
        valid_queue = [r for r in rows if r['fork_queue']['waiting'] is not None and r['fork_queue']['eligible_waiting'] is not None]
        checks = dict(fixed_clock_complete=[r['time_seconds'] for r in rows] == expected,
            exact_native_cutoff=float(cutoff_snapshot.time_seconds) == self.cutoff,
            final_ledger_exact=bool(rows) and all(rows[-1][key] == getattr(cutoff_snapshot, key)
                for key in ('time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
                            'accrued_wait_seconds', 'accrued_in_vehicle_seconds')),
            fork_vehicle_baseline_visible=self.fork_baseline_visible,
            real_capacity_observed=bool(valid_capacity), fork_queue_observed=bool(valid_queue),
            native_arrival_observed=self.event_counts['arrival'] > 0,
            native_departure_observed=self.event_counts['departure'] > 0,
            native_service_ready_observed=self.event_counts['service_ready'] > 0)
        clock_path, events_path = self.path.parent/'service_clock.json', self.path.parent/'service_events.json'
        for path, value in ((clock_path, dict(schema='v20-native-ledger-1s', start_time=self.start,
                cutoff_time=self.cutoff, rows=rows)),
            (events_path, dict(schema='v20-native-service-events', events=self.events))):
            with path.open('x') as handle:
                json.dump(value, handle, allow_nan=False, separators=(',', ':'))
                handle.write('\n')
        first, last = rows[0], rows[-1]
        wait = last['accrued_wait_seconds']-first['accrued_wait_seconds']
        ride = last['accrued_in_vehicle_seconds']-first['accrued_in_vehicle_seconds']
        result = dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), start_time=self.start,
            cutoff_time=self.cutoff, clock_rows=len(rows), event_counts=dict(self.event_counts),
            line_event_counts=dict(self.line_counts), missing_counts=dict(self.missing),
            coverage=dict(valid_capacity_events=len(valid_capacity), fork_queue_samples=len(valid_queue),
                fork_vehicle_baseline_visible=self.fork_baseline_visible),
            fixed300=dict(dt=self.duration, wait=wait, ride=ride, cost=2*wait+ride),
            paths=dict(clock='service_clock.json', events='service_events.json', trace=self.path.name),
            clocks='Clock rows are ledger delivery times; service_ready retains the bridge-calculated history time, native_detection_time and observed_at. Person changes retain observation intervals. Use cumulative ledger differences.',
            service_scope='All native bus service boundaries and every-tick actual person-set changes including the cutoff; each-second queue at the fork shared station. Passenger changes are interval-observed, simultaneous events have no asserted physical ordering. No future observation enters training.')
        self.writer.flush()
        return result

    def close(self):
        if self.closed:
            return
        for name, (_, instance_value) in self.original.items():
            if instance_value is _ABSENT:
                delattr(self.bridge, name)
            else:
                setattr(self.bridge, name, instance_value)
        self.writer.close()
        self.closed = True
