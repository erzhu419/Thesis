"""Passive native service evidence for the fixed tail-policy exchange."""
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path


class TailServiceTrace:
    """Keep event detail resident; query every station only at explicit milestones.

    Arrival/departure are separate physical observations. The bridge's legacy
    passenger synchronization itself does not advance SUMO, so observations
    around that method are not described as before/after physical boarding.
    """

    def __init__(self, bridge, path, *, traci=None, detail_from=16200.):
        self.bridge = bridge
        self.api = traci if traci is not None else importlib.import_module(
            bridge.__class__.__module__).traci
        self.detail_from = float(detail_from)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = self.path.open('x')
        self.seen = set()
        self.services = []
        self.milestones = {}
        self.missing = Counter()
        self.event_counts = Counter()
        self.closed = False
        self.original = {name: (getattr(bridge, name), bridge.__dict__.get(name))
            for name in ('_synchronize_bus_passenger_transition', '_collect_new_events',
                '_advance_one_step')}
        self.had_instance = {name: name in bridge.__dict__ for name in self.original}
        self._install()

    def _install(self):
        transition = self.original['_synchronize_bus_passenger_transition'][0]
        collect = self.original['_collect_new_events'][0]
        advance = self.original['_advance_one_step'][0]

        def observed_transition(bus_id, previous_state, current_state, bus_obj, native_time):
            result = transition(bus_id, previous_state, current_state, bus_obj, native_time)
            if previous_state != 'Stop' and current_state == 'Stop':
                self._history(bus_id, bus_obj, 'arrival', bus_obj.arriver_stop_time_d, native_time)
            if previous_state == 'Stop' and current_state != 'Stop':
                self._history(bus_id, bus_obj, 'departure', bus_obj.depart_stop_time_d, native_time)
            return result

        def observed_collect(native_time):
            result = collect(native_time)
            # Completion histories include services released without a control
            # issue, including isolated buses. No policy/event mutation occurs.
            for bus_id in self.bridge.active_bus_ids:
                bus = self.bridge.bus_obj_dic[bus_id]
                if bus.bus_state_s != 'Stop':
                    continue
                for station in bus.just_server_stop_data_d:
                    history = bus.service_completion_history.get(station, ())
                    self._history(bus_id, bus, 'service_ready', {station: history}, native_time)
            return result

        def observed_advance(*args, **kwargs):
            result = advance(*args, **kwargs)
            # The native method advances SUMO and then the passenger ledger.
            # EventStream can return only at a later control event, so capture
            # these physical ticks here rather than after stream.step returns.
            clock = float(self.bridge.passenger_time_ledger.snapshot().time_seconds)
            if clock in (18300., 20100., 24000.):
                label = str(int(clock))
                if label not in self.milestones:
                    self.snapshot(label, clock=clock)
            return result

        self.bridge._synchronize_bus_passenger_transition = observed_transition
        self.bridge._collect_new_events = observed_collect
        self.bridge._advance_one_step = observed_advance

    def _write(self, row):
        self.writer.write(json.dumps(row, allow_nan=False)+'\n')

    def _get(self, field, getter):
        try:
            return getter()
        except Exception:
            # A real terminal vehicle can be removed before its next sample.
            # Preserve missing observations instead of inventing empty sets.
            self.missing[field] += 1
            return None

    def _passenger(self, passenger_id, station):
        passenger = self.bridge.passenger_obj_dic.get(passenger_id)
        row = dict(passenger_id=str(passenger_id), station_id=str(station))
        if passenger is None:
            self.missing['passenger_metadata'] += 1
            return dict(row, passable_lines=None, metadata_observed=False)
        row.update(metadata_observed=True,
            passable_lines=sorted(str(line) for line in passenger.passable_line_l),
            original_stop=str(passenger.start_stop_id_s),
            transfer_stop=str(passenger.transfer_stop_id_s),
            destination_stop=str(passenger.end_stop_id_s),
            native_person_state=str(passenger.passenger_state_s),
            departure_time=self.bridge.departed_passenger_times.get(passenger_id),
            completed_legs=len(passenger.travel_data_l))
        return row

    def _service_state(self, bus_id, bus, station):
        queue = self._get('service_queue_ids', lambda: tuple(self.api.busstop.getPersonIDs(station)))
        onboard = self._get('service_onboard_ids', lambda: tuple(self.api.vehicle.getPersonIDList(bus_id)))
        capacity = self._get('service_capacity', lambda: float(self.api.vehicle.getPersonCapacity(bus_id)))
        line = str(bus.belong_line_id_s)
        unknown = [] if queue is None else [person for person in queue
            if person not in self.bridge.passenger_obj_dic]
        eligible = None if queue is None or unknown else sum(line in
            self.bridge.passenger_obj_dic[person].passable_line_l for person in queue)
        return dict(queued_passenger_ids=None if queue is None else sorted(set(queue)),
            waiting=None if queue is None else len(set(queue)), eligible_waiting=eligible,
            unknown_eligibility_ids=sorted(unknown),
            onboard_passenger_ids=None if onboard is None else sorted(set(onboard)),
            onboard=None if onboard is None else len(set(onboard)), capacity=capacity,
            free_capacity=None if onboard is None or capacity is None
                else max(0., capacity-len(set(onboard))),
            native_board_count=bus.board_num_d.get(station),
            native_alight_count=bus.alight_num_d.get(station),
            native_stranded_count=bus.strand_num_d.get(station),
            native_want_board_count=bus.want_board_num_d.get(station))

    def _history(self, bus_id, bus, kind, history, observed_at):
        for station, values in history.items():
            values = values if isinstance(values, (tuple, list)) else (values,)
            for occurrence, clock in enumerate(values):
                key = (str(bus_id), str(station), kind, occurrence)
                if key in self.seen:
                    continue
                self.seen.add(key)
                line = str(bus.belong_line_id_s)
                stops = self.bridge.line_obj_dic[line].stop_id_l
                event = dict(record='service', kind=kind, bus_id=str(bus_id),
                    line_id=line, station_id=str(station), clock_seconds=float(clock),
                    observed_at=float(observed_at), terminal=str(station) in
                        (str(stops[0]), str(stops[-1])),
                    observation_order=sum(self.event_counts.values()))
                if float(observed_at) >= self.detail_from:
                    event.update(self._service_state(bus_id, bus, station))
                self._write(event)
                compact = {key: value for key, value in event.items()
                    if key not in ('queued_passenger_ids', 'onboard_passenger_ids')}
                self.services.append(compact)
                self.event_counts[kind] += 1

    def snapshot(self, label, clock=None):
        """Read actual stop memberships once; never infer people from counts."""
        ledger = self.bridge.passenger_time_ledger.snapshot()
        now = float(ledger.time_seconds)
        if clock is not None and now != float(clock):
            raise ValueError('The requested passenger milestone is not the actual ledger clock')
        people, duplicates, missing_stations = {}, {}, []
        raw_count = 0
        for station in self.bridge.stop_obj_dic:
            ids = self._get('milestone_queue_ids', lambda s=station:
                tuple(self.api.busstop.getPersonIDs(s)))
            if ids is None:
                missing_stations.append(str(station))
                continue
            raw_count += len(ids)
            for person in ids:
                if person in people:
                    duplicates.setdefault(str(person), [people[person]['station_id']]).append(str(station))
                else:
                    people[person] = self._passenger(person, station)
        result = dict(record='milestone', label=str(label), time_seconds=now,
            ledger_waiting=int(ledger.waiting_passengers),
            ledger_in_vehicle=int(ledger.in_vehicle_passengers),
            accrued_wait_seconds=float(ledger.accrued_wait_seconds),
            accrued_in_vehicle_seconds=float(ledger.accrued_in_vehicle_seconds),
            waiting_unique=len(people), waiting_raw_memberships=raw_count,
            waiting_passengers=[people[person] for person in sorted(people)],
            duplicate_station_memberships=duplicates, missing_stations=missing_stations,
            waiting_inventory_complete=not missing_stations,
            waiting_matches_ledger=not missing_stations and len(people) == ledger.waiting_passengers)
        self.milestones[str(label)] = result
        self._write(result)
        return result

    def flush(self):
        self.writer.flush()

    def child_detach(self, path):
        """Reopen only child output; retain inherited passive service history."""
        self.writer.close()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = self.path.open('x')

    def finish(self):
        clock = float(self.bridge.passenger_time_ledger.snapshot().time_seconds)
        final = self.milestones.get('24000') if clock == 24000. else None
        if final is None:
            final = self.snapshot('final')
        groups = defaultdict(list)
        for passenger in final['waiting_passengers']:
            lines = passenger['passable_lines']
            groups[(passenger['station_id'], None if lines is None else tuple(lines))].append(passenger)
        summaries = []
        for (station, lines), passengers in sorted(groups.items(), key=lambda pair: str(pair[0])):
            service = [event for event in self.services if event['station_id'] == station
                and lines is not None and event['line_id'] in lines]
            last = {kind: next((event for event in reversed(service) if event['kind'] == kind), None)
                for kind in ('arrival', 'service_ready', 'departure')}
            summaries.append(dict(station_id=station,
                passable_lines=None if lines is None else list(lines), waiting=len(passengers),
                passenger_ids=[person['passenger_id'] for person in passengers],
                compatible_service_events=len(service), last_compatible_service=last))
        self.flush()
        summary_path = self.path.with_name('service_summary.json')
        result = dict(final=final, milestones=self.milestones,
            final_waiting_groups=summaries, service_event_counts=dict(self.event_counts),
            missing_observations=dict(self.missing), resident_service_path=str(self.path),
            resident_summary_path=str(summary_path),
            service_sample_scope='All prefix service clocks; physical inventories from16200; '
                'arrival and departure are distinct native observations; zero extra simulation steps.')
        with summary_path.open('x') as output:
            json.dump(result, output, allow_nan=False)
            output.write('\n')
        return result

    def close(self):
        if self.closed:
            return
        self.writer.close()
        for name, (_, value) in self.original.items():
            if self.had_instance[name]:
                setattr(self.bridge, name, value)
            else:
                delattr(self.bridge, name)
        self.closed = True
