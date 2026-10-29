"""Completed native departure-to-arrival history; never future service clocks."""
from collections import Counter, defaultdict
import numpy as np

LOOKAHEAD = 3
SCALE_SECONDS = 60.
_ABSENT = object()


def route_segments(bridge, issued):
    line = bridge.line_obj_dic[str(issued['line_id'])]
    stops = list(line.stop_id_l)
    index = stops.index(str(issued['station_id']))
    return [dict(from_stop=stops[i], to_stop=stops[i+1],
                 edge_path=list(line.edge_between_stop_d[stops[i+1]]))
            for i in range(index, min(index+LOOKAHEAD, len(stops)-1))]


def segment_key(segment):
    return segment['from_stop'], segment['to_stop'], tuple(segment['edge_path'])


def native_running_reference(bus, origin, destination):
    """The existing static native timetable increment, without stop service.

    get_arriver_timetable sums edge_length/(hourly edge_average_speed*0.8).
    This reference is descriptive only: the frozen102-interval physical model
    and its decision-ready-to-next-ready semantics remain untouched.
    """
    timetable = bus.arriver_timetable_d
    if origin not in timetable or destination not in timetable:
        return None
    value = float(timetable[destination])-float(timetable[origin])
    return value if np.isfinite(value) and value > 0 else None


class NativeRunHistory:
    def __init__(self, bridge):
        self.bridge, self.rows, self.counts = bridge, defaultdict(list), Counter()
        self.seen, self.lengths = set(), {}
        name = '_synchronize_bus_passenger_transition'
        self.original = getattr(bridge, name)
        self.instance_original = bridge.__dict__.get(name, _ABSENT)

        def observed(bus_id, previous_state, current_state, bus, native_time):
            result = self.original(bus_id, previous_state, current_state, bus, native_time)
            self.observe_bus(str(bus_id), bus, float(native_time))
            return result

        bridge._synchronize_bus_passenger_transition = observed
        now = float(bridge.passenger_time_ledger.snapshot().time_seconds)
        for bus_id, bus in bridge.bus_obj_dic.items():
            self.observe_bus(str(bus_id), bus, now)

    def observe_bus(self, bus_id, bus, observed_at):
        arrivals, departures = bus.arriver_stop_time_d, bus.depart_stop_time_d
        shape = len(arrivals), len(departures)
        if self.lengths.get(bus_id) == shape:
            return
        self.lengths[bus_id] = shape
        line_id = str(bus.belong_line_id_s)
        line = self.bridge.line_obj_dic[line_id]
        stops = list(line.stop_id_l)
        for origin, destination in zip(stops, stops[1:]):
            if origin not in departures or destination not in arrivals:
                continue
            start, end = float(departures[origin]), float(arrivals[destination])
            identity = bus_id, origin, destination, end
            if identity in self.seen:
                continue
            self.seen.add(identity)
            if not start <= end <= observed_at:
                self.counts['invalid_native_clock'] += 1
                continue
            reference = native_running_reference(bus, origin, destination)
            if reference is None:
                self.counts['reference_missing'] += 1
                continue
            segment = dict(from_stop=origin, to_stop=destination,
                edge_path=list(line.edge_between_stop_d[destination]))
            self.rows[segment_key(segment)].append(dict(bus_id=bus_id, line_id=line_id,
                departed_at=start, completed_at=end, observed_at=float(observed_at),
                duration=end-start, native_running_reference=reference,
                residual=end-start-reference))
            self.counts['completed_intervals'] += 1

    def snapshot(self, issued):
        now = float(issued['started_at'])
        segments = route_segments(self.bridge, issued)
        result = []
        for segment in segments:
            eligible = [row for row in self.rows.get(segment_key(segment), ())
                        if row['completed_at'] < now and row['observed_at'] < now]
            # Equal-clock histories are averaged; dictionary iteration cannot
            # choose one tied vehicle's residual as a hidden ordering feature.
            newest = max((row['completed_at'] for row in eligible), default=None)
            latest = sorted((row for row in eligible if row['completed_at'] == newest),
                            key=lambda row:(row['bus_id'],row['line_id']))
            result.append(dict(**segment, available=bool(latest), latest=None if not latest else dict(
                residual=float(np.mean([r['residual'] for r in latest])),
                duration=float(np.mean([r['duration'] for r in latest])),
                native_running_reference=float(np.mean([r['native_running_reference'] for r in latest])),
                completed_at=newest, observed_at=max(r['observed_at'] for r in latest),
                age=now-newest, tied_completed_intervals=len(latest),
                contributing_lines=sorted({r['line_id'] for r in latest}))))
        available = len(result) == LOOKAHEAD and all(r['available'] for r in result)
        return dict(status='supported' if available else 'history_missing',
            issued_at=now, segments=result,
            mean_residual_scaled=float(np.mean([r['latest']['residual'] for r in result]))/SCALE_SECONDS
                if available else None,
            reference='native_static_arriver_timetable_increment',
            semantics='Completed same directed stop-pair AND edge-path runs only; completion<issue, observation<issue. No service time or future timetable outcome is an input.')

    def close(self):
        if self.instance_original is _ABSENT:
            del self.bridge._synchronize_bus_passenger_transition
        else:
            self.bridge._synchronize_bus_passenger_transition = self.instance_original


def future_services(bridge, issued, cutoff):
    """Diagnostic outcomes, separate from snapshot inputs; retain censoring."""
    bus = bridge.bus_obj_dic[str(issued['trip_id'])]
    started = float(issued['started_at'])
    segments = route_segments(bridge, issued)
    stations = [s['to_stop'] for s in segments]
    value = dict(stations=stations, arrival=[], departure=[], status=dict(arrival=[], departure=[]))
    for kind, history in (('arrival', bus.arriver_stop_time_d), ('departure', bus.depart_stop_time_d)):
        for stop in stations:
            clock = history.get(stop)
            complete = clock is not None and started <= float(clock) <= float(cutoff)
            value[kind].append(float(clock)-started if complete else None)
            value['status'][kind].append('complete' if complete else 'censored')
    return value
