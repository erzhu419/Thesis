"""Causal, all-vehicle execution context for the ordinary factual stream.

This collector reads only already updated native bus attributes. It never
queries TraCI, scans a completed bus history, selects a diagnostic vehicle, or
changes a policy/training input. Command amounts are audit data, not features.
"""
from __future__ import annotations

from experiments.mcwm_closed_loop_v1.stream import EventStream


LINE_FEATURES = (
    "recent_apply_available", "recent_vehicle_count_over_50",
    "phase_observed_fraction", "stopped_fraction", "departure_observed_fraction",
    "mean_apply_age_over_300", "mean_observed_departure_age_over_300",
)
_ABSENT = object()


class ExecutionMemory:
    """Observe all native applies and native phase updates after they return.

    The most recent applied visit of each trip remains eligible for 300 s.
    A departure is recorded only when the ordinary native update has observed
    Stop -> Edge/Signal. Predicted departure fields are deliberately unused.
    """

    def __init__(self, bridge, window_seconds=300.):
        self.bridge = bridge
        self.window_seconds = float(window_seconds)
        if self.window_seconds != 300.:
            raise ValueError("The v32 context uses the fixed 300-second window")
        self.line_ids = tuple(sorted(str(line) for line in bridge.line_obj_dic))
        self.feature_names = tuple(f"{line}:{name}" for line in self.line_ids for name in LINE_FEATURES)
        self.memory_dim = len(self.feature_names)
        self.records = {}
        self.apply_count = self.phase_observation_count = self.snapshot_count = 0
        self.causal_errors = 0
        self.line_apply_counts = dict.fromkeys(self.line_ids, 0)
        self.latest_observed_at = None
        self._instance_originals = {name: bridge.__dict__.get(name, _ABSENT) for name in
            ('apply_action', '_synchronize_bus_passenger_transition')}
        original_apply = bridge.apply_action
        original_sync = bridge._synchronize_bus_passenger_transition

        def apply_action(event, command):
            result = original_apply(event, command)
            # This field is assigned by the real native implementation only
            # after its actual apply path, including its ordinary warning path.
            if 'applied_additional_hold_seconds' in event.metadata:
                self.observe_apply(event, self._clock())
            return result

        def synchronize(bus_id, previous_state, current_state, bus_obj, simulation_current_time):
            result = original_sync(bus_id, previous_state, current_state, bus_obj, simulation_current_time)
            self.observe_phase(bus_id, previous_state, current_state, bus_obj,
                float(simulation_current_time), self._clock())
            return result

        bridge.apply_action = apply_action
        bridge._synchronize_bus_passenger_transition = synchronize

    def _clock(self):
        return float(self.bridge.passenger_time_ledger.snapshot().time_seconds)

    def _observe_clock(self, now):
        now = float(now)
        if self.latest_observed_at is not None and now < self.latest_observed_at:
            self.causal_errors += 1
            raise ValueError("Native memory observations moved backwards")
        self.latest_observed_at = now
        return now

    @staticmethod
    def _cached_physical(bus):
        if bus is None:
            return dict(phase=None, cached_speed_mps=None, cached_distance_m=None, cached_lane_id=None)
        phase = getattr(bus, 'bus_state_s', None)
        return dict(phase=phase if phase in ('Stop', 'Edge', 'Signal') else None,
            cached_speed_mps=getattr(bus, 'bus_speed_n', None),
            cached_distance_m=getattr(bus, 'distance_n', None),
            cached_lane_id=getattr(bus, 'bus_cur_lane_s', None))

    def observe_apply(self, event, now):
        now = self._observe_clock(now)
        trip, line = str(event.bus_id), str(event.line_id)
        if line not in self.line_apply_counts:
            raise ValueError(f"Native apply has an unregistered line: {line}")
        bus = self.bridge.bus_obj_dic.get(trip)
        physical = self._cached_physical(bus)
        self.records[trip] = dict(trip_id=trip, line_id=line, stop_id=str(event.stop_id),
            applied_at=now, observed_at=now, departure_at=None, departure_observed_at=None,
            command_hold_seconds=float(event.metadata['applied_additional_hold_seconds']), **physical)
        self.apply_count += 1
        self.line_apply_counts[line] += 1

    def observe_phase(self, trip_id, previous_state, current_state, bus_obj, physical_time, observed_at):
        observed_at = self._observe_clock(observed_at)
        if physical_time > observed_at:
            self.causal_errors += 1
            raise ValueError("A physical observation cannot precede its own native event time")
        row = self.records.get(str(trip_id))
        if row is None:
            return
        row.update(self._cached_physical(bus_obj), observed_at=observed_at)
        if previous_state == 'Stop' and current_state != 'Stop' and row['departure_at'] is None:
            row['departure_at'] = float(physical_time)
            row['departure_observed_at'] = observed_at
        self.phase_observation_count += 1

    def snapshot(self, now):
        """Return the currently stored context; past snapshots are not rebuilt."""
        now = float(now)
        if self.latest_observed_at is not None and self.latest_observed_at > now:
            self.causal_errors += 1
            raise ValueError("A memory snapshot would include a future observation")
        recent = [row for row in self.records.values() if 0. <= now-row['applied_at'] <= self.window_seconds]
        vector, lines = [], []
        for line in self.line_ids:
            rows = [row for row in recent if row['line_id'] == line]
            count = len(rows)
            observed = [row for row in rows if row['phase'] is not None]
            departed = [row for row in rows if row['departure_observed_at'] is not None]
            values = [float(bool(rows)), count/50., len(observed)/count if count else 0.,
                sum(row['phase'] == 'Stop' for row in rows)/count if count else 0.,
                len(departed)/count if count else 0.,
                sum(now-row['applied_at'] for row in rows)/(count*300.) if count else 0.,
                sum(now-row['departure_at'] for row in departed)/(len(departed)*300.) if departed else 0.]
            vector.extend(values)
            lines.append(dict(line_id=line, recent_vehicle_count=count,
                phase_observed_count=len(observed), stopped_count=sum(row['phase'] == 'Stop' for row in rows),
                departure_observed_count=len(departed), available=bool(rows), vector=values))
        self.snapshot_count += 1
        return dict(observed_at=now, vector=vector, feature_names=list(self.feature_names),
            line_ids=list(self.line_ids), lines=lines,
            metadata=dict(latest_native_observation_at=self.latest_observed_at,
                command_is_feature=False, selected_diagnostic_vehicle=False,
                recent_trip_count=len(recent),
                latest_recent_apply_at=max((row['applied_at'] for row in recent), default=None),
                latest_recent_physical_observation_at=max((row['observed_at'] for row in recent), default=None)))

    def summary(self):
        checks = dict(no_future_or_backwards_observations=self.causal_errors == 0,
            fixed_full_line_context=len(self.line_ids)*len(LINE_FEATURES) == self.memory_dim,
            apply_line_accounting=sum(self.line_apply_counts.values()) == self.apply_count)
        return dict(line_ids=list(self.line_ids), feature_names=list(self.feature_names),
            feature_dimension=len(self.feature_names), window_seconds=self.window_seconds,
            apply_count=self.apply_count, phase_observation_count=self.phase_observation_count,
            snapshot_count=self.snapshot_count, line_apply_counts=self.line_apply_counts.copy(),
            latest_native_observation_at=self.latest_observed_at,
            command_is_feature=False, selected_diagnostic_vehicle=False,
            extra_traci_queries=0, history_backfill=False,
            integrity=dict(all_pass=all(checks.values()), checks=checks))

    def close(self):
        """Restore the exact prior descriptor/instance-wrapper arrangement."""
        for name, original in self._instance_originals.items():
            if original is _ABSENT:
                delattr(self.bridge, name)
            else:
                setattr(self.bridge, name, original)


class MemoryEventStream(EventStream):
    """Original stream with causal memory metadata on each actual transition.

    Install ``stream.memory = ExecutionMemory(env._bridge)`` after ``reset``.
    The issue snapshot precedes the user's callback and the native apply. The
    next-state snapshot precedes the original close at the same ledger clock.
    """

    def __init__(self, *args, **kwargs):
        self.memory = None
        self._memory_issues = {}
        self._user_on_issue = kwargs.pop('on_issue', None)
        super().__init__(*args, on_issue=self._observe_issue, **kwargs)

    def _observe_issue(self, event):
        if self.memory is None:
            raise RuntimeError("ExecutionMemory must be installed before the first issue")
        snapshot = self.memory.snapshot(event['started_at'])
        event['memory_s'] = snapshot
        self._memory_issues[event['sequence']] = snapshot
        if self._user_on_issue is not None:
            self._user_on_issue(event)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        pending = self._pending
        sequence = pending['sequence'] if pending is not None else None
        memory_ns = self.memory.snapshot(snapshot.time_seconds) if pending is not None else None
        queue_size = len(self._queue)
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if sequence is not None:
            memory_s = self._memory_issues.pop(sequence)
            if len(self._queue) != queue_size:
                transition = self._queue[-1]
                if transition.sequence != sequence:
                    raise RuntimeError("Memory closure did not match the original transition")
                transition.memory_s, transition.memory_ns = memory_s, memory_ns
