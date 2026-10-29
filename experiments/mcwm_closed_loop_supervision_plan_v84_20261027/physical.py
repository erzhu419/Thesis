"""Read-only physical witnesses for one remaining-day action intervention."""
from __future__ import annotations

from experiments.mcwm_closed_loop_physical_execution_v30_20261023.suffix import (
    NativePhysicalObserver,
)


class RootPhysicalProbe:
    def __init__(self, bridge, selected, *, traci=None, horizon_seconds=24000.):
        self.bridge, self.selected = bridge, selected
        self.observer = NativePhysicalObserver(bridge, traci=traci)
        self.observer.activate(selected)
        # v30 used a 300-second endpoint; this probe observes the real episode.
        self.observer.cutoff = float(horizon_seconds)
        self.original_apply = bridge.apply_action
        self.original_advance = bridge._advance_one_step
        self.native_action = None
        self.finished = False
        self.closed = False
        bridge.apply_action = self.apply
        bridge._advance_one_step = self.advance

    def apply(self, event, command):
        selected = (self.native_action is None
                    and str(event.bus_id) == str(self.selected['trip_id'])
                    and str(event.stop_id) == str(self.selected['station_id']))
        if selected:
            self.observer.apply_before = self.observer.sample()
            self.observer.observe(self.observer.apply_before)
        result = self.original_apply(event, command)
        if selected:
            self.observer.apply_after = self.observer.sample()
            self.observer.observe(self.observer.apply_after)
            self.native_action = dict(
                bus_id=str(event.bus_id), station_id=str(event.stop_id),
                event_time=float(event.sim_time),
                issue_ledger_time=float(self.selected['started_at']),
                real_clock_before=self.observer.apply_before['native_clock_seconds'],
                real_clock_after=self.observer.apply_after['native_clock_seconds'],
                assigned_action=float(event.assigned_action),
                applied_hold_seconds=float(event.metadata['applied_additional_hold_seconds']))
        return result

    def advance(self, *args, **kwargs):
        result = self.original_advance(*args, **kwargs)
        if self.native_action is not None and not self.finished:
            self.observer.observe(self.observer.sample(), tick=True)
            departure = self.observer.departure or self.observer.history_departure
            arrival = self.observer.next_arrival or self.observer.history_arrival
            self.finished = departure is not None and arrival is not None
        return result

    def summary(self, endpoint_reached=False):
        value = self.observer.summary(endpoint_reached)
        return {key: value[key] for key in (
            'apply_before', 'apply_after', 'departure', 'next_same_vehicle_arrival',
            'native_history_departure', 'native_history_next_arrival',
            'coverage', 'missing', 'native_tick_count')} | dict(
                native_action=self.native_action,
                sampling_stopped_after_next_arrival=self.finished)

    def close(self):
        if not self.closed:
            self.bridge.apply_action = self.original_apply
            self.bridge._advance_one_step = self.original_advance
            self.closed = True


def install(bridge, selected, *, traci=None, horizon_seconds=24000.):
    return RootPhysicalProbe(bridge, selected, traci=traci,
                             horizon_seconds=horizon_seconds)
