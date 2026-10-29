"""Real native event chains with a separate exact300 passenger ledger."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.collect import (
    ContextForkStream, transition_dict,
)
from experiments.mcwm_closed_loop_v1.stream import pack_event_batch_v7


def ledger(snapshot):
    return {key: float(getattr(snapshot, key)) for key in (
        'time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
        'accrued_wait_seconds', 'accrued_in_vehicle_seconds')}


def cost_between(start, stop):
    wait = stop['accrued_wait_seconds']-start['accrued_wait_seconds']
    ride = stop['accrued_in_vehicle_seconds']-start['accrued_in_vehicle_seconds']
    return dict(dt=stop['time_seconds']-start['time_seconds'], wait=wait,
                ride=ride, cost=2.*wait+ride)


class ChainRecorder:
    def __init__(self, out, selected):
        self.out, self.selected = Path(out), selected
        self.events = (self.out/'events.jsonl').open('x', buffering=1)
        self.ticks = (self.out/'ticks.jsonl').open('x', buffering=1)
        self.rows = self.zero_duration = self.tick_count = 0
        self.last_clock = None
        self.previous = None
        self.reward_exact = self.continuous = self.causal = True

    @staticmethod
    def _write(handle, value):
        handle.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')

    def observe_tick(self, stream):
        snapshot = stream._snapshot()
        current = ledger(snapshot)
        if current['time_seconds'] == self.last_clock:
            return
        relative = cost_between(self.selected['issue_ledger'], current)
        self._write(self.ticks, dict(clock_seconds=current['time_seconds'],
            relative_seconds=relative['dt'], ledger=current, **{key: relative[key] for key in ('wait', 'ride', 'cost')}))
        self.last_clock = current['time_seconds']
        self.tick_count += 1

    def close(self, row, start, stop, *, kind):
        value = transition_dict(row)
        start, stop = ledger(start), ledger(stop)
        segment = cost_between(start, stop)
        self.reward_exact &= row.r == -segment['cost'] and row.dt == segment['dt']
        self.causal &= (row.memory_s['observed_at'] == row.started_at
                        and row.memory_ns['observed_at'] == row.completed_at)
        if self.previous is not None:
            self.continuous &= (self.previous['completed_at'] == value['started_at']
                and self.previous['ns'] == value['s']
                and self.previous['memory_ns'] == value['memory_s'])
        else:
            self.continuous &= (row.sequence == self.selected['sequence']
                                and row.started_at == self.selected['started_at'])
        self._write(self.events, dict(kind=kind, transition=value,
            ledger_s=start, ledger_ns=stop))
        self.previous = value
        self.rows += 1
        self.zero_duration += int(row.dt == 0.)

    def summary(self):
        checks = dict(raw_reward_ledger_exact=self.reward_exact,
            real_state_memory_chain_continuous=self.continuous,
            memory_causal=self.causal)
        return dict(events_path=str(self.out/'events.jsonl'), ticks_path=str(self.out/'ticks.jsonl'),
            transition_count=self.rows, zero_duration_count=self.zero_duration,
            ledger_tick_count=self.tick_count,
            integrity=dict(all_pass=all(checks.values()), checks=checks))


class CreditEventStream(ContextForkStream):
    """Stop at the first real control >=300, observing but never issuing it."""
    def __init__(self, *args, max_overrun_seconds=120., **kwargs):
        self.recorder = None
        self.boundary300 = self.real_endpoint = self.stop_snapshot = None
        self.endpoint_pending_partial = None
        self.max_overrun_seconds = float(max_overrun_seconds)
        self._endpoint_closure = False
        super().__init__(*args, **kwargs)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        pending = self._pending
        count = len(self._queue)
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if self.recorder is not None and pending is not None and len(self._queue) > count:
            self.recorder.close(self._queue[-1], pending['snapshot'], snapshot,
                kind='endpoint_actual_without_issue' if self._endpoint_closure else 'global_transition')

    def _observe300(self):
        snapshot = self._snapshot()
        if self.boundary300 is None and snapshot.time_seconds == self.cutoff_time:
            self.cutoff_snapshot = snapshot
            controls = [event for event in self._batch() if not event['terminal']]
            self.boundary300 = dict(clock=float(snapshot.time_seconds), ledger=ledger(snapshot),
                memory=self.memory.snapshot(snapshot.time_seconds),
                actual_control_at_clock=bool(controls), state=None)

    def _advance_bounded_native(self, commands):
        env, bridge = self.env, self.env._bridge
        for event in env._current_events:
            bridge.apply_action(event, commands.get(event.bus_id, [0., 1.]))
        env.state, env.reward, env._current_events = {}, {}, []
        while not bridge.done:
            self._observe300()
            if bridge.decision_queue:
                events, done, _departed = bridge.fetch_events()
                env.done, env._current_events = bool(done), events
                for event in events:
                    env.state[event.bus_id] = [[env._event_to_obs(event).tolist()]]
                    env.reward[event.bus_id] = env._compute_reward(event)
                # The caller's close hook observes this true control state.
                # A terminal-only batch follows the unchanged native path.
                return
            if self._snapshot().time_seconds >= self.cutoff_time+self.max_overrun_seconds:
                return
            bridge._advance_one_step()
        env.done = True

    def _capture_cutoff(self):
        self._observe300()
        snapshot = self._snapshot()
        if snapshot.time_seconds < self.cutoff_time:
            return
        controls = [event for event in self._batch() if not event['terminal']]
        if controls:
            local = np.stack([event['event'] for event in controls])
            raw = pack_event_batch_v7(local, 0, np.full(len(controls), -1., np.float32), self.contract)
            self._endpoint_closure = True
            self._close(raw, snapshot, controls[0]['vehicle_id'])
            self._endpoint_closure = False
            self.real_endpoint = dict(status='observed', clock=float(snapshot.time_seconds),
                s=self._canonical(raw).tolist(), raw_s=raw.tolist(),
                memory=self.memory.snapshot(snapshot.time_seconds), ledger=ledger(snapshot),
                vehicle_id=controls[0]['vehicle_id'], trip_id=controls[0]['trip_id'],
                station_id=controls[0]['station_id'], issued=False, executed=False)
            if snapshot.time_seconds == self.cutoff_time:
                self.boundary300['actual_control_at_clock'] = True
                self.boundary300['state'] = self.real_endpoint['s']
        elif snapshot.time_seconds < self.cutoff_time+self.max_overrun_seconds:
            return
        else:
            self.real_endpoint = dict(status='missing', clock=None, s=None, raw_s=None,
                memory=None, ledger=None, issued=False, executed=False)
            pending = self._pending
            if pending is not None:
                self.endpoint_pending_partial = dict(sequence=pending['sequence'],
                    started_at=pending['started_at'], completed_at=float(snapshot.time_seconds),
                    partial_cost=cost_between(ledger(pending['snapshot']), ledger(snapshot)),
                    ns=None, r=None, bootstrap=None)
        self.stop_snapshot, self.cutoff_reached = snapshot, True

