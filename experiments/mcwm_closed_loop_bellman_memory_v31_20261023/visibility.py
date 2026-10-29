"""Causal suffix observations; none enters replay, policy state or training."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream
from experiments.mcwm_closed_loop_v1.stream import Transition, pack_event_batch_v7, transition_reward_v7


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


class VisibilityRecorder:
    """Save observed memory at issue time, never reconstruct it from the future."""

    def __init__(self, out, selected, physical, evaluator=None):
        self.out, self.selected, self.physical = Path(out), selected, physical
        self.evaluator = evaluator
        self.events = (self.out/'visibility.jsonl').open('x', buffering=1)
        self.ticks = (self.out/'ledger_ticks.jsonl').open('x', buffering=1)
        self.applied = None
        self.issues = {}
        self.occurrences = {}
        self.closed_sequences = set()
        self.transition_count = self.issue_count = self.tick_count = 0
        self.zero_duration_count = self.partial_count = self.evaluation_count = 0
        self.rng_restored = self.learner_rng_restored = self.target_path_exact = True
        self.memory_causal = True
        self.first_transition = self.last_tick_clock = None

    @staticmethod
    def _write(handle, row):
        handle.write(json.dumps(plain(row), separators=(',', ':'), allow_nan=False)+'\n')

    def observe_apply(self, native_application):
        """Called only after the real native apply_action returned."""
        self.applied = copy.deepcopy(native_application)

    def memory(self, clock, actor_vehicle_id=''):
        row = self.physical.previous
        applied = self.applied
        departed = self.physical.departure or self.physical.history_departure
        arrived = self.physical.next_arrival or self.physical.history_arrival
        # A history event can be discovered after its historical event clock.
        # Store flags only when its observation has actually occurred.
        departed_seen = departed is not None and departed['observed_at'] <= clock
        arrived_seen = arrived is not None and arrived['observed_at'] <= clock
        physical_clock = row.get('native_clock_seconds') if row is not None else None
        if physical_clock is not None and physical_clock > clock:
            self.memory_causal = False
        application_clock = applied.get('real_clock_after') if applied is not None else None
        if application_clock is not None and application_clock > clock:
            self.memory_causal = False
        return plain(dict(observed_at=float(clock), selected_trip_id=self.selected['trip_id'],
            selected_vehicle_id=self.selected['vehicle_id'], selected_station_id=self.selected['station_id'],
            actor_is_selected_vehicle=str(actor_vehicle_id) == self.selected['vehicle_id'],
            execution_applied=applied is not None,
            executed_hold_seconds=applied['applied_hold_seconds'] if applied is not None else None,
            applied_real_clock=application_clock,
            applied_event_clock=applied['time'] if applied is not None else None,
            applied_issue_ledger_clock=applied['issue_ledger_time'] if applied is not None else None,
            elapsed_since_apply_seconds=float(clock)-application_clock if application_clock is not None else None,
            registered_stop_duration_seconds=row['registered_stop_duration_seconds'] if row is not None else None,
            departed_observed=departed_seen, next_same_vehicle_arrived_observed=arrived_seen,
            latest_physical=copy.deepcopy(row)))

    def observe_issue(self, event, stream):
        clock = float(stream._snapshot().time_seconds)
        key = (str(event['line_id']), str(event['trip_id']), str(event['station_id']))
        occurrence = self.occurrences.get(key, 0)
        self.occurrences[key] = occurrence+1
        self.issues[event['sequence']] = dict(
            identity=dict(line_id=key[0], trip_id=key[1], station_id=key[2], occurrence=occurrence,
                vehicle_id=event['vehicle_id']),
            memory=self.memory(clock, event['vehicle_id']))
        self.issue_count += 1
        if self.tick_count == 0:
            self.observe_tick(stream)

    def observe_tick(self, stream):
        snapshot = stream._snapshot()
        clock = float(snapshot.time_seconds)
        if self.last_tick_clock == clock:
            return
        wait = float(snapshot.accrued_wait_seconds-stream.fork_snapshot.accrued_wait_seconds)
        ride = float(snapshot.accrued_in_vehicle_seconds-stream.fork_snapshot.accrued_in_vehicle_seconds)
        self._write(self.ticks, dict(clock_seconds=clock,
            relative_seconds=clock-float(self.selected['started_at']), wait=wait, ride=ride, cost=2*wait+ride,
            waiting_passengers=float(snapshot.waiting_passengers),
            in_vehicle_passengers=float(snapshot.in_vehicle_passengers),
            physical=copy.deepcopy(self.physical.previous), memory=self.memory(clock)))
        self.last_tick_clock = clock
        self.tick_count += 1

    def close_transition(self, transition, snapshot, *, kind='global_transition'):
        if transition.sequence in self.closed_sequences:
            raise RuntimeError('A suffix transition was recorded twice')
        self.closed_sequences.add(transition.sequence)
        issue = self.issues.get(transition.sequence)
        if issue is None:
            raise RuntimeError('A suffix transition has no causal issue observation')
        bellman = self.evaluator.evaluate(transition) if self.evaluator is not None else None
        if bellman is not None:
            checks = bellman['integrity']
            self.rng_restored &= bool(checks['rng_restored'])
            self.learner_rng_restored &= bool(checks['learner_rng_restored'])
            self.target_path_exact &= bool(checks['target_path_exact'])
            self.evaluation_count += 1
        row = dict(kind=kind, cutoff_partial=False,
            identity=issue['identity'], transition=asdict(transition),
            memory_s=issue['memory'], memory_ns=self.memory(float(snapshot.time_seconds), transition.next_vehicle_id),
            bellman=bellman)
        self._write(self.events, row)
        self.transition_count += 1
        self.zero_duration_count += int(transition.dt == 0)
        if self.first_transition is None:
            self.first_transition = dict(sequence=transition.sequence, dt=transition.dt,
                started_at=transition.started_at, completed_at=transition.completed_at)

    def cutoff_partial(self, stream):
        pending = stream._pending
        if pending is None or pending['sequence'] in self.closed_sequences:
            return
        start, stop = pending['snapshot'], stream.cutoff_snapshot
        wait = float(stop.accrued_wait_seconds-start.accrued_wait_seconds)
        ride = float(stop.accrued_in_vehicle_seconds-start.accrued_in_vehicle_seconds)
        issue = self.issues[pending['sequence']]
        # No next event exists here. Keep the realized partial cost without
        # creating ns, a terminal label or a bootstrap from a fictional state.
        self._write(self.events, dict(kind='cutoff_partial', cutoff_partial=True,
            identity=issue['identity'], transition=dict(sequence=pending['sequence'],
                started_at=pending['started_at'], completed_at=float(stop.time_seconds),
                dt=float(stop.time_seconds)-pending['started_at'], terminal=False,
                vehicle_id=pending['vehicle_id'], line_id=pending['line_id'],
                trip_id=pending['trip_id'], station_id=pending['station_id'],
                s=pending['s'], a=pending['a'], raw_s=pending['raw_s'], ns=None, raw_ns=None, r=None),
            partial_cost=dict(wait=wait, ride=ride, cost=2*wait+ride),
            memory_s=issue['memory'], memory_ns=None, bellman=None))
        self.partial_count += 1

    def summary(self):
        checks = dict(memory_observed_causally=self.memory_causal,
            evaluator_rng_restored=self.rng_restored,
            evaluator_learner_rng_restored=self.learner_rng_restored,
            evaluator_target_path_exact=self.target_path_exact,
            each_actual_transition_evaluated=self.evaluator is not None and self.evaluation_count == self.transition_count,
            no_training_memory_injection=True)
        return dict(events_path='visibility.jsonl', ledger_ticks_path='ledger_ticks.jsonl',
            transition_count=self.transition_count, issue_count=self.issue_count,
            ledger_tick_count=self.tick_count, zero_duration_count=self.zero_duration_count,
            cutoff_partial_count=self.partial_count, first_transition=self.first_transition,
            evaluation_count=self.evaluation_count,
            integrity=dict(all_pass=all(checks.values()), checks=checks),
            budget=dict(optimizer_updates=0, training_labels=0))


class VisibilityEventStream(ForkEventStream):
    """Original parent path with read-only child closure hooks."""

    def __init__(self, *args, **kwargs):
        self.visibility_recorder = None
        super().__init__(*args, **kwargs)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        pending = self._pending
        sequence = pending['sequence'] if pending is not None else None
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if self.visibility_recorder is not None and sequence is not None:
            if not self._queue or self._queue[-1].sequence != sequence:
                raise RuntimeError('Visibility diagnostics require every global transition')
            self.visibility_recorder.close_transition(self._queue[-1], snapshot)

    def _capture_cutoff(self):
        super()._capture_cutoff()
        if self.cutoff_reached and self.visibility_recorder is not None:
            pending = self._pending
            if pending is None or pending['sequence'] in self.visibility_recorder.closed_sequences:
                return
            # The original stream intentionally does not issue a control at
            # the cutoff. If a real batch already exists there, observe its
            # first actual state without changing pending/queue/counters.
            controls = [event for event in self._batch() if not event['terminal']]
            if controls:
                local = np.stack([event['event'] for event in controls])
                raw = pack_event_batch_v7(local, 0,
                    np.full(len(controls), -1., np.float32), self.contract)
                stop = self.cutoff_snapshot
                row = Transition(pending['s'], pending['a'], self._canonical(raw),
                    float(transition_reward_v7(pending['snapshot'], stop, self.contract)),
                    float(stop.time_seconds)-pending['started_at'], False,
                    self.episode_id, self.domain, pending['started_at'],
                    float(stop.time_seconds), float(stop.time_seconds)+self.delay_seconds,
                    pending['vehicle_id'], pending['line_id'], pending['station_id'],
                    controls[0]['vehicle_id'], pending['sequence'], pending['raw_s'],
                    raw.copy(), pending['trip_id'])
                self.visibility_recorder.close_transition(row, stop,
                    kind='cutoff_actual_without_issue')
            else:
                self.visibility_recorder.cutoff_partial(self)
