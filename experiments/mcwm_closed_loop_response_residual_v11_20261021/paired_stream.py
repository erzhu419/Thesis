"""Controlled calibration prefixes and one bounded native action intervention.

Repeated prefix controls follow v10. At the held-out fork, sibling controls
retain their nominal policy actions while recorded states retain the command
actually executed. The suffix ends at the native 300-second ledger clock.
"""

from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.controlled_stream import ControlledStream
from experiments.mcwm_closed_loop_v1.stream import (
    pack_event_batch_v7, policy_action_to_hold_seconds_v7, terminal_raw_state_v7,
)


def _action(value):
    value = np.asarray(value, np.float32).reshape(-1)
    if value.shape != (1,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1.00001):
        raise ValueError('Executed command must be one finite action in [-1,1]')
    return value


class PairedControlledStream(ControlledStream):
    """Keep one common calibration prefix, then replace one executed command.

    ``selector`` may repeatedly replace ordinary calibration commands.
    ``fork_selector`` is called until it returns one executed action. No label
    is admitted to training by this stream; the runner owns that decision.
    Endpoints not observed by the cutoff remain pending for explicit missing
    reporting by the runner rather than extending the suffix.
    """

    def __init__(self, *args, fork_selector=None, fork_duration_seconds=300., **kwargs):
        super().__init__(*args, **kwargs)
        self.fork_selector = fork_selector
        self.fork_duration_seconds = float(fork_duration_seconds)
        if self.fork_duration_seconds <= 0:
            raise ValueError('Fork duration must be positive')
        self.fork = self.fork_transition = self.fork_snapshot = self.cutoff_snapshot = None
        self.cutoff_reached = False
        self.fork_batch_other_controls = []

    @property
    def native_suffix_steps(self):
        return 0 if self.fork is None else int(self.env._bridge.steps)-self.fork['native_step']

    @property
    def cutoff_time(self):
        return None if self.fork is None else self.fork['started_at']+self.fork_duration_seconds

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        capture = (self._pending is not None and self.fork is not None
                   and self._pending['sequence'] == self.fork['sequence'])
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if capture:
            if not self._queue or self._queue[-1].sequence != self.fork['sequence']:
                raise RuntimeError('The paired native fork transition was thinned')
            self.fork_transition = self._queue[-1]

    def _advance_bounded_native(self, commands):
        env, bridge = self.env, self.env._bridge
        for event in env._current_events:
            bridge.apply_action(event, commands.get(event.bus_id, [0., 1.]))
        env.state, env.reward, env._current_events = {}, {}, []
        while not bridge.done:
            if bridge.decision_queue:
                events, done, _departed = bridge.fetch_events()
                env.done, env._current_events = bool(done), events
                for event in events:
                    env.state[event.bus_id] = [[env._event_to_obs(event).tolist()]]
                    env.reward[event.bus_id] = env._compute_reward(event)
                return
            now = float(bridge.passenger_time_ledger.snapshot().time_seconds)
            if now >= self.cutoff_time:
                if now != self.cutoff_time:
                    raise RuntimeError('Native stepping overshot the exact paired cutoff')
                return
            # The attached v10 clock hook observes the actual post-tick ledger.
            bridge._advance_one_step()
        env.done = True

    def _capture_cutoff(self):
        snapshot = self._snapshot()
        if float(snapshot.time_seconds) != self.cutoff_time:
            return
        batch = self._batch()
        # Actual events delivered at the boundary still close physical/response
        # endpoints, including terminal arrivals. No action is issued there.
        for event in batch:
            self._close_physical(event)
        if self._pending is not None:
            controls = [event for event in batch if not event['terminal']]
            if controls:
                local = np.stack([event['event'] for event in controls])
                prior = np.full(len(controls), -1., np.float32)
                raw = pack_event_batch_v7(local, 0, prior, self.contract)
                self._close(raw, snapshot, controls[0]['vehicle_id'])
        self.cutoff_snapshot, self.cutoff_reached = snapshot, True

    def step(self, policy):
        if self.episode_id < 0:
            self.reset()
        if self.done or self.cutoff_reached:
            return self.release()
        batch, commands = self._batch(), {}
        for event in batch:
            self._close_physical(event)
            if event['terminal']:
                commands[event['agent']] = [0., 1.]
        controls = [event for event in batch if not event['terminal']]
        selected_in_batch = False
        if controls:
            snapshot = self._snapshot(controls[0]['agent'])
            local = np.stack([event['event'] for event in controls])
            prior = np.full(len(controls), -1., np.float32)
            nominal_prior = prior.copy()
            for index, event in enumerate(controls):
                raw = pack_event_batch_v7(local, index, prior, self.contract)
                state = self._canonical(raw)
                self._close(raw, snapshot, event['vehicle_id'])
                nominal_raw = (pack_event_batch_v7(local, index, nominal_prior, self.contract)
                               if selected_in_batch else raw)
                nominal_state = self._canonical(nominal_raw) if selected_in_batch else state
                proposed = _action(policy(nominal_state))
                issued = dict(event, s=state.copy(), raw_s=raw.copy(), a=proposed.copy(),
                    proposed_a=proposed.copy(), started_at=float(snapshot.time_seconds),
                    sequence=self.issued_count, episode_id=self.episode_id, domain=self.domain)
                # A sibling's normal controller sees its unperturbed prefix.
                choice_issue = (dict(issued, s=nominal_state.copy(), raw_s=nominal_raw.copy())
                                if selected_in_batch else issued)
                replacement = self.selector(choice_issue)
                nominal_action = proposed.copy() if replacement is None else _action(replacement)
                action = nominal_action.copy()
                issued['a'] = action.copy()
                fork_replacement = None
                if self.fork is None and self.fork_selector is not None:
                    fork_replacement = self.fork_selector(issued)
                    if fork_replacement is not None:
                        if self.record_every != 1:
                            raise ValueError('Paired native diagnostics require record_every=1')
                        action = _action(fork_replacement)
                        issued['a'] = action.copy()
                        self.fork = dict(issued, nominal_a=nominal_action.copy(),
                            native_step=int(self.env._bridge.steps))
                        self.fork_snapshot = snapshot
                        selected_in_batch = True
                elif selected_in_batch:
                    self.fork_batch_other_controls.append(dict(sequence=self.issued_count,
                        line_id=event['line_id'], vehicle_id=event['vehicle_id'],
                        a=action.tolist(), hold_seconds=float(
                            policy_action_to_hold_seconds_v7(action[0], self.contract))))
                if self.on_issue is not None:
                    self.on_issue(issued)
                self._pending = dict(issued, snapshot=snapshot)
                self._vehicle_pending[self._vehicle_key(event)] = issued
                prior[index], nominal_prior[index] = action[0], nominal_action[0]
                hold = float(policy_action_to_hold_seconds_v7(action[0], self.contract))
                commands[event['agent']] = [hold, 1.]
                if replacement is not None or fork_replacement is not None:
                    self.executed_controls.append(dict(sequence=self.issued_count, hold_seconds=hold,
                        action=float(action[0]), time=float(snapshot.time_seconds), line_id=event['line_id']))
                self.issued_count += 1
        if self.fork is None:
            self.env.step_to_event(commands)
        else:
            self._advance_bounded_native(commands)
        self.env._bridge.consume_completed_trip_passenger_times()
        if self.done:
            self.final_snapshot = self._snapshot()
            if self.final_snapshot.time_seconds != self.contract.horizon_seconds:
                raise RuntimeError('Native simulator did not reach the configured target horizon')
            if self._pending is not None:
                self._close(terminal_raw_state_v7(self._pending['raw_s'], self.final_snapshot),
                            self.final_snapshot, terminal=True)
            self.uncompleted_interval_count = len(self._vehicle_pending)
        if self.fork is not None:
            self._capture_cutoff()
        return self.release()
