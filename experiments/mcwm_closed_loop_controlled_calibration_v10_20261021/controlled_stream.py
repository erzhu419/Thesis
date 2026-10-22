"""Repeated executed-action calibration, with native response-clock observations."""
import numpy as np

from experiments.mcwm_closed_loop_v1.stream import (
    EventStream, pack_event_batch_v7, policy_action_to_hold_seconds_v7, terminal_raw_state_v7,
)


class ControlledStream(EventStream):
    def __init__(self, *args, selector, endpoints, response_memory, **kwargs):
        if kwargs.get('execute_transform', True):
            raise ValueError('Supply already executed actions, applying F exactly once')
        super().__init__(*args, **kwargs)
        self.selector, self.endpoints, self.response_memory = selector, endpoints, response_memory
        self.native_ticks = 0
        self.executed_controls = []

    def attach_native_clock(self):
        bridge = self.env._bridge
        original = bridge._advance_one_step

        def tick():
            result = original()
            self.native_ticks += 1
            self.endpoints.observe_tick(bridge.passenger_time_ledger.snapshot())
            return result

        bridge._advance_one_step = tick

    def _close_physical(self, event):
        self.endpoints.observe_event(event, self._snapshot())
        self.response_memory.observe_event(event)
        return super()._close_physical(event)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        sequence = None if self._pending is None else self._pending['sequence']
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if sequence is not None and self._queue and self._queue[-1].sequence == sequence:
            self.endpoints.observe_global(self._queue[-1], snapshot)

    def step(self, policy):
        if self.episode_id < 0:
            self.reset()
        if self.done:
            return self.release()
        batch, commands = self._batch(), {}
        for event in batch:
            self._close_physical(event)
            if event['terminal']:
                commands[event['agent']] = [0., 1.]
        controls = [event for event in batch if not event['terminal']]
        if controls:
            snapshot = self._snapshot(controls[0]['agent'])
            local = np.stack([event['event'] for event in controls])
            prior = np.full(len(controls), -1., np.float32)
            for index, event in enumerate(controls):
                raw = pack_event_batch_v7(local, index, prior, self.contract)
                state = self._canonical(raw)
                self._close(raw, snapshot, event['vehicle_id'])
                action = np.asarray(policy(state), np.float32).reshape(-1)
                issued = dict(event, s=state.copy(), raw_s=raw.copy(), a=action.copy(),
                    proposed_a=action.copy(), started_at=float(snapshot.time_seconds),
                    sequence=self.issued_count, episode_id=self.episode_id, domain=self.domain)
                replacement = self.selector(issued)
                if replacement is not None:
                    action = np.asarray(replacement, np.float32).reshape(-1)
                    issued['a'] = action.copy()
                if action.shape != (1,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.00001):
                    raise ValueError('Executed command must be one finite action in [-1,1]')
                if self.on_issue is not None:
                    self.on_issue(issued)
                self._pending = dict(issued, snapshot=snapshot)
                self._vehicle_pending[self._vehicle_key(event)] = issued
                prior[index] = action[0]
                hold = float(policy_action_to_hold_seconds_v7(action[0], self.contract))
                commands[event['agent']] = [hold, 1.]
                if replacement is not None:
                    self.executed_controls.append(dict(sequence=self.issued_count, hold_seconds=hold,
                        action=float(action[0]), time=float(snapshot.time_seconds), line_id=event['line_id']))
                self.issued_count += 1
        self.env.step_to_event(commands)
        self.env._bridge.consume_completed_trip_passenger_times()
        if self.done:
            self.final_snapshot = self._snapshot()
            if self.final_snapshot.time_seconds != self.contract.horizon_seconds:
                raise RuntimeError('Native simulator did not reach the configured target horizon')
            if self._pending is not None:
                self._close(terminal_raw_state_v7(self._pending['raw_s'], self.final_snapshot),
                            self.final_snapshot, terminal=True)
            self.uncompleted_interval_count = len(self._vehicle_pending)
        return self.release()
