"""One native action intervention followed by an exact, nonterminal cutoff.

The prefix uses the original event advancement.  A selector is called before
the existing issue callback and may replace exactly one *executed* action.
Later controls in that atomic batch use their unperturbed policy states while
their recorded states retain the action actually executed by the intervention.
"""

from __future__ import annotations

import numpy as np

from experiments.mcwm_closed_loop_v1.stream import (
    EventStream, pack_event_batch_v7, policy_action_to_hold_seconds_v7,
    terminal_raw_state_v7,
)


def _action(value):
    value = np.asarray(value, np.float32).reshape(-1)
    if value.shape != (1,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1.00001):
        raise ValueError("Policy and fork actions must be finite scalars in [-1, 1]")
    return np.clip(value, -1., 1.)


class ForkEventStream(EventStream):
    """EventStream with ``selector(issued) -> action | None`` and a 300 s suffix.

    ``fork`` is the selected issue dictionary. ``fork_transition`` is its
    actual next global control transition, including a same-time zero-dt row.
    It is captured on completion, independently of feedback delay. No suffix
    labels are inserted into training by this class. ``cutoff_snapshot`` is
    the native passenger ledger at exactly ``fork['started_at'] + duration``;
    reaching it leaves ``env.done`` false and does not create a terminal row.
    """

    def __init__(self, *args, selector, fork_duration_seconds=300., **kwargs):
        if kwargs.get("execute_transform", True):
            raise ValueError("Fork actions must already use the executed-action transform")
        super().__init__(*args, **kwargs)
        self.selector = selector
        self.fork_duration_seconds = float(fork_duration_seconds)
        if self.fork_duration_seconds <= 0:
            raise ValueError("Fork duration must be positive")
        self.fork = self.fork_transition = self.fork_snapshot = self.cutoff_snapshot = None
        self.cutoff_reached = False
        self.fork_batch_other_controls = []

    @property
    def native_suffix_steps(self):
        return 0 if self.fork is None else int(self.env._bridge.steps) - self.fork["native_step"]

    @property
    def cutoff_time(self):
        return None if self.fork is None else self.fork["started_at"] + self.fork_duration_seconds

    def _close(self, raw_ns, snapshot, next_vehicle_id="", terminal=False):
        capture = (self._pending is not None and self.fork is not None
                   and self._pending["sequence"] == self.fork["sequence"])
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if capture:
            # Fork diagnostics use every issued event; thinning would conceal
            # the true global next event, including simultaneous other buses.
            if not self._queue or self._queue[-1].sequence != self.fork["sequence"]:
                raise RuntimeError("The fork transition was thinned")
            self.fork_transition = self._queue[-1]

    def _advance_bounded_native(self, commands):
        """Apply native commands and run full bridge ticks up to the cutoff.

        ``fetch_events`` ordinarily blocks until an event. Here the bridge's
        actual one-tick path updates passengers, SUMO, and the welfare ledger;
        fetch is called only when its real decision queue is populated.
        """
        env, bridge = self.env, self.env._bridge
        if bridge is None:
            raise ValueError("The action-fork suffix requires the native SUMO bridge")
        for event in env._current_events:
            bridge.apply_action(event, commands.get(event.bus_id, [0., 1.]))
        env.state, env.reward, env._current_events = {}, {}, []
        while not bridge.done:
            if bridge.decision_queue:
                events, done, _departed = bridge.fetch_events()
                env.done = bool(done)
                env._current_events = events
                for event in events:
                    env.state[event.bus_id] = [[env._event_to_obs(event).tolist()]]
                    env.reward[event.bus_id] = env._compute_reward(event)
                return
            now = float(bridge.passenger_time_ledger.snapshot().time_seconds)
            if now >= self.cutoff_time:
                if now != self.cutoff_time:
                    raise RuntimeError("Native stepping overshot the exact fork cutoff")
                return
            bridge._advance_one_step()
        env.done = True

    def _capture_cutoff(self):
        snapshot = self._snapshot()
        if float(snapshot.time_seconds) != self.cutoff_time:
            return
        # An actual event may have become available exactly at the endpoint.
        # Observe its first canonical state without issuing another control.
        if self.fork_transition is None:
            controls = [event for event in self._batch() if not event["terminal"]]
            if controls:
                local = np.stack([event["event"] for event in controls])
                raw = pack_event_batch_v7(local, 0, np.full(len(controls), -1., np.float32), self.contract)
                self._close(raw, snapshot, controls[0]["vehicle_id"])
        self.cutoff_snapshot, self.cutoff_reached = snapshot, True

    def step(self, policy):
        if self.episode_id < 0:
            self.reset()
        if self.done or self.cutoff_reached:
            return self.release()
        batch, commands = self._batch(), {}
        for event in batch:
            self._close_physical(event)
            if event["terminal"]:
                commands[event["agent"]] = [0., 1.]
        controls = [event for event in batch if not event["terminal"]]
        selected_in_batch = False
        if controls:
            snapshot = self._snapshot(controls[0]["agent"])
            local = np.stack([event["event"] for event in controls])
            prior = np.full(len(controls), -1., np.float32)
            nominal_prior = prior.copy()
            for index, event in enumerate(controls):
                raw = pack_event_batch_v7(local, index, prior, self.contract)
                state = self._canonical(raw)
                self._close(raw, snapshot, event["vehicle_id"])
                policy_state = state
                if selected_in_batch:
                    policy_state = self._canonical(pack_event_batch_v7(local, index, nominal_prior, self.contract))
                proposed = _action(policy(policy_state))
                action = proposed.copy()
                issued = dict(event, s=state.copy(), raw_s=raw.copy(), a=action.copy(),
                    proposed_a=proposed.copy(), started_at=float(snapshot.time_seconds),
                    sequence=self.issued_count, episode_id=self.episode_id, domain=self.domain)
                if self.fork is None:
                    replacement = self.selector(issued)
                    if replacement is not None:
                        if self.record_every != 1:
                            raise ValueError("Native fork diagnostics require record_every=1")
                        action = _action(replacement)
                        issued["a"] = action.copy()
                        self.fork = dict(issued, nominal_a=proposed.copy(),
                            native_step=int(self.env._bridge.steps))
                        self.fork_snapshot = snapshot
                        selected_in_batch = True
                elif selected_in_batch:
                    self.fork_batch_other_controls.append(dict(sequence=self.issued_count,
                        line_id=event["line_id"], vehicle_id=event["vehicle_id"],
                        a=action.tolist(), hold_seconds=float(policy_action_to_hold_seconds_v7(action[0], self.contract))))
                if self.on_issue is not None:
                    self.on_issue(issued)
                self._pending = dict(issued, snapshot=snapshot)
                self._vehicle_pending[self._vehicle_key(event)] = issued
                nominal_prior[index], prior[index] = proposed[0], action[0]
                commands[event["agent"]] = [float(policy_action_to_hold_seconds_v7(action[0], self.contract)), 1.]
                self.issued_count += 1
        if self.fork is None:
            self.env.step_to_event(commands)
        else:
            self._advance_bounded_native(commands)
        if getattr(self.env, "_bridge", None) is not None:
            self.env._bridge.consume_completed_trip_passenger_times()
        if self.done:
            self.final_snapshot = self._snapshot()
            if self.final_snapshot.time_seconds != self.contract.horizon_seconds:
                raise RuntimeError("Simulator stopped before the configured horizon; no terminal label was fabricated")
            if self._pending is not None:
                self._close(terminal_raw_state_v7(self._pending["raw_s"], self.final_snapshot),
                            self.final_snapshot, terminal=True)
            self.uncompleted_interval_count = len(self._vehicle_pending)
        if self.fork is not None:
            self._capture_cutoff()
        return self.release()
