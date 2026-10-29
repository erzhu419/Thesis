"""Two parallel native suffixes copied from one live, paused factual prefix."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from experiments.mcwm_closed_loop_fast_branch_v24r1_20261023.fork_state import fork_children
from experiments.mcwm_closed_loop_v1.stream import policy_action_to_hold_seconds_v7


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def selected_state(event, snapshot):
    selected = {key: event[key] for key in ('sequence', 'started_at', 'line_id',
        'vehicle_id', 'trip_id', 'station_id', 's', 'raw_s', 'event')}
    selected['issue_ledger'] = {key: float(getattr(snapshot, key)) for key in (
        'time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
        'accrued_wait_seconds', 'accrued_in_vehicle_seconds')}
    return plain(selected)


def endpoint(start, stop, *, status='complete'):
    if stop is None:
        return dict(status=status, dt=None, wait=None, ride=None, cost=None)
    wait = float(stop.accrued_wait_seconds-start.accrued_wait_seconds)
    ride = float(stop.accrued_in_vehicle_seconds-start.accrued_in_vehicle_seconds)
    return dict(status=status, dt=float(stop.time_seconds-start.time_seconds),
        wait=wait, ride=ride, cost=2*wait+ride)


class NativeTruthPair:
    """Parent resumes its original action; children replace just the selected one.

    The caller records the common nominal issue once before ``fork`` and uses
    the existing ForkEventStream. Children never return to main-day training:
    call ``observe_issue`` for every child issue, ``observe_progress`` after
    each child step, and ``finish_child`` at the bounded native cutoff.
    """

    def __init__(self, out, expected, counts, external_seed):
        self.out, self.expected = Path(out), expected
        self.counts, self.external_seed = counts, int(external_seed)
        self.in_child, self.arm = False, None
        self.receipt, self.values = None, None
        self.selected = self.execution = self.next_snapshot = None
        self.native_actions = []
        self.before = None
        self.started_wall = None

    def fork(self, event, stream, trace, prepare_child=None):
        selected = self.expected['selected']
        if (event['sequence'] != selected['sequence']
                or event['started_at'] != selected['started_at']
                or not np.array_equal(event['s'], np.asarray(selected['s'], np.float32))):
            raise RuntimeError('The declared v28 tail state was not exactly reproduced')
        self.before = copy.deepcopy(self.counts())

        def prepare(arm):
            self.in_child, self.arm = True, arm
            self.started_wall = time.time()
            # Open route parsers were already detached by the approved helper.
            # Detach the Python trace/logs before the half-issued event resumes.
            trace.writer.close()
            if trace.reference is not None:
                trace.reference.close()
            trace.path = self.out/arm/'suffix_trace.jsonl'
            trace.writer = trace.path.open('x', encoding='utf-8')
            trace.reference = None
            self._install_native_observer(stream)
            if prepare_child is not None:
                prepare_child(arm)
            logfd = os.open(self.out/(arm+'.log'), os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
            os.dup2(logfd, 1)
            os.dup2(logfd, 2)
            os.close(logfd)

        result = fork_children(self.out, trace, lambda: stream.now, self.counts, prepare)
        if isinstance(result, str):
            return np.asarray(self.expected['actions'][result], np.float32)
        self.receipt = result
        self.values = {arm: json.loads((self.out/arm/'summary.json').read_text())
            for arm in ('low', 'high')}
        if self.values['low']['selected'] != self.values['high']['selected']:
            raise RuntimeError('Parallel suffixes changed the inherited selected state')
        if (self.values['low']['native']['other_batch_actions'] !=
                self.values['high']['native']['other_batch_actions']):
            raise RuntimeError('Nonselected controls in the fork batch differ')
        return None

    def _install_native_observer(self, stream):
        bridge = stream.env._bridge
        apply_action = bridge.apply_action

        def observe_native(event, command):
            apply_action(event, command)
            selected = self.selected
            # A queued decision's event clock can precede the current passenger
            # ledger clock. Identify this first application by the selected
            # trip/stop, preserving both clocks rather than equating them.
            if selected is not None and not self.native_actions and (
                    str(event.bus_id) == selected['trip_id']
                    and str(event.stop_id) == selected['station_id']):
                self.native_actions.append(dict(bus_id=str(event.bus_id),
                    station_id=str(event.stop_id), time=float(event.sim_time),
                    issue_ledger_time=selected['started_at'],
                    assigned_action=float(event.assigned_action),
                    applied_hold_seconds=float(event.metadata['applied_additional_hold_seconds']),
                    predicted_departure_time_seconds=float(event.metadata['predicted_departure_time_seconds'])))

        # This assignment happens only after the child exists; the paused
        # parent's bridge and all native action semantics remain untouched.
        bridge.apply_action = observe_native

    def observe_issue(self, event, stream):
        if not self.in_child:
            return
        if event['sequence'] == self.expected['selected']['sequence']:
            self.selected = selected_state(event, stream.fork_snapshot)
            action = np.asarray(event['a'], np.float32)
            actual = float(policy_action_to_hold_seconds_v7(action[0], stream.contract))
            requested = 0. if self.arm == 'low' else .9375
            self.execution = dict(requested_hold_seconds=requested,
                executed_hold_seconds=actual, executed_action=action.tolist(),
                nominal_action=plain(stream.fork['nominal_a']))
            if (actual != requested or not np.array_equal(action,
                    np.asarray(self.expected['actions'][self.arm], np.float32))):
                raise RuntimeError('The native intervention did not execute the saved F candidate')
        self.observe_progress(stream)

    def observe_progress(self, stream):
        transition = stream.fork_transition
        if self.next_snapshot is None and transition is not None:
            snapshot = stream._snapshot()
            if float(snapshot.time_seconds) == transition.completed_at:
                self.next_snapshot = snapshot

    def build_summary(self, stream):
        self.observe_progress(stream)
        transition, start = stream.fork_transition, stream.fork_snapshot
        next_truth = endpoint(start, self.next_snapshot, status=(
            'complete' if self.next_snapshot is not None else 'missing'))
        next_truth.update(completed_at=transition.completed_at if transition is not None else None,
            next_vehicle_id=transition.next_vehicle_id if transition is not None else None,
            terminal=transition.terminal if transition is not None else None,
            zero_duration=transition.dt == 0 if transition is not None else None)
        fixed = endpoint(start, stream.cutoff_snapshot,
            status='complete' if stream.cutoff_reached else 'truncated')
        after = copy.deepcopy(self.counts())
        delta = {key: after[key]-self.before[key] for key in self.before}
        native_action = self.native_actions[0] if len(self.native_actions) == 1 else None
        if self.execution is not None:
            self.execution['native_applied_hold_seconds'] = (
                native_action['applied_hold_seconds'] if native_action is not None else None)
            self.execution['native_application'] = native_action
        checks = dict(live_prefix_inherited=self.in_child and self.selected is not None,
            actual_action_exact=self.execution is not None and
                self.execution['executed_hold_seconds'] == (0. if self.arm == 'low' else .9375),
            native_action_applied=native_action is not None and
                native_action['applied_hold_seconds'] == native_action['assigned_action'] ==
                (0. if self.arm == 'low' else .9375),
            suffix_zero_training=after == self.before and all(value == 0 for value in delta.values()),
            native_suffix_exact=stream.cutoff_reached and stream.native_suffix_steps == 300,
            fixed300_truth=fixed['status'] == 'complete' and fixed['dt'] == 300.,
            target_clock_unchanged=stream.contract.horizon_seconds == 24000.,
            no_terminal_fabrication=not stream.done and stream.final_snapshot is None)
        return plain(dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), arm=self.arm,
            selected=self.selected, external_seed=self.external_seed, execution=self.execution,
            truth=dict(next_event=next_truth, fixed300=fixed),
            global_transition=asdict(transition) if transition is not None else None,
            native=dict(other_batch_actions=stream.fork_batch_other_controls,
                suffix_steps=stream.native_suffix_steps),
            counters=dict(before=self.before, after=after, delta=delta),
            budget=dict(suffix_sim_seconds=stream.now-start.time_seconds,
                replay_target_labels=0, replay_policy_updates=0, new_truth_labels=1),
            execution_wall=dict(pid=os.getpid(), native_started_wall=self.started_wall,
                native_finished_wall=time.time())))

    def finish_child(self, stream):
        summary = self.build_summary(stream)
        with (self.out/self.arm/'summary.json').open('x') as output:
            json.dump(summary, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary['integrity']['all_pass'] else 1)
