"""Paired actions with a common frozen controller and dynamic F endpoints."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children
from experiments.mcwm_closed_loop_physical_execution_v30_20261023.suffix import (
    NativePhysicalObserver, NativeTruthTriple, endpoint, plain,
    selected_state as _native_selected_state,
)
from experiments.mcwm_closed_loop_v1.stream import policy_action_to_hold_seconds_v7


ARMS = ('low', 'high')
COMMAND_ATOL_SECONDS = 1e-5
REGISTERED_ATOL_SECONDS = .0011


def selected_state(event, snapshot):
    selected = _native_selected_state(event, snapshot)
    if 'memory_s' in event:
        selected['memory_s'] = plain(event['memory_s'])
    return selected


def _close(left, right, tolerance=COMMAND_ATOL_SECONDS):
    return left is not None and right is not None and abs(float(left)-float(right)) <= tolerance


def pair_comparison(values):
    """Physical separability is an outcome, never a reason to reselect a state."""
    low, high = values['low'], values['high']
    departures = [value['physical']['departure'] for value in (low, high)]
    arrivals = [value['physical']['next_same_vehicle_arrival'] for value in (low, high)]

    def gap(events):
        return (float(events[1]['clock_seconds'])-float(events[0]['clock_seconds'])
            if all(event['status'] == 'observed' for event in events) else None)

    departure_gap, arrival_gap = gap(departures), gap(arrivals)
    departure_separable = None if departure_gap is None else departure_gap != 0.
    arrival_separable = None if arrival_gap is None else arrival_gap != 0.
    separable = (True if departure_separable is True or arrival_separable is True else
        False if departure_separable is False and arrival_separable is False else None)
    separation = dict(status='physical_unknown' if separable is None else
        'separable' if separable else 'nonseparable',
        physical_separable=separable, departure_separable=departure_separable,
        next_arrival_separable=arrival_separable,
        departure_gap_seconds=departure_gap, next_same_vehicle_arrival_gap_seconds=arrival_gap,
        low_departure_status=departures[0]['status'], high_departure_status=departures[1]['status'],
        low_next_arrival_status=arrivals[0]['status'], high_next_arrival_status=arrivals[1]['status'])
    differences = {}
    for name in ('next_event', 'fixed300'):
        left, right = low['truth'][name], high['truth'][name]
        complete = left['status'] == right['status'] == 'complete'
        differences[name] = dict(status='complete' if complete else 'missing',
            same_duration=left['dt'] == right['dt'] if complete else None,
            **{key: float(right[key]-left[key]) if complete else None
               for key in ('dt', 'wait', 'ride', 'cost')})
    return dict(physical_separation=separation, high_minus_low=differences)


class NativeTruthPair(NativeTruthTriple):
    """Fork only the selected executed action; parent and controller stay fixed.

    expected contains selected, actions[low/high], holds[low/high], and optional
    extra issue fields (e.g. memory_s). The original physical observer is reused
    unchanged; no all-event Bellman/visibility traces are produced.
    """

    def __init__(self, out, expected, counts, external_seed):
        super().__init__(out, expected, counts, external_seed)
        self.comparison = None

    def fork(self, event, stream, trace, prepare_child=None):
        actual = selected_state(event, stream._snapshot())
        for key, value in self.expected['selected'].items():
            if key not in actual or plain(value) != actual[key]:
                raise RuntimeError(f'The selected common-prefix state differs: {key}')
        for key, value in self.expected.get('extra', {}).items():
            if key not in event or plain(event[key]) != plain(value):
                raise RuntimeError(f'The selected causal issue context differs: {key}')
        self.before = copy.deepcopy(self.counts())

        def prepare(arm):
            self.in_child, self.arm = True, arm
            self.started_wall = time.time()
            trace.writer.close()
            if trace.reference is not None:
                trace.reference.close()
            trace.path = self.out/arm/'suffix_trace.jsonl'
            trace.writer = trace.path.open('x', encoding='utf-8')
            trace.reference = None
            self._install_native_observer(stream)
            if prepare_child is not None:
                prepare_child(arm)

        result = fork_children(self.out, trace, lambda: stream.now, self.counts, prepare)
        if isinstance(result, str):
            return np.asarray(self.expected['actions'][result], np.float32)
        self.receipt = result
        self.values = {arm: json.loads((self.out/arm/'summary.json').read_text()) for arm in ARMS}
        if self.values['low']['selected'] != self.values['high']['selected']:
            raise RuntimeError('Paired suffixes changed the inherited selected state/context')
        if self.values['low']['native']['other_batch_actions'] != self.values['high']['native']['other_batch_actions']:
            raise RuntimeError('Nonselected controls in the common fork batch differ')
        if any(value['external_seed'] != self.external_seed for value in self.values.values()):
            raise RuntimeError('Paired suffixes have different exogenous seeds')
        self.comparison = pair_comparison(self.values)
        with (self.out/'pair_comparison.json').open('x') as output:
            json.dump(self.comparison, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
        # Returning None leaves ForkEventStream.fork unset in the parent. Its
        # selector may therefore select another declared line/window later.
        return None

    def observe_issue(self, event, stream):
        if not self.in_child:
            return
        if event['sequence'] == self.expected['selected']['sequence']:
            self.selected = selected_state(event, stream.fork_snapshot)
            self.physical.activate(self.selected)
            action = np.asarray(event['a'], np.float32)
            actual = float(policy_action_to_hold_seconds_v7(action[0], stream.contract))
            requested = float(self.expected['holds'][self.arm])
            self.execution = dict(requested_hold_seconds=requested,
                executed_hold_seconds=actual, executed_action=action.tolist(),
                nominal_action=plain(stream.fork['nominal_a']))
            if not _close(actual, requested) or not np.array_equal(action,
                    np.asarray(self.expected['actions'][self.arm], np.float32)):
                raise RuntimeError('The intervention did not execute its dynamic saved F candidate')
        self.observe_progress(stream)

    def build_summary(self, stream):
        self.observe_progress(stream)
        transition, start = stream.fork_transition, stream.fork_snapshot
        next_truth = endpoint(start, self.next_snapshot,
            status='complete' if self.next_snapshot is not None else 'missing')
        next_truth.update(completed_at=transition.completed_at if transition is not None else None,
            next_vehicle_id=transition.next_vehicle_id if transition is not None else None,
            terminal=transition.terminal if transition is not None else None,
            zero_duration=transition.dt == 0 if transition is not None else None)
        fixed = endpoint(start, stream.cutoff_snapshot,
            status='complete' if stream.cutoff_reached else 'truncated')
        after = copy.deepcopy(self.counts())
        delta = {key: after[key]-self.before[key] for key in self.before}
        native_action = self.native_actions[0] if len(self.native_actions) == 1 else None
        physical = self.physical.summary(stream.cutoff_reached) if self.physical is not None else None
        requested = float(self.expected['holds'][self.arm])
        if self.execution is not None:
            self.execution['native_applied_hold_seconds'] = native_action['applied_hold_seconds'] if native_action else None
            self.execution['native_application'] = native_action
            registered = physical['apply_after']['registered_stop_duration_seconds'] if physical else None
            self.execution['registered_hold_seconds'] = registered
            self.execution['registered_matches_requested'] = (_close(registered, requested, REGISTERED_ATOL_SECONDS)
                if registered is not None else None)
        checks = dict(live_prefix_inherited=self.in_child and self.selected is not None,
            actual_action_exact=self.execution is not None and _close(self.execution['executed_hold_seconds'], requested),
            native_action_applied=native_action is not None and
                _close(native_action['applied_hold_seconds'], requested) and
                _close(native_action['assigned_action'], requested),
            registered_hold_when_observed=self.execution is not None and
                self.execution['registered_matches_requested'] is not False,
            physical_issue_observed=physical is not None and all(physical['coverage'][key] for key in
                ('actual_clock_observed', 'action_step_length_observed', 'issue_stop_state_observed')),
            physical_native_ticks_exact=physical is not None and physical['native_tick_count'] == stream.native_suffix_steps,
            real_application_clock=native_action is not None and
                native_action['real_clock_before'] == native_action['real_clock_after'] == start.time_seconds,
            suffix_zero_training=after == self.before and all(value == 0 for value in delta.values()),
            native_suffix_exact=stream.cutoff_reached and stream.native_suffix_steps == 300,
            fixed300_truth=fixed['status'] == 'complete' and fixed['dt'] == 300.,
            target_clock_unchanged=stream.contract.horizon_seconds == 24000.,
            no_terminal_fabrication=not stream.done and stream.final_snapshot is None)
        return plain(dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()), checks=checks), arm=self.arm,
            selected=self.selected, external_seed=self.external_seed, execution=self.execution, physical=physical,
            truth=dict(next_event=next_truth, fixed300=fixed),
            global_transition=asdict(transition) if transition is not None else None,
            native=dict(other_batch_actions=stream.fork_batch_other_controls, suffix_steps=stream.native_suffix_steps),
            counters=dict(before=self.before, after=after, delta=delta),
            budget=dict(suffix_sim_seconds=stream.now-start.time_seconds,
                replay_target_labels=0, replay_policy_updates=0, new_truth_labels=1),
            execution_wall=dict(pid=os.getpid(), native_started_wall=self.started_wall,
                native_finished_wall=time.time())))
