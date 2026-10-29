"""Passive issue-time native context observation; no simulation or training API."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from experiments.mcwm_closed_loop_capacity_context_v78_20261027.context import observe_context


LEDGER_FIELDS = ('time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
                 'accrued_wait_seconds', 'accrued_in_vehicle_seconds')


def statistics(values):
    return dict(count=len(values), min=min(values) if values else None,
        max=max(values) if values else None,
        mean=sum(values)/len(values) if values else None)


class NativeContextObserver:
    def __init__(self, bridge, path, *, window=(12000., 16200.), state_probe=None,
                 state_equal=None, api=None):
        self.bridge, self.window = bridge, tuple(window)
        self.api = api if api is not None else sys.modules[type(bridge).__module__].traci
        self.state_probe, self.state_equal = state_probe, state_equal
        if state_probe is not None and state_equal is None:
            from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal
            self.state_equal = _equal
        self.writer = Path(path).open('x')
        self.original = bridge.apply_action
        self.had_instance_action = 'apply_action' in bridge.__dict__
        self.saved_instance_action = bridge.__dict__.get('apply_action')
        self.pending, self.order, self.closed = {}, 0, False
        self.audit, self.reasons, self.plans = Counter(), Counter(), Counter()
        self.overall, self.by_line = Counter(), defaultdict(Counter)
        self.groups = {key: Counter() for key in
            ('onboard_positive', 'nonempty_queue', 'release_positive')}
        self.membership_unknown = Counter()
        self.clock_lags, self.observation_lags = [], []
        self.stage_counts = {group: Counter() for group in ('onboard', 'queue')}
        self.stage_examples = {group: [] for group in ('onboard', 'queue')}
        self.nonempty_line_examples = {group: [] for group in ('onboard', 'queue')}
        self.examples = []
        bridge.apply_action = self._apply

    def _write(self, row):
        self.writer.write(json.dumps(row, allow_nan=False, ensure_ascii=False)+'\n')

    def _next_order(self):
        self.order += 1
        return self.order

    def _in_window(self, clock):
        return self.window[0] <= clock < self.window[1]

    def _terminal(self, event):
        stops = self.bridge.line_obj_dic[event.line_id].stop_id_l
        return event.stop_idx >= len(stops)-1 or event.stop_id == stops[-1]

    def _guard(self):
        snapshot = self.bridge.passenger_time_ledger.snapshot()
        return dict(clock=float(self.api.simulation.getTime()),
            ledger={key: float(getattr(snapshot, key)) for key in LEDGER_FIELDS},
            rng=None if self.state_probe is None else self.state_probe())

    def _guard_checks(self, before, after, scope):
        checks = dict(clock=before['clock'] == after['clock'],
            ledger=before['ledger'] == after['ledger'],
            rng=self.state_probe is not None and self.state_equal(before['rng'], after['rng']))
        for key, value in checks.items():
            self.audit[scope+'_'+key+'_checks'] += 1
            self.audit[scope+'_'+key+'_failures'] += not value
        return checks

    def _next_stop_plan(self, native, expected):
        """A native stop plan is known intent, never a future arrival observation."""
        try:
            rows = self.api.vehicle.getStops(native.bus_id, 2)
        except Exception:
            return dict(status='missing', scorable=False, matches=None)
        stops = [getattr(row, 'stoppingPlaceID', None) for row in rows]
        if not stops:
            return dict(status='empty', scorable=False, matches=None)
        if stops[0] == native.stop_id:
            stops = stops[1:]
        if not stops:
            return dict(status='current_only', scorable=False, matches=None)
        if not isinstance(stops[0], str) or not stops[0]:
            return dict(status='unsupported_stop_id', scorable=False, matches=None)
        matched = stops[0] == expected
        return dict(status='matched' if matched else 'mismatch', scorable=True,
                    matches=matched, planned_next_stop=stops[0])

    def _context_api(self):
        memberships = {}

        def onboard(trip):
            values = self.api.vehicle.getPersonIDList(trip)
            memberships['onboard'] = set(values)
            return values

        def queue(stop):
            values = self.api.busstop.getPersonIDs(stop)
            memberships['queue'] = set(values)
            return values

        def stage(person, index):
            group = 'onboard' if person in memberships.get('onboard', ()) else 'queue'
            counts = self.stage_counts[group]
            counts['calls'] += 1
            try:
                value = self.api.person.getStage(person, index)
            except Exception:
                counts['missing'] += 1
                raise
            kind, line, destination = (getattr(value, field, None)
                for field in ('type', 'line', 'destStop'))
            counts['type_'+str(kind)] += 1
            counts['line_nonempty'] += isinstance(line, str) and bool(line.strip())
            counts['destination_nonempty'] += isinstance(destination, str) and bool(destination)
            example = dict(scope=group+'_current_stage0', stage_type=kind,
                line_type=type(line).__name__, line=line if isinstance(line, str) else None,
                destination_type=type(destination).__name__,
                destination_stop=destination if isinstance(destination, str) else None)
            if example not in self.stage_examples[group] and len(self.stage_examples[group]) < 4:
                self.stage_examples[group].append(example)
            if isinstance(line, str) and line.strip():
                samples = self.nonempty_line_examples[group]
                if example not in samples and len(samples) < 2:
                    samples.append(example)
            return value

        return SimpleNamespace(simulation=self.api.simulation,
            vehicle=SimpleNamespace(getPersonIDList=onboard,
                getPersonCapacity=self.api.vehicle.getPersonCapacity,
                getLine=self.api.vehicle.getLine),
            person=SimpleNamespace(getStage=stage),
            busstop=SimpleNamespace(getPersonIDs=queue))

    def capture(self, issued, native_event):
        issue_clock = float(issued['started_at'])
        if not self._in_window(issue_clock):
            self.audit['outside_window_issues'] += 1
            return None
        if self._terminal(native_event):
            self.audit['terminal_issue_excluded'] += 1
            return None
        identity = (str(native_event.bus_id), str(native_event.line_id), str(native_event.stop_id))
        expected = (issued['trip_id'], issued['line_id'], issued['station_id'])
        if identity != expected or identity in self.pending:
            raise RuntimeError('Native context issue identity is not unique and current')
        start_order = self._next_order()
        before = self._guard()
        stops = self.bridge.line_obj_dic[native_event.line_id].stop_id_l
        context = observe_context(self._context_api(), trip_id=native_event.bus_id,
            line_id=native_event.line_id, station_index=native_event.stop_idx,
            route_stops=stops, issue_clock=issue_clock, event_clock=float(native_event.sim_time))
        plan = self._next_stop_plan(native_event, context['next_stop'])
        after = self._guard()
        checks = self._guard_checks(before, after, 'sample')
        checks['issue_clock'] = before['clock'] == before['ledger']['time_seconds'] == issue_clock
        self.audit['issue_clock_failures'] += not checks['issue_clock']
        end_order = self._next_order()
        supported = bool(context['observed'])
        observed = supported and all(checks.values()) and plan['status'] != 'mismatch'
        context_reasons = list(context['reasons'])
        if plan['status'] == 'mismatch':
            context_reasons.append('native_next_stop_plan_mismatch')
        if not all(checks.values()):
            context_reasons.append('sampling_changed_clock_ledger_or_rng')
        self.reasons.update(context_reasons)
        self.plans[plan['status']] += 1
        for target in (self.overall, self.by_line[str(native_event.line_id)]):
            target['total'] += 1
            target['supported'] += supported
            target['observed'] += observed
            target['unsupported'] += not observed
        counts = context['counts']
        for group, field in (('onboard_positive', 'onboard'), ('nonempty_queue', 'next_queue'),
                             ('release_positive', 'release_next')):
            if counts[field] is None:
                self.membership_unknown[group] += 1
            elif counts[field] > 0:
                target = self.groups[group]
                target['total'] += 1
                target['supported'] += supported
                target['observed'] += observed
                target['unsupported'] += not observed
        self.clock_lags.append(issue_clock-float(native_event.sim_time))
        if context['clocks']['obs_start'] is not None:
            self.observation_lags.append(context['clocks']['obs_start']-issue_clock)
        capture_seq = self.overall['total']-1
        action = np.clip(np.asarray(issued['a'][0], np.float32), -1., 1.)
        hold = float(.5*(action+1.)*float(self.bridge.max_hold_seconds))
        row = dict(record='context', capture_seq=capture_seq,
            observation_start_order=start_order, observation_end_order=end_order,
            issue_sequence=int(issued['sequence']), trip_id=identity[0], line_id=identity[1],
            station_id=identity[2], issue_clock=issue_clock,
            event_clock=float(native_event.sim_time), supported=supported, observed=observed,
            reasons=context_reasons, context=context, next_stop_plan=plan, guard_checks=checks)
        self._write(row)
        if len(self.examples) < 4 or (context_reasons and not any(
                example['reasons'] == context_reasons for example in self.examples)):
            if len(self.examples) < 8:
                self.examples.append({key: value for key, value in row.items()
                    if key not in ('trip_id', 'issue_sequence', 'capture_seq')})
        if not all(checks.values()):
            raise RuntimeError('Native context sampling changed clock, ledger or RNG')
        self.pending[identity] = dict(native=native_event, capture_seq=capture_seq,
            issue_clock=issue_clock, sample_end_order=end_order, expected_hold=hold)
        return row

    def _apply(self, event, command):
        self.audit['calls_total'] += 1
        clock = float(self.bridge.passenger_time_ledger.snapshot().time_seconds)
        if not self._in_window(clock):
            self.audit['outside_window_applies'] += 1
            return self.original(event, command)
        if self._terminal(event):
            self.audit['terminal_excluded'] += 1
            return self.original(event, command)
        self.audit['window_controls'] += 1
        identity = (str(event.bus_id), str(event.line_id), str(event.stop_id))
        captured = self.pending.get(identity)
        if captured is None:
            self.audit['uncaptured'] += 1
            raise RuntimeError('Native action reached execution before its context sample')
        if captured['native'] is not event:
            self.audit['wrong_event'] += 1
            raise RuntimeError('Native action does not use the captured event object')
        hold = float(np.asarray(command).reshape(-1)[0])
        if hold != captured['expected_hold']:
            self.audit['holding_mismatch'] += 1
            raise RuntimeError('Observed execution holding differs from issued action')
        if clock != captured['issue_clock']:
            self.audit['apply_clock_mismatch'] += 1
            raise RuntimeError('Native time advanced between issue observation and execution')
        apply_order = self._next_order()
        if not captured['sample_end_order'] < apply_order:
            self.audit['order_failure'] += 1
            raise RuntimeError('Context observation did not finish before action execution')
        before = self._guard()
        result = self.original(event, command)
        after = self._guard()
        checks = self._guard_checks(before, after, 'apply')
        self.audit['applied'] += 1
        self.audit['exact_event'] += 1
        self.audit['holding_exact'] += 1
        self.audit['capture_before_apply'] += 1
        self._write(dict(record='apply_audit', capture_seq=captured['capture_seq'],
            effect_seq=self.audit['calls_total']-1, apply_start_order=apply_order,
            apply_end_order=self._next_order(), sample_end_order=captured['sample_end_order'],
            holding_seconds=hold, guard_checks=checks,
            apply_clock_before=before['clock'], apply_clock_after=after['clock']))
        del self.pending[identity]
        if not all(checks.values()):
            raise RuntimeError('Native action application changed clock, ledger or RNG')
        return result

    def summary(self):
        def coverage(counter):
            return {key: counter[key] for key in ('total', 'supported', 'observed', 'unsupported')}
        checks = dict(rng_checked=self.state_probe is not None,
            captures_executed=self.overall['total'] == self.audit['applied'] and not self.pending,
            capture_before_actual_apply=self.audit['uncaptured'] == self.audit['order_failure'] == 0,
            exact_native_event=self.audit['wrong_event'] == 0,
            holding_unchanged=self.audit['holding_mismatch'] == 0,
            actual_issue_clock=self.audit['issue_clock_failures'] == self.audit['apply_clock_mismatch'] == 0,
            native_next_stop_scored_matches=self.plans['mismatch'] == 0)
        for scope in ('sample', 'apply'):
            for field in ('clock', 'ledger', 'rng'):
                checks[scope+'_'+field+'_unchanged'] = self.audit[scope+'_'+field+'_failures'] == 0
        overall = coverage(self.overall)
        overall['terminal'] = self.audit['terminal_excluded']
        return dict(all_pass=all(checks.values()), checks=checks, window=list(self.window),
            right_open=True, captures=overall, apply_audit=dict(self.audit),
            coverage=dict(overall=overall,
                by_line={line: coverage(value) for line, value in sorted(self.by_line.items())},
                subgroups={key: coverage(value) for key, value in self.groups.items()},
                membership_unknown={key: self.membership_unknown[key] for key in self.groups}),
            reasons=dict(self.reasons), clocks=dict(issue_minus_event=statistics(self.clock_lags),
                observation_minus_issue=statistics(self.observation_lags)),
            stage_semantics={group: dict(counts=dict(self.stage_counts[group]),
                examples=self.stage_examples[group],
                nonempty_line_examples=self.nonempty_line_examples[group])
                for group in ('onboard', 'queue')},
            next_stop_plan=dict(self.plans), examples=self.examples,
            interpretation='Actual native stage0 observations only. Stop plans are known intent; '
                'current-only or absent plans do not fabricate future service or invalidate fixed-route identity.')

    def close(self):
        if self.closed:
            return
        self.writer.close()
        if self.had_instance_action:
            self.bridge.apply_action = self.saved_instance_action
        else:
            del self.bridge.apply_action
        self.closed = True
