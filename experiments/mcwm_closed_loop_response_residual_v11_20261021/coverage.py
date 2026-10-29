"""Two independently reserved held-out diagnostic actions per affected line."""

from collections import Counter

import numpy as np

from experiments.mcwm_closed_loop_controlled_calibration_v10_20261021.calibration import (
    ControlledCalibration, LINES,
)


class HeldoutCoverage:
    """Choose before outcomes; count only the action supplied to native execution.

    Each line owns its first two jointly executable diagnostic issues after the
    actual final update. There is no window shared with another line. An
    optional ``action_for_hold(state, seconds)`` reuses the training selector's
    existing F-inversion method; by default the same v10 method is used here.
    """

    def __init__(self, protocol, seed, transform, *, station_id, start_after,
                 lines=LINES, action_for_hold=None):
        self.transform, self.lines = transform, tuple(lines)
        self.station_id, self.start_after = str(station_id), float(start_after)
        self.latest_issue = (float(protocol['horizon_seconds'])
            - float(protocol.get('response_endpoint_seconds', 300.))
            - float(protocol['feedback_delay_seconds']))
        self._action_for_hold_callback = action_for_hold
        self.action_sequences = {}
        for index, line in enumerate(self.lines):
            # Separate generators make one line's traffic and missed issues
            # unable to change another line's prespecified low/high order.
            rng = np.random.default_rng(int(seed)
                + int(protocol.get('heldout_coverage_seed_offset', 520000))+index)
            self.action_sequences[line] = [0., 7.5] if rng.integers(2) == 0 else [7.5, 0.]
        self.line_successes = Counter()
        self.attempts, self.issued = {}, {}

    def _action_for_hold(self, state, hold):
        if self._action_for_hold_callback is not None:
            return self._action_for_hold_callback(state, hold)
        return ControlledCalibration._action_for_hold(self, state, hold)

    def propose(self, event):
        now, line = float(event['started_at']), str(event['line_id'])
        if not (self.start_after < now <= self.latest_issue):
            return None
        if line not in self.lines or str(event['station_id']) != self.station_id:
            return None
        index, sequence = self.line_successes[line], int(event['sequence'])
        if index >= 2 or sequence in self.issued:
            return None
        target = self.action_sequences[line][index]
        choices = {hold: self._action_for_hold(event['s'], hold) for hold in (0., 7.5)}
        attempt = dict(role='heldout', sequence=sequence, line_id=line,
            started_at=now, coverage_index=index, planned_hold_seconds=target)
        if any(value is None for value in choices.values()):
            actions = [self.transform.numpy(event['s'], np.asarray([u], np.float32))
                       for u in (-1., 1.)]
            holds = [float((np.asarray(action).reshape(-1)[0]+1.)*.5*self.transform.max_hold)
                     for action in actions]
            attempt.update(status='collapsed' if abs(holds[1]-holds[0]) <= 1e-5 else 'ineligible',
                minimum_executable_hold=holds[0], maximum_executable_hold=holds[1])
            self.attempts[sequence] = attempt
            return None
        proposal, action, actual = choices[target]
        self.attempts[sequence] = dict(attempt, status='proposed')
        return dict(attempt, proposal=proposal, action=action,
            assignment='low' if target == 0. else 'high', window_index=index,
            target_hold_seconds=target, executed_hold_seconds=float(actual),
            jointly_executable=True)

    def commit(self, event, decision):
        """Retain the actual hold, including an explicitly overridden fork.

        A fork runner can copy a decision and replace its action/target fields;
        ``planned_hold_seconds`` still records the common-prefix assignment.
        This keeps branch overrides separate from nominal coverage evidence.
        """
        sequence, line = int(event['sequence']), str(event['line_id'])
        if sequence in self.issued:
            raise ValueError('The same held-out issue cannot fill another coverage slot')
        if (sequence != int(decision['sequence']) or line != str(decision['line_id'])
                or decision.get('role') != 'heldout'
                or int(decision['coverage_index']) != self.line_successes[line]):
            raise ValueError('Coverage commit must match the proposed chronological issue')
        if sequence not in self.attempts or self.attempts[sequence]['status'] != 'proposed':
            raise ValueError('Only a jointly executable proposed issue can be committed')
        action = np.asarray(event['a'], np.float32).reshape(-1)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Coverage requires the actual scalar executed command')
        hold = float((action[0]+1.)*.5*self.transform.max_hold)
        expected = float(decision['executed_hold_seconds'])
        if not (abs(hold-expected) <= 1e-5 and
                any(abs(hold-target) <= 1e-5 for target in (0., 7.5))):
            raise ValueError('Logged coverage holding differs from the actual 0 / 7.5 command')
        planned = self.action_sequences[line][self.line_successes[line]]
        record = dict(role='heldout', sequence=sequence, line_id=line,
            started_at=float(event['started_at']), coverage_index=self.line_successes[line],
            planned_hold_seconds=planned, executed_hold_seconds=hold,
            assignment='low' if abs(hold) <= 1e-5 else 'high',
            branch_override=abs(hold-planned) > 1e-5, jointly_executable=True)
        self.issued[sequence] = record
        self.attempts[sequence].update(status='issued', executed_hold_seconds=hold,
            branch_override=record['branch_override'])
        self.line_successes[line] += 1
        return record

    def summary(self, now=None):
        closed = now is not None and float(now) >= self.latest_issue
        coverage = {}
        for line in self.lines:
            records = [record for record in self.issued.values() if record['line_id'] == line]
            by_action = {name: sum(record['assignment'] == name for record in records)
                         for name in ('low', 'high')}
            coverage[line] = dict(n=len(records), **by_action,
                planned_order_seconds=self.action_sequences[line],
                missing_actions=[hold for name, hold in (('low', 0.), ('high', 7.5))
                                 if not by_action[name]],
                remaining_issue_slots=2-len(records),
                branch_overrides=sum(record['branch_override'] for record in records))
        return dict(start_after=self.start_after, latest_issue=self.latest_issue,
            required_per_line=2, complete=all(row['low'] and row['high'] for row in coverage.values()),
            observation_window_closed=bool(closed), coverage=coverage,
            missing_by_line={line: row['missing_actions'] for line, row in coverage.items()
                             if row['missing_actions']},
            statuses=dict(Counter(row['status'] for row in self.attempts.values())),
            assignments=list(self.issued.values()), attempts=list(self.attempts.values()))
