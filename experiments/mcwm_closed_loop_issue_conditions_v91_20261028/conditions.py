"""Strictly historical shared-corridor service slowdown from native issue logs."""
from __future__ import annotations

from collections import Counter
import math
from statistics import mean, median

LINES = ('7X', '102X', '705X')
UPSTREAM = '7X13_102X26_705X27'
DOWNSTREAM = '7X14_102X27_705X28'
BASELINE_BEFORE = 9000.
MIN_BASELINE = 3
MAX_AGE = 900.
SCALE_SECONDS = 60.
MAX_HOLD_SECONDS = 60.
SEMANTICS = ('Decision-ready issue interval minus the already-executed origin holding command; '
             'includes next-stop service and native/issue discretization, not pure departure-to-arrival runtime')


class CorridorHistory:
    def __init__(self, event_index):
        self.event_index = dict(event_index)
        self.previous = {}
        self.intervals = []
        self.counts = Counter()

    def observe(self, row):
        """Ingest factual parent/reference issues only; no label field is read."""
        self.counts['observed_issues'] += 1
        line = row['line_id']
        if line not in LINES:
            return
        point = dict(line_id=line, trip_id=row['trip_id'], station_id=row['station_id'],
            sequence=int(row['sequence']), issue_clock=float(row['started_at']),
            event_clock=float(row['event'][self.event_index['sim_time_seconds']]),
            station_index=float(row['event'][self.event_index['station_index']]),
            action=float(row['a'][0]))
        key = line, point['trip_id']
        previous = self.previous.get(key)
        self.previous[key] = point
        if previous is None or previous['station_id'] != UPSTREAM or point['station_id'] != DOWNSTREAM:
            return
        if point['station_index'] != previous['station_index']+1:
            self.counts['nonadjacent_corridor_issues'] += 1
            return
        duration = point['issue_clock']-previous['issue_clock']
        hold = MAX_HOLD_SECONDS*(previous['action']+1.)/2.
        nonholding = duration-hold
        if (not all(math.isfinite(value) for value in (duration, hold, nonholding))
                or duration <= 0 or not 0 <= hold <= MAX_HOLD_SECONDS or nonholding < 0
                or previous['event_clock'] > previous['issue_clock']
                or point['event_clock'] > point['issue_clock']):
            self.counts['invalid_corridor_intervals'] += 1
            return
        self.intervals.append(dict(line_id=line, trip_id=point['trip_id'],
            start_sequence=previous['sequence'], end_sequence=point['sequence'],
            started_at=previous['issue_clock'], completed_at=point['issue_clock'],
            observed_at=point['issue_clock'], start_event_clock=previous['event_clock'],
            end_event_clock=point['event_clock'], commanded_hold_seconds=hold,
            ready_interval_seconds=duration, nonholding_ready_seconds=nonholding))
        self.counts['corridor_intervals'] += 1

    def snapshot(self, issue_clock):
        """Exclude all same-clock completions, regardless of ingestion order."""
        now = float(issue_clock)
        past = [row for row in self.intervals if row['completed_at'] < now and row['observed_at'] < now]
        baseline = {line: [row for row in past if row['line_id'] == line
            and row['completed_at'] < BASELINE_BEFORE and row['observed_at'] < BASELINE_BEFORE]
            for line in LINES}
        centers = {line: median(row['nonholding_ready_seconds'] for row in values) if values else None
            for line, values in baseline.items()}
        newest = max((row['completed_at'] for row in past), default=None)
        latest = sorted([row for row in past if row['completed_at'] == newest],
            key=lambda row: (row['line_id'], row['trip_id'], row['end_sequence']))
        age = now-newest if newest is not None else None
        reason = None
        if any(len(values) < MIN_BASELINE for values in baseline.values()):
            reason = 'baseline_coverage_missing'
        elif newest is None:
            reason = 'corridor_history_missing'
        elif age > MAX_AGE:
            reason = 'latest_corridor_interval_stale'
        sources = [dict(row, baseline_nonholding_seconds=centers[row['line_id']],
            normalized_slowdown=(row['nonholding_ready_seconds']-centers[row['line_id']])/SCALE_SECONDS
                if centers[row['line_id']] is not None else None) for row in latest]
        value = mean(row['normalized_slowdown'] for row in sources) if reason is None else None
        used = [row for values in baseline.values() for row in values]+latest
        return dict(status='supported' if reason is None else 'unsupported', reason=reason, value=value,
            historical_only=all(row['completed_at'] < now and row['observed_at'] < now for row in used),
            issue_clock=now, baseline_before=BASELINE_BEFORE,
            baseline_counts={line: len(values) for line, values in baseline.items()},
            baseline_medians_seconds=centers,
            baseline_latest_completed_at={line: max((row['completed_at'] for row in values), default=None)
                for line, values in baseline.items()},
            baseline_latest_observed_at={line: max((row['observed_at'] for row in values), default=None)
                for line, values in baseline.items()},
            latest_completed_at=newest, latest_observed_at=max((row['observed_at'] for row in latest), default=None),
            latest_age_seconds=age, latest_sources=sources, semantics=SEMANTICS)

    def summary(self):
        return dict(corridor=dict(upstream=UPSTREAM, downstream=DOWNSTREAM), counts=dict(self.counts),
            intervals_by_line={line: sum(row['line_id'] == line for row in self.intervals) for line in LINES},
            baseline_intervals_by_line={line: sum(row['line_id'] == line and row['completed_at'] < BASELINE_BEFORE
                and row['observed_at'] < BASELINE_BEFORE for row in self.intervals) for line in LINES},
            minimum_baseline_intervals=MIN_BASELINE, maximum_latest_age_seconds=MAX_AGE,
            normalization_seconds=SCALE_SECONDS, semantics=SEMANTICS)
