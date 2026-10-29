"""Pure waiting-ledger and maturity readouts; no selection or model updates."""
from collections import defaultdict
import math


def waiting_intervals(events, horizon=24000):
    """Build every person's waiting intervals from sorted membership changes."""
    intervals, active = defaultdict(list), {}
    for event in events:
        clock = event['time']
        if clock > horizon:
            break
        for person in event['left']:
            if person not in active:
                raise ValueError('Waiting exit without an entry: ' + str(person))
            intervals[person].append([active.pop(person), clock])
        for person in event['entered']:
            if person in active:
                raise ValueError('Repeated waiting entry: ' + str(person))
            active[person] = clock
    for person, start in active.items():
        intervals[person].append([start, horizon])
    return dict(intervals)


def sum_overlap(intervals, start, end):
    """Sum waiting seconds inside [start,end] without changing the ledger."""
    return sum(max(0, min(right, end) - max(left, start))
               for left, right in intervals)


def maturity(root_selection, closed_at, nonempty, later_issue_times, cutoff=9000):
    """Describe original eligibility and hypothetical delayed release timing.

    Empty cohorts can close at issue with zero cost, but are not cost evidence.
    An extended packet is descriptive only and never changes the selected data.
    """
    end = root_selection['window_end']
    packet = root_selection['packet_deadline']
    within = closed_at is not None and closed_at <= end
    eligible = within and closed_at + 30 <= packet
    extended = None
    if closed_at is not None and closed_at + 30 <= 18000:
        extended = max(packet, 1200 + 300 * max(0, math.ceil((closed_at + 30 - 1200) / 300)))
    return dict(
        closed_at=closed_at, nonempty=bool(nonempty), empty_cohort=not nonempty,
        evidence=bool(nonempty) and closed_at is not None,
        original60_end=end, original_packet=packet,
        within_original_window=within, eligible_original_packet=eligible,
        common_initialization_available=eligible and packet <= cutoff,
        earliest_extended_packet=extended,
        later_issue_count=None if extended is None else
            sum(issue > extended for issue in later_issue_times),
        extension_is_counterfactual_only=True)
