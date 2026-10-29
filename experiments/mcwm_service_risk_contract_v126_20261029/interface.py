"""Factual binary targets from the original window; no model or native imports."""
import math


def overrun_target(record):
    """Return a new target without changing the frozen departure-time label."""
    selection, action = record['selection'], record['action']
    labels = record['labels']
    departure = labels['receiver_departure']
    start, end = selection['window_start'], selection['window_end']
    feedback, packet = selection['feedback_available_at'], selection['packet_deadline']
    if end != start + 60 or feedback != end + 30 or packet < feedback:
        raise ValueError('Expected the original 60-second window and delayed packet')
    deadline = math.ceil(action['applied_at'] + action['hold_seconds'])
    value, censored = departure['value'], departure['right_censored']
    observed_until = departure.get('observed_until')
    available = labels['available_at']
    if available > end or (observed_until is not None and observed_until > end):
        raise ValueError('Only evidence from the original window is admissible')
    target = dict(value=None, clock_deadline=deadline, window_end=end,
                  feedback_available_at=feedback, packet_deadline=packet,
                  departure_time_censored=censored, reason='incomplete_observation')
    if value is not None:
        if censored or not start < value <= min(end, available):
            raise ValueError('Departure must be an observed event within its window')
        if observed_until is not None and value > observed_until:
            raise ValueError('Departure lies beyond observed coverage')
        target.update(value=int(value > deadline), reason='observed_departure')
    elif not censored:
        raise ValueError('Absent departure must retain its original censoring flag')
    elif observed_until == end and available == end:
        if deadline <= end:
            target.update(value=1, reason='still_at_stop_through_clock_deadline')
        else:
            target['reason'] = 'clock_deadline_after_window'
    return target


def can_release(target, issue_time):
    """Known labels remain unavailable until strictly after their original packet."""
    return target['value'] is not None and target['packet_deadline'] < issue_time


def risk_inputs(record, observation):
    """Join one passive observation to its exact root; return only raw n and h."""
    fields = ('source_group', 'sequence', 'observed_at', 'receiver', 'station')
    for field in fields:
        if record['context']['meta'][field] != observation['meta'][field]:
            raise ValueError('Receiver observation root mismatch: ' + field)
    receiver = observation['receiver']
    count = receiver['current_leg_dest_count']
    if not receiver['current_stage0_complete'] or count is None:
        return None
    return [count, record['action']['hold_seconds']]
