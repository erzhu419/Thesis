"""Read existing packet diagnostics; do not reconstruct a policy by SGD.

The hold-probability metric is from the current sampled actor minibatch before
its actor optimizer step.  A same-row deterministic proposal only certifies
which side of the Hurdle p=0.5 boundary that state occupies.
"""
from collections import Counter
import json
import math
from pathlib import Path


METRICS = ('train/hold_probability', 'train/hybrid_gate_entropy',
           'train/positive_action_mean', 'train/executed_positive_action_mean',
           'train/executed_zero_action_mean')


def lines(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_log(record):
    packet = record['packet']
    metric = record['metrics']
    actor_before = 8*(packet-1)
    if (record['updates'] != 8*packet or
            metric['train/update'] != 50000+actor_before+1 or
            metric['train/online_critic_update'] != actor_before+1):
        raise ValueError('Packet metrics no longer describe the first of eight updates')
    return dict(packet=packet, processed_at=record['receipt']['processed_at'],
        policy_updates_before=actor_before,
        metric_update_completed=actor_before+1,
        packet_updates_completed=record['updates'],
        p_hold=metric[METRICS[0]], gate_entropy=metric[METRICS[1]],
        positive_mean=metric[METRICS[2]],
        executed_positive_mean=metric[METRICS[3]],
        executed_zero_mean=metric[METRICS[4]])


def fixed_state_crossings(scores):
    grouped = {}
    for record in scores:
        key = record['packet'], record['phase']
        if key in grouped:
            raise ValueError('Duplicate packet score phase')
        grouped[key] = record
    detail, counts = [], Counter()
    packets = sorted({packet for packet, _ in grouped})
    for packet in packets:
        before = grouped[(packet, 'packet_before')]
        after = grouped[(packet, 'packet_after')]
        if len(before['rows']) != len(after['rows']):
            raise ValueError('Before and after scores have different rows')
        for low, high in zip(before['rows'], after['rows']):
            identity = ('sequence', 'line_id', 'issue', 'context_observed_at')
            if any(low[key] != high[key] for key in identity):
                raise ValueError('Cannot interpret different scored states as a gate crossing')
            a, b = low['proposal_seconds'], high['proposal_seconds']
            if a < 0 or b < 0:
                raise ValueError('Hurdle proposal seconds cannot be negative')
            was_positive, is_positive = a > 0., b > 0.
            counts['same_state_comparisons'] += 1
            counts['positive_before'] += was_positive
            counts['positive_after'] += is_positive
            if was_positive != is_positive:
                direction = 'positive_to_zero' if was_positive else 'zero_to_positive'
                counts[direction] += 1
                detail.append(dict(packet=packet, processed_at=before['processed_at'],
                    policy_update_interval=[8*(packet-1), 8*packet],
                    line_id=low['line_id'], sequence=low['sequence'],
                    issue=low['issue'], direction=direction))
    counts['zero_to_positive'] += 0
    counts['positive_to_zero'] += 0
    # Existing logs establish threshold crossings, not probabilities or an
    # interpolation of unsaved policy weights.
    return dict(counts=dict(counts), first_crossing=detail[0] if detail else None,
        first24_crossings=[row for row in detail if row['packet'] <= 3],
        by_line={line: dict(Counter(row['direction'] for row in detail if row['line_id']==line))
                 for line in sorted({row['line_id'] for row in detail})},
        all_crossing_packets=sorted({row['packet'] for row in detail}),
        probability_values_recoverable=False)


def summarize_model(root, *, expected_packets=57):
    root = Path(root)
    source = lines(root/'packets.jsonl')
    if [row['packet'] for row in source] != list(range(1, expected_packets+1)):
        raise ValueError('Missing or reordered source packets')
    rows = [compact_log(record) for record in source]
    groups = {}
    for row in rows:
        clock = row['processed_at']
        label = 'first24' if row['packet'] <= 3 else 'before9000' if clock < 9000 else \
            '9000-12000' if clock < 12000 else '12000-18000' if clock < 18000 else 'tail'
        groups.setdefault(label, []).append(row)
    windows = {}
    for label, values in groups.items():
        windows[label] = dict(packets=len(values),
            policy_update_before_range=[values[0]['policy_updates_before'], values[-1]['policy_updates_before']],
            processed_at_range=[values[0]['processed_at'], values[-1]['processed_at']],
            p_hold_mean=math.fsum(v['p_hold'] for v in values)/len(values),
            p_hold_min=min(v['p_hold'] for v in values),
            p_hold_max=max(v['p_hold'] for v in values),
            gate_entropy_mean=math.fsum(v['gate_entropy'] for v in values)/len(values))
    thresholds = {str(p): next((row for row in rows if row['p_hold'] >= p), None)
                  for p in (.5, .8, .9)}
    picks = {1, 2, 3, expected_packets}
    picks.update(row['packet'] for row in thresholds.values() if row is not None)
    scores = lines(root/'scores.jsonl')
    if {record['packet'] for record in scores} != set(range(1, expected_packets+1)):
        raise ValueError('Scores do not cover the source packet calendar')
    return dict(source=str(root), packets=len(rows), updates=source[-1]['updates'],
        warmup=[row for row in rows if row['packet'] <= 3],
        milestones=[row for row in rows if row['packet'] in picks],
        sampled_minibatch_windows=windows, sampled_minibatch_first_threshold=thresholds,
        fixed_state_gate_threshold=fixed_state_crossings(scores),
        logged_metric_semantics='current_actor_minibatch_before_current_actor_optimizer_step',
        fixed_state_probability_series_available=False,
        intermediate_weights_saved=False,
        missing='No per-state gate/mean/std or intermediate actor checkpoint; no within-first24 fixed-state probability trajectory.',
        sampled_mean_is_fixed_state_drift=False,
        new_updates=0, new_native_seconds=0, new_feedback=0)
