"""Causal factual departure calibration; no simulator or controller imports."""
from collections import defaultdict
import math
from statistics import median


def residual(row):
    label = row['labels']['receiver_departure']
    if label['right_censored']:
        if label['value'] is not None:
            raise ValueError('Censored departure must retain an unknown point target')
        return None
    action = row['action']
    return label['value'] - (action['applied_at'] + action['hold_seconds'])


def forecast_scene(records, initial_residuals, frozen_median, cutoff=9000):
    values = list(initial_residuals)
    groups = defaultdict(list)
    for row in records:
        deadline = row['selection']['packet_deadline']
        if deadline > cutoff:
            groups[deadline].append(row)
    deadlines = sorted(groups)
    predictions, updates = [], []
    cursor, current_median, latest_deadline = 0, frozen_median, cutoff

    def release(deadline, used_by_prediction):
        nonlocal current_median, latest_deadline
        packet = groups[deadline]
        new = [value for row in packet if (value := residual(row)) is not None]
        values.extend(new)
        previous = current_median
        if new:
            current_median = median(values)
        latest_deadline = deadline
        updates.append(dict(deadline=deadline, sequences=[r['selection']['sequence'] for r in packet],
            mature_added=len(new), censored=len(packet)-len(new), calibration_records=len(values),
            median_residual=current_median, median_changed=current_median != previous,
            recalculated=bool(new), used_by_later_prediction=used_by_prediction))

    for row in sorted(records, key=lambda r: (r['selection']['window_start'], r['selection']['sequence'])):
        issue = row['selection']['window_start']
        if issue <= cutoff:
            continue
        while cursor < len(deadlines) and deadlines[cursor] < issue:
            release(deadlines[cursor], True)
            cursor += 1
        action = row['action']
        nominal = action['applied_at'] + action['hold_seconds']
        # No access to this root's target before its predictions are constructed.
        predictions.append(dict(sequence=row['selection']['sequence'], issue_time=issue,
            predictions=dict(frozen=nominal+frozen_median, fresh=nominal+current_median,
                clock_reference=math.ceil(nominal), raw_command_reference=nominal),
            median_residual=current_median, calibration_records=len(values),
            latest_feedback_deadline=latest_deadline))
    for deadline in deadlines[cursor:]:
        release(deadline, False)
    stats = dict(initial_records=len(initial_residuals), initial_median=frozen_median,
        final_records=len(values), final_median=current_median,
        packet_recalculations=sum(u['recalculated'] for u in updates),
        recalculations_used_by_predictions=sum(u['recalculated'] and u['used_by_later_prediction'] for u in updates),
        recalculations_after_last_prediction=sum(u['recalculated'] and not u['used_by_later_prediction'] for u in updates),
        median_changes=sum(u['median_changed'] for u in updates),
        added_mature_labels=sum(u['mature_added'] for u in updates),
        censored_feedback_slots=sum(u['censored'] for u in updates),
        prediction_median_min=min(p['median_residual'] for p in predictions),
        prediction_median_max=max(p['median_residual'] for p in predictions))
    return predictions, updates, stats
