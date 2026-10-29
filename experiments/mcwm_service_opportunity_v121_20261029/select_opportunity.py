"""One declared pre-action opportunity substitution per original feedback packet."""
from collections import Counter, defaultdict
import math


def opportunity(context):
    return any(feeder['x']['feeder_next_stop_matches'] is True
               and feeder['x']['current_leg_dest_count'] is not None
               and feeder['x']['current_leg_dest_count'] > 0
               for feeder in context['feeders'])


def select_packets(contexts, original_packets, protocol):
    """No commands, costs, outcomes or person identities participate in selection."""
    first = protocol['feedback_first_deadline']
    last = protocol['feedback_last_deadline']
    interval = protocol['feedback_interval_seconds']
    limit = protocol['feedback_records_per_packet']
    latest, latest_opportunity, counts = defaultdict(list), {}, Counter()
    for context in contexts:
        meta = context['meta']
        available = meta['observed_at']+60+protocol['feedback_delay_seconds']
        deadline = first+max(0, math.ceil((available-first)/interval))*interval
        if deadline > last:
            continue
        item = (available, meta['sequence'], context)
        counts[deadline] += 1
        latest[deadline].append(item)
        latest[deadline].sort(key=lambda row: row[:2])
        latest[deadline] = latest[deadline][-limit:]
        if opportunity(context) and (deadline not in latest_opportunity
                or item[:2] > latest_opportunity[deadline][:2]):
            latest_opportunity[deadline] = item
    originals = {row['deadline']: row for row in original_packets}
    deadlines = list(range(first, last+1, interval))
    if set(originals) != set(deadlines):
        raise ValueError('Original packet schedule differs')
    plans, selected = [], {}
    for deadline in deadlines:
        base = latest[deadline]
        base_ids = [row[1] for row in base]
        original = originals[deadline]
        if base_ids != original['selected_sequences'] or len(base) != limit:
            raise ValueError('Original latest5 does not reproduce at '+str(deadline))
        changed, chosen = None, base
        if any(opportunity(row[2]) for row in base):
            reason = 'original_contains_opportunity'
        elif deadline not in latest_opportunity:
            reason = 'no_eligible_opportunity'
        else:
            replacement = latest_opportunity[deadline]
            chosen = sorted([*base[1:], replacement], key=lambda row: row[:2])
            changed = dict(removed_sequence=base[0][1], added_sequence=replacement[1])
            reason = 'one_opportunity_substitution'
        ids = [row[1] for row in chosen]
        plans.append(dict(deadline=deadline,
            release_lower_exclusive=original['release_lower_exclusive'],
            release_upper_inclusive=deadline, candidate_records=counts[deadline],
            original_sequences=base_ids, selected_sequences=ids,
            selected_records=len(ids), reason=reason, substitution=changed))
        for available, sequence, context in chosen:
            if sequence in selected:
                raise ValueError('Root selected in multiple packets')
            selected[sequence] = (deadline, available, context)
    if len(selected) > protocol['target_record_budget']:
        raise ValueError('Selection exceeded the unchanged root-action budget')
    return plans, selected
