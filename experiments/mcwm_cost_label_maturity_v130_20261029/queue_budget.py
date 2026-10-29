"""Hypothetical bounded FIFO delivery of fixed receiver-ride labels only."""


def replay_queue(roots):
    """Return (summary, deliveries), without changing roots or their labels.

    Source counts include empty cohorts. Backlog is measured after each packet
    delivers at most five already-mature roots; values never choose a root.
    """
    roots = list(roots)
    sequences = [r['selection']['sequence'] for r in roots]
    if len(sequences) != len(set(sequences)):
        raise ValueError('Each original root may enter the queue only once')
    if len({r['meta']['source_group'] for r in roots}) > 1:
        raise ValueError('Queue replay is confined to one scenario')
    remaining, deliveries = list(roots), []
    max_backlog = max_ready = backlog = 0
    for deadline in range(1200, 18001, 300):
        ready = [r for r in remaining
                 if r['receiver_ride']['maturity']['earliest_extended_packet'] is not None
                 and r['receiver_ride']['maturity']['earliest_extended_packet'] <= deadline
                 and r['selection']['packet_deadline'] <= deadline]
        ready.sort(key=lambda r: (r['selection']['packet_deadline'],
                                  r['selection']['window_start'], r['selection']['sequence']))
        chosen = ready[:5]
        max_ready = max(max_ready, len(ready))
        backlog = len(ready) - len(chosen)
        max_backlog = max(max_backlog, backlog)
        used = {r['selection']['sequence'] for r in chosen}
        remaining = [r for r in remaining if r['selection']['sequence'] not in used]
        for root in chosen:
            selection, label = root['selection'], root['receiver_ride']
            deliveries.append(dict(sequence=selection['sequence'],
                original_packet=selection['packet_deadline'], assigned_packet=deadline,
                issue_time=selection['window_start'], nonempty=label['maturity']['nonempty'],
                value=label['value'], closed_at=label['maturity']['closed_at']))

    maturity = [r['receiver_ride']['maturity'] for r in roots]
    last_issue = max((r['selection']['window_start'] for r in roots), default=None)
    has_later = lambda d: last_issue is not None and last_issue > d['assigned_packet']
    summary = dict(total=len(roots), delivered=len(deliveries),
        nonempty=sum(m['nonempty'] for m in maturity),
        empty=sum(not m['nonempty'] for m in maturity),
        original60=sum(m['eligible_original_packet'] for m in maturity),
        original60_nonempty=sum(m['eligible_original_packet'] and m['nonempty'] for m in maturity),
        nonemptydelivered=sum(d['nonempty'] for d in deliveries),
        emptydelivered=sum(not d['nonempty'] for d in deliveries),
        delivered_after_original_packet=sum(d['assigned_packet'] > d['original_packet'] for d in deliveries),
        init_by9000=sum(d['assigned_packet'] <= 9000 for d in deliveries),
        init_nonempty_by9000=sum(d['assigned_packet'] <= 9000 and d['nonempty'] for d in deliveries),
        max_backlog=max_backlog, end_backlog=backlog, max_ready_before_delivery=max_ready,
        unavailable_at18000=sum(m['earliest_extended_packet'] is None or
            m['earliest_extended_packet'] > 18000 for m in maturity),
        final18000deliveries=sum(d['assigned_packet'] == 18000 for d in deliveries),
        deliveries_with_later_selected_issue=sum(has_later(d) for d in deliveries),
        nonempty_deliveries_with_later_selected_issue=sum(d['nonempty'] and has_later(d) for d in deliveries),
        hypothetical_only=True, original_selection_and_labels_unchanged=True)
    return summary, deliveries
