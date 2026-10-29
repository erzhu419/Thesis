"""The frozen v126 chronological schedule, without reading or fitting targets."""


def causal_events(roots, cutoff=9000):
    """Return one scene's initialization, past-packet releases and predictions.

    The caller validates panel counts and shares the common-prefix initialization
    once across scenes. Extra root fields, including targets, are never accessed.
    """
    rows = []
    groups, sequences = set(), set()
    for root in roots:
        meta, selection = root['meta'], root['selection']
        sequence, issue = meta['sequence'], meta['observed_at']
        deadline = selection['packet_deadline']
        if sequence != selection['sequence'] or issue != selection['window_start']:
            raise ValueError('Root identity and selection clock disagree')
        if sequence in sequences:
            raise ValueError('A root appears more than once')
        if deadline < issue:
            raise ValueError('A packet cannot precede its own issue')
        sequences.add(sequence)
        groups.add(meta['source_group'])
        rows.append((issue, sequence, deadline))
    if len(groups) != 1:
        raise ValueError('Supply exactly one nonempty source group')
    rows.sort(key=lambda row: row[:2])
    prefix = [sequence for issue, sequence, deadline in rows if deadline <= cutoff]
    suffix = [row for row in rows if row[2] > cutoff]
    if not prefix:
        raise ValueError('The common-prefix initialization is missing')
    if any(issue <= cutoff for issue, _, _ in suffix):
        raise ValueError('Formal predictions must be strictly after the cutoff')

    events = [dict(kind='initialize', sequences=prefix)]
    packets = {}
    for _, sequence, deadline in suffix:
        packets.setdefault(deadline, []).append(sequence)
    deadlines = sorted(packets)
    released = 0
    for issue, sequence, _ in suffix:
        while released < len(deadlines) and deadlines[released] < issue:
            deadline = deadlines[released]
            events.append(dict(kind='release', packet_deadline=deadline,
                               sequences=packets[deadline]))
            released += 1
        events.append(dict(kind='predict', sequence=sequence, issue_time=issue))
    return events
