"""Bind the complete frozen root manifest to its original records."""


def bind_selection(manifest, records, case):
    ordered = sorted(manifest, key=lambda r: (r['selection']['window_start'], r['selection']['sequence']))
    selected = {r['selection']['sequence']: r for r in records}
    if len(ordered) != case['roots'] or len(selected) != len(records) or len(records) != case['roots']:
        raise ValueError('Expected every frozen selected root exactly once')
    packets = {r['selection']['packet_deadline'] for r in ordered}
    if len(packets) != case['packets']:
        raise ValueError('Frozen packet count differs')
    targets, contexts = {}, {}
    for root in ordered:
        meta, selection = root['meta'], root['selection']
        seq = selection['sequence']
        row = selected.get(seq)
        if row is None or row['context']['meta'] != meta or row['selection'] != selection:
            raise ValueError('Selection or root identity differs from v126 manifest')
        if seq in targets or meta['sequence'] != seq or meta['observed_at'] != selection['window_start']:
            raise ValueError('Duplicate root or inconsistent issue clock')
        targets[seq] = dict(sequence=seq, receiver=meta['receiver'], station=meta['station'],
                            issue_clock=meta['observed_at'])
        contexts[seq] = row['context']
    if (ordered[-1]['meta'] != case['final_root'] or
            ordered[-1]['meta']['observed_at'] != case['stop_before_apply_clock'] or
            ordered[-1]['meta']['sequence'] != case['stop_before_apply_sequence']):
        raise ValueError('Final selected root is not the declared stop point')
    return targets, contexts
