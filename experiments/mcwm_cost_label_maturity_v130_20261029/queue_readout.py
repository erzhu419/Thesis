"""Resident hypothetical budget readout; no model fitting or actual delivery."""
import json
from pathlib import Path
from queue_budget import replay_queue

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / 'queue_readout'
    output.mkdir(exist_ok=False)
    cases, initial = {}, {}
    for scene in ('stable', 'local_speed_drop'):
        source = HERE / 'readout' / (scene + '_cost_roots.jsonl')
        roots = [json.loads(line) for line in source.read_text().splitlines()]
        summary, deliveries = replay_queue(roots)
        initial[scene] = [d for d in deliveries if d['assigned_packet'] <= 9000]
        packets = [dict(deadline=t,
            delivered=sum(d['assigned_packet'] == t for d in deliveries),
            nonempty=sum(d['assigned_packet'] == t and d['nonempty'] for d in deliveries))
            for t in range(1200, 18001, 300)]
        new = [d for d in deliveries if d['assigned_packet'] > 9000]
        nonempty_new = [d for d in new if d['nonempty']]
        issues = [r['selection']['window_start'] for r in roots
                  if r['selection']['packet_deadline'] > 9000]
        summary.update(post9000_deliveries=len(new),
            post9000_nonempty_deliveries=len(nonempty_new),
            post9000_packets_with_nonempty=sum(p['deadline'] > 9000 and p['nonempty'] > 0 for p in packets),
            suffix_roots=len(issues),
            suffix_roots_after_new_nonempty_delivery=sum(
                any(d['assigned_packet'] < t for d in nonempty_new) for t in issues),
            packets_with_unused_capacity=sum(p['delivered'] < 5 for p in packets),
            unused_capacity=285-len(deliveries),
            earliest_nonempty_delivery=min((d['assigned_packet'] for d in deliveries if d['nonempty']), default=None))
        cases[scene] = summary
        (output / (scene + '_deliveries.jsonl')).write_text(''.join(json.dumps(d) + '\n' for d in deliveries))
        (output / (scene + '_packets.json')).write_text(json.dumps(packets, indent=2) + '\n')
    shared = initial['stable'] == initial['local_speed_drop']
    result = dict(status='complete' if shared else 'HOLD_DIFFERENT_INITIALIZATION',
        hypothetical_only=True, cases=cases,
        initialization=dict(exact_equal=shared,
            total=len(initial['stable']), nonempty=sum(d['nonempty'] for d in initial['stable']),
            source='Natural closure plus30, original packet floor, then FIFO capacity; delivered by9000.'),
        actual_admissions=0, actual_fits=0, new_native_seconds=0,
        old_selection_labels_and_control_unchanged=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
