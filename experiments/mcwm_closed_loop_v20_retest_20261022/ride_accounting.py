"""Resident paired ride ledger accounting from native passenger exchanges."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    main = read(root/'service_summary.json')
    assert main['integrity']['all_pass']
    rows = []
    for case in main['cases']:
        line, selected = case['line_id'], case['selected']
        grouped, final_stock, exchanges = defaultdict(float), 0, {}
        stops = {}
        for arm, sign in (('low', 1), ('high', -1)):
            branch = root/'branches/native'/line/arm
            summary = read(branch/'summary.json')
            assert summary['integrity']['all_pass']
            assert not summary['service']['missing_counts']
            events = read(branch/'service_events.json')['events']
            exchanges[arm] = 0
            for event in events:
                bus = event.get('bus_id')
                if event['kind'] in ('board', 'alight'):
                    count = event['passenger_count']
                    assert count == len(event['passenger_ids'])
                    direction = 1 if event['kind'] == 'board' else -1
                    group = 'controlled_trip' if bus == selected['trip_id'] else (
                        'other_target_line' if event['line_id'] == line else 'other_lines')
                    grouped[group] += sign*direction*count*(300-event['relative_seconds'])
                    final_stock += sign*direction*count
                    exchanges[arm] += 1
                if bus == selected['trip_id'] and event['kind'] in ('arrival', 'departure', 'issue'):
                    stop = event['station_id']
                    entry = stops.setdefault(stop, dict(first=event['clock_seconds'], low={}, high={}))
                    entry['first'] = min(entry['first'], event['clock_seconds'])
                    value = event['clock_seconds']-selected['started_at']
                    if event['kind'] == 'issue':
                        value = [value, 30*(event['action'][0]+1)]
                    entry[arm][event['kind']] = value
        actual = case['clock_decomposition']['final_gain']['ride']
        total = sum(grouped.values())
        rows.append(dict(line=line, ride_gain_by_group=dict(grouped), reconstructed=total,
            ledger=actual, error=total-actual, exchange_events=exchanges,
            actual_cutoff_onboard_low_minus_high=final_stock,
            trip=[[stop]+[stops[stop][arm].get(kind) for kind in ('arrival', 'departure', 'issue')
                for arm in ('low', 'high')] for stop in sorted(stops, key=lambda s:stops[s]['first'])]))
    return dict(status='complete', exact=all(r['error']==0 for r in rows), cases=rows,
        schema=dict(trip_fields=['stop','arrival_low','arrival_high','departure_low','departure_high','issue_low','issue_high'],
            clock='All trip clocks are relative to the fixed fork. Issue is [relative seconds, executed holding].',
            accounting='Same-prefix initial onboard differences are zero. Integrate signed native exchanges count*(300-relative event time), board positive/alight negative, then low minus high. Event clocks match set_counts before advance_to of the next tick. No waiting-cost attribution.',
            scope='Paired ride accounting by observed vehicle group; a complete ledger identity requires exact=true. Actual cutoff stock can differ from the last ledger inventory, observed one tick earlier.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.root)
    data = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(data.encode()) <= 5000
    with args.out.open('x') as output:
        output.write(data)
    print(json.dumps(dict(status=value['status'], exact=value['exact'], bytes=len(data.encode()))))
