"""Read retained logs on the server; emit only compact contexts and service events."""
import argparse
import json
from pathlib import Path

from features import build_context, bind_action


def load(path):
    return json.loads(path.read_text())


def window_rows(path, start, end):
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['time'] > end:
                break
            if row['time'] > start:
                yield row


def extract(root):
    exp = root/'experiments'
    result = dict(cases={}, record_schema={}, budget=dict(native_seconds=0, optimizer_updates=0))
    cases = [
        ('stable', exp/'mcwm_connection_full_day_v118_20261029/runs/stable/root_capture.json',
         exp/'mcwm_connection_full_day_v118_20261029/runs/stable', 'v114-low/stable/root18966'),
        ('local_speed_drop', exp/'mcwm_logged_action_observation_v116_20261029/runs/local_speed_drop/capture.json',
         exp/'mcwm_same_state_holding_v114_20261029/runs/local_speed_drop', 'v114-shared/shock/root14547')]
    for scene, path, run, source_group in cases:
        capture = load(path)
        if not capture['summary']['all_pass']:
            raise ValueError('Historical pre-action capture failed: '+scene)
        context = build_context(capture, source_group=source_group)
        start = context['meta']['observed_at']
        end, receiver, station = start+60, context['meta']['receiver'], context['meta']['station']
        cohorts = {f['trip_id']: set(f['cohort_ids']) for f in context['feeders']}
        arms = {}
        for arm in ('low', 'high'):
            folder = run/'branches'/arm
            observer = load(folder/'observer_summary.json')
            if observer['status'] != 'complete' or not all(observer['checks'].values()) or observer['end'] < end:
                raise ValueError('Historical observation coverage failed: '+scene+'/'+arm)
            with (folder/'commands.jsonl').open() as handle:
                command = next(json.loads(line) for line in handle
                    if (lambda r: r['trip_id'] == receiver and r['stop_id'] == station
                        and r['applied_at'] == start)(json.loads(line)))
            bound = bind_action(context, command)
            visits = [e for e in window_rows(folder/'visit_events.jsonl', start, end)
                      if e['stop_id'] == station and e['trip_id'] in {receiver, *cohorts}]
            exchanges = []
            for event in window_rows(folder/'exchanges.jsonl', start, end):
                trip = event['trip_id']
                if event['stop_id'] != station or trip not in {receiver, *cohorts}:
                    continue
                if not event['observed_at_stop']:
                    continue
                exchanges.append(dict(time=event['time'], trip_id=trip, stop_id=station,
                    boarded_ids=event['boarded_ids'] if trip == receiver else [],
                    alighted_ids=sorted(cohorts.get(trip, set()).intersection(event['alighted_ids']))))
            arms[arm] = dict(action=bound['action'], evidence=dict(start=start, end=end,
                continuous=True, visit_events=visits, exchanges=exchanges),
                sources=dict(observer=str(folder/'observer_summary.json'),
                    commands=str(folder/'commands.jsonl'), visits=str(folder/'visit_events.jsonl'),
                    exchanges=str(folder/'exchanges.jsonl')))
        result['cases'][scene] = dict(context=context, arms=arms, capture_source=str(path))
    # Inspect schema only. Do not return full action vectors or ticks.
    run = cases[0][2]
    for name in ('actions', 'ticks', 'commands', 'exchanges', 'visit_events'):
        with (run/(name+'.jsonl')).open() as handle:
            row = json.loads(next(handle))
        schema = dict(fields=sorted(row))
        if name == 'ticks':
            schema['vehicle_fields'] = sorted(row['vehicles'][0]) if row['vehicles'] else []
        result['record_schema'][name] = schema
    result['independent_root_states'] = len(result['cases'])
    result['root_action_records'] = sum(len(c['arms']) for c in result['cases'].values())
    result['usage'] = 'Exposed mechanism cases only. Repeated stable v116/v117/v118 roots are not new samples.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.thesis_root)
    raw = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(raw.encode()) > 24000:
        raise ValueError('Keep complete readout resident; compact export exceeded24KB')
    args.out.write_text(raw+'\n')
    print(raw)
