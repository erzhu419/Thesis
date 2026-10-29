"""Read resident v71 evidence; emit bounded selected fields, never real person IDs."""
import argparse
import json
from pathlib import Path


BASE = 'experiments/mcwm_closed_loop_tail_policy_exchange_v71_20261026/runs'
ARMS = ('fresh', 'no_fresh')
FIELDS = ('kind', 'bus_id', 'line_id', 'station_id', 'clock_seconds', 'observed_at',
          'waiting', 'eligible_waiting', 'onboard', 'capacity', 'free_capacity',
          'native_board_count', 'native_stranded_count')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def compact(event):
    return [event.get(key) for key in FIELDS] if event else None


def people(summary, milestone):
    point = (summary or {}).get('milestones', {}).get(milestone)
    if point is None or point.get('waiting_inventory_complete') is not True:
        return None
    return {row['passenger_id']: row for row in point['waiting_passengers']}


def group_key(row):
    lines = row.get('passable_lines')
    return row['station_id'], None if lines is None else tuple(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/home/zhengliang01/scheduleurm_work/Thesis'))
    args = parser.parse_args()
    output = dict(event_fields=FIELDS, seeds=[], audit=[], missing=[],
        observation_scope='Existing prefix and diagonal suffix service logs only; '
        'first queue/onboard samples are observations, not exact arrival/boarding times. '
        'Pre18300 inventories start16200; absent samples remain null.')
    for seed in (6201, 6203):
        directories = {arm: args.root/BASE/f'seed{seed}'/arm for arm in ARMS}
        summaries = {}
        for arm, directory in directories.items():
            branch = 'low' if arm == 'fresh' else 'high'
            summaries[arm] = read(directory/'suffixes'/branch/'service_summary.json')
            prefix = read(directory/'prefix_summary.json')
            fork = read(directory/'suffixes'/'live_fork.json')
            if prefix is None or fork is None or summaries[arm] is None:
                output['missing'].append([seed, arm, 'summary_or_fork'])
            output['audit'].append(dict(seed=seed, arm=arm,
                prefix=None if prefix is None else {key: prefix.get(key) for key in
                    ('status', 'integrity', 'prefix_clock', 'trace_rows', 'matched_prefix_rows', 'budget')},
                fork=None if fork is None else {key: fork.get(key) for key in
                    ('all_pass', 'native_before', 'native_after', 'counters_before', 'counters_after',
                     'parent_rng_unchanged', 'parent_trace_cookie_unchanged', 'overlap_seconds',
                     'children', 'route_streams')}))
        final = {arm: people(summaries[arm], '24000') for arm in ARMS}
        early = {arm: people(summaries[arm], '18300') for arm in ARMS}
        if any(final[arm] is None for arm in ARMS):
            output['missing'].append([seed, 'complete_final_inventories'])
            continue
        identities = sorted(set(final['fresh']) ^ set(final['no_fresh']))
        cases, private = [], {}
        for ordinal, identity in enumerate(identities, 1):
            owner = next(arm for arm in ARMS if identity in final[arm])
            meta = final[owner][identity]
            case = dict(case_id=f'{seed}-{ordinal:02d}', final_only_in=owner,
                station_id=meta['station_id'], passable_lines=meta.get('passable_lines'),
                departure_time=meta.get('departure_time'), original_stop=meta.get('original_stop'),
                transfer_stop=meta.get('transfer_stop'), destination_stop=meta.get('destination_stop'),
                completed_legs=meta.get('completed_legs'), paths={})
            for arm in ARMS:
                row = None if early[arm] is None else early[arm].get(identity)
                case['paths'][arm] = dict(waiting_at18300=None if early[arm] is None
                    else identity in early[arm], station_at18300=None if row is None else row['station_id'],
                    first_queue_sample=None, first_onboard_sample=None,
                    latest_compatible_departure_while_queued=None,
                    latest_pre18300_compatible_departure_while_queued=None)
            cases.append(case)
            private[identity] = case
        groups = {}
        for identity, case in private.items():
            key = (case['station_id'], None if case['passable_lines'] is None
                   else tuple(case['passable_lines']))
            group = groups.setdefault(key, dict(station_id=key[0], passable_lines=key[1],
                selected_count=dict.fromkeys(ARMS, 0), departure_times=[], last_services={}))
            group['selected_count'][case['final_only_in']] += 1
            if case['departure_time'] is not None:
                group['departure_times'].append(case['departure_time'])
        for key, group in groups.items():
            times = group.pop('departure_times')
            group['selected_departure_time_range'] = [min(times), max(times)] if times else None
            for arm in ARMS:
                found = next((row for row in summaries[arm].get('final_waiting_groups', [])
                              if group_key(row) == key), None)
                group['last_services'][arm] = None if found is None else {
                    kind: compact(event) for kind, event in found.get('last_compatible_service', {}).items()}
        for arm, directory in directories.items():
            branch = 'low' if arm == 'fresh' else 'high'
            paths = [directory/'service_trace.jsonl',
                     directory/'suffixes'/branch/'service_trace.jsonl']
            for path in paths:
                if not path.is_file():
                    output['missing'].append([seed, arm, str(path.relative_to(args.root))])
                    continue
                with path.open() as source:
                    for line in source:
                        event = json.loads(line)
                        if event.get('record') != 'service':
                            continue
                        queued = set(event.get('queued_passenger_ids') or ()) & private.keys()
                        onboard = set(event.get('onboard_passenger_ids') or ()) & private.keys()
                        for identity in queued | onboard:
                            case = private[identity]
                            row = case['paths'][arm]
                            if identity in onboard and row['first_onboard_sample'] is None:
                                row['first_onboard_sample'] = compact(event)
                            if identity not in queued or event['station_id'] != case['station_id']:
                                continue
                            if row['first_queue_sample'] is None:
                                row['first_queue_sample'] = compact(event)
                            if event['kind'] == 'departure' and case['passable_lines'] is not None \
                                    and event['line_id'] in case['passable_lines']:
                                row['latest_compatible_departure_while_queued'] = compact(event)
                                if event['observed_at'] < 18300:
                                    row['latest_pre18300_compatible_departure_while_queued'] = compact(event)
        output['seeds'].append(dict(seed=seed, final_exclusive_counts={
            arm: len(set(final[arm])-set(final[other])) for arm, other in
            (('fresh', 'no_fresh'), ('no_fresh', 'fresh'))}, groups=list(groups.values()), cases=cases))
    raw = json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    assert len(raw.encode()) <= 48000, f'Bounded output exceeded: {len(raw.encode())}'
    print(raw)


if __name__ == '__main__':
    main()
