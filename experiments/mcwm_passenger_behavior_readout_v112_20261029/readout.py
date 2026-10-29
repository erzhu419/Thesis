"""Reduce v111 resident ledgers without importing a learner or simulator."""
import json
from pathlib import Path
import sys

N = 15079
BOUNDARIES = (0, 9000, 12000, 18300, 24000)
FIELDS = ('accrued_wait_seconds', 'accrued_in_vehicle_seconds')


def interval(fresh, control, start, end):
    differences = [(fresh[end][key]-fresh[start][key])
                   -(control[end][key]-control[start][key]) for key in FIELDS]
    wait, ride = differences
    return dict(start=start, end=end, wait_person_seconds=wait, ride_person_seconds=ride,
        weighted_wait_per_departed=2*wait/N, ride_per_departed=ride/N,
        cost_per_departed=(2*wait+ride)/N,
        time_mean_waiting_count_difference=wait/(end-start),
        time_mean_onboard_count_difference=ride/(end-start))


def analyze(base):
    timelines, summaries, scenes = {}, {}, {}
    for scene in ('stable', 'local_speed_drop'):
        for arm in ('fresh', 'no_fresh'):
            folder = base/'runs'/scene/arm
            summary = json.loads((folder/'summary.json').read_text())
            records = json.loads((folder/'metrics.json').read_text())
            if summary['passenger_denominator'] != N:
                raise ValueError('Cannot compare a different passenger cohort')
            table = {row['time_seconds']: row for row in records}
            table[0] = {key: 0 for key in FIELDS}
            table[24000] = summary['ledger']
            timelines[scene+'/'+arm] = table
            summaries[scene+'/'+arm] = summary
        a, b = (timelines[scene+'/'+arm] for arm in ('fresh', 'no_fresh'))
        if set(a) != set(b) or any(t not in a for t in BOUNDARIES):
            raise ValueError('No exact paired ledger anchors')
        times = sorted(a)
        intervals = [interval(a, b, start, end) for start, end in zip(times, times[1:])]
        full = interval(a, b, 0, 24000)
        # These identities detect wrong fields or dropped bins in the reduction.
        for key in ('wait_person_seconds', 'ride_person_seconds'):
            if sum(row[key] for row in intervals) != full[key]:
                raise ValueError('Interval costs do not sum to the full-day ledger')
        positive = [row for row in intervals if row['ride_person_seconds'] > 0]
        negative = [row for row in intervals if row['ride_person_seconds'] < 0]
        cumulative = [dict(time=t, ride_person_seconds=a[t][FIELDS[1]]-b[t][FIELDS[1]],
            wait_person_seconds=a[t][FIELDS[0]]-b[t][FIELDS[0]]) for t in times]
        first_ride = next((row for row in cumulative if row['ride_person_seconds'] != 0), None)
        first_onboard = next((dict(time=t, fresh=a[t]['in_vehicle_passengers'],
            no_fresh=b[t]['in_vehicle_passengers']) for t in times if t and
            a[t]['in_vehicle_passengers'] != b[t]['in_vehicle_passengers']), None)
        p, q = (summaries[scene+'/'+arm]['passengers'] for arm in ('fresh', 'no_fresh'))
        journey = {key: p['completed_journey_seconds'][key]-q['completed_journey_seconds'][key]
                   for key in ('mean', 'p50', 'p90', 'p95')}
        scenes[scene] = dict(full_day=full,
            segments=[interval(a, b, start, end) for start, end in zip(BOUNDARIES, BOUNDARIES[1:])],
            first_nonzero_cumulative_ride=first_ride, first_observed_onboard_count_difference=first_onboard,
            cumulative_ride_peak=max(cumulative, key=lambda row: row['ride_person_seconds']),
            cumulative_ride_minimum=min(cumulative, key=lambda row: row['ride_person_seconds']),
            ledger_intervals=len(intervals), positive_ride_intervals=len(positive),
            negative_ride_intervals=len(negative), zero_ride_intervals=len(intervals)-len(positive)-len(negative),
            gross_ride_increases_person_seconds=sum(row['ride_person_seconds'] for row in positive),
            gross_ride_decreases_person_seconds=sum(row['ride_person_seconds'] for row in negative),
            largest_ride_increase_intervals=sorted(positive, key=lambda row: row['ride_person_seconds'], reverse=True)[:5],
            largest_ride_decrease_intervals=sorted(negative, key=lambda row: row['ride_person_seconds'])[:3],
            completed_passengers=dict(fresh=p['completed_passengers'], no_fresh=q['completed_passengers']),
            completed_journey_distribution_difference_seconds=journey,
            action_issue_counts={arm: summaries[scene+'/'+arm]['action_audit']['issues'] for arm in ('fresh', 'no_fresh')})
    shared = {}
    for arm in ('fresh', 'no_fresh'):
        a, b = (timelines[scene+'/'+arm] for scene in ('stable', 'local_speed_drop'))
        times = [t for t in sorted(a) if t <= 9000]
        shared[arm] = dict(points=len(times), equal=all(a[t] == b[t] for t in times))
    sample = summaries['stable/fresh']
    shared_output = Path(sample['runtime']['sumo_root'])/'online_control/output_file'
    return dict(status='complete', new_native_seconds=0, new_optimizer_updates=0,
        denominator=N, comparator='v111 fresh minus v111 matched no_fresh', scenes=scenes,
        shared_pre_shock_ledgers=shared,
        shared_output_inventory=dict(directory=str(shared_output), exists=shared_output.exists(),
            files=[dict(name=p.name, bytes=p.stat().st_size) for p in shared_output.iterdir() if p.is_file()]
            if shared_output.exists() else []),
        interpretation='Ride counts all time onboard, regardless of holding, service or motion. '
        'Time-mean count differences come from exact ledger increments, not sampled endpoint averages. '
        'Interval rankings are descriptive only. Completed-cohort distributions are not same-passenger effects.')


if __name__ == '__main__':
    result = analyze(Path(sys.argv[1]))
    raw = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(raw.encode()) > 18000:
        raise ValueError('Reduce output before transfer')
    print(raw)
