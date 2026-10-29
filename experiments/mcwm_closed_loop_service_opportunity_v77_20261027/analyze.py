"""Resident service-opportunity readout. No simulator, learner or weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_service_segments_v74_20261027.core import load_services
from core import compare_services
from opportunities import waiting_opportunities

EXPERIMENTS = HERE.parent
V75 = EXPERIMENTS/'mcwm_closed_loop_frozen_validation_v75_20261027'
V76 = EXPERIMENTS/'mcwm_closed_loop_mid_policy_exchange_v76_20261027'
LO, HI, DENOMINATOR = 12000, 16200, 15079


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Small readout exceeds48KB: '+str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(raw)


def source(seed, arm):
    return V75/f'runs/executed_cql/validation/seed{seed}/stable/{arm}'


def replay_source(seed, arm):
    return V76/f'runs/seed{seed}/{arm}'


def issues(path):
    keys = ('trip_id', 'vehicle_id', 'line_id', 'station_id', 'started_at', 'a')
    result = []
    for raw in Path(path).open():
        row = json.loads(raw)
        result.append({key: row[key] for key in keys})
    return result


def service_replay(original, replay_path):
    """Compare the same service identity, excluding passive writer ordering."""
    fields = ('bus_id', 'line_id', 'station_id', 'kind')
    expected = {tuple(row[key] for key in fields)+(row['occurrence'],): row
                for row in original if LO <= row['observed_at'] < HI}
    observed, occurrences = {}, Counter()
    for raw in replay_path.open():
        row = json.loads(raw)
        if row.get('record') != 'service':
            continue
        base = tuple(row[key] for key in fields)
        occurrence = occurrences[base]
        occurrences[base] += 1
        if LO <= row['observed_at'] < HI:
            observed[base+(occurrence,)] = row
    mismatch, order_differences = None, 0
    excluded = {'occurrence', '_ordinal', 'observation_order'}
    for identity in sorted(expected.keys() & observed.keys()):
        a, b = expected[identity], observed[identity]
        order_differences += a['observation_order'] != b['observation_order']
        different = [key for key in (a.keys() | b.keys())-excluded if a.get(key) != b.get(key)]
        if different and mismatch is None:
            mismatch = dict(identity=identity, fields=sorted(different))
    return dict(exact=expected.keys() == observed.keys() and mismatch is None,
        v75_rows=len(expected), v76_rows=len(observed), first_difference=mismatch,
        same_service_identities=expected.keys() == observed.keys(),
        passive_writer_order_differences=order_differences,
        compared='All clocks, inventories, capacity and legacy fields by native service identity; '
            'passive observation_order excluded, physical departure ordering retained.',
        window='12000<=observed_at<16200; actual fork tick excluded')


def ledger_points(root):
    result = {}
    for raw in (root/'metrics.jsonl').open():
        row = json.loads(raw)
        clock = row['time_seconds']
        if LO <= clock <= HI:
            result[int(clock)] = row
    if set(result) != set(range(LO, HI+1, 300)):
        raise ValueError('The real300-second ledger endpoints are incomplete')
    return result


def difference(a, b, start, end):
    def cost(points, key):
        return points[end][key]-points[start][key]
    wait = cost(a, 'accrued_wait_seconds')-cost(b, 'accrued_wait_seconds')
    ride = cost(a, 'accrued_in_vehicle_seconds')-cost(b, 'accrued_in_vehicle_seconds')
    return dict(start=start, end=end, delta_wait_person_seconds=wait,
        delta_ride_person_seconds=ride, delta_weighted_wait_per_passenger=2*wait/DENOMINATOR,
        delta_ride_per_passenger=ride/DENOMINATOR,
        delta_total_per_passenger=(2*wait+ride)/DENOMINATOR,
        end_waiting_count_difference=a[end]['waiting_passengers']-b[end]['waiting_passengers'])


def analyze_seed(seed):
    services, commands, metadata, points = {}, {}, {}, {}
    for arm in ('fresh', 'no_fresh'):
        root, replay = source(seed, arm), replay_source(seed, arm)
        summary, prefix = read(root/'summary.json'), read(replay/'prefix_summary.json')
        if summary['status'] != 'complete' or prefix['status'] != 'complete' \
                or not summary['integrity']['all_pass'] or not prefix['integrity']['all_pass']:
            raise ValueError('Existing source or exact replay failed; cannot identify service differences')
        rows, service_audit = load_services([root/'service_trace.jsonl'])
        exact = service_replay(rows, replay/'service_trace.jsonl')
        if not exact['exact']:
            raise ValueError('The original and restored service observations differ: '+str(exact))
        services[arm] = rows
        commands[arm] = issues(root/'prefix_trace.jsonl')
        points[arm] = ledger_points(root)
        metadata[arm] = dict(root=str(root), replay_root=str(replay),
            service_replay=exact, service_audit=service_audit,
            source_integrity=True, prefix_integrity=True,
            prefix_trace_rows=prefix['trace_rows'], matched_prefix_rows=prefix['matched_prefix_rows'],
            prefix_update_rows_compared=prefix['prefix_update_rows_compared'])
    a, b = points['fresh'], points['no_fresh']
    costs = dict(metric='fresh minus no_fresh; negative cost is better', denominator=DENOMINATOR,
        full_window=difference(a, b, LO, HI),
        bins=[difference(a, b, time, time+300) for time in range(LO, HI, 300)],
        cumulative=[dict(time=t,
            weighted_wait=2*(a[t]['accrued_wait_seconds']-b[t]['accrued_wait_seconds'])/DENOMINATOR,
            ride=(a[t]['accrued_in_vehicle_seconds']-b[t]['accrued_in_vehicle_seconds'])/DENOMINATOR,
            total=(2*(a[t]['accrued_wait_seconds']-b[t]['accrued_wait_seconds'])+
                a[t]['accrued_in_vehicle_seconds']-b[t]['accrued_in_vehicle_seconds'])/DENOMINATOR,
            waiting_count_difference=a[t]['waiting_passengers']-b[t]['waiting_passengers'])
            for t in range(LO, HI+1, 300)])
    service = compare_services(services['fresh'], services['no_fresh'],
        commands['fresh'], commands['no_fresh'], lo=LO, hi=HI)
    opportunities = waiting_opportunities(services['fresh'], services['no_fresh'], lo=LO, hi=HI)
    result = dict(version=77, seed=seed, scenario='stable', status='complete',
        cost=costs, service=service, opportunities=opportunities,
        new_simulated_seconds=0, new_training_updates=0, new_feedback_slots=0,
        old_code_and_results_modified=False)
    evidence = dict(seed=seed, sources=metadata,
        semantic_limits=[
            'Actual holding commands are distinct from physical dwell and service-ready time.',
            'Legacy board/alight/stranded dictionaries are not physical passenger-flow totals; null is missing.',
            'Service-ready inventory is sampled at observed_at, not backdated to a lagged history clock.',
            'Queue membership does not imply individual line eligibility; no final-stage metadata backfill.',
            'Service endpoint inventories cannot partition the exact global waiting integral.',
            'This is retrospective association on exposed trajectories, not a new causal intervention.'])
    return result, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    args = parser.parse_args()
    evidence, overview = [], []
    for seed in (6208, 6207):
        result, proof = analyze_seed(seed)
        write(args.out/f'seed{seed}.json', result)
        evidence.append(proof)
        overview.append(dict(seed=seed, full_window=result['cost']['full_window'],
            result=str(args.out/f'seed{seed}.json')))
        print(json.dumps(dict(seed=seed, status='complete', cost=result['cost']['full_window'])), flush=True)
    write(args.out/'evidence.json', dict(version=77, status='complete', entries=evidence,
        overview=overview, priority='high', new_simulated_seconds=0, new_training_updates=0,
        new_feedback_slots=0, old_code_and_results_modified=False))


if __name__ == '__main__':
    main()
