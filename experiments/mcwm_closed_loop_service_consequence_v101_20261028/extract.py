"""Read existing T/C native logs once; no simulator, fitting or new observations."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path

LINES = ('7X', '102X', '705X')
SCENES = ('stable', 'local_speed_drop')
COHORTS = (
    ('T', 'mcwm_closed_loop_collection_domain_v97_20261028'),
    ('C', 'mcwm_closed_loop_confidence_calibration_v99_20261028'),
)
EVENT_COLUMNS = ['line', 'trip', 'station', 'station_index', 'occurrence',
    'reference_issue', 'candidate_issue', 'reference_event', 'candidate_event',
    'reference_onboard', 'candidate_onboard', 'reference_stranded', 'candidate_stranded',
    'reference_dwell', 'candidate_dwell', 'reference_action', 'candidate_action']


def read(path):
    return json.loads(path.read_text())


def cost(wait, ride):
    return [wait, ride, 2*wait+ride]


def difference(a, b):
    return [x-y for x,y in zip(a,b)]


def ledger(value):
    return dict(cost=cost(value['accrued_wait_seconds'],value['accrued_in_vehicle_seconds']),
        counts=[value.get('waiting_passengers'),value.get('in_vehicle_passengers')])


def endpoint(summary):
    initial, remaining = summary['root']['issue_ledger'], summary['remaining']
    return cost(initial['accrued_wait_seconds']+remaining['wait'],
                initial['accrued_in_vehicle_seconds']+remaining['ride'])


class Trace:
    def __init__(self):
        self.events = {}
        self.ledgers = {}
        self.visits = Counter()
        self.rows = 0
        self.first_sequence = self.last_sequence = None
        self.last_issue = None
        self.conflicts = 0
        self.first_conflict = None

    def conflict(self, reason):
        self.conflicts += 1
        if self.first_conflict is None:
            self.first_conflict = reason

    def add_ledger(self, now, value):
        old = self.ledgers.get(now)
        if old is not None:
            if old['cost'] != value['cost']:
                self.conflict(dict(kind='same_clock_cost_conflict',time=now))
            if all(x is not None for x in old['counts']+value['counts']) and old['counts'] != value['counts']:
                self.conflict(dict(kind='same_clock_count_conflict',time=now))
            if any(x is None for x in old['counts']) and all(x is not None for x in value['counts']):
                old['counts'] = value['counts']
        else:
            self.ledgers[now] = value

    def add(self, row):
        sequence = row['sequence']
        if self.last_sequence is not None and sequence != self.last_sequence+1:
            self.conflict(dict(kind='nonconsecutive_sequence',previous=self.last_sequence,sequence=sequence))
        if self.first_sequence is None:
            self.first_sequence = sequence
        self.last_sequence = sequence
        self.last_issue = row['started_at']
        self.rows += 1
        point = row['issue_ledger']
        if point['time_seconds'] != row['started_at']:
            self.conflict(dict(kind='issue_ledger_clock_mismatch',sequence=sequence))
        self.add_ledger(point['time_seconds'],ledger(point))
        event = row['event']
        route_index = int(event[1])
        base = (row['trip_id'],row['line_id'],row['station_id'],route_index)
        self.visits[base] += 1
        key = (*base,self.visits[base])
        self.events[key] = dict(sequence=sequence,issue=row['started_at'],event_time=event[2],
            onboard=event[8],stranded=event[7],dwell=event[10],action=row['a'][0])

    def finish(self, summary):
        initial = summary['root']['issue_ledger']
        self.add_ledger(initial['time_seconds'],ledger(initial))
        tail = summary.get('tail_boundary')
        if tail is not None:
            if tail['time_seconds'] != 18300.:
                self.conflict(dict(kind='nonexact_saved_tail',time=tail['time_seconds']))
            else:
                self.add_ledger(18300.,ledger(tail))
        # These summaries retain final cumulative costs, not final passenger counts.
        self.add_ledger(24000.,dict(cost=endpoint(summary),counts=[None,None]))


def trace_rows(path, trace):
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            trace.add(row)
            yield row


def reference_group(paths, summaries):
    """Compare all raw fields at common reference sequence numbers while reading."""
    traces = [Trace() for _ in paths]
    iterators = [trace_rows(path,trace) for path,trace in zip(paths,traces)]
    heads = [next(it,None) for it in iterators]
    start = max(summary['root']['sequence'] for summary in summaries)
    for i in range(len(heads)):
        while heads[i] is not None and heads[i]['sequence'] < start:
            heads[i] = next(iterators[i],None)
    common = unequal = 0
    unshared = [0]*len(paths)
    first_difference = None
    last_common = None
    while any(row is not None for row in heads):
        active = [row for row in heads if row is not None]
        maximum = max(row['sequence'] for row in active)
        if len(active) == len(heads) and all(row['sequence'] == maximum for row in heads):
            common += 1
            last_common = maximum
            fields = sorted({key for row in heads[1:] for key in heads[0].keys() | row.keys()
                             if heads[0].get(key) != row.get(key) or (key in heads[0]) != (key in row)})
            if fields:
                unequal += 1
                if first_difference is None:
                    first_difference = dict(sequence=maximum,fields=fields)
            heads = [next(it,None) for it in iterators]
        else:
            for i,row in enumerate(heads):
                if row is not None and (len(active) < len(heads) or row['sequence'] < maximum):
                    unshared[i] += 1
                    heads[i] = next(iterators[i],None)
    for trace,summary in zip(traces,summaries):
        trace.finish(summary)
    endpoints = [endpoint(summary) for summary in summaries]
    same_endpoint = all(value == endpoints[0] for value in endpoints)
    result = dict(all_pass=common > 0 and unequal == 0 and not any(unshared)
        and same_endpoint and all(trace.conflicts == 0 for trace in traces),
        common_start_sequence=start,last_common_sequence=last_common,compared_rows=common,
        unequal_rows=unequal,unshared_tail_rows=dict(zip(LINES,unshared)),
        trace_extents=[[trace.rows,trace.first_sequence,trace.last_sequence,trace.last_issue] for trace in traces],
        first_difference=first_difference,reference_endpoints=endpoints,same_endpoint=same_endpoint)
    return traces,result


def event_row(key, ref, cand):
    trip,line,station,index,occurrence = key
    return [line,trip,station,index,occurrence,ref['issue'],cand['issue'],
        ref['event_time'],cand['event_time'],ref['onboard'],cand['onboard'],
        ref['stranded'],cand['stranded'],ref['dwell'],cand['dwell'],ref['action'],cand['action']]


def event_order(item):
    key,ref,cand = item
    return (min(ref['issue'],cand['issue']),max(ref['issue'],cand['issue']),key)


def changed(ref, cand):
    return any(ref[key] != cand[key] for key in ('issue','onboard','stranded'))


def match_events(reference, candidate, root):
    ref, cand = reference.events, candidate.events
    common = sorted(((key,ref[key],cand[key]) for key in ref.keys() & cand.keys()),key=event_order)
    nonroot = [item for item in common if not (item[1]['sequence'] == root['sequence']
                                               and item[2]['sequence'] == root['sequence'])]
    root_vehicle = [item for item in nonroot if item[0][0] == root['trip_id'] and item[0][1] == root['line_id']]
    first = lambda rows:next((event_row(*item) for item in rows if changed(item[1],item[2])),None)
    by_line = {}
    for line in LINES:
        rows = [item for item in nonroot if item[0][1] == line]
        shifts = [c['issue']-r['issue'] for _,r,c in rows]
        onboard = [c['onboard']-r['onboard'] for _,r,c in rows]
        stranded = [c['stranded']-r['stranded'] for _,r,c in rows]
        first_time = next(([r['issue'],c['issue']] for _,r,c in rows if r['issue'] != c['issue']),None)
        first_counts = next(([r['issue'],c['issue']] for _,r,c in rows
            if r['onboard'] != c['onboard'] or r['stranded'] != c['stranded']),None)
        by_line[line] = dict(matched=len(rows),missing_reference=sum(key[1] == line for key in cand.keys()-ref.keys()),
            missing_candidate=sum(key[1] == line for key in ref.keys()-cand.keys()),
            issue_shift=[sum(shifts),sum(shifts)/len(shifts) if shifts else None,
                min(shifts) if shifts else None,max(shifts) if shifts else None,sum(x != 0 for x in shifts)],
            onboard=[sum(onboard),sum(x != 0 for x in onboard)],
            stranded=[sum(stranded),sum(x != 0 for x in stranded)],
            first_time=first_time,first_counts=first_counts,
            action_changes=sum(r['action'] != c['action'] for _,r,c in rows),
            first_action=next(([r['issue'],c['issue'],r['action'],c['action']] for _,r,c in rows
                               if r['action'] != c['action']),None))
    return dict(by_line=by_line,root_next_three=[event_row(*item) for item in root_vehicle[:3]],
        root_vehicle_matched_events=len(root_vehicle),
        first_root_vehicle_change=first(root_vehicle),first_network_change=first(nonroot))


def decompose(reference, candidate, root_time):
    times = sorted(reference.ledgers.keys() & candidate.ledgers.keys())
    times = [time for time in times if time >= root_time]
    def delta(time):
        r,c = reference.ledgers[time],candidate.ledgers[time]
        counts = difference(c['counts'],r['counts']) if all(x is not None for x in c['counts']+r['counts']) else None
        return difference(c['cost'],r['cost']),counts
    first = next((time for time in times if delta(time)[0] != [0.,0.,0.] or delta(time)[1] not in (None,[0.,0.])),None)
    first_cost = next((time for time in times if delta(time)[0] != [0.,0.,0.]),None)
    requested = [root_time+offset for offset in (300.,900.,1800.,3600.)]+[18300.,20100.,24000.]
    rows,previous = [],[0.,0.,0.]
    for target in requested:
        index = bisect_left(times,target)
        observed = times[index] if index < len(times) and times[index] <= target+60. else None
        if observed is None:
            rows.append([target,None,None,None,None,None])
            continue
        cumulative,counts = delta(observed)
        rows.append([target,observed,observed-target,cumulative,difference(cumulative,previous),counts])
        previous = cumulative
    return dict(common_clocks=len(times),first_common_divergence=first,first_common_cost_divergence=first_cost,
        anchors=rows,interval_sum=[sum(row[4][i] for row in rows if row[4] is not None) for i in range(3)])


def summarize_pair(reference, candidate, low, high):
    root = low['root']
    total = cost(high['remaining']['wait']-low['remaining']['wait'],high['remaining']['ride']-low['remaining']['ride'])
    windows = decompose(reference,candidate,root['started_at'])
    events = match_events(reference,candidate,root)
    physical = [low['executed_hold_seconds'],high['executed_hold_seconds'],
        low['physical']['departure']['clock_seconds'],high['physical']['departure']['clock_seconds'],
        low['physical']['next_same_vehicle_arrival']['clock_seconds'],high['physical']['next_same_vehicle_arrival']['clock_seconds']]
    checks = dict(source_complete=all(s['status'] == 'complete' and s['integrity']['all_pass'] for s in (low,high)),
        same_root=low['root'] == high['root'],
        real_terminal=all(s['final_clock'] == s['remaining']['observed_at'] == 24000. for s in (low,high)),
        root_present_in_trace=reference.first_sequence == candidate.first_sequence == root['sequence']
            and reference.rows > 0 and candidate.rows > 0,
        trace_consistent=reference.conflicts == candidate.conflicts == 0,
        exact_decomposition=windows['interval_sum'] == total)
    windows.pop('interval_sum')
    return dict(line=root['line_id'],root=[root['started_at'],root['sequence'],root['trip_id'],root['station_id']],
        all_pass=all(checks.values()),checks=checks,D=total,physical=physical,
        trace_extents=[[trace.rows,trace.first_sequence,trace.last_sequence,trace.last_issue] for trace in (reference,candidate)],
        trace_conflicts=[reference.first_conflict,candidate.first_conflict],
        decomposition=windows,events=events)


def extract(root):
    groups = []
    for cohort,name in COHORTS:
        base = root/'experiments'/name
        protocol = read(base/'protocol.json')
        for seed in protocol['external_seeds']:
            for scene in SCENES:
                paths = [base/'runs'/str(seed)/scene/'pairs'/line for line in LINES]
                lows = [read(path/'low'/'summary.json') for path in paths]
                highs = [read(path/'high'/'summary.json') for path in paths]
                references,comparison = reference_group([path/'low'/'trace.jsonl' for path in paths],lows)
                pairs = []
                for path,reference,low,high in zip(paths,references,lows,highs):
                    candidate = Trace()
                    for _ in trace_rows(path/'high'/'trace.jsonl',candidate):
                        pass
                    candidate.finish(high)
                    pairs.append(summarize_pair(reference,candidate,low,high))
                groups.append(dict(cohort=cohort,seed=seed,scenario=scene,
                    reference_consistency=comparison,pairs=pairs))
    pair_count = sum(len(group['pairs']) for group in groups)
    passed = len(groups) == 12 and pair_count == 36 and all(group['reference_consistency']['all_pass']
        and all(pair['all_pass'] for pair in group['pairs']) for group in groups)
    return dict(status='complete' if passed else 'failed',all_pass=passed,groups=groups,
        accounting=dict(groups=len(groups),roots=pair_count,traces=2*pair_count,new_native_seconds=0,new_labels=0,new_fits=0),
        schema=dict(cost_columns=['wait_passenger_seconds','ride_passenger_seconds','generalized_passenger_seconds'],
            passenger_denominator=15079,candidate_minus_reference=True,
            anchor_columns=['requested','observed','offset','cumulative_D','D_since_previous_observed_anchor','count_delta'],
            count_columns=['system_waiting','system_onboard'],event_columns=EVENT_COLUMNS,
            root_columns=['issue_time','sequence','trip','station'],
            physical_columns=['reference_hold','candidate_hold','reference_departure','candidate_departure','reference_next_arrival','candidate_next_arrival'],
            issue_shift_columns=['sum','mean','min','max','nonzero_events'],snapshot_delta_columns=['sum','nonzero_events']),
        interpretation=dict(anchor_rule='First common exact ledger clock at or after request within60s; no interpolation. Actual observed boundaries define intervals.',
            divergence_scope='Earliest observed common ledger clock, not an unseen first physical change.',
            visit_key='trip,line,station,(event[1] station_index,occurrence within the suffix)',
            event_times='event is native decision-ready event time; issue is controller issue time; neither is an arrival timestamp.',
            snapshot_units='Recorded post-service stop-event snapshots, not unique passengers or gross boarding/alighting.',
            stranded_semantics='Current queue with line in passenger_obj.passable_line_l; not current-stage compatibility or proven capacity denial.',
            onboard_semantics='Recorded event onboard count; the source getter has an except-to-zero fallback.',
            action_semantics='Recorded executed normalized a; first_action lists reference/candidate issue times then their a values.',
            first_event_order='Earlier of the two matched issue clocks, then later clock and visit identity; root issue excluded.',
            missing_events='Unmatched observed log events, including possible unflushed EOF rows; these counts do not establish missed service.',
            trace_extents='rows,first_sequence,last_sequence,last_issue. Unshared reference tails are persisted-coverage failures, not proof of trajectory divergence; the source children exited without an explicit trace flush.',
            unavailable='No persisted full300s metrics, terminal counts, person-level wait/ride, boarding/alighting identities or realized transfer journeys.',
            limits='Query-consequence readout only; no refit, new queries, deployment decision or reopening frozen C/V gates.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Output already exists; preserve the prior readout')
    result = extract(args.root)
    raw = json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode()) > 131072:
        raise ValueError('Readout exceeds131072bytes; keep all roots and ask the owner to choose output reductions')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result['status'],bytes=len(raw.encode()),**result['accounting'])))
    return result


if __name__ == '__main__':
    main()
