"""Resident factual passenger-cost and maturity audit; no model or simulator."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from core import waiting_intervals, sum_overlap, maturity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HORIZON = 24000


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def ride_intervals(events, horizon=HORIZON):
    active, intervals, boards = {}, [], defaultdict(list)
    for event in events:
        clock, trip = event['time'], event['trip_id']
        for person in event['alighted_ids']:
            start = active.pop((person, trip))
            intervals.append(dict(person=person, trip=trip, start=start, end=clock,
                closed=True, exit_at_stop=event['observed_at_stop'], stop=event['stop_id']))
        for person in event['boarded_ids']:
            if (person, trip) in active:
                raise ValueError('Repeated onboard entry')
            active[person, trip] = clock
            boards[person].append(clock)
    for (person,trip),start in active.items():
        intervals.append(dict(person=person, trip=trip, start=start, end=horizon,
            closed=False, exit_at_stop=False, stop=None))
    for clocks in boards.values():
        clocks.sort()
    return intervals, boards


def receiver_label(current, members, selection, issues):
    issue = selection['window_start']
    cost = sum(r['end']-issue for r in current)
    closed = max((r['end'] for r in current), default=issue) if all(r['closed'] for r in current) else None
    return dict(person_count=len(members), value=cost if closed is not None else None,
        observed_cost=cost, censored_people=sum(not r['closed'] for r in current), unknown_people=0,
        off_stop_membership_exits=sum(r['closed'] and not r['exit_at_stop'] for r in current),
        maturity=maturity(selection,closed,bool(members),issues))


def feeder_label(record, by_pair, boards, waiting, passengers, issues):
    selection, context = record['selection'], record['context']
    issue, station = selection['window_start'], context['meta']['station']
    people, details = set(), []
    for feeder in context['feeders']:
        for person in feeder['cohort_ids']:
            if person in people:
                raise ValueError('Root feeder cohorts overlap')
            people.add(person)
            current = [r for r in by_pair.get((person,feeder['trip_id']),[]) if r['start']<=issue<r['end']]
            if len(current)!=1:
                raise ValueError('Current feeder membership is not in the original exchange ledger')
            ride = current[0]
            if not ride['closed'] or not ride['exit_at_stop'] or ride['stop']!=station:
                details.append(dict(person=person,value=None,observed_cost=None,closed_at=None,
                    reason='target_station_alighting_not_observed'))
                continue
            alight = ride['end']
            next_board = next((t for t in boards.get(person,[]) if t>=alight),None)
            completion = passengers[person]['completion_time']
            if completion is not None and completion<alight:
                raise ValueError('Completion precedes current-leg alighting')
            ends = [t for t in (next_board,completion) if t is not None]
            end = min(ends) if ends else None
            cost = sum_overlap(waiting.get(person,[]),alight,HORIZON if end is None else end)
            details.append(dict(person=person,value=cost if end is not None else None,
                observed_cost=cost,closed_at=end,alighted_at=alight,
                elapsed_to_close=None if end is None else end-alight,
                reason='observed_boarding_or_completion' if end is not None else 'still_unclosed_at_horizon'))
    inputs_complete = record['cohort_definition_complete'] is True
    closed = max((d['closed_at'] for d in details),default=issue) if inputs_complete and all(d['closed_at'] is not None for d in details) else None
    observed = sum(d['observed_cost'] for d in details if d['observed_cost'] is not None)
    return dict(person_count=len(people),value=observed if closed is not None else None,
        observed_cost=observed,cohort_inputs_complete=inputs_complete,
        unknown_people=sum(d['reason']=='target_station_alighting_not_observed' for d in details),
        censored_people=sum(d['reason']=='still_unclosed_at_horizon' for d in details),
        closed_people=sum(d['closed_at'] is not None for d in details),
        closed_with_positive_wait=sum(d['value'] is not None and d['value']>0 for d in details),
        elapsed_is_not_wait_people=sum(d['closed_at'] is not None and d['elapsed_to_close']!=d['value'] for d in details),
        maturity=maturity(selection,closed,bool(people),issues),people=details)


def summarize(labels):
    buckets = Counter()
    for r in labels:
        m = r['maturity']
        if r.get('cohort_inputs_complete',True) is not True:
            name = 'unknown_cohort_inputs'
        elif not m['nonempty']:
            name = 'empty_cohort'
        elif m['closed_at'] is None:
            name = 'unclosed_or_unknown'
        elif m['eligible_original_packet']:
            name = 'original60_complete'
        elif m['closed_at']+30 <= m['original_packet']:
            name = 'beyond60_but_before_original_packet'
        elif m['earliest_extended_packet'] is not None:
            name = 'later_packet_before18000'
        else:
            name = 'after_feedback_end'
        buckets[name] += 1
    nonempty = [r for r in labels if r['maturity']['nonempty']]
    completed = [r for r in nonempty if r['value'] is not None]
    return dict(roots=len(labels),nonempty_roots=len(nonempty),buckets=dict(buckets),
        unknown_cohort_input_roots=sum(r.get('cohort_inputs_complete',True) is not True for r in labels),
        original60_nonempty=sum(r['maturity']['eligible_original_packet'] for r in nonempty),
        original60_nonempty_positive=sum(r['maturity']['eligible_original_packet'] and r['value']>0 for r in nonempty),
        potentially_released_before18000=sum(r['maturity']['earliest_extended_packet'] is not None for r in nonempty),
        potentially_released_with_later_root=sum(r['maturity']['later_issue_count'] is not None and r['maturity']['later_issue_count']>0 for r in nonempty),
        closed_nonempty_roots=len(completed),closed_positive_roots=sum(r['value']>0 for r in completed),
        closed_cost_seconds=dict(minimum=min((r['value'] for r in completed),default=None),
            maximum=max((r['value'] for r in completed),default=None)),
        person_memberships=sum(r['person_count'] for r in labels),
        unknown_person_memberships=sum(r['unknown_people'] for r in labels),
        censored_person_memberships=sum(r['censored_people'] for r in labels),
        off_stop_membership_exits=sum(r.get('off_stop_membership_exits',0) for r in labels),
        elapsed_is_not_wait_person_memberships=sum(r.get('elapsed_is_not_wait_people',0) for r in labels))


def scene_audit(scene, out):
    source = ROOT/'experiments/mcwm_service_replay_v120_20261029'
    qualified = json.loads((source/'review'/scene/'summary.json').read_text())
    if qualified['status']!='complete' or not qualified['integrity']['all_pass']:
        raise ValueError('Original native logs are not qualified')
    base = source/'runs'/scene
    passengers = {r['passenger_id']:r for r in rows(base/'passengers.jsonl')}
    waiting = waiting_intervals(rows(base/'waiting_changes.jsonl'))
    rides, boards = ride_intervals(rows(base/'exchanges.jsonl'))
    person_ride, by_trip, by_pair = Counter(), defaultdict(list), defaultdict(list)
    for r in rides:
        person_ride[r['person']] += r['end']-r['start']
        by_trip[r['trip']].append(r)
        by_pair[r['person'],r['trip']].append(r)
    gaps = {pid:dict(wait=sum_overlap(waiting.get(pid,[]),0,HORIZON)-r['wait'],
                     ride=person_ride[pid]-r['ride']) for pid,r in passengers.items()}
    bad = {pid:g for pid,g in gaps.items() if g['wait'] or g['ride']}
    if bad or (set(waiting)|set(person_ride))-set(passengers):
        raise ValueError('Reconstructed memberships do not match the original passenger ledger')
    selected = list(rows(ROOT/'experiments/mcwm_service_opportunity_v121_20261029/runs'/scene/'selected_records.jsonl'))
    packets = Counter(r['selection']['packet_deadline'] for r in selected)
    if len(selected)!=285 or len(packets)!=57 or set(packets.values())!={5}:
        raise ValueError('Original selected-root budget differs')
    issues = sorted(r['selection']['window_start'] for r in selected)
    all_roots = []
    for record in selected:
        meta, selection = record['context']['meta'], record['selection']
        issue, trip = meta['observed_at'], meta['receiver']
        capture = json.loads((ROOT/'experiments/mcwm_full_receiver_observation_v127_20261029/runs'/scene/'captures'/(str(meta['sequence'])+'.json')).read_text())
        observation = capture['observation']
        if observation['meta']!=meta or not all(capture['passivity'].values()):
            raise ValueError('Receiver observation binding differs')
        members = {p['person_id'] for p in observation['people']}
        current = [r for r in by_trip[trip] if r['start']<=issue<r['end']]
        if {r['person'] for r in current}!=members or len(members)!=record['context']['receiver']['onboard']:
            raise ValueError('Root receiver membership differs from recorded getters')
        all_roots.append(dict(meta=meta,selection=selection,
            receiver_ride=receiver_label(current,members,selection,issues),
            feeder_wait=feeder_label(record,by_pair,boards,waiting,passengers,issues)))
    with (out/(scene+'_cost_roots.jsonl')).open('x') as stream:
        for row in all_roots:
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    result = dict(roots=len(all_roots),packets=len(packets),passenger_ledger_exact=True,
        passengers_checked=len(passengers),receiver_membership_exact_roots=len(all_roots))
    for phase,subset in (('all',all_roots),('original_prefix',[r for r in all_roots if r['selection']['packet_deadline']<=9000]),
                         ('original_suffix',[r for r in all_roots if r['selection']['packet_deadline']>9000])):
        result[phase] = {name:summarize([r[name] for r in subset]) for name in ('receiver_ride','feeder_wait')}
    return result, all_roots


def main():
    started = time.monotonic()
    out = HERE/'readout'
    out.mkdir(exist_ok=False)
    cases, roots = {}, {}
    for scene in ('stable','local_speed_drop'):
        cases[scene], roots[scene] = scene_audit(scene,out)
    prefix = {scene:{r['meta']['sequence']:r for r in values if r['selection']['packet_deadline']<=9000}
              for scene,values in roots.items()}
    if prefix['stable'].keys()!=prefix['local_speed_drop'].keys() or len(prefix['stable'])!=135:
        raise ValueError('Original shared-prefix roots differ')
    comparison = {}
    for name in ('receiver_ride','feeder_wait'):
        exact = eligible = different = crossing = unknown = 0
        for seq,a in prefix['stable'].items():
            b = prefix['local_speed_drop'][seq]
            x,y = a[name],b[name]
            known = x['value'] is not None and y['value'] is not None
            both_unknown = x['value'] is None and y['value'] is None
            same = known and (x['value'],x['maturity']['closed_at'])==(y['value'],y['maturity']['closed_at'])
            exact += same
            unknown += both_unknown
            different += not same and not both_unknown
            crossing += any(z['maturity']['nonempty'] and (z['maturity']['closed_at'] is None or z['maturity']['closed_at']>9000) for z in (x,y))
            if x['maturity']['nonempty'] and x['maturity']['common_initialization_available'] and y['maturity']['common_initialization_available']:
                eligible += 1
                if not same:
                    raise ValueError('Legally available common-prefix cost labels differ')
        comparison[name] = dict(original_common_roots=135,same_known_full_value_and_close=exact,
            both_unknown_full_values=unknown,different_value_or_close_or_one_unknown=different,
            nonempty_not_closed_by9000_or_unknown=crossing,
            legally_available_nonempty_common_roots=eligible)
    result = dict(status='complete',cases=cases,shared_prefix_costs=comparison,
        elapsed_seconds=time.monotonic()-started,new_native_seconds=0,new_fits=0,training_admissions=0,
        changed_original_labels=0,control_benefit_evidence=False,
        interpretation='Root cohorts overlap; their costs cannot be summed as network cost or counterfactual action effect. Extended packets are timing diagnostics only, without allocation into the5-slot budget.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    raw = json.dumps(result,separators=(',',':'),allow_nan=False)
    if len(raw.encode())+1>16000:
        raise ValueError('Summary exceeds the small-transfer bound')
    print(raw)


if __name__=='__main__':
    main()
