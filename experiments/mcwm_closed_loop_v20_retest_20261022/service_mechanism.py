"""Resident same-person service associations; no simulation or fitting."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

LINES = ('7X', '102X', '705X')
GROUPS = ('controlled_trip', 'fork_stop', 'target_line', 'other_lines')
DELAY_FIELDS = ('n', 'earlier', 'same_sample', 'later', 'min', 'median', 'max',
                'mean', 'possible_min', 'possible_max', 'missing_intervals')


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    if isinstance(value,float): return int(value) if value.is_integer() else float(format(value,'.8g'))
    if isinstance(value,dict): return {k:compact(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [compact(v) for v in value]
    return value


def group(event, selected):
    if event.get('bus_id') == selected['trip_id']: return GROUPS[0]
    if event.get('station_id') == selected['station_id']: return GROUPS[1]
    return GROUPS[2] if event.get('line_id') == selected['line_id'] else GROUPS[3]


def people(events, kind):
    result = defaultdict(list)
    for e in events:
        if e.get('kind') == kind:
            for p in e.get('passenger_ids', []): result[p].append(e)
    for rows in result.values(): rows.sort(key=lambda e:e['clock_seconds'])
    return result


def delays(pairs):
    values, bounds, missing = [], [], 0
    for lo, hi in pairs:
        a, b = lo.get('person_sample_interval'), hi.get('person_sample_interval')
        if a is None or b is None:
            missing += 1
            continue
        values.append(b[1]-a[1]); bounds.append((b[0]-a[1], b[1]-a[0]))
    if not values: return [0,0,0,0,None,None,None,None,None,None,missing]
    s = sorted(values); n = len(s)
    return [n,sum(x<0 for x in s),sum(x==0 for x in s),sum(x>0 for x in s),
        s[0],(s[(n-1)//2]+s[n//2])/2,s[-1],sum(s)/n,
        min(x[0] for x in bounds),max(x[1] for x in bounds),missing]


def pair_people(low, high, selected, kind):
    lo, hi = people(low, kind), people(high, kind)
    common = lo.keys() & hi.keys(); result = {}
    for name in GROUPS:
        ids = [p for p in common if group(lo[p][0], selected) == name]
        result[name] = dict(delay=delays((lo[p][0],hi[p][0]) for p in ids), counts=[len(ids),
            sum(group(lo[p][0],selected)==name for p in lo.keys()-hi.keys()),
            sum(group(hi[p][0],selected)==name for p in hi.keys()-lo.keys()),
            sum(group(hi[p][0],selected)!=name for p in ids)])
        if name=='controlled_trip' and ids:
            p=min(ids)
            result[name]['one_interval_example']=[[e.get('bus_id'),e.get('station_id'),e.get('person_sample_interval')]
                for e in (lo[p][0],hi[p][0])]
    return result, lo, hi


def repeats(events):
    b, a = people(events, 'board'), people(events, 'alight')
    transfer = [p for p in b if any(x['bus_id'] != y['bus_id'] and
        x['clock_seconds'] <= y['clock_seconds'] for x in a.get(p,[]) for y in b[p])]
    return dict(repeat_board=sum(len(v)>1 for v in b.values()),
        repeat_alight=sum(len(v)>1 for v in a.values()),
        multiple_board_vehicles=sum(len({r['bus_id'] for r in v})>1 for v in b.values()),
        observed_transfer_pattern=len(transfer),
        transfer_with_same_sample=sum(any(x['bus_id'] != y['bus_id'] and
            x['clock_seconds']==y['clock_seconds'] for x in a.get(p,[]) for y in b[p]) for p in transfer))


def capacities(events, selected):
    result = {}
    for name in GROUPS:
        rows = [e for e in events if group(e,selected)==name and e.get('kind')!='issue']
        free = [e['effective_capacity'] for e in rows if e.get('effective_capacity') is not None]
        queue = [e['eligible_waiting'] for e in rows if e.get('eligible_waiting') is not None]
        both = [e for e in rows if e.get('effective_capacity') is not None and e.get('eligible_waiting') is not None]
        result[name] = [len(rows),len(free),min(free) if free else None,max(free) if free else None,
            sum(x==0 for x in free),len(queue),min(queue) if queue else None,max(queue) if queue else None,
            sum(e['eligible_waiting']>e['effective_capacity'] for e in both),
            sum(e.get('station_interval_uncertainty') is True for e in rows)]
    return result


def order(events, selected):
    result = {}
    for kind in ('arrival','departure'):
        by_clock = defaultdict(list)
        for e in events:
            if e.get('kind')==kind and e.get('station_id')==selected['station_id']:
                by_clock[e['clock_seconds']].append([e['line_id'],e['bus_id']])
        result[kind] = dict(groups=len(by_clock), first_four=[[t,sorted(by_clock[t])]
            for t in sorted(by_clock)[:4]])
    return result


def trip_trace(events, selected):
    stops = {}
    for arm, rows in events.items():
        for e in rows:
            if e.get('bus_id')!=selected['trip_id'] or e.get('kind') not in ('arrival','departure','service_ready','issue'): continue
            stop = e.get('station_id')
            if stop is None: continue
            item = stops.setdefault(stop,dict(first=e['clock_seconds'],low=defaultdict(list),high=defaultdict(list)))
            item['first'] = min(item['first'],e['clock_seconds']); kind=e['kind']
            value = e['clock_seconds']
            if kind=='issue':
                a=e.get('action'); value=[value,None if not a else 30*(a[0]+1)]
            elif kind=='service_ready': value=[value,e.get('native_detection_time'),e.get('observed_at')]
            item[arm][kind].append(value)
    return dict(stops=len(stops),first_four=[[s]+[stops[s][a][k] for k in ('arrival','departure','service_ready','issue')
        for a in ('low','high')] for s in sorted(stops,key=lambda s:(stops[s]['first'],s))[:4]])


def analyze(root):
    main = read(root/'service_summary.json')
    if main.get('status')!='complete' or main.get('integrity',{}).get('all_pass') is not True:
        raise RuntimeError('The six service outcomes must pass before mechanism analysis')
    cases = []
    for line in LINES:
        values = {arm:read(root/'branches/native'/line/arm/'summary.json') for arm in ('low','high')}
        for v in values.values():
            delta = v.get('suffix',{}).get('training_delta',{})
            if v.get('status')!='complete' or v.get('integrity',{}).get('all_pass') is not True or not (
                v['integrity']['checks'].get('old_native_truth_exact') is True and delta and all(x==0 for x in delta.values())):
                raise RuntimeError('Native truth and zero-training gate failed: '+line)
        selected = values['low']['selected']
        events = {a:read(root/'branches/native'/line/a/'service_events.json')['events'] for a in values}
        board,blo,bhi = pair_people(events['low'],events['high'],selected,'board')
        alight,alo,ahi = pair_people(events['low'],events['high'],selected,'alight')
        candidates = (alo.keys() & ahi.keys())-blo.keys()-bhi.keys()
        initial = {g:dict(n=sum(group(alo[p][0],selected)==g for p in candidates),
            delay=delays((alo[p][0],ahi[p][0]) for p in candidates if group(alo[p][0],selected)==g)) for g in GROUPS}
        case = next(c for c in main['cases'] if c['line_id']==line)
        missing = {a:dict(events_without_person_ids=sum(e.get('kind') in ('board','alight') and
            not e.get('passenger_ids') for e in rows), terminal_samples=sum(e.get('kind')=='passenger_sample_missing' for e in rows),
            terminal_people=sum(e.get('previous_onboard_count',0) for e in rows if e.get('kind')=='passenger_sample_missing'),
            unknown_station=sum(e.get('kind') in ('board','alight') and e.get('station_id') is None for e in rows))
            for a,rows in events.items()}
        cases.append(dict(line=line, board=board, alight=alight, initial_onboard_candidates=initial,
            repeat={a:repeats(rows) for a,rows in events.items()}, missing=missing,
            capacity={a:capacities(rows,selected) for a,rows in events.items()},
            fork_stop_order={a:order(rows,selected) for a,rows in events.items()},
            controlled_trip=trip_trace(events,selected),
            ledger_stages=case['clock_decomposition']['stages']))
    return dict(status='complete', cases=cases, schema=dict(delay_fields=DELAY_FIELDS,
        person_count_fields=('matched','low_only','high_only','high_context_changed'),
        capacity_fields=('observations','valid_free','free_min','free_max','zero_free','valid_eligible',
            'eligible_min','eligible_max','eligible_exceeds_free','uncertain_station'),
        trip_fields=('stop','arrival_low','arrival_high','departure_low','departure_high','ready_low','ready_high','issue_low','issue_high'),
        trip_timing='Ready records are [bridge history clock,detection clock,observed_at]; issue records are [delivery clock,executed hold seconds], hold=30*(action+1) under the unchanged 60-second contract.',
        groups='Disjoint priority: controlled trip, fork stop, anchor line, other lines. Matched groups use low-arm first event; high-only groups use high arm.',
        timing='High minus low first observed event within this window; sample-interval bounds are retained in possible_min/max. Ties remain ties.',
        capacity='Observed at the event observation clock; counts are event observations, not independent vehicles.',
        initial='Alight in both arms but no boarding observed in either: initial-onboard candidates only.',
        limitation='No full initial ID baseline or waiting-entry/walking observations. Same-person service associations do not allocate complete passenger cost.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); value = compact(analyze(args.root))
    data = json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(data.encode())>12000: raise RuntimeError('Reduce the resident identity aggregate before export')
    with args.out.open('x') as f: f.write(data)
    print(json.dumps(dict(status='complete',bytes=len(data.encode()))))


if __name__=='__main__': main()
