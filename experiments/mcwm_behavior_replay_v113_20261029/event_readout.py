"""Small matched stop/road examples from completed resident behavior logs."""
import json
from pathlib import Path
import sys


def read(folder):
    rows=[json.loads(line) for line in (folder/'visits.jsonl').open()]
    return {(r['trip_id'],r['stop_id']):r for r in rows}


def details(row, exchanges):
    ids=set()
    for event in exchanges:
        if event['trip_id']==row['trip_id'] and row['departure_time'] is not None and event['time']<=row['departure_time']:
            ids.difference_update(event['alighted_ids']);ids.update(event['boarded_ids'])
    return {**{k:row[k] for k in ('trip_id','stop_id','arrival_time','service_ready_time','action_time','command_hold_seconds','departure_time','segment_end')},
        'road_seconds':None if row['departure_time'] is None else row['segment_end']-row['departure_time'],
        'onboard_at_departure':len(ids),'boardings_in_visit':len(row['boarded_ids']),
        'alightings_in_visit':len(row['alighted_ids'])}


def analyze(here):
    here=Path(here); out={}
    for scene in ('stable','local_speed_drop'):
        folders={a:here/'runs'/scene/a for a in ('fresh','no_fresh')}
        records={a:read(f) for a,f in folders.items()}
        f,n=records['fresh'],records['no_fresh']; common=f.keys()&n.keys()
        peak=sorted(common,key=lambda k:f[k]['periods']['12600_12900']['ride']-n[k]['periods']['12600_12900']['ride'],reverse=True)[:2]
        keys=peak+([('705X_9','705X16'),('705X_8','705X16')] if scene=='stable' else [('311X_19','122X22_311X22_406S29')])
        triples={k[0] for k in keys}
        exchanges={a:[e for e in map(json.loads,(folder/'exchanges.jsonl').open()) if e['trip_id'] in triples] for a,folder in folders.items()}
        rows=[]
        for key in dict.fromkeys(keys):
            a,b=f[key],n[key]
            rows.append(dict(fresh=details(a,exchanges['fresh']),no_fresh=details(b,exchanges['no_fresh']),
                full_delta_ride=a['ride']-b['ride'],full_delta_phases={p:a['ride_phases'][p]-b['ride_phases'][p] for p in a['ride_phases']},
                peak12600_12900_delta_ride=a['periods']['12600_12900']['ride']-b['periods']['12600_12900']['ride']))
        out[scene]=rows
    return out


if __name__=='__main__':
    print(json.dumps(analyze(Path(sys.argv[1])),separators=(',',':'),allow_nan=False))
