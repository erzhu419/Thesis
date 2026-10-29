"""Read the twelve frozen departure cases from resident native logs only."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def extract_scene(root, scene, ids, out):
    native = root/'experiments/mcwm_service_replay_v120_20261029/runs'/scene
    selected = root/'experiments/mcwm_service_opportunity_v121_20261029/runs'/scene
    wanted = set(ids['non_clock_exact_sequences']+ids['censored_sequences'])
    records = {r['selection']['sequence']:r for r in rows(selected/'selected_records.jsonl')
               if r['selection']['sequence'] in wanted}
    if set(records) != wanted:
        raise ValueError('Frozen root list missing from v121 selection')
    commands = list(rows(native/'commands.jsonl'))
    roots, by_visit = {}, {}
    for seq, record in records.items():
        matching = [r for r in commands if r.get('sequence')==seq]
        if len(matching)!=1:
            raise ValueError('Root does not bind exactly one native command')
        command = matching[0]
        key = (command['trip_id'],command['visit_index'])
        if key in by_visit:
            raise ValueError('Frozen roots unexpectedly share one native visit')
        by_visit[key]=seq
        roots[seq]=dict(record=record,command=command,visit=None,events=[],exchanges=[],ticks=[])
    for visit in rows(native/'visits.jsonl'):
        key=(visit['trip_id'],visit['visit_index'])
        if key in by_visit:
            roots[by_visit[key]]['visit']=visit
    for name in ('visit_events','exchanges'):
        for row in rows(native/(name+'.jsonl')):
            key=(row['trip_id'],row['visit_index'])
            if key in by_visit:
                roots[by_visit[key]]['events' if name=='visit_events' else 'exchanges'].append(row)
    for data in roots.values():
        visit=data['visit']
        if visit is None or visit['departure_time'] is None:
            raise ValueError('Frozen diagnostic visit has no observed native departure')
        data['same_visit_commands']=[r for r in commands if r['trip_id']==visit['trip_id']
            and r['visit_index']==visit['visit_index']]
    low=min(r['visit']['arrival_time'] for r in roots.values())
    high=max(r['visit']['departure_time'] for r in roots.values())
    for tick in rows(native/'ticks.jsonl'):
        if tick['start']<low:
            continue
        if tick['start']>high:
            break
        for vehicle in tick['vehicles']:
            key=(vehicle['trip_id'],vehicle['visit_index'])
            if key in by_visit:
                data=roots[by_visit[key]]
                if data['visit']['arrival_time'] <= tick['start'] <= data['visit']['departure_time']:
                    data['ticks'].append(dict(start=tick['start'],end=tick['end'],**vehicle))
    summaries=[]
    for seq in sorted(roots):
        data=roots[seq]
        row,command,visit=data['record'],data['command'],data['visit']
        issue=row['selection']['window_start']
        nominal=command['applied_at']+command['requested_additional_hold_seconds']
        clock=math.ceil(nominal)
        departure=visit['departure_time']
        extra=[t for t in data['ticks'] if clock<=t['start']<departure]
        exchanges=[e for e in data['exchanges'] if visit['arrival_time']<=e['time']<=departure]
        after=[e for e in exchanges if issue<e['time']<=min(departure,issue+60)]
        mature=row['labels']['receiver_departure']['value']
        checks=dict(root_identity_matches=(row['context']['meta']['receiver']==visit['trip_id']
                and row['context']['meta']['station']==visit['stop_id']
                and command['applied_at']==issue and command['accepted']),
            hold_matches=command['requested_additional_hold_seconds']==row['action']['hold_seconds'],
            original_label_preserved=(mature==departure if seq not in ids['censored_sequences']
                else mature is None and departure>issue+60),
            one_departure_event=sum(e['event']=='departure' and e['time']==departure for e in data['events'])==1,
            extra_ticks_accounted=sum(t['end']-t['start'] for t in extra)==departure-clock,
            extra_ticks_physically_at_stop=all(t['at_stop'] and t['stop_id']==visit['stop_id'] for t in extra),
            root_boarding_label_matches=sum(len(e['boarded_ids']) for e in after)==row['labels']['receiver_post_issue_boarded_count']['observed_count_lower_bound'])
        if not all(checks.values()):
            raise ValueError('Native timeline mismatch: '+str(seq)+' '+str(checks))
        reason_seconds=Counter()
        for tick in extra:
            for start,end,phase,reason in tick['segments']:
                reason_seconds[phase+'/'+reason]+=end-start
        summaries.append(dict(sequence=seq,trip=visit['trip_id'],stop=visit['stop_id'],visit_index=visit['visit_index'],
            arrival=visit['arrival_time'],service_ready=command['service_ready_time'],issue=issue,
            hold=command['requested_additional_hold_seconds'],clock_departure=clock,actual_departure=departure,
            extra_seconds=departure-clock,original_60s_censored=mature is None,
            pre_action_receiver=row['context']['receiver'],
            commands=[dict(sequence=c['sequence'],at=c['applied_at'],hold=c['requested_additional_hold_seconds'],
                accepted=c['accepted']) for c in data['same_visit_commands']],
            exchanges=[[e['time'],len(e['boarded_ids']),len(e['alighted_ids']),e['observed_at_stop']] for e in exchanges],
            extra_tick_states=dict(Counter(str(t['native_stop_state']) for t in extra)),
            extra_tick_speed_range=[min(t['speed'] for t in extra),max(t['speed'] for t in extra)] if extra else None,
            extra_phase_seconds=dict(reason_seconds),all_checks_pass=True))
    out.mkdir()
    write(out/'full_evidence.json',roots)
    write(out/'summary.json',summaries)
    return summaries


def run(root,out):
    protocol=json.loads((HERE/'protocol.json').read_text())
    out.mkdir(exist_ok=False)
    cases={scene:extract_scene(root,scene,ids,out/scene) for scene,ids in protocol['fixed_roots'].items()}
    totals={scene:dict(roots=len(values),mature_extra_seconds=sum(r['extra_seconds'] for r in values
        if not r['original_60s_censored']),censored_roots=sum(r['original_60s_censored'] for r in values))
        for scene,values in cases.items()}
    if totals['stable']['mature_extra_seconds']!=18 or totals['local_speed_drop']['mature_extra_seconds']!=16:
        raise ValueError('Diagnostic roots do not reproduce the frozen34 seconds of prediction error')
    result=dict(status='complete',all_pass=True,cases=cases,totals=totals,
        exchange_columns=['time','boarded','alighted','observed_at_stop'],
        semantics='Observer after_hold identifies an interval, not its physical cause. Later censored-root departure is diagnostic only; original labels remain unchanged.',
        budget=dict(new_native_seconds=0,optimizer_updates=0,new_collection=0,new_predictor_fits=0,control_changes=0))
    write(out/'summary.json',result)
    payload=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(payload.encode())>12000:
        raise ValueError('Retain full evidence remotely; compact summary exceeds12KB')
    print(payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.thesis_root,args.out)
