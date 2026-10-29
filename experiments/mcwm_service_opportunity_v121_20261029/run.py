"""Read-only one-slot coverage study on retained v120 service observations."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from select_opportunity import opportunity, select_packets
from experiments.mcwm_service_replay_v120_20261029.samples import rows, event_index, window, unknown_fields
from experiments.mcwm_service_replay_v120_20261029.analyze_data import marks, analyze_data
from experiments.mcwm_service_contract_v119_20261029.features import bind_action
from experiments.mcwm_service_contract_v119_20261029.labels import mature_labels


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def jsonl(path, records):
    with path.open('x') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')


def attach_new_labels(source, selected, added):
    commands = defaultdict(list)
    for command in rows(source/'commands.jsonl'):
        if command.get('sequence') in added:
            commands[command['sequence']].append(command)
    keys = set()
    for seq in added:
        context = selected[seq][2]
        station = context['meta']['station']
        keys.add((context['meta']['receiver'], station))
        keys.update((f['trip_id'], station) for f in context['feeders'])
    visits = event_index(source/'visit_events.jsonl', keys)
    exchanges = event_index(source/'exchanges.jsonl', keys, exchanges=True)
    records = {}
    for seq in sorted(added):
        deadline, available, context = selected[seq]
        start, station = context['meta']['observed_at'], context['meta']['station']
        checks = context['capture']['checks']
        complete = all(checks[k] is True for k in ('receiver_native_present',
            'receiver_at_declared_stop', 'feeder_selection_complete')) and all(
                f['current_stage0_complete'] for f in context['feeders'])
        record = dict(context=context, action=None, labels=None, errors=[],
            selection=dict(sequence=seq, packet_deadline=deadline, window_start=start,
                window_end=start+60, feedback_available_at=available),
            missing_inputs=dict(unknown_fields(context)), cohort_definition_complete=complete)
        try:
            if len(commands[seq]) != 1:
                raise ValueError('Selected root must have exactly one recorded command')
            bound = bind_action(context, commands[seq][0])
            related = {context['meta']['receiver'], *(f['trip_id'] for f in context['feeders'])}
            evidence = dict(start=start, end=start+60, continuous=True,
                visit_events=[e for trip in related for e in window(visits,(trip,station),start,start+60)],
                exchanges=[e for trip in related for e in window(exchanges,(trip,station),start,start+60)])
            record.update(action=bound['action'], labels=mature_labels(bound,evidence,available_at=start+60))
        except (ValueError, KeyError) as error:
            record['errors'].append(str(error))
        records[seq] = record
    return records


def metrics(records):
    counts = Counter({name:0 for name in ('records','opportunity','valid_mature',
        'observed_cohort_boarding_positive','mature_cohort_boarding_positive','departure_censored','errors')})
    for row in records:
        value = marks(row)
        counts['records'] += 1
        counts['opportunity'] += opportunity(row['context'])
        counts['valid_mature'] += value['valid_mature']
        counts['observed_cohort_boarding_positive'] += value['root_cohort_boarding_positive']
        counts['mature_cohort_boarding_positive'] += value['root_cohort_boarding_positive'] and value['valid_mature']
        counts['departure_censored'] += value['departure_censored']
        counts['errors'] += bool(row['errors'])
    return dict(counts)


def process_scene(raw, old, out, feedback):
    qualified = json.loads((old/'summary.json').read_text())
    if qualified['status'] != 'complete' or not qualified['integrity']['all_pass']:
        raise ValueError('Use the qualified v120 review, not the original strict-order failure')
    original_packets = list(rows(old/'packets.jsonl'))
    plans, chosen = select_packets(rows(raw/'contexts.jsonl'), original_packets, feedback)
    out.mkdir()
    # Persist the entire outcome-blind choice before opening any outcome file.
    jsonl(out/'selection_plan.jsonl', plans)
    original = {r['selection']['sequence']:r for r in rows(old/'selected_records.jsonl')}
    added, removed = set(chosen)-set(original), set(original)-set(chosen)
    new = attach_new_labels(raw,chosen,added)
    records=[]
    for packet in plans:
        for seq in packet['selected_sequences']:
            records.append(deepcopy(original[seq]) if seq in original else new[seq])
    by_sequence = {r['selection']['sequence']:r for r in records}
    inherited = set(original).intersection(chosen)
    checks = dict(original_latest5_reproduced=True,
        retained_rows_exact=all(by_sequence[s]==original[s] for s in inherited),
        same_budget=len(records)==len(original)==feedback['target_record_budget'],
        unique_roots=len(by_sequence)==len(records),
        one_change_per_packet=all(len(set(p['original_sequences'])-set(p['selected_sequences']))<=1
            and len(set(p['selected_sequences'])-set(p['original_sequences']))<=1 for p in plans),
        label_and_action_builds_valid=not any(r['errors'] for r in records),
        unchanged_time_contract=all(r['selection']['window_end']==r['context']['meta']['observed_at']+60
            and r['selection']['feedback_available_at']==r['selection']['window_end']+feedback['feedback_delay_seconds']
            and r['selection']['feedback_available_at']<=r['selection']['packet_deadline']
            and r['labels'] is not None and r['labels']['available_at']==r['selection']['window_end'] for r in records))
    jsonl(out/'selected_records.jsonl',records)
    jsonl(out/'packets.jsonl',plans)
    positive_rows=[]
    partitions={name:[] for name in ('feedback_by_9000','roots_from_9000','boundary')}
    for row in records:
        selection=row['selection']
        partition=('feedback_by_9000' if selection['feedback_available_at']<=9000 else
                   'roots_from_9000' if selection['window_start']>=9000 else 'boundary')
        partitions[partition].append(row)
        if marks(row)['root_cohort_boarding_positive']:
            positive_rows.append(dict(sequence=selection['sequence'],clock=selection['window_start'],
                packet_deadline=selection['packet_deadline'],partition=partition,
                added=selection['sequence'] in added,
                mature=marks(row)['valid_mature'],
                cohort_boarded=row['labels']['cohort_boarded_count']['observed_count_lower_bound']))
    after=metrics(records)
    summary=dict(status='complete' if all(checks.values()) else 'technical_hold',
        integrity=dict(all_pass=all(checks.values()),checks=checks),
        service_records=dict(selected_records=len(records),packet_slots=len(plans),
            statistics=dict(valid_mature_records=after['valid_mature'])),
        coverage=dict(before=metrics(original.values()),after=after,
            added=metrics(new.values()),removed=metrics(original[s] for s in removed)),
        packet_reasons=dict(Counter(p['reason'] for p in plans)),
        added_records=len(added),removed_records=len(removed),retained_records=len(inherited),
        partitions={k:metrics(v) for k,v in partitions.items()},
        positive_records=positive_rows[:8],positive_records_total=len(positive_rows),
        source=dict(raw=str(raw),qualified_review=str(old)),
        budget=dict(native_seconds=0,optimizer_updates=0,fit_models=0,training_admissions=0),
        scope='Exposed-data coverage only; original actions, outcomes and v120 selections unchanged.')
    write(out/'summary.json',summary)
    return summary,positive_rows


def run(root,out):
    protocol=json.loads((HERE/'protocol.json').read_text())
    source=root/protocol['source_root']
    out.mkdir(exist_ok=False)
    cases,positive_groups={},set()
    for scene in protocol['scenarios']:
        case,positives=process_scene(source/'runs'/scene,source/'review'/scene,out/scene,
                                     protocol['feedback_contract'])
        cases[scene]=case
        for row in positives:
            group='shared_pre9000' if row['clock']+60<=9000 else scene
            positive_groups.add((group,row['sequence'],row['clock']))
    profile=analyze_data(out)
    technical=all(c['integrity']['all_pass'] for c in cases.values()) and profile['data_reconciliation_passed']
    positive=sum(c['coverage']['after']['observed_cohort_boarding_positive'] for c in cases.values())
    result=dict(status='complete' if technical else 'technical_hold',all_pass=technical,
        decision=('ADMIT_LEARNING_VALIDATION_CONTRACT_ONLY' if positive else 'CLOSE_FIXED_SAMPLING_CANDIDATE')
            if technical else 'TECHNICAL_HOLD',
        cases=cases,profile=profile,positive_record_slots=positive,
        different_positive_root_groups=len(positive_groups),
        budget=dict(new_native_seconds=0,optimizer_updates=0,fitted_models=0,training_admissions=0),
        automatic_followup=False)
    write(out/'summary.json',result)
    payload=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(payload.encode())>24000:
        raise ValueError('Keep full output resident; compact summary exceeded24KB')
    print(payload)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--thesis-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.thesis_root,args.out)
    raise SystemExit(0 if result['all_pass'] else 1)
