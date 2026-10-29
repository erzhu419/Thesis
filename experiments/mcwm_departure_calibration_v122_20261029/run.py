"""Evaluate a frozen, deterministic calibration contract on resident v121 labels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, median

from core import forecast_scene, residual

HERE = Path(__file__).resolve().parent
SCENES = ('stable', 'local_speed_drop')
MODELS = ('frozen', 'fresh', 'clock_reference', 'raw_command_reference')


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def write_rows(path, values):
    with path.open('x') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')


def load_source(source):
    result = json.loads((source/'summary.json').read_text())
    if not result['all_pass'] or result['status'] != 'complete':
        raise ValueError('v121 source did not complete its declared technical checks')
    cases = {}
    for scene in SCENES:
        folder = source/scene
        records = read_rows(folder/'selected_records.jsonl')
        packets = read_rows(folder/'packets.jsonl')
        indexed = {r['selection']['sequence']: r for r in records}
        if len(indexed) != len(records) or len(records) != 285:
            raise ValueError('Changed selected-record budget or duplicate root')
        if [p['deadline'] for p in packets] != list(range(1200,18001,300)):
            raise ValueError('Changed packet deadlines')
        listed = []
        for packet in packets:
            seqs = packet['selected_sequences']
            if len(seqs) != 5 or any(indexed[s]['selection']['packet_deadline'] != packet['deadline'] for s in seqs):
                raise ValueError('Packet membership changed')
            listed.extend(seqs)
        if sorted(listed) != sorted(indexed):
            raise ValueError('Packet membership does not cover selected records once')
        for row in records:
            selection, meta, action = row['selection'], row['context']['meta'], row['action']
            if (row['errors'] or selection['sequence'] != meta['sequence']
                    or selection['window_start'] != meta['observed_at']
                    or action['applied_at'] != meta['observed_at']
                    or selection['window_end'] != selection['window_start']+60
                    or selection['feedback_available_at'] != selection['window_end']+30
                    or selection['feedback_available_at'] > selection['packet_deadline']):
                raise ValueError('Original identity, action or label-release clock mismatch')
        cases[scene] = records
    return cases


def initialize(cases):
    prefixes = {scene: {r['selection']['sequence']: r for r in records
        if r['selection']['packet_deadline'] <= 9000} for scene, records in cases.items()}
    stable, shock = prefixes['stable'], prefixes['local_speed_drop']
    if set(stable) != set(shock) or len(stable) != 135:
        raise ValueError('Shared initialization root set differs from the frozen contract')
    for seq in stable:
        left, right = deepcopy(stable[seq]), deepcopy(shock[seq])
        for value in (left,right):
            value['context']['meta'].pop('source_group', None)
        if left != right:
            raise ValueError('Shared prefix payload differs')
    values = [value for row in stable.values() if (value := residual(row)) is not None]
    return values, dict(shared_roots=len(stable), mature_departures=len(values),
        censored_departures=len(stable)-len(values), median_residual=median(values),
        initial_fits=1, source_scenes_verified=list(SCENES))


def score(records, predictions):
    indexed = {r['selection']['sequence']: r for r in records}
    errors, censored = [], []
    for prediction in predictions:
        row = indexed[prediction['sequence']]
        label = row['labels']['receiver_departure']
        if row['labels']['available_at'] != row['selection']['window_end']:
            raise ValueError('Label observation window changed')
        if label['right_censored']:
            if label['value'] is not None:
                raise ValueError('Censored root gained a point target')
            censored.append(prediction['sequence'])
            continue
        target = label['value']
        if not row['selection']['window_start'] < target <= row['selection']['window_end']:
            raise ValueError('Departure target lies outside the original observation window')
        nominal = row['action']['applied_at']+row['action']['hold_seconds']
        signed = {k:prediction['predictions'][k]-target for k in MODELS}
        errors.append(dict(sequence=prediction['sequence'], issue_time=prediction['issue_time'],
            actual_departure=target, nominal_departure=nominal,
            residual=target-nominal, signed_errors=signed,
            absolute_errors={k:abs(value) for k,value in signed.items()},
            median_residual=prediction['median_residual']))
    mae = {k:mean(r['absolute_errors'][k] for r in errors) for k in MODELS}
    signed_mean = {k:mean(r['signed_errors'][k] for r in errors) for k in MODELS}
    differences = [r['absolute_errors']['fresh']-r['absolute_errors']['frozen'] for r in errors]
    counts = dict(improved=sum(x<0 for x in differences), tied=sum(x==0 for x in differences),
                  worsened=sum(x>0 for x in differences))
    detail = dict(evaluated=len(errors), predicted=len(predictions), censored=len(censored),
        censored_sequences=censored, mae_seconds=mae, mean_signed_error_seconds=signed_mean,
        fresh_minus_frozen_mae=mae['fresh']-mae['frozen'],
        fresh_minus_clock_mae=mae['fresh']-mae['clock_reference'], paired_counts=counts,
        observed_residual=dict(median=median(r['residual'] for r in errors),
            minimum=min(r['residual'] for r in errors), maximum=max(r['residual'] for r in errors)),
        clock_exact_targets=sum(r['absolute_errors']['clock_reference']==0 for r in errors),
        largest_paired_changes=[dict(sequence=r['sequence'],issue_time=r['issue_time'],
            delta_absolute_error=r['absolute_errors']['fresh']-r['absolute_errors']['frozen'],
            actual_residual=r['residual'],used_median=r['median_residual'])
            for r in sorted(errors,key=lambda r:abs(r['absolute_errors']['fresh']-r['absolute_errors']['frozen']),reverse=True)[:3]])
    return detail, errors


def run(source, out):
    cases = load_source(source)
    initial, initialization = initialize(cases)
    out.mkdir(exist_ok=False)
    results = {}
    for scene, records in cases.items():
        folder = out/scene
        folder.mkdir()
        predictions, updates, stats = forecast_scene(records,initial,initialization['median_residual'])
        # Persist all causal predictions before evaluating their realized targets.
        write_rows(folder/'predictions.jsonl', predictions)
        write_rows(folder/'calibration_updates.jsonl',updates)
        detail, errors = score(records,predictions)
        write_rows(folder/'errors.jsonl', errors)
        causal = all(p['latest_feedback_deadline'] < p['issue_time'] for p in predictions)
        result = dict(integrity=dict(all_pass=causal, strict_feedback_before_issue=causal),
            records=285, packets=57, initialization_boundary_roots=sum(r['selection']['window_start']==9000 for r in records),
            delayed_pre9000_feedback_roots=sum(r['selection']['window_start']<9000 and r['selection']['packet_deadline']>9000 for r in records),
            prediction=detail,calibration=stats)
        write(folder/'summary.json',result)
        results[scene]=result
    checks = dict(fresh_not_worse_than_frozen_each_scene=all(r['prediction']['fresh_minus_frozen_mae']<=0 for r in results.values()),
        fresh_strictly_better_than_frozen_some_scene=any(r['prediction']['fresh_minus_frozen_mae']<0 for r in results.values()),
        fresh_not_worse_than_clock_each_scene=all(r['prediction']['fresh_minus_clock_mae']<=0 for r in results.values()))
    technical = all(r['integrity']['all_pass'] for r in results.values())
    admitted = technical and all(checks.values())
    result=dict(status='complete' if technical else 'technical_hold', all_pass=technical,
        source=str(source),initialization=initialization,cases=results,
        admission=dict(passed=admitted,checks=checks),
        decision=('ADMIT_SCALAR_CALIBRATION_FOR_NEXT_DESIGN' if admitted else 'CLOSE_FIXED_CUMULATIVE_MEDIAN_CANDIDATE')
            if technical else 'TECHNICAL_HOLD',
        budget=dict(new_native_seconds=0,optimizer_updates=0,new_collection=0,checkpoint_loads=0,
            connection_head_fits=0,control_changes=0,shared_median_initializations=1,
            subsequent_median_recalculations=sum(r['calibration']['packet_recalculations'] for r in results.values())),
        F='unchanged',control_cost_gates='unchanged; not evaluated by factual prediction',automatic_followup=False)
    write(out/'summary.json',result)
    payload=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(payload.encode())>18000:
        raise ValueError('Summary exceeds18KB; retain complete artifacts on server')
    print(payload)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.source,args.out)
    raise SystemExit(0 if result['all_pass'] else 1)
