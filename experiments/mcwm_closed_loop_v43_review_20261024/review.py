"""Audit completed v43 logs and align real reward clocks; never fit a model."""
import json
import math
from pathlib import Path
import time

from reward_clock import summarize_pair

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_sync_pilot_v43_20261024'


def read(path):
    return json.loads(path.read_text())


def records(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_audit(case):
    root = SOURCE/'runs'/case
    source = read(root/'pilot_summary.json')
    native = source['native']
    packets = records(root/'learning/packets.jsonl')
    credit = records(root/'learning/credit_packets.jsonl')
    snapshot = read(root/'behavior_snapshot.json')
    ckpt = Path(snapshot['resident_checkpoint'])
    adam_continuous = all(
        b['optimizer_steps'][name][bound]-a['optimizer_steps'][name][bound] == 8.
        for a, b in zip(packets, packets[1:]) for name in a['optimizer_steps'] for bound in ('min','max'))
    checks = dict(source_complete=source['status'] == 'complete' and source['integrity']['all_pass'],
        actual_snapshot_present=ckpt.is_file() and ckpt.stat().st_size > 0,
        post24_metadata=snapshot['total_updates'] == 50024 and snapshot['feedback_slots'] == 15
            and snapshot['clock'] == native['behavior_saved_at'] < 9000.,
        packets57=len(packets) == 57 and [row['packet'] for row in packets] == list(range(1,58)),
        packet_updates_exact=all(row['updates'] == row['packet']*8 for row in packets),
        optimizer_objects_retained=all(row['optimizer_objects_retained'] for row in packets),
        adam_continuous_all_packets=adam_continuous,
        snapshot_adam_matches_packet3=all(snapshot['optimizer_steps'][name][bound]
            == packets[2]['optimizer_steps'][name][bound]
            for name in packets[2]['optimizer_steps'] for bound in ('min','max')),
        feedback_admission_order=all(row['feedback_slots'] == (5*row['packet'] if row['packet'] <= 3 else 285)
            for row in packets),
        response_at9630=all(row['receipt']['processed_at'] == 9630.
            and row['receipt']['all285_previously_admitted'] for row in packets[3:]),
        credit_all55_boundaries=[row['updates'] for row in credit] == list(range(24,457,8)))
    zero = credit[0]
    rows0 = [row for chain in zero['chains'] for row in chain['rows']]
    checks['post24_point_exact_and_integral_normalized'] = len(rows0) == 264 and all(row['c'] == 1.
        and abs(row['overlap']-1.) < 5e-12 and row['log_c'] == 0.
        and abs(row['expected_log_c']) < 5e-12 for row in rows0)
    checks['all_credit_normalized'] = all(chain['maximum_normalization_error'] < 5e-12
        and chain['maximum_expectation_identity_error'] < 5e-12 for packet in credit for chain in packet['chains'])
    pairs = []
    branch_proofs = []
    for pair in native['response']['pairs']:
        line = pair['line_id']
        arms = {arm: read(Path(pair['branches'][arm]['summary_path'])) for arm in ('low','high')}
        live = read(Path(pair['branches']['low']['parallel_receipt']))
        proof = dict(line=line, selected_equal=arms['low']['selected'] == arms['high']['selected'] == pair['selected'],
            external_seed_equal=arms['low']['external_seed'] == arms['high']['external_seed'] == 106243,
            suffix_zero_training=all(arm['counters']['before'] == arm['counters']['after'] for arm in arms.values()),
            actual_holds=[arms[arm]['execution']['native_applied_hold_seconds'] for arm in ('low','high')],
            shared_first_batch_actions=arms['low']['native']['other_batch_actions'] == arms['high']['native']['other_batch_actions'],
            parent_unchanged=live['native_before'] == live['native_after']
                and live['counters_before'] == live['counters_after']
                and live['parent_rng_unchanged'] and live['parent_trace_cookie_unchanged']
                and live['route_streams']['parent_offsets_unchanged'],
            actual_parallel_seconds=live['overlap_seconds'])
        branch_proofs.append(proof)
        pairs.append(summarize_pair(Path(pair['branches']['low']['events_path']),
            Path(pair['branches']['high']['events_path']), root/'branches'/line/'low/suffix_trace.jsonl',
            root/'branches'/line/'high/suffix_trace.jsonl', credit, line_id=line))
    checks['all_pair_proofs'] = all(proof['selected_equal'] and proof['external_seed_equal']
        and proof['suffix_zero_training'] and proof['shared_first_batch_actions'] and proof['parent_unchanged']
        and proof['actual_holds'] == [0.,1.875] and proof['actual_parallel_seconds'] > 0. for proof in branch_proofs)
    checks['all_reward_ledgers_exact'] = all(check['reward_ledger_consistent']
        for pair in pairs for check in pair['ledger_checks'].values())
    curve = []
    means = []
    for packet in credit:
        rows = [row for chain in packet['chains'] for row in chain['rows']]
        mean_p = math.fsum(row['target_parameters']['positive_probability'] for row in rows)/len(rows)
        mean_o = math.fsum(row['overlap'] for row in rows)/len(rows)
        means.append([packet['updates'], mean_p, mean_o])
        if packet['updates'] in (24,32,48,80,144,272,400,456):
            curve.append([packet['updates'], mean_p, mean_o,
                min(row['overlap'] for row in rows),
                math.fsum(chain['prefixes']['full44']['L_old'] for chain in packet['chains'])/6,
                math.fsum(chain['prefixes']['full44']['L_geom'] for chain in packet['chains'])/6])
    return dict(integrity=dict(all_pass=all(checks.values()), checks=checks),
        snapshot=dict(clock=snapshot['clock'], total_updates=snapshot['total_updates'], feedback_slots=15,
            bytes=ckpt.stat().st_size, actual_file_present=True, checkpoints_loaded=0,
            adam_after24=packets[2]['optimizer_steps'], adam_after456=packets[-1]['optimizer_steps']),
        full_day=native['budget'], update_log=dict(packets=57, response_packets=len(packets[3:]),
            response_update_calls=len(packets[3:])*8, all_feedback_before_response=285,
            source_tail_zero=native['integrity']['checks']['post_response_zero_training']),
        new_contexts=dict(continuation_per_boundary=len(rows0), boundaries=len(credit),
            actor_input_differences=sum(row['behavior_input_differs'] for row in rows0),
            F_input_differences=sum(row['behavior_F_input_differs'] for row in rows0)),
        first_gate_mean_at_least_half=next((row for row in means if row[1] >= .5),None),
        first_gate_mean_at_least_065=next((row for row in means if row[1] >= .65),None),
        minimum_mean_overlap=min(means,key=lambda row:row[2]), curve=curve,
        paired_proofs=branch_proofs, reward_clock=pairs)


def compact(value):
    if isinstance(value,float):return float(f'{value:.9g}')
    if isinstance(value,list):return [compact(item) for item in value]
    if isinstance(value,dict):return {key:compact(item) for key,item in value.items()}
    return value


def main():
    started=time.time()
    source=read(SOURCE/'analysis/summary.json')
    cases={case:case_audit(case) for case in ('stable','local_speed_drop')}
    warmups={case:records(SOURCE/'runs'/case/'learning/packets.jsonl')[:3] for case in cases}
    warmup_equal=all(a['sampling']==b['sampling'] and a['rng']==b['rng']
        and a['optimizer_steps']==b['optimizer_steps'] for a,b in zip(warmups['stable'],warmups['local_speed_drop']))
    result=dict(status='complete', integrity=dict(all_pass=source['integrity']['all_pass']
        and all(case['integrity']['all_pass'] for case in cases.values()) and warmup_equal),
        common_before9000=source['common_before_shock'], warmup_sampling_rng_adam_equal=warmup_equal,
        priority='high', source_summary_bytes=(SOURCE/'analysis/summary.json').stat().st_size,
        cases=cases, new_optimizer_updates=0,new_native_seconds=0,checkpoints_loaded=0,
        curve_columns=['updates','fixed_state_target_p_mean','F_overlap_mean','minimum_F_overlap',
            'mean_full44_L_actual','mean_full44_L_cond_geom'],
        elapsed_seconds=time.time()-started)
    encoded=json.dumps(compact(result),separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode()) > 24000:raise RuntimeError('Review summary exceeds24KB; reduce on server without rerunning v43')
    out=HERE/'evidence.json'
    with out.open('x') as handle:handle.write(encoded)
    print(json.dumps(dict(status=result['status'], integrity=result['integrity'],bytes=len(encoded.encode()))))


if __name__=='__main__':main()
