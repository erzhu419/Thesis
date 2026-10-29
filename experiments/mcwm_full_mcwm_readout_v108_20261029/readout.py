"""Read v107 resident JSON logs; never import a learner or construct an environment."""
import json
from pathlib import Path
import sys

import numpy as np
from exposure import reconstruct

VERSION = 'mcwm_full_mcwm_dripping_pilot_v107_20261029'
ANCHORS = (0, 1200, 9000, 12000, 18000, 18300, 24000)
N = 15079


def read(path):
    return json.loads(path.read_text())


def point(row):
    return {key: row[key] for key in ('accrued_wait_seconds', 'accrued_in_vehicle_seconds',
                                    'waiting_passengers', 'in_vehicle_passengers')}


def cost(a, b):
    wait = (b['accrued_wait_seconds']-a['accrued_wait_seconds'])/N
    ride = (b['accrued_in_vehicle_seconds']-a['accrued_in_vehicle_seconds'])/N
    return dict(weighted_wait=2*wait, ride=ride, cost=2*wait+ride)


def delta(a, b):
    return {key: a[key]-b[key] for key in a}


def analyze(root):
    base = root/'experiments'/VERSION
    experiment = read(base/'summary.json')
    protocol = read(base/'protocol.json')
    if not experiment['complete'] or not experiment['integrity']['all_pass']:
        raise ValueError('This readout requires the completed v107 fixed study')
    result = dict(source=str(base), new_native_seconds=0, new_optimizer_updates=0,
        new_feedback_labels=0, cells={}, matched={}, shared_pre_shock={},
        fixed_segment_boundaries=list(ANCHORS))
    timelines = {}
    for scene in ('stable', 'local_speed_drop'):
        for arm in ('fresh', 'no_fresh'):
            key = scene+'/'+arm
            folder = base/'runs'/scene/arm
            summary = read(folder/'summary.json')
            records = read(folder/'metrics.json')
            updates = [json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
            packets = read(folder/'packet_receipts.json')
            table = {float(row['time_seconds']): point(row) for row in records}
            table[0.] = dict(accrued_wait_seconds=0., accrued_in_vehicle_seconds=0.,
                            waiting_passengers=0, in_vehicle_passengers=0)
            table[24000.] = point(summary['ledger'])
            if any(t not in table for t in ANCHORS):
                raise ValueError('Missing exact ledger anchor: '+key)
            if summary['passenger_denominator'] != N:
                raise ValueError('The fixed cohort differs: '+key)
            end = table[24000.]
            actual = (2*end['accrued_wait_seconds']+end['accrued_in_vehicle_seconds'])/N
            if actual != summary['periods']['full_day']['cost_per_passenger']:
                raise ValueError('Ledger differs from the reported full day: '+key)
            replay = summary['replay']
            ids = [seq for packet in packets for seq in packet['sequence_ids']]
            expected = (285, 456, 29184, 29184 if arm == 'fresh' else 0)
            observed = (len(set(ids)), replay['sample_mixed_calls'],
                        replay['private_draws'], replay['replaced_rows'])
            if observed != expected or len(ids) != 285:
                raise ValueError('Actual replay use differs from the fixed index reconstruction: '+key)
            if len(packets) != 57 or len(updates) != 57:
                raise ValueError('Missing packet/update receipts: '+key)
            if summary['runtime']['numpy'] != np.__version__:
                raise ValueError('Use the original NumPy runtime for the private draw reconstruction')
            init = summary['initialization']
            result['cells'][key] = dict(status=summary['status'], integrity=summary['integrity']['all_pass'],
                actual_counts=summary['counts'], budget=summary['budget'], replay=replay,
                initialization={k:init[k] for k in ('actor_exact','offline_rows','source_replay_rows',
                    'original_source_events','real_td_coefficient','model_td_coefficient','model_family',
                    'actor_learning_rates','online_backup','mixed_backup_lambda','reward_scale')},
                action_audit=summary['action_audit'], final_passenger_states=summary['passengers']['state_counts_at_horizon'],
                packet_timing=dict(first=packets[0]['processed_at'], last=packets[-1]['processed_at']),
                feedback_ids_first5=ids[:5], feedback_ids_last5=ids[-5:],
                exact_anchor_inventory={str(t):{k:table[t][k] for k in ('waiting_passengers','in_vehicle_passengers')}
                    for t in ANCHORS})
            timelines[key] = table
        fresh, control = (timelines[scene+'/'+arm] for arm in ('fresh','no_fresh'))
        segments = []
        for start, end in zip(ANCHORS, ANCHORS[1:]):
            segments.append(dict(start=start, end=end,
                fresh_minus_matched=delta(cost(fresh[start],fresh[end]),cost(control[start],control[end]))))
        common = sorted(fresh.keys() & control.keys())
        differences = [(t, delta(cost(fresh[0],fresh[t]),cost(control[0],control[t]))) for t in common]
        nonzero = next(((t,d) for t,d in differences if any(d.values())), None)
        result['matched'][scene] = dict(segments=segments,
            anchors={str(t):delta(cost(fresh[0],fresh[t]),cost(control[0],control[t])) for t in ANCHORS},
            first_different_snapshot=None if nonzero is None else dict(time=nonzero[0],difference=nonzero[1]),
            snapshot_resolution_seconds=300,
            timing_scope='First observed ledger difference, not the time or causal update that produced it')
    for arm in ('fresh', 'no_fresh'):
        a,b = (timelines[scene+'/'+arm] for scene in ('stable','local_speed_drop'))
        keys_a,keys_b = ({t for t in table if t <= 9000} for table in (a,b))
        mismatches = [t for t in sorted(keys_a & keys_b) if a[t] != b[t]]
        result['shared_pre_shock'][arm] = dict(all_equal=keys_a == keys_b and not mismatches,
            compared_snapshots=len(keys_a & keys_b), differing_times=mismatches,
            scope='All saved ledger fields through9000; no claim about unrecorded full states')
    exposure = reconstruct(protocol)
    exposure.pop('row_counts', None)
    result['feedback_exposure'] = exposure
    result['exposure_evidence'] = dict(numpy=np.__version__,actual_replay_counters_matched=True,
        scope='Deterministic index reconstruction from the frozen sampler and its private RNG; '
            'not a replay of learning or a gradient attribution')
    return result


if __name__ == '__main__':
    value = analyze(Path(sys.argv[1]))
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(text.encode()) > 24000:
        raise ValueError('Reduce the resident summary before any transfer')
    print(text)
