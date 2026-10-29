"""Resident exact comparison of the two new execution-parity suffixes."""
import argparse
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'mcwm_closed_loop_service_shift_v21_20261022'
FIELDS=('selected','prefix','prediction','service_history','service_prediction','execution',
        'truth','global_transition','service_shift','native')


def analyze():
    cases=[]
    for arm in ('low','high'):
        folder=HERE/'parity'/arm
        if (folder/'failure.json').exists():
            return dict(status='failed',integrity=dict(all_pass=False),failure=json.loads((folder/'failure.json').read_text()))
        f=folder/'summary.json'
        if not f.exists():return dict(status='incomplete')
        value=json.loads(f.read_text())
        old=json.loads((ORIGINAL/'stage_a/runs/stable/seed6201/training_branches/0'/arm/'summary.json').read_text())
        checks={k:value.get(k)==old.get(k) for k in FIELDS}
        checks['native_integrity']=value['integrity']['all_pass'] is True
        cases.append(dict(arm=arm,checks=checks,budget=value['budget']))
    passed=all(all(c['checks'].values()) for c in cases)
    return dict(status='complete' if passed else 'integrity_failed',integrity=dict(all_pass=passed),cases=cases,
        budget=dict(new_outcomes=2,suffix_seconds=sum(c['budget']['suffix_sim_seconds'] for c in cases),
            prefix_replay_seconds=sum(c['budget']['prefix_sim_seconds'] for c in cases),
            replayed_target_labels=sum(c['budget']['replay_target_labels'] for c in cases),
            replayed_policy_updates=sum(c['budget']['replay_policy_updates'] for c in cases),
            labels_enter_training=0),
        scope='Execution equivalence only; no new training labels and no scientific performance evaluation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    while True:
        value=analyze()
        if value['status']!='incomplete' or not args.wait:break
        time.sleep(20)
    data=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    assert len(data.encode())<6000
    with (HERE/'parity_gate.json').open('x') as f:f.write(data)
    print(json.dumps(dict(status=value['status'],integrity=value.get('integrity'),bytes=len(data.encode()))))
