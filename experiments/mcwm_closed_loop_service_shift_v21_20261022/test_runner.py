"""Stage separation and two-slot cache provenance for the actual v21 runner."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('v21_native_runner_test',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_stage_a_never_constructs_or_fits_a_300_second_candidate():
    p=runner.load_protocol()
    assert list(runner.make_readouts(p,service_only=True))==['shared','route']
    assert list(runner.make_readouts(p))==['shared','route','candidate']
    assert p['preflight']['target_seed_offset']+6201==85201
    assert p['target_seed_offset']+6202==86202
    assert (p['target_record_budget'],p['horizon_seconds'],p['initializer_updates'])==(285,24000,50000)
    assert p['compute_budget']['total_training_labels_maximum']==1140


def test_native_plan_keeps_nine_two_slot_pairs_and_six_heldouts():
    p=runner.load_protocol();plan=runner.planned_cells(p,6201)
    assert sum(r['role']=='train' for r in plan)==9
    assert sum(r['role']=='heldout' for r in plan)==6
    assert p['training_outcome_slots_per_pair']==2
    assert min(r['window']['start'] for r in plan if r['role']=='heldout')>18000


def transition():
    return runner.transition_from_dict(dict(s=[0.]*229,a=[-1.],ns=[1.]*229,r=-1.,dt=2.,terminal=False,
        episode_id=0,domain='target',started_at=1250.,completed_at=1252.,available_at=1282.,
        vehicle_id='bus',line_id='7X',station_id='stop',next_vehicle_id='other',sequence=20,
        raw_s=[0.]*229,raw_ns=[1.]*229,trip_id='trip'))


def test_two_slot_pair_cache_keeps_issue_history_and_delayed_service_targets(tmp_path):
    budget=runner.PairPacketBudget(runner.load_protocol());budget.reserve(20,'train/7X/early')
    pair=dict(id='train/7X/early',index=0,role='train',source_sequence=20,issued_at=1250.,line_id='7X',
        features=[1.,0.,0.,.2],service_history=dict(status='supported',mean_residual_scaled=-.1),
        service_targets=dict(stations=['a','b','c'],arrival=[1.,None,-2.],departure=[2.,None,-3.]),
        outcomes={arm:dict(truth=dict(status='complete',dt=300.,wait=100.,ride=200.,cost=400.),
            transition=transition()) for arm in ('low','high')})
    normalized=budget.add_pair(pair)
    (tmp_path/'pair_cache').mkdir()
    (tmp_path/'pair_cache/0.json').write_text(json.dumps(runner.encode_pair(normalized)))
    restored=runner.cached_pair(tmp_path,0)
    assert restored['service_history']==pair['service_history']
    assert restored['service_targets']==pair['service_targets']
    assert restored['available_at']==1580. and restored['selected_for_training'] is False
    assert [restored['outcomes'][a]['transition'].sequence for a in ('low','high')]==[-1,-2]
    assert restored['outcomes']['low']['transition'].a.dtype==np.float32
