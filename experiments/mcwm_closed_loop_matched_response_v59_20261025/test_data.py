"""Detect outcome leakage and broken two-slot accounting before paid collection."""
import json
from pathlib import Path

import pytest

from experiments.mcwm_closed_loop_matched_response_v59_20261025.data import load_dataset
from experiments.mcwm_closed_loop_matched_response_v59_20261025.audit import overlap


def dataset():
    packets, ids = [], list(range(-18,0))+list(range(267))
    # Nine pairs in separate five-slot packets.
    ordinary = iter(range(267))
    for index in range(57):
        selected = [-(2*index+1), -(2*index+2)] if index < 9 else []
        selected += [next(ordinary) for _ in range(5-len(selected))]
        rows = [dict(s=[0.]*229, a=[-1.], memory_s=dict(observed_at=100.,vector=[0.]*84),
                     sequence=sequence,line_id='7X',started_at=100.) for sequence in selected]
        packets.append(dict(index=index+1,rows=rows,receipt=dict(processed_at=1200+index*300,
            sequence_ids=selected)))
    return dict(proof=dict(all_pass=True),metadata=dict(scenario='stable',initialization_seed=6202),packets=packets)


def load(tmp_path, payload):
    path=tmp_path/'covariates_dataset.json'
    path.write_text(json.dumps(payload))
    return load_dataset(path,arm='no_fresh',scenario='stable',seed=6202)


def test_outcome_free_complete_budget(tmp_path):
    result=load(tmp_path,dataset())
    assert len(result['packets']) == 57
    assert not hasattr(result['packets'][0]['rows'][0], 'r')


@pytest.mark.parametrize('field', ['r','ns','dt','terminal','memory_ns','truth'])
def test_no_fresh_rejects_outcome_fields(tmp_path,field):
    payload=dataset();payload['packets'][0]['rows'][0][field]=0
    with pytest.raises(ValueError,match='only current covariates'):
        load(tmp_path,payload)


def test_duplicate_slot_fails(tmp_path):
    payload=dataset();payload['packets'][0]['rows'][1]['sequence']=-1
    with pytest.raises(ValueError,match='receipt'):
        load(tmp_path,payload)


def test_parallel_overlap_uses_update_intervals():
    assert overlap([(0,2),(4,6)],[(1,3),(5,7)]) == 2
    assert overlap([(0,2)],[(2,4)]) == 0
