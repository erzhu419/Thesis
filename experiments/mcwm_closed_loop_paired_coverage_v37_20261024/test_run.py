"""Budgeted new facts, legal zero-duration TD and post-freeze heldout isolation."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_paired_coverage_v37_20261024 import run
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import attach_shared_level
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import attach_memory
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _restore, _state


def transition(sequence, issue, *, arm=None, line='7X'):
    value = rows(.6)[0]
    duration = 0. if arm == 'low' else 1.
    state, next_state = value.s.copy(), value.ns.copy()
    state[0], next_state[0] = issue/24000., (issue+duration)/24000.
    return dict(sequence=sequence, line_id=line, started_at=issue,
        completed_at=issue+duration, available_at=issue+duration+30.,
        s=state.tolist(), ns=next_state.tolist(), a=[-1. if arm == 'low' else -.9375],
        r=0. if arm == 'low' else -3., dt=duration, terminal=False,
        memory_s=dict(vector=[.1]*84, observed_at=issue),
        memory_ns=dict(vector=[.2]*84, observed_at=issue+duration))


def dataset():
    packets, ordinary = [], 1
    for index in range(57):
        processed = 1200.+300.*index
        packet_rows = []
        if index < 9:
            for offset, arm in ((1, 'low'), (2, 'high')):
                packet_rows.append(transition(-(2*index+offset), processed-330., arm=arm))
        for _ in range(5-len(packet_rows)):
            packet_rows.append(transition(ordinary, processed-31.))
            ordinary += 1
        packets.append(dict(index=index+1, receipt=dict(processed_at=processed,
            sequence_ids=[value['sequence'] for value in packet_rows]), rows=packet_rows))
    return dict(proof=dict(all_pass=True, checks=dict(fork_parent_invariant=True)),
        metadata=dict(scenario='stable', initialization_seed=6202), packets=packets)


def heldout():
    result = []
    for index, (line, when) in enumerate((line, when) for line in ('7X', '102X', '705X')
                                        for when in ('early', 'tail')):
        issue = (9400. if when == 'early' else 18400.)+index
        pair = {arm: transition(1000+index*2+offset, issue, arm=arm, line=line)
                for offset, arm in enumerate(('low', 'high'))}
        result.append(dict(slot=line+'/'+when,
            selected=dict(s=pair['low']['s'], memory_s=pair['low']['memory_s'], sequence=1000+index*2),
            actions={arm: value['a'] for arm, value in pair.items()}, holds=dict(low=0., high=1.875),
            paired_transitions=pair))
    return result


def test_new_285_slot_stream_keeps_zero_time_arms_and_enforces_pair_release(tmp_path):
    path = tmp_path/'dataset.json'
    value = dataset()
    path.write_text(json.dumps(value))
    actual = run.load_dataset(path, scenario='stable', seed=6202)
    assert actual['ordinary_rows'] == 267 and actual['paired_rows'] == 18
    assert actual['paired_zero_duration_rows'] == 9
    assert all(pair['low'].dt == 0. for pair in actual['pairs'])
    assert actual['pairs'][0]['low'].sequence == -1 and actual['pairs'][0]['high'].sequence == -2
    # Next-event labels already exist, but the independently measured 300s
    # pair endpoint is still one second short of its release requirement.
    changed = deepcopy(value)
    for record in changed['packets'][0]['rows'][:2]:
        for key in ('started_at', 'completed_at', 'available_at'):
            record[key] += 1.
        record['memory_s']['observed_at'] += 1.
        record['memory_ns']['observed_at'] += 1.
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='before both300-second'):
        run.load_dataset(path, scenario='stable', seed=6202)


def test_actual_pair_scoring_retains_zero_dt_native_bootstrap_and_rng():
    learner = attach_shared_level(initial())
    run.freeze(learner)
    pair = {arm: run.row_from_record(transition(-offset, 18400., arm=arm))
            for arm, offset in (('low', 1), ('high', 2))}
    before, rng = deepcopy(_state(learner)), _rng()
    result = run.pair_prediction(learner, pair, seed=336202)
    assert result['prediction']['holding_seconds'] == [0., 1.875]
    low, high = (result['bellman_by_action'][arm] for arm in ('low', 'high'))
    assert low['dt'] == 0 and not low['terminal'] and low['reward_scaled'] == 0.
    assert low['target'] == low['bootstrap_contribution']
    assert result['bellman_high_minus_low']['reward_scaled'] == high['reward_scaled']
    assert len(result['prediction']['min_heads']) == 2
    assert _equal(before, _state(learner)) and _equal(rng, _rng())


def test_full_worker_456_steps_never_opens_heldout_until_frozen_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    learner = attach_shared_level(initial())
    learner.total_updates = 50000
    monkeypatch.setattr(run, 'make_learner', lambda *args, **kwargs:
        (learner, learner.replay_buffer, None, None, 'same_resident_initializer'))
    facts, inputs, out = tmp_path/'dataset.json', tmp_path/'heldout_prediction_inputs.json', tmp_path/'worker'
    facts.write_text(json.dumps(dataset()))
    inputs.write_text(json.dumps(heldout()))
    original_read = Path.read_text
    opened = []

    def read(path, *args, **kwargs):
        if path == inputs:
            assert learner.total_updates == 50456 and (out/'final_model.pt').exists()
            assert all(not value.requires_grad for name in run.MODULES for value in getattr(learner, name).parameters())
            opened.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    run.worker(SimpleNamespace(out=out, seed=6202, scenario='stable', child_arm='shared_level',
        dataset=facts, heldout=inputs), dict(diagnostic_seed=336202))
    result = json.loads((out/'summary.json').read_text())
    assert result['integrity']['all_pass'] and opened == [inputs]
    assert result['budget']['feedback_slots'] == 285 and result['budget']['shadow_updates'] == 456
    assert len(run._read_lines(out/'heldout.jsonl')) == 6
    targets = run._read_lines(out/'pair_targets.jsonl')
    assert len(targets) == 27
    assert {value['phase'] for value in targets} == {'packet_before', 'packet_after', 'frozen_final'}
    assert all(value['bellman_by_action']['low']['dt'] == 0 for value in targets)


def test_matched_worker_gate_detects_any_sampling_or_rng_disagreement(tmp_path):
    common = [dict(packet=index+1, receipt=dict(processed_at=1200+index*300), updates=8*(index+1),
        sampling=[dict(recent=[step]) for step in range(8)], rng=dict(torch=[index], sampler=[1]),
        optimizer_steps=dict(critic=dict(states=2, min=8*(index+1), max=50008+index*8))) for index in range(57)]
    for arm in run.ARMS:
        root = tmp_path/arm
        root.mkdir()
        (root/'packets.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in common))
        (root/'scores.jsonl').write_text(json.dumps(dict(packet=1, phase='packet_before', rows=[dict(q=1.)]))+'\n')
    assert len(run.compare_workers(tmp_path)['zero_memory']) == 57
    path = tmp_path/'shared_level'/'packets.jsonl'
    changed = deepcopy(common)
    changed[1]['sampling'][7]['recent'] = [99]
    path.write_text(''.join(json.dumps(value)+'\n' for value in changed))
    with pytest.raises(RuntimeError, match='packet2'):
        run.compare_workers(tmp_path)


def test_three_native_trainers_keep_same_sampler_rng_with_zero_time_pairs():
    torch.set_num_threads(1)
    origin = initial()
    learners = dict(zero_memory=deepcopy(origin), per_head_level=attach_memory(deepcopy(origin)),
                    shared_level=attach_shared_level(deepcopy(origin)))
    initial_rng, traces, first_scores = _rng(), {}, []
    packets = dataset()['packets'][:2]
    for arm, learner in learners.items():
        _restore(initial_rng)
        traces[arm] = []
        for index, packet in enumerate(packets):
            admitted = [run.row_from_record(value) for value in packet['rows']]
            if index == 0:
                first_scores.append(run.probe(learner, admitted, seed=336202))
            learner.replay_buffer.add_target(admitted)
            sampling = []
            for step in range(8):
                learner.continue_step(collect_diagnostics=step == 0)
                sampling.append(deepcopy(learner.replay_buffer.last_sampling))
            traces[arm].append(dict(sampling=sampling, rng=run._training_rng(learner)))
    assert all(value == first_scores[0] for value in first_scores[1:])
    assert all(value == traces['zero_memory'] for value in traces.values())
    assert run.optimizer_steps(learners['per_head_level']) == run.optimizer_steps(learners['shared_level'])
