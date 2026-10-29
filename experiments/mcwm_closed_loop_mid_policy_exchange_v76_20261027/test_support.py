"""Visible actor provenance, real cut costs and detached suffix references."""
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.mcwm_closed_loop_mid_policy_exchange_v76_20261027 import support as S
from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace


def test_actor_uses_only_packet51_even_when_peer_processed_clock_is_later(tmp_path):
    path = tmp_path/'actor_at_packet51.pt'
    actor = torch.nn.Linear(2, 1)
    metadata = S.publish_actor(path, SimpleNamespace(actor=actor), seed=6208,
        arm='fresh', counts=dict(S.COUNTS), published_clock=16207., max_available_at=16199.)
    assert metadata['processed_at'] == 16207 and metadata['data_deadline'] == 16200
    assert metadata['complete_outcomes'] == 255 and not metadata['future_training_used']
    weights, selected = S.opposite_actor(path, seed=6208, arm='fresh')
    assert selected == metadata
    assert all(torch.equal(weights[key], value) for key, value in actor.state_dict().items())
    bad = torch.load(path, weights_only=False)
    bad['metadata']['counts']['actor'] = 456
    torch.save(bad, path)
    with pytest.raises(RuntimeError, match='future training'):
        S.opposite_actor(path, seed=6208, arm='fresh')


def test_final_actor_or_future_feedback_cannot_be_published(tmp_path):
    learner = SimpleNamespace(actor=torch.nn.Linear(1, 1))
    final = dict(S.COUNTS, actor=456)
    with pytest.raises(RuntimeError, match='exactly408'):
        S.publish_actor(tmp_path/'final.pt', learner, seed=6207, arm='no_fresh',
            counts=final, published_clock=16204, max_available_at=16200)
    with pytest.raises(RuntimeError, match='visible feedback deadline'):
        S.publish_actor(tmp_path/'future.pt', learner, seed=6207, arm='no_fresh',
            counts=dict(S.COUNTS), published_clock=16504, max_available_at=16500)


def test_suffix_reference_is_detached_but_issue_counter_is_inherited(tmp_path):
    source = tmp_path/'source.jsonl'
    source.write_text(json.dumps(dict(sequence=7))+'\n')
    old = NativePrefixTrace(tmp_path/'prefix.jsonl', source)
    old.observed_issues = old.matched_issues = 6
    child = S.trace_child(old, tmp_path/'child.jsonl')
    assert child.reference is None
    assert child.observed_issues == child.matched_issues == 6
    assert child.path.read_text() == ''
    child.close()


def test_remaining_cost_uses_actual_cut_not_a_fabricated16200_endpoint():
    cut = S.snapshot(SimpleNamespace(time_seconds=16207, accrued_wait_seconds=100.,
        accrued_in_vehicle_seconds=200., waiting_passengers=3, in_vehicle_passengers=8))
    metrics = [dict(time_seconds=t, accrued_wait_seconds=w, accrued_in_vehicle_seconds=r)
               for t,w,r in [(16200,90.,180.), (18300,120.,230.), (20100,130.,240.), (24000,150.,260.)]]
    points, costs = S.cost_windows(metrics, 15079, cut)
    assert set(points) == {'18300','20100','24000'}
    assert costs['remaining']['wait'] == 50 and costs['remaining']['ride'] == 60
    assert costs['remaining']['cost'] == 160
    assert costs['tail']['cost'] == 90 and costs['post20100']['cost'] == 60
    with pytest.raises(RuntimeError, match='Missing actual'):
        S.cost_windows(metrics[:-1], 15079, cut)


def test_only_measured_profiler_timing_is_removed_from_update_replay():
    a = dict(packet=51, processed_at=16203, td=dict(dt=7, loss=.2), regularizer_wall_seconds=.001)
    b = dict(a, regularizer_wall_seconds=.2)
    assert S.strip_profiler(a) == S.strip_profiler(b)
    assert S.strip_profiler(a) != S.strip_profiler(dict(b, processed_at=16204))
