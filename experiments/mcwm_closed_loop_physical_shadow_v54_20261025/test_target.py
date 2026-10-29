"""Detect target/optimizer parity loss, unsupported credit and RNG leakage."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import attach_shared_level
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial
from experiments.mcwm_closed_loop_paired_recovery_v37r1_20261024.test_run import transition
from experiments.mcwm_closed_loop_sync_collection_v53_20261025.evaluator import FrozenIssueEvaluator
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state
from experiments.mcwm_closed_loop_physical_credit_v50_20261025.credit import physical_trace
from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import (
    CAPACITY, ETA, install_reference_sync_m8, install_sync_target,
    restore_target_state, set_lambda, target_state, target_stats,
)


def ready():
    torch.set_num_threads(1)
    model = attach_shared_level(initial())
    model.config = replace(model.config, batch_size=256)
    model.total_updates = 50024
    evaluator = FrozenIssueEvaluator(model, 1800.)
    rows, clock = [], 9000.
    current = transition(1, clock)
    for index in range(CAPACITY):
        row = deepcopy(current)
        duration = 0. if index % 3 == 0 else 2.
        row.update(sequence=index+1, started_at=clock, completed_at=clock+duration,
            available_at=clock+duration+30., dt=duration, r=-3.-index*.01,
            line_id=('7X', '102X', '705X')[index % 3])
        row['ns'][0] = (clock+duration)/24000.
        row['memory_ns'] = dict(vector=[.2+index*.001]*84, observed_at=clock+duration)
        rows.append(row)
        clock += duration
        current = deepcopy(row)
        current['s'], current['memory_s'] = deepcopy(row['ns']), deepcopy(row['memory_ns'])
    # The replay retains exactly fourteen ordinary labels and 271 new labels.
    ordinary = [SimpleNamespace(**deepcopy(rows[0])) for _ in range(14)]
    model.replay_buffer.add_target(ordinary+[SimpleNamespace(**row) for row in rows])
    return model, evaluator, rows


def draw(model):
    offline, source = model.replay_buffer.sample_mixed(256, .5, rng=model.rng)
    return model._concatenate_batches(offline, source)


def test_lambda_zero_eight_actual_updates_preserve_parameters_Adam_and_training_RNG():
    origin, evaluator, rows = ready()
    candidate = install_sync_target(deepcopy(origin), evaluator, rows, [1.]*270)
    reference = install_reference_sync_m8(deepcopy(origin), evaluator)
    future = candidate.future_rng.clone(), candidate.future_extra_rng.clone()
    for _ in range(8):
        start = torch.get_rng_state().clone()
        reference.continue_step()
        end = torch.get_rng_state().clone()
        torch.set_rng_state(start)
        candidate.continue_step()
        assert torch.equal(end, torch.get_rng_state())
        assert _equal(_state(reference), _state(candidate))
        assert torch.equal(reference.mean8_extra_rng_state, candidate.mean8_extra_rng_state)
        assert reference.replay_buffer.last_sampling == candidate.replay_buffer.last_sampling
        assert torch.equal(candidate.target_last['native'], candidate.target_last['target'])
    assert target_stats(candidate)['future_calls'] == 0
    assert torch.equal(future[0], candidate.future_rng)
    assert torch.equal(future[1], candidate.future_extra_rng)
    assert all(not parameter.requires_grad for parameter in candidate.sync_actor.parameters())


def test_trace_keeps_native_first_draw_and_extra_rng_and_matches_independent_real_chain():
    origin, evaluator, rows = ready()
    single = install_sync_target(deepcopy(origin), evaluator, rows, [1.]*270)
    trace = install_sync_target(deepcopy(origin), evaluator, rows, [1.]*270, lambda_trace=1.)
    batch = draw(single)
    trace.replay_buffer.last_sampling = deepcopy(single.replay_buffer.last_sampling)
    start = torch.get_rng_state().clone()
    native = single._target(batch, backup='h2oplus')
    end = torch.get_rng_state().clone()
    torch.set_rng_state(start)
    actual = trace._target(batch, backup='h2oplus')
    assert torch.equal(end, torch.get_rng_state())
    assert torch.equal(single.mean8_extra_rng_state, trace.mean8_extra_rng_state)
    assert torch.equal(native, trace.target_last['native'])
    ids = trace.target_last['continuous_ids']
    assert torch.equal(native[ids < 0], actual[ids < 0])
    assert bool((native[ids >= 0] != actual[ids >= 0]).any())
    future = trace.target_last['future']
    supplied = [dict(native_target=float(future['native'][i]),
        reward_scaled=float(future['reward'][i]), bootstrap_contribution=float(future['bootstrap'][i]),
        recorded_next_target_q=float(future['matching_q'][i]) if i < CAPACITY-1 else None,
        terminal=False, dt=rows[i]['dt']) for i in range(CAPACITY)]
    expected = physical_trace(supplied, [1.]*270, 1.)
    assert np.array_equal(np.asarray(expected['targets'], np.float32), future['physical'].numpy())
    scale = actual.new_tensor(ETA)
    selected = ids >= 0
    assert torch.equal(actual[selected], native[selected]+scale*(
        future['physical']-future['native'])[ids[selected]])
    assert sorted(future['target_subset'].tolist()) == [0, 1]
    assert target_stats(trace)['trace_corrected_rows'] == int(selected.sum())
    json.dumps(target_stats(trace), allow_nan=False)


def test_missing_support_breaks_link_without_removing_zero_time_or_final_bootstrap():
    origin, evaluator, rows = ready()
    ratios = [1.]*270
    ratios[0] = None
    model = install_sync_target(origin, evaluator, rows, ratios, lambda_trace=1.)
    assert len(model.physical_batch['rewards']) == 271
    assert model.physical_coefficients[0] == 0
    assert model.physical_coefficients[3] == 1  # Supported zero-duration link remains.
    assert model.physical_batch['terminals'].sum() == 0
    batch = {key: value[[0, 1, 270]] for key, value in model.physical_batch.items()}
    batch['continuous_ids'] = torch.tensor([0, 1, 270])
    result = model._target(batch, backup='h2oplus')
    assert result[0] == model.target_last['native'][0]
    assert result[-1] == model.target_last['native'][-1]
    assert model.target_last['future']['bootstrap'][-1] != 0


def test_target_state_restores_streams_and_frozen_actor_isolated_from_live_actor():
    model, evaluator, rows = ready()
    model = install_sync_target(model, evaluator, rows, [1.]*270, lambda_trace=1.)
    batch = draw(model)
    state, main = target_state(model), torch.get_rng_state().clone()
    first = model._target(batch, backup='h2oplus')
    frozen = deepcopy(model.sync_actor.state_dict())
    restore_target_state(model, state)
    torch.set_rng_state(main)
    with torch.no_grad():
        next(model.actor.parameters()).add_(1.)
    second = model._target(batch, backup='h2oplus')
    assert torch.equal(first, second)
    assert _equal(frozen, model.sync_actor.state_dict())
    assert evaluator.unchanged()
    set_lambda(model, 0.)
    future = model.future_rng.clone()
    model._target(batch, backup='h2oplus')
    assert torch.equal(future, model.future_rng)


def test_changed_chain_or_wrong_recent_identity_cannot_receive_trace():
    model, evaluator, rows = ready()
    bad = deepcopy(rows)
    bad[4]['memory_s']['vector'][0] += 1.
    with pytest.raises(ValueError, match='adjacent'):
        install_sync_target(deepcopy(model), evaluator, bad, [1.]*270)
    model = install_sync_target(model, evaluator, rows, [1.]*270, lambda_trace=1.)
    batch = draw(model)
    index = 64+next(i for i, value in enumerate(model.replay_buffer.last_sampling['recent']) if value >= 14)
    batch['next_observations'][index, 229] += 1.
    with pytest.raises(RuntimeError, match='memory mismatch'):
        model._target(batch, backup='h2oplus')
