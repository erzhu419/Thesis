"""Catch shadow optimizer/RNG leakage, false Local reproduction and moving labels."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_probes import learner
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    BRANCHES, FrozenBellmanEvaluator, PacketShadowProbe, _candidate_pair, _learner_state,
    _run_packet, restore_rng, rng_state, transition_batch,
)


def packet(item, offset=0):
    table = item.replay_buffer.history
    return [SimpleNamespace(s=table['s'][i].copy(), a=table['a'][i].copy(),
        ns=table['ns'][i].copy(), r=float(table['r'][i]), dt=float(table['dt'][i]),
        terminal=False, available_at=9000.+offset, completed_at=8970.+offset)
        for i in range(5)]


def warmed():
    item = learner()
    # A nonvacuous cost correction; other heads remain active as well.
    for readout in item.world.local_readouts.values():
        readout[:, 1].fill_(.3)
    item.continue_step(collect_diagnostics=True)
    item.world.capture_counterfactual = False
    for key in ('counterfactual_batch', 'head_counterfactual_batches',
                'head_counterfactual_weights', 'head_counterfactual_outputs'):
        setattr(item.world, key, None)
    return item


def test_complete_eight_steps_preserve_live_optimizer_and_all_rng_and_reproduce_local():
    item = warmed()
    ordinary = copy.deepcopy(item)
    probe = PacketShadowProbe(deadlines=(8400,))
    random.seed(725)
    np.random.seed(923)
    torch.manual_seed(849)
    initial = rng_state()
    _run_packet(ordinary)
    ordinary_rng = rng_state()
    restore_rng(initial)
    token = probe.begin_packet(item, packet(item), 8400, 9000)
    _run_packet(item)
    result = probe.finish_packet(item, token)
    assert_same(_learner_state(item), _learner_state(ordinary))
    assert_same(rng_state(), ordinary_rng)
    assert result['steps'] == result['active_model_batches'] == 8
    assert result['trust_weight_rows'] == 64
    assert result['local_exact_active'] is True
    assert set(result['branches']) == set(BRANCHES)
    assert result['branches']['Local-C']['critic_parameter_difference_versus_local']['l2'] > 0
    assert result['branches']['Base']['critic_parameter_difference_versus_local']['l2'] > 0
    assert result['branches']['Local']['executed_versus_local']['max_absolute_delta'] == 0
    assert probe.summary()['pending_anchor_count'] == 1


def test_future_packet_uses_one_frozen_anchor_target_before_training_and_is_noninvasive():
    item = warmed()
    probe = PacketShadowProbe(deadlines=(8400,))
    token = probe.begin_packet(item, packet(item), 8400, 9000)
    _run_packet(item)
    probe.finish_packet(item, token)
    future = packet(item, 300)
    held = probe._pending[0]
    batch = transition_batch(future, item.device)
    target = held['evaluator'].target(batch)
    with torch.no_grad():
        expected = {name: float((critic(batch['observations'], batch['actions'])-target[None]).abs().mean())
                    for name, critic in held['critics'].items()}
    live_before = copy.deepcopy(_learner_state(item))
    rng_before = rng_state()
    result = probe.resolve_next_packet(future, 9300)
    assert_same(_learner_state(item), live_before)
    assert_same(rng_state(), rng_before)
    assert len(result) == 1 and result[0]['rows'] == 5
    assert result[0]['common_frozen_anchor_target'] is True
    assert {name: value['mean_absolute_td_error'] for name, value in result[0]['branches'].items()} == expected
    assert probe.summary()['pending_anchor_count'] == 0
    assert probe.resolve_next_packet(future, 9300) == []


def test_future_label_not_yet_available_fails_without_dropping_pending_anchor():
    item = warmed()
    probe = PacketShadowProbe(deadlines=(8400,))
    token = probe.begin_packet(item, packet(item), 8400, 9000)
    _run_packet(item)
    probe.finish_packet(item, token)
    with pytest.raises(ValueError, match='before its outcomes'):
        probe.resolve_next_packet(packet(item, 300), 9200)
    assert probe.summary()['pending_anchor_count'] == 1


def test_frozen_target_uses_matching_draws_and_no_replay_or_world_and_marks_merged_candidates():
    item = warmed()
    frozen = FrozenBellmanEvaluator(item, seed=931)
    assert not hasattr(frozen.learner, 'world') or frozen.learner.world is None
    assert not hasattr(frozen.learner, 'replay_buffer')
    batch = transition_batch(packet(item), item.device)
    before = rng_state()
    first = frozen.target(batch)
    torch.rand(13)
    np.random.rand(3)
    after_external = rng_state()
    second = frozen.target(batch)
    assert torch.equal(first, second)
    assert_same(rng_state(), after_external)
    restore_rng(before)
    # Reachable v19 completion floor makes every proposal execute 60 seconds.
    from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import state
    s = torch.as_tensor(state(time=9000., bwd_present=0, load=5.)[1][None])
    _, _, valid, separation = _candidate_pair(s, item.action_transform)
    assert not valid.any() and float(separation[0]) == 0
