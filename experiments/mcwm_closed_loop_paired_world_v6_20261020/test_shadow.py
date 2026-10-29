"""Detect mixed draws, incomplete optimization and leakage into active control."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_probes import learner
from experiments.mcwm_closed_loop_paired_world_v6_20261020.paired_world import PairedWorld
from experiments.mcwm_closed_loop_paired_world_v6_20261020.shadow import (
    BRANCHES, PacketShadowProbe, _learner_state, _run_packet,
    restore_rng, rng_state, transition_batch,
)


def packet(item, offset=0):
    table = item.replay_buffer.history
    return [SimpleNamespace(s=table['s'][i].copy(), a=table['a'][i].copy(),
        ns=table['ns'][i].copy(), r=float(table['r'][i]), dt=float(table['dt'][i]),
        terminal=False, available_at=9000.+offset, completed_at=8970.+offset)
        for i in range(5)]


def warmed():
    item = learner(enabled=False)
    physical = item.world.physical
    physical.shock_at = 9000.
    item.replay_buffer.recent.clear()
    item.replay_buffer.add_target(packet(item))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(601)
        item.world = PairedWorld(item.replay_buffer, seed=41, updates=1,
            action_transform=item.action_transform, physical=physical)
    # Initialize twin before live fitting, then make the two neural predictions
    # distinct. The intervention is in the network, never a local readout.
    item.world.refit(2)
    with torch.no_grad():
        for model in item.world.models.values():
            model.head[-1].bias[1].add_(.3)
    _run_packet(item)
    return item


def test_eight_steps_reproduce_active_adam_and_preserve_every_active_rng():
    item = warmed()
    ordinary = copy.deepcopy(item)
    frozen_before = {key: copy.deepcopy(model.state_dict())
                     for key, model in item.world.frozen_models.items()}
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
    # Passive twin training/logging also runs exactly once on the live learner.
    assert_same(item.world.twin_trust.net.state_dict(), ordinary.world.twin_trust.net.state_dict())
    assert_same(item.world.twin_trust.optimizer.state_dict(), ordinary.world.twin_trust.optimizer.state_dict())
    assert item.world.trust_updates == ordinary.world.trust_updates
    assert item.world.trust_records == ordinary.world.trust_records
    assert result['steps'] == result['active_model_batches'] == 8
    assert result['trust_weight_rows'] == 64
    assert result['online_exact_active'] is True
    assert result['common_state_rows_matched'] is True
    assert result['common_rng_streams_matched'] is True
    assert result['trust_weights_fixed_to_active'] is True
    assert result['common_family'] == item.world.family
    assert set(result['branches']) == set(BRANCHES)
    assert result['branches']['Frozen']['critic_parameter_difference_versus_online']['l2'] > 0
    assert result['branches']['Online']['executed_versus_online']['max_absolute_delta'] == 0
    assert probe.summary()['pending_anchor_count'] == 1
    for key, model in item.world.frozen_models.items():
        assert_same(frozen_before[key], model.state_dict())


def test_next_packet_uses_one_preupdate_target_and_no_future_learning():
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
        expected = {name: float((critic(batch['observations'], batch['actions'])-target[None])
                               .square().mean()) for name, critic in held['critics'].items()}
    # Changing the active value standard after the anchor must not move labels.
    with torch.no_grad():
        for p in item.target_critic.parameters():
            p.add_(.5)
    live_before = copy.deepcopy(_learner_state(item))
    rng_before = rng_state()
    result = probe.resolve_next_packet(future, 9300)
    assert_same(_learner_state(item), live_before)
    assert_same(rng_state(), rng_before)
    assert result[0]['rows'] == 5
    assert result[0]['common_frozen_anchor_target'] is True
    assert {name: row['mean_squared_td_error'] for name, row in result[0]['branches'].items()} == expected
    assert probe.summary()['pending_anchor_count'] == 0
    assert probe.resolve_next_packet(future, 9300) == []


def test_future_label_availability_and_complete_packet_gates():
    item = warmed()
    probe = PacketShadowProbe(deadlines=(8400,))
    assert probe.begin_packet(item, packet(item), 8100, 9000) is None
    token = probe.begin_packet(item, packet(item), 8400, 9000)
    _run_packet(item)
    probe.finish_packet(item, token)
    with pytest.raises(ValueError, match='before its outcomes'):
        probe.resolve_next_packet(packet(item, 300), 9200)
    assert probe.summary()['pending_anchor_count'] == 1
    token = probe.begin_packet(item, packet(item), 8400, 9000)
    item.continue_step(collect_diagnostics=True)
    with pytest.raises(RuntimeError, match='exactly eight'):
        probe.finish_packet(item, token)


def test_frozen_shadow_substitution_uses_anchor_snapshot_not_later_live_weights():
    item = warmed()
    ordinary = copy.deepcopy(item)
    probe = PacketShadowProbe(deadlines=(8400,))
    other_probe = PacketShadowProbe(deadlines=(8400,))
    before = rng_state()
    token = probe.begin_packet(item, packet(item), 8400, 9000)
    _run_packet(item)
    first = probe.finish_packet(item, token)
    restore_rng(before)
    other_token = other_probe.begin_packet(ordinary, packet(ordinary), 8400, 9000)
    _run_packet(ordinary)
    # There is no training path to the frozen snapshot; this deliberate later
    # mutation checks that the fork reads the captured anchor copy instead.
    with torch.no_grad():
        for model in ordinary.world.frozen_models.values():
            model.head[-1].bias[1].add_(100.)
    second = other_probe.finish_packet(ordinary, other_token)
    assert first == second
    for branch in BRANCHES:
        assert_same(probe._pending[0]['critics'][branch].state_dict(),
                    other_probe._pending[0]['critics'][branch].state_dict())
