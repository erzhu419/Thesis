"""Check native-first parity, joint reduction, persistent extra RNG and pickling."""
from copy import deepcopy
import io
import os
from pathlib import Path

import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import (
    MemoryOutcomeTrainer, critic_states,
)
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.test_frozen import learner
from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_probe import chain
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _batch, _equal, _rng, _state,
)
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import row_from_record
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.mean_target import (
    Mean8OutcomeTrainer, install_mean8, mean8_state, mean8_stats,
    native_scoring, restore_mean8_state,
)


def batch_for(model):
    first, second = [row_from_record(value) for value in chain()]
    batch = _batch([first, second], model.device)
    memory = torch.tensor([first.memory_ns['vector'], second.memory_ns['vector']])
    batch['next_observations'] = critic_states(model, batch['next_observations'], memory)
    return batch


def test_native_m1_path_is_untouched_and_m8_keeps_first_target_bits_and_training_rng():
    native = learner()
    assert native.__class__ is MemoryOutcomeTrainer
    candidate = deepcopy(native)
    before_model, rng = deepcopy(_state(candidate)), _rng()
    install_mean8(candidate, seed=476202)
    assert _equal(rng, _rng()) and _equal(before_model, _state(candidate))
    assert native.__class__ is MemoryOutcomeTrainer
    assert '_target' not in Mean8OutcomeTrainer.__dict__
    batch = batch_for(native)
    start = torch.get_rng_state()
    expected = native._target(batch, backup=native.config.online_backup)
    expected_end = torch.get_rng_state()
    torch.set_rng_state(start)
    actual = candidate._target(batch, backup=candidate.config.online_backup)
    assert torch.equal(expected_end, torch.get_rng_state())
    last = candidate.mean8_last
    canonical = batch['next_observations'][:, :229]
    bootstrap = .1*candidate.value(canonical)
    bootstrap = bootstrap+.9*last['native_sac']
    discount = batch['discounts'].reshape(-1)*(1.-batch['terminals'].reshape(-1))
    reconstructed = candidate.config.reward_scale*batch['rewards'].reshape(-1)+discount*bootstrap
    assert torch.equal(expected, reconstructed)
    joint_mean = last['positive_joint'].mean(0)
    mean_sac = last['zero_probability']*last['zero_value']+last['hold_probability']*joint_mean
    assert torch.equal(last['positive_joint_mean'], joint_mean)
    assert torch.equal(last['mean8_sac'], mean_sac)
    assert torch.equal(actual, candidate.config.reward_scale*batch['rewards'].reshape(-1)
        +discount*(.1*candidate.value(canonical)+.9*mean_sac))
    assert last['native_q_batch_rows'] == 2*len(actual)
    assert sorted(last['target_subset'].tolist()) == [0, 1]
    stats = mean8_stats(candidate)
    assert stats['calls'] == 1 and stats['extra_q_forwards'] == 7
    assert stats['extra_actor_forwards'] == 7 and stats['extra_target_head_draws'] == 0
    assert stats['extra_positive_samples'] == 7*len(actual)
    assert _equal(before_model, _state(candidate))


def test_extra_stream_advances_across_calls_restores_and_survives_checkpoint_pickle():
    model = install_mean8(learner())
    batch = batch_for(model)
    initial = mean8_state(model)
    global_rng = torch.get_rng_state()
    first = model._target(batch, backup=model.config.online_backup)
    after_first = mean8_state(model)
    torch.set_rng_state(global_rng)
    second = model._target(batch, backup=model.config.online_backup)
    assert not torch.equal(initial['extra_rng'], after_first['extra_rng'])
    assert not torch.equal(after_first['extra_rng'], model.mean8_extra_rng_state)
    assert not torch.equal(after_first['last']['positive_joint'][1:], model.mean8_last['positive_joint'][1:])
    assert model.mean8_compute['extra_q_forwards'] == 14
    restore_mean8_state(model, initial)
    torch.set_rng_state(global_rng)
    assert torch.equal(first, model._target(batch, backup=model.config.online_backup))
    saved = io.BytesIO()
    torch.save(model, saved)
    saved.seek(0)
    restored = torch.load(saved, weights_only=False, map_location='cpu')
    assert restored.__class__ is Mean8OutcomeTrainer
    assert _equal(mean8_state(model), mean8_state(restored))
    restore_mean8_state(restored, after_first)
    torch.set_rng_state(global_rng)
    assert torch.equal(second, restored._target(batch, backup=restored.config.online_backup))


def test_native_scoring_keeps_parameters_extra_state_and_all_global_rng():
    model = install_mean8(learner())
    batch = batch_for(model)
    model._target(batch, backup=model.config.online_backup)
    private, global_rng = mean8_state(model), _rng()
    params = deepcopy(_state(model))
    with native_scoring(model):
        assert model.__class__ is MemoryOutcomeTrainer
        model._target(batch, backup=model.config.online_backup)
    assert model.__class__ is Mean8OutcomeTrainer
    assert _equal(private, mean8_state(model)) and _equal(global_rng, _rng())
    assert _equal(params, _state(model))
    native = learner()
    with native_scoring(native):
        assert native.__class__ is MemoryOutcomeTrainer


def test_true_terminal_still_has_zero_bootstrap():
    model = install_mean8(learner())
    batch = batch_for(model)
    batch['terminals'].fill_(1.)
    actual = model._target(batch, backup=model.config.online_backup)
    assert torch.equal(actual, model.config.reward_scale*batch['rewards'].reshape(-1))


def test_native_update_routing_retains_adam_and_sampler_rng_across_two_packets():
    from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial, rows
    from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import attach_shared_level

    native = attach_shared_level(initial())
    candidate = install_mean8(deepcopy(native))
    adam_id = id(candidate.critic_optimizer)
    parameter_ids = [id(value) for group in candidate.critic_optimizer.param_groups for value in group['params']]
    initial_state = mean8_state(candidate)['extra_rng']
    for packet in range(2):
        values = rows(.72+packet*.02)
        native.replay_buffer.add_target(values)
        candidate.replay_buffer.add_target(values)
        for step in range(2):
            start = torch.get_rng_state()
            native.continue_step(collect_diagnostics=step == 0)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(start)
            candidate.continue_step(collect_diagnostics=step == 0)
            assert torch.equal(expected_rng, torch.get_rng_state())
            assert _equal(native.rng.bit_generator.state, candidate.rng.bit_generator.state)
            assert _equal(native.diagnostic_rng.bit_generator.state, candidate.diagnostic_rng.bit_generator.state)
            assert _equal(native.replay_buffer.last_sampling, candidate.replay_buffer.last_sampling)
        assert id(candidate.critic_optimizer) == adam_id
        assert parameter_ids == [id(value) for group in candidate.critic_optimizer.param_groups for value in group['params']]
    assert native.total_updates == candidate.total_updates
    # Native online updates concatenate factual and source partitions and
    # dispatch their shared mixed-backup target once per optimizer update.
    assert candidate.mean8_compute['calls'] == 4
    assert candidate.mean8_compute['extra_q_forwards'] == 28
    assert not torch.equal(initial_state, candidate.mean8_extra_rng_state)
    assert any(not torch.equal(left, right) for left, right in zip(native.critic.parameters(), candidate.critic.parameters()))
