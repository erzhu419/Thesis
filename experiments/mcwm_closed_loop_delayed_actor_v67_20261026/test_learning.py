"""Check the real delayed schedule, optimizer continuity, sampling and targets."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_delayed_actor_v67_20261026 import learning as v
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import make_replay, table
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.test_learning import exact
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


def trainer(arm, *, vanilla=False, fresh=None):
    torch.set_num_threads(1)
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        bootstrap_probability=1., offline_objective='iql', online_backup='h2oplus',
        mixed_backup_lambda=.1, online_fraction=.5, online_update_order='actor-before-critic')
    cls = v.FreshValueTrainer if vanilla else v.DelayedActorTrainer
    learner = cls(make_replay(arm, fresh=fresh), config=config, device='cpu', seed=42,
        action_transform=v.ExecutionAction(TransitMDPConfigV7()))
    if vanilla:
        learner.config = replace(learner.config, **v.SCHEDULE)
    else:
        learner.activate_delayed_schedule()
    return learner


def optimizer_steps(optimizer):
    return [int(state['step']) for state in optimizer.state.values() if 'step' in state]


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_same_native_delayed_schedule_reproduces_parameters_adam_and_rng_exact(arm):
    vanilla, audited = trainer(arm, vanilla=True), trainer(arm)
    for packet in range(2):
        before = torch.get_rng_state().clone()
        audited.begin_actor_packet()
        assert torch.equal(before, torch.get_rng_state())
        reference = {key: value.clone() for key, value in audited._packet_actor.state_dict().items()}
        for step in range(8):
            before = torch.get_rng_state().clone()
            expected = vanilla.continue_step(collect_diagnostics=step == 0)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = audited.continue_step(collect_diagnostics=step == 0)
            exact(expected, actual)
            assert torch.equal(after, torch.get_rng_state())
            assert vanilla.rng.bit_generator.state == audited.rng.bit_generator.state
            assert vanilla.replay_buffer.last_sampling == audited.replay_buffer.last_sampling
            exact(reference, audited._packet_actor.state_dict())
            assert actual['train/actor_updated'] == float(step == 7)
            assert not any(key.startswith('train/actor_packet_') for key in actual)
        for name in ('actor', 'critic', 'target_critic', 'value'):
            exact(getattr(vanilla, name).state_dict(), getattr(audited, name).state_dict())
        for name in ('actor', 'critic', 'value', 'alpha'):
            exact(getattr(vanilla, name+'_optimizer').state_dict(),
                  getattr(audited, name+'_optimizer').state_dict())
        exact(vanilla.log_alpha, audited.log_alpha)
    assert audited.actor_schedule_audit()['actor_update_after_critic_steps'] == [8, 8]


def test_first_seven_critic_steps_leave_actor_alpha_and_their_adam_untouched():
    learner = trainer('fresh')
    learner.begin_actor_packet()
    before_actor = {name: value.clone() for name, value in learner.actor.state_dict().items()}
    before_alpha = learner.log_alpha.detach().clone()
    before_actor_adam = learner.actor_optimizer.state_dict()
    before_alpha_adam = learner.alpha_optimizer.state_dict()
    before_critic = {name: value.clone() for name, value in learner.critic.state_dict().items()}
    for _ in range(7):
        learner.continue_step()
    exact(before_actor, learner.actor.state_dict())
    exact(before_alpha, learner.log_alpha)
    exact(before_actor_adam, learner.actor_optimizer.state_dict())
    exact(before_alpha_adam, learner.alpha_optimizer.state_dict())
    assert any(not torch.equal(value, learner.critic.state_dict()[key])
               for key, value in before_critic.items())
    learner.continue_step()
    audit = learner.packet_actor_audit()
    assert audit == dict(packet=1, value_updates=8, critic_updates=8, actor_updates=1,
        alpha_updates=1, actor_update_after_critic_steps=8, optimizer_order_exact=True,
        complete=True)
    assert set(optimizer_steps(learner.value_optimizer)) == {8}
    assert set(optimizer_steps(learner.critic_optimizer)) == {8}
    assert set(optimizer_steps(learner.actor_optimizer)) == {1}
    assert set(optimizer_steps(learner.alpha_optimizer)) == {1}
    assert any(not torch.equal(value, learner.actor.state_dict()[key])
               for key, value in before_actor.items())


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_full57_packets_count456_critic_value_and57_actor_alpha_with_original_targets(arm):
    learner = trainer(arm)
    target_draws = []
    soft_components = learner.actor.soft_action_components

    def record(states):
        target_draws.append(len(states))
        return soft_components(states)

    learner.actor.soft_action_components = record
    for _ in range(57):
        learner.begin_actor_packet()
        for _ in range(8):
            learner.continue_step()
    audit = learner.actor_schedule_audit()
    assert audit['packets'] == 57
    assert audit['critic_updates'] == audit['value_updates'] == 456
    assert audit['actor_updates'] == audit['alpha_updates'] == 57
    assert audit['packet_critic_steps'] == audit['packet_value_steps'] == [8]*57
    assert audit['packet_actor_steps'] == audit['packet_alpha_steps'] == [1]*57
    assert audit['actor_update_after_critic_steps'] == [8]*57
    assert audit['optimizer_order_exact'] and not audit['kl_penalty']
    assert audit['reference_for_diagnostics_only'] and audit['additional_random_draws'] == 0
    assert audit['schedule'] == v.SCHEDULE
    assert learner.total_updates == learner.online_critic_updates == 456
    assert len(target_draws) == 456+57  # One original target draw per step and actor draw per packet.
    targets = learner.target_audit()
    assert targets['parent_mixed_target_calls'] == 456
    assert targets['fresh_value_replacement_calls'] == (456 if arm == 'fresh' else 0)
    assert targets['no_fresh_unchanged_returns'] == (0 if arm == 'fresh' else 456)
    for family, expected in (('critic', 456), ('value', 456), ('actor', 57), ('alpha', 57)):
        assert set(optimizer_steps(getattr(learner, family+'_optimizer'))) == {expected}


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_changed_factual_outcomes_reach_delayed_actor_only_in_fresh_arm(arm):
    altered = table(.7)
    altered['r'] -= 1e6
    altered['ns'][:, 0] = .95
    left, right = trainer(arm), trainer(arm, fresh=altered)
    left.begin_actor_packet()
    right.begin_actor_packet()
    for _ in range(8):
        before = torch.get_rng_state().clone()
        left.continue_step()
        torch.set_rng_state(before)
        right.continue_step()
    equal = all(torch.equal(value, right.actor.state_dict()[key])
                for key, value in left.actor.state_dict().items())
    assert equal == (arm == 'no_fresh')


def test_probe_is_passive_and_schedule_changes_only_three_config_fields():
    learner = trainer('fresh')
    original = asdict(learner.config)
    original.update(online_actor_update_interval=1, online_actor_batch_mode='latest',
                    online_update_order='actor-before-critic')
    changed = {key for key, value in asdict(learner.config).items() if value != original[key]}
    assert changed == {'online_actor_update_interval', 'online_update_order'}
    learner.begin_actor_packet()
    before = torch.get_rng_state().clone()
    learner.actor_comparison(torch.as_tensor(table(.1)['s']))
    assert torch.equal(before, torch.get_rng_state())
    assert not any(parameter.requires_grad for parameter in learner._packet_actor.parameters())
    with pytest.raises(RuntimeError, match='preceding'):
        learner.begin_actor_packet()
    for _ in range(8):
        learner.continue_step()
    with pytest.raises(RuntimeError, match='Begin'):
        learner.continue_step()
