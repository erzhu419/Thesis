"""Resident frozen-model parity; no collection, replay sampling or updates."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import learning as v75
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import _batch, _join
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    OPTIMIZERS, _equal, _rng, _restore, _state)
from experiments.mcwm_closed_loop_capacity_context_v78_20261027.critic import CapacityContextQEnsemble


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_frozen_validation_v75_20261027'


def _random_state(learner):
    return dict(global_state=_rng(), learner=deepcopy(learner.rng.bit_generator.state),
        outcome=deepcopy(learner.replay_buffer.outcome_rng.bit_generator.state))


def _restore_random(learner, state):
    _restore(state['global_state'])
    learner.rng.bit_generator.state = deepcopy(state['learner'])
    learner.replay_buffer.outcome_rng.bit_generator.state = deepcopy(state['outcome'])


def _facts(seed, arm):
    path = SOURCE/f'runs/executed_cql/validation/seed{seed}/stable/{arm}/prefix_trace.jsonl'
    rows = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if 12000 <= row['started_at'] < 16200:
                rows.append(row)
                if len(rows) == 16:
                    break
    if len(rows) != 16:
        raise ValueError(f'{path}: expected16 existing factual issues in[12000,16200)')
    return rows


def _note(checks, errors, name, left, right):
    checks[name] = _equal(left, right)
    pairs = zip(left, right) if isinstance(left, tuple) else [(left, right)]
    differences = [float((a.detach()-b.detach()).abs().max())
                   for a, b in pairs if isinstance(a, torch.Tensor)]
    error = max(differences, default=0.)
    errors[name] = error if np.isfinite(error) else None


def _actor_loss(learner, reduced_q, components, size):
    zero_q, positive_q = reduced_q.split(size)
    zero = learner.alpha.detach()*components['zero_log_prob']-zero_q
    positive = learner.alpha.detach()*components['positive_log_prob']-positive_q
    return (components['zero_probability']*zero+components['hold_probability']*positive).mean()


@torch.no_grad()
def _candidate_target(learner, candidate, batch):
    states = batch['next_observations']
    mixing = learner.config.mixed_backup_lambda
    bootstrap = mixing*learner.value(states)
    components = learner.actor.soft_action_components(states)
    size = len(states)
    actions = torch.cat((learner.action_transform(states, components['zero_action']),
                         learner.action_transform(states, components['positive_action'])))
    chosen = learner._sample_target_subset()
    features = candidate.encode_states(states)
    values = candidate.forward_features_subset(torch.cat((features, features)), actions,
        torch.full((2*size,), float('nan'), device=states.device),
        torch.zeros(2*size, dtype=torch.bool, device=states.device), chosen)
    zero_q, positive_q = values.split(size, dim=1)
    zero_value, positive_value = zero_q.min(0).values, positive_q.min(0).values
    if learner.config.backup_entropy:
        zero_value -= learner.alpha.detach()*components['zero_log_prob']
        positive_value -= learner.alpha.detach()*components['positive_log_prob']
    sac = (components['zero_probability']*zero_value
           + components['hold_probability']*positive_value)
    bootstrap = bootstrap+(1.-mixing)*sac
    discount = batch['discounts'].reshape(-1)*(1.-batch['terminals'].reshape(-1))
    return learner.config.reward_scale*batch['rewards'].reshape(-1)+discount*bootstrap


def _run_parity(seed):
    """Probe archived rows with the original offline initializer, never its rollout."""
    learner, replay, _, _, initializer = v75.load_learner(seed, arm='fresh')
    before_rng = _random_state(learner)
    before_state = deepcopy(_state(learner))
    native_objects = {name: getattr(learner, name) for name in
        ('actor', 'value', 'critic', 'target_critic', 'action_transform')}
    counters = {name: getattr(learner, name) for name in (
        'parent_mixed_target_calls', 'parent_mixed_target_rows', 'no_fresh_unchanged_returns',
        'fresh_mixed_target_calls', 'fresh_mixed_target_rows')}
    parameters = tuple(learner.critic.parameters())
    optimizer_objects = {name: getattr(learner, name) for name in OPTIMIZERS}
    optimizer_ids = {name: ([id(p) for group in opt.param_groups for p in group['params']],
                           {id(p): id(state) for p, state in opt.state.items()})
                     for name, opt in optimizer_objects.items()}
    checks, errors = {}, {}
    try:
        candidate = CapacityContextQEnsemble(learner.critic)
        target = CapacityContextQEnsemble(learner.target_critic)
        checks['construction_all_rng_unchanged'] = _equal(before_rng, _random_state(learner))
        checks['native_parameter_objects_retained'] = all(
            left is right for left, right in zip(parameters, candidate.base.parameters()))
        rows = {arm: _facts(seed, arm) for arm in ('fresh', 'no_fresh')}
        joined = rows['fresh']+rows['no_fresh']
        states = torch.as_tensor(np.asarray([row['s'] for row in joined]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray([row['a'] for row in joined]), dtype=torch.float32).reshape(-1, 1)
        missing = (torch.full((len(states),), float('nan')), torch.zeros(len(states), dtype=torch.bool))
        stress = (torch.linspace(-1., 1., len(states)), torch.arange(len(states))%3 != 0)
        for model_name, wrapped in (('critic', candidate), ('target', target)):
            base = wrapped.base
            for context_name, (z, observed) in (('missing', missing), ('synthetic', stress)):
                prefix = model_name+'_'+context_name
                features = wrapped.encode_states(states)
                _note(checks, errors, prefix+'_encoder', features, base.encode_states(states))
                for label, subset in (('full', None), ('reverse', torch.tensor([base.ensemble_size-1, 0])),
                                      ('fixed_first_last', torch.tensor([0, base.ensemble_size-1]))):
                    a, b = actions.clone().requires_grad_(), actions.clone().requires_grad_()
                    native = base(states, a) if subset is None else base.forward_subset(states, a, subset)
                    actual = wrapped(states, b, z, observed) if subset is None else wrapped.forward_subset(states, b, z, observed, subset)
                    feature_q = (wrapped.forward_features(features, b, z, observed) if subset is None
                        else wrapped.forward_features_subset(features, b, z, observed, subset))
                    key = prefix+'_'+label
                    _note(checks, errors, key+'_heads', native, actual)
                    _note(checks, errors, key+'_features', native, feature_q)
                    _note(checks, errors, key+'_min', native.min(0).values, actual.min(0).values)
                    _note(checks, errors, key+'_gradient', torch.autograd.grad(native.sum(), a, retain_graph=True)[0],
                        torch.autograd.grad(actual.sum(), b, retain_graph=True)[0])
                    _note(checks, errors, key+'_min_gradient', torch.autograd.grad(native.min(0).values.sum(), a)[0],
                        torch.autograd.grad(actual.min(0).values.sum(), b)[0])
            low, high = torch.full_like(actions, -1.), torch.full_like(actions, -.9375)
            z, observed = stress
            _note(checks, errors, model_name+'_advantage', base(states, high)-base(states, low),
                wrapped(states, high, z, observed)-wrapped(states, low, z, observed))
        # Identical components and unchanged F; reductions use paired original RNG.
        actor_before = learner.actor(states, deterministic=True)[0].detach().clone()
        value_before = learner.value(states).detach().clone()
        executed_before = learner.action_transform(states, actor_before).detach().clone()
        components = learner.actor.soft_action_components(states)
        actor_actions = torch.cat((learner.action_transform(states, components['zero_action']),
                                   learner.action_transform(states, components['positive_action'])))
        features = learner.critic.encode_states(states)
        twice = torch.cat((features, features))
        native_q = learner.critic.forward_features(twice, actor_actions)
        new_q = candidate.forward_features(twice, actor_actions, stress[0].repeat(2), stress[1].repeat(2))
        reduction_rng = _random_state(learner)
        reduced = learner._reduce_actor_q(native_q)
        reduction_end = _random_state(learner)
        _restore_random(learner, reduction_rng)
        new_reduced = learner._reduce_actor_q(new_q)
        checks['actor_reduction_rng_exact'] = _equal(reduction_end, _random_state(learner))
        loss, new_loss = (_actor_loss(learner, q, components, len(states)) for q in (reduced, new_reduced))
        _note(checks, errors, 'actor_loss', loss, new_loss)
        actor_parameters = tuple(learner.actor.parameters())
        _note(checks, errors, 'actor_parameter_gradients',
            torch.autograd.grad(loss, actor_parameters, retain_graph=True),
            torch.autograd.grad(new_loss, actor_parameters))
        batch = _join(_batch(replay.history, np.arange(16), learner.device),
                      _batch(replay.source, np.arange(16), learner.device))
        target_rng = _random_state(learner)
        expected = learner._target(batch, backup='h2oplus')
        target_end = _random_state(learner)
        _restore_random(learner, target_rng)
        actual = _candidate_target(learner, target, batch)
        _note(checks, errors, 'mixed_target', expected, actual)
        checks['mixed_target_sampling_rng_exact'] = _equal(target_end, _random_state(learner))
        # Hand-written nonzero weights check wiring only; never optimizer.step.
        with torch.no_grad():
            candidate.action_weight.copy_(torch.arange(1, candidate.ensemble_size+1)*2.)
            high_q, low_q = candidate(states, high, *stress), candidate(states, low, *stress)
            delta = high_q-low_q
            native_delta = learner.critic(states, high)-learner.critic(states, low)
            expected_delta = candidate.action_weight[:, None]*torch.where(stress[1], stress[0], 0.)[None, :]/32.
            checks['nonzero_structural_action_interaction'] = torch.allclose(delta-native_delta, expected_delta,
                rtol=0., atol=4*torch.finfo(delta.dtype).eps*max(1., float(high_q.abs().max()), float(low_q.abs().max()))) and bool(torch.any(delta != native_delta))
            _note(checks, errors, 'nonzero_missing_bypass', learner.critic(states, actions), candidate(states, actions, *missing))
            candidate.action_weight.zero_()
        _note(checks, errors, 'actor_output_unchanged', actor_before, learner.actor(states, deterministic=True)[0])
        _note(checks, errors, 'value_output_unchanged', value_before, learner.value(states))
        _note(checks, errors, 'F_output_unchanged', executed_before, learner.action_transform(states, actor_before))
    finally:
        for name, value in counters.items():
            setattr(learner, name, value)
        _restore_random(learner, before_rng)
    checks['pre_probe_all_rng_restored'] = _equal(before_rng, _random_state(learner))
    checks['native_modules_Adam_and_update_state_unchanged'] = _equal(before_state, _state(learner))
    checks['actor_value_F_and_native_critics_not_replaced'] = all(
        getattr(learner, name) is obj for name, obj in native_objects.items())
    checks['optimizer_objects_and_state_objects_retained'] = all(
        getattr(learner, name) is opt and optimizer_ids[name] == (
            [id(p) for group in opt.param_groups for p in group['params']],
            {id(p): id(state) for p, state in opt.state.items()})
        for name, opt in optimizer_objects.items())
    checks['no_replay_admission'] = replay.target_received == len(replay.recent) == len(replay.fresh_outcomes) == 0
    checks['no_updates'] = learner.total_updates == 50000 and learner.online_critic_updates == 0
    return dict(seed=seed, all_pass=all(checks.values()), checks=checks, maximum_absolute_errors=errors,
        initializer=str(initializer), initializer_updates=50000, factual_rows={arm: len(value) for arm, value in rows.items()},
        historical_transition_rows=16, source_transition_rows=16, optimizer_updates=0, native_simulation_seconds=0,
        new_feedback_slots=0, synthetic_context_only=True, native_joint_context_coverage_verified=False,
        scope='Frozen structural/target/actor-gradient parity, not optimizer-step or deployment validation.')


def run_parity(seed):
    outer_rng = _rng()
    try:
        return _run_parity(seed)
    finally:
        _restore(outer_rng)


def main():
    result = dict(seeds=[run_parity(seed) for seed in (6208, 6207)])
    result['all_pass'] = all(row['all_pass'] for row in result['seeds'])
    destination = HERE/'results'/'model_parity.json'
    destination.parent.mkdir(exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if len(encoded.encode()) > 48*1024:
        raise ValueError('Frozen parity summary exceeds48KiB')
    destination.write_text(encoded)
    print(json.dumps(dict(all_pass=result['all_pass'], output=str(destination))))


if __name__ == '__main__':
    main()
