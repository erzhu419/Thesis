"""Issue-visible frozen actor; native V, alpha and target Q remain live.

This module scores existing native rows and never installs a training target.
M1 follows the native two-action batch and subset draw. M8 averages only the
positive joint Q-minus-entropy term, with its first draw shared with M1.
"""
from __future__ import annotations

from copy import copy, deepcopy

import numpy as np
import torch

from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    actor_parameters, execution_measure, ratio_from_measures,
)
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import conditional_overlap


def _get(row, field):
    return row[field] if isinstance(row, dict) else getattr(row, field)


def _scalar(value):
    return float(value.reshape(-1)[0])


def _adam_versions(learner):
    result = []
    for name in ('actor_optimizer', 'critic_optimizer', 'value_optimizer', 'alpha_optimizer'):
        optimizer = getattr(learner, name)
        result.append((name, id(optimizer), tuple(id(parameter) for group in optimizer.param_groups
            for parameter in group['params'])))
        result.extend((name, id(parameter), field, id(value), value._version)
            for parameter, state in optimizer.state.items() for field, value in state.items()
            if isinstance(value, torch.Tensor))
    return result


class FrozenIssueEvaluator:
    def __init__(self, reference, snapshot_clock):
        if reference.total_updates != 50024 or not snapshot_clock < 9000:
            raise ValueError('The evaluator requires the actual issue-pre post24 snapshot')
        self.actor = deepcopy(reference.actor).eval()
        self.action_transform = deepcopy(reference.action_transform)
        self.device = reference.device
        self.snapshot_clock = float(snapshot_clock)
        self.total_updates = reference.total_updates
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self._original = {key: value.detach().clone() for key, value in self.actor.state_dict().items()}

    def unchanged(self):
        return all(torch.equal(value, self._original[key])
                   for key, value in self.actor.state_dict().items())

    @torch.no_grad()
    def check_wiring(self, learner, row, *, seed=536202):
        """Challenge the native path using only a private copy of its live actor."""
        original_actor = learner.actor
        actor_before = {key: value.detach().clone() for key, value in learner.actor.state_dict().items()}
        adam_before, rng_before = _adam_versions(learner), torch.get_rng_state().clone()
        before = self.score(learner, row, seed=seed)
        private = copy(learner)
        private.actor = deepcopy(learner.actor)
        # The supported v7 Hurdle head has exactly gate, positive mean, log std.
        bias = private.actor.network.network[-1].bias
        if bias.shape != (3,):
            raise ValueError('The native Hurdle output must contain gate, mean and log std')
        bias[:2].add_(bias.new_tensor([2., 1.5]))
        altered = self.score(private, row, seed=seed)
        checks = dict(frozen_M1_exact=before['m1'] == altered['m1'],
            frozen_M8_exact=before['m8'] == altered['m8'],
            perturbed_live_target_changes=before['current_actor_M1_target'] != altered['current_actor_M1_target'],
            finite_targets=all(np.isfinite(score['current_actor_M1_target']) for score in (before, altered)),
            main_actor_values_unchanged=all(torch.equal(value, actor_before[key])
                for key, value in learner.actor.state_dict().items()),
            main_actor_object_retained=learner.actor is original_actor and private.actor is not original_actor,
            main_Adam_unchanged=adam_before == _adam_versions(learner),
            torch_RNG_unchanged=torch.equal(rng_before, torch.get_rng_state()),
            immutable_evaluator_unchanged=self.unchanged())
        result = dict(all_pass=all(checks.values()), checks=checks,
            perturbation=dict(gate_bias_add=2., positive_mean_bias_add=1.5,
                private_actor_only=True),
            synchronized_M1_target=before['m1']['native_target'],
            synchronized_M8_target=before['m8']['native_target'],
            original_live_M1_target=before['current_actor_M1_target'],
            perturbed_live_M1_target=altered['current_actor_M1_target'],
            main_updates=learner.total_updates, new_optimizer_updates=0)
        if not result['all_pass']:
            raise RuntimeError('Native frozen-evaluator wiring challenge failed: '+str(result))
        return result

    @torch.no_grad()
    def issue(self, behavior_s, issue_clock=None):
        state = np.asarray(behavior_s['policy_input'], np.float32).reshape(229)
        execution = np.asarray(behavior_s['execution_input'], np.float32).reshape(229)
        actual = np.asarray(behavior_s['executed_action'], np.float32).reshape(1)
        proposal = np.asarray(behavior_s['proposal'], np.float32).reshape(1)
        own = actor_parameters(self, state)
        behavior = execution_measure(self.action_transform, execution, actual, behavior_s['parameters'])
        target = execution_measure(self.action_transform, state, actual, own)
        rn = ratio_from_measures(behavior, target)
        overlap = conditional_overlap(self.action_transform, execution, behavior_s['parameters'], own,
            target_transform=self.action_transform, target_execution_state=state)
        # The shared mathematical helper also serves old counterfactual logs;
        # this caller measures a newly collected issue under its real behavior.
        overlap.pop('old_log_authorized_for_training', None)
        overlap['counterfactual_only'] = False
        repeated_F = self.action_transform.numpy(execution, proposal)
        checks = dict(snapshot_precedes_issue=issue_clock is None or self.snapshot_clock < issue_clock,
            actor_F_inputs_exact=np.array_equal(state, execution),
            actor_parameters_exact=own == behavior_s['parameters'],
            actual_F_exact=np.array_equal(repeated_F, actual),
            nominal_actual_exact=np.array_equal(behavior_s['nominal_executed_action'], actual),
            no_intervention=behavior_s.get('intervention') is False,
            frozen_actor_unchanged=self.unchanged(),
            supported=rn['status'] == 'supported', pointwise_c_one=rn['c'] == 1.,
            whole_F_overlap_one=overlap['status'] == 'supported'
                and abs(overlap['total_overlap']-1.) < 5e-12)
        return dict(all_pass=all(checks.values()), checks=checks, ratio=rn,
            overlap=overlap, evaluator_parameters=own,
            snapshot_clock=self.snapshot_clock, scoring_updates=0)

    @torch.no_grad()
    def score(self, learner, row, following=None, *, seed=536202):
        """Score one real TD with isolated M1/M8 samples and native M1 oracle.

        A shallow native view substitutes only the frozen actor and its F. It
        shares the live V/Q/alpha but calls no optimizer or mutable replay API.
        """
        config = learner.config
        if (learner.device != self.device or config.mixed_backup_lambda != .1
                or config.ensemble_size != 2 or config.target_subset_size != 2
                or config.target_subset_replacement or config.online_backup != 'h2oplus'):
            raise ValueError('v53 retains native two-of-two .1V/.9SAC CPU scoring')
        states = torch.as_tensor(np.asarray(_get(row, 'ns')), dtype=torch.float32,
                                 device=self.device).reshape(1, 229)
        memory = torch.as_tensor(_get(row, 'memory_ns')['vector'], dtype=states.dtype,
                                 device=self.device).reshape(1, 84)
        explicit = torch.cat((states, memory), dim=1)
        batch = dict(next_observations=explicit,
            rewards=states.new_tensor([_get(row, 'r')]), discounts=states.new_ones(1),
            terminals=states.new_tensor([float(_get(row, 'terminal'))]))
        reward = config.reward_scale*batch['rewards']
        discount = batch['discounts']*(1.-batch['terminals'])
        features = learner.target_critic.encode_states(explicit)
        repeated = torch.cat((features, features))
        value = learner.value(states)
        view = copy(learner)
        view.actor, view.action_transform, view._collect_td = self.actor, self.action_transform, False
        devices = [] if self.device.type == 'cpu' else [self.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            positives, positive_qs, first = [], [], None
            for index in range(8):
                torch.manual_seed(int(seed)+104729*index)
                parts = self.actor.soft_action_components(states)
                chosen = learner._sample_target_subset() if index == 0 else chosen
                actions = torch.cat((self.action_transform(states, parts['zero_action']),
                                     self.action_transform(states, parts['positive_action'])))
                selected = learner.target_critic.forward_features_subset(repeated, actions, chosen)
                zero_heads, positive_heads = selected.split(1, dim=1)
                zero_q, positive_q = zero_heads.min(0).values, positive_heads.min(0).values
                zero, positive = zero_q.clone(), positive_q.clone()
                if config.backup_entropy:
                    zero -= learner.alpha.detach()*parts['zero_log_prob']
                    positive -= learner.alpha.detach()*parts['positive_log_prob']
                if first is None:
                    first = dict(parts=parts, zero=zero, zero_q=zero_q, actions=actions)
                positives.append(positive)
                positive_qs.append(positive_q)

            def assemble(positive, positive_q, count):
                p = first['parts']
                sac = p['zero_probability']*first['zero']+p['hold_probability']*positive
                sac_q = p['zero_probability']*first['zero_q']+p['hold_probability']*positive_q
                bootstrap = discount*(.1*value+.9*sac)
                target = reward+bootstrap
                # lambda=0 is the cached synchronized target, without rebuilding it.
                copied = target.clone()
                return dict(estimator_M=count, native_target=_scalar(target),
                    lambda_zero_target=_scalar(copied), lambda_zero_native_exact=torch.equal(
                        target.view(torch.int32), copied.view(torch.int32)),
                    reward_scaled=_scalar(reward), bootstrap_contribution=_scalar(bootstrap),
                    value_contribution=_scalar(discount*.1*value),
                    sac_contribution=_scalar(discount*.9*sac), next_value=_scalar(value),
                    sac_q_without_entropy=_scalar(sac_q), sac_entropy_bonus=_scalar(sac-sac_q),
                    target_subset=chosen.tolist())

            m1 = assemble(positives[0], positive_qs[0], 1)
            m8 = assemble(torch.stack(positives).mean(0), torch.stack(positive_qs).mean(0), 8)
            torch.manual_seed(int(seed))
            native = view._target(batch, backup=config.online_backup)
            exact = np.float32(m1['native_target']).view(np.uint32) == np.float32(_scalar(native)).view(np.uint32)
            torch.manual_seed(int(seed))
            live_view = copy(learner)
            live_view._collect_td = False
            live = live_view._target(batch, backup=config.online_backup)
            matching_q, missing = None, None
            if _get(row, 'terminal'):
                missing = 'terminal_transition'
            elif following is None:
                missing = 'no_next_real_recorded_action'
            elif (_get(row, 'completed_at') != _get(following, 'started_at')
                    or not np.array_equal(_get(row, 'ns'), _get(following, 's'))
                    or _get(row, 'memory_ns')['observed_at'] != _get(following, 'memory_s')['observed_at']
                    or not np.array_equal(_get(row, 'memory_ns')['vector'],
                                          _get(following, 'memory_s')['vector'])):
                raise RuntimeError('Matching Q requires the genuine contiguous state and causal memory')
            else:
                action = states.new_tensor(np.asarray(_get(following, 'a')).reshape(1, 1))
                q = learner.target_critic.forward_features_subset(repeated,
                    torch.cat((action, action)), chosen)
                matching_q = _scalar(q[:, 0].min())
        checks = dict(native_synchronized_M1_exact=bool(exact),
            M1_lambda_zero_exact=m1['lambda_zero_native_exact'],
            M8_lambda_zero_exact=m8['lambda_zero_native_exact'],
            frozen_actor_unchanged=self.unchanged())
        if not all(checks.values()):
            raise RuntimeError('Frozen issue-evaluator target parity failed: '+str(checks))
        return dict(all_pass=True, checks=checks, m1=m1, m8=m8,
            current_actor_M1_target=_scalar(live),
            synchronous_minus_current_actor_M1=m1['native_target']-_scalar(live),
            recorded_next_target_q=matching_q, baseline_missing_reason=missing,
            snapshot_clock=self.snapshot_clock, target_actor='new_post24_from14',
            current_alpha=_scalar(learner.alpha.detach()), current_V_Q_preserved=True,
            response_training_updates=0)
