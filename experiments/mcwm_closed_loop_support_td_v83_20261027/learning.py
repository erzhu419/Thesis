"""Block only singleton-support factual TD gradients into the capacity slope.

The active trainer uses critic.forward for its combined factual TD predictions.
Its actor and CQL objectives use forward_features; target/value calls use the
separate target critic. Those paths and all forward values remain unchanged.
"""
import torch

from experiments.mcwm_closed_loop_context_wiring_v80_20261027 import learning as previous
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng

ContextTrainer = previous.ContextTrainer
ContextReplay = previous.ContextReplay


class SupportTDQEnsemble(previous.ContextQEnsemble):
    def forward(self, states, actions):
        canonical, context = previous.split_states(states)
        with torch.no_grad():
            lower = self._support_transform(canonical, torch.full_like(actions, -1.))
            upper = self._support_transform(canonical, torch.ones_like(actions))
            singleton = (lower == upper).reshape(-1)
        observed = context[:, 1].bool()
        z = torch.where(observed, context[:, 0], torch.zeros_like(context[:, 0]))
        feature = z*((actions[:, 0]+1.)*.5)
        # torch.where selects exactly the same stored theta values. Only the
        # parameter derivative differs; base-Q and action derivatives do not.
        weight = torch.where(singleton[None], self.action_weight.detach()[:, None],
                             self.action_weight[:, None])
        prediction = self.base(canonical, actions)+weight*feature[None]
        if torch.is_grad_enabled():
            audit = self._singleton_td_counts
            audit['calls'] += 1
            audit['rows'] += len(actions)
            audit['observed_rows'] += int(observed.sum())
            audit['unobserved_rows'] += int((~observed).sum())
            audit['singleton_rows'] += int(singleton.sum())
            audit['singleton_observed_rows'] += int((singleton & observed).sum())
            audit['choice_observed_rows'] += int((~singleton & observed).sum())
            audit['blocked_observed_nonzero_feature_rows'] += int(
                (singleton & observed & (feature.detach() != 0.)).sum())
        return prediction


def attach_support_td(learner):
    """Retain the v80 objects, parameters, target, Adam and random streams."""
    if type(learner) is not ContextTrainer or type(learner.critic) is not previous.ContextQEnsemble:
        raise ValueError('v83 attaches to the actual v80 contextual learner only')
    before = _rng()
    learner.critic.__class__ = SupportTDQEnsemble
    # ExecutionAction is a plain callable, not a torch module or parameter.
    learner.critic._support_transform = learner.action_transform
    learner.critic._singleton_td_counts = {key:0 for key in (
        'calls','rows','observed_rows','unobserved_rows','singleton_rows',
        'singleton_observed_rows','choice_observed_rows','blocked_observed_nonzero_feature_rows')}
    if not _equal(before,_rng()):
        raise RuntimeError('Support-gradient attachment consumed a random draw')
    return learner


def singleton_td_audit(learner):
    counts = dict(learner.critic._singleton_td_counts)
    checks = dict(observation_partition=counts['rows'] ==
            counts['observed_rows']+counts['unobserved_rows'],
        observed_support_partition=counts['observed_rows'] ==
            counts['singleton_observed_rows']+counts['choice_observed_rows'],
        blocked_rows_within_singleton_observed=0 <= counts['blocked_observed_nonzero_feature_rows']
            <= counts['singleton_observed_rows'] <= counts['singleton_rows'],
        target_critic_unchanged_type=type(learner.target_critic) is previous.ContextQEnsemble,
        critic_type=type(learner.critic) is SupportTDQEnsemble,
        shared_original_F=learner.critic._support_transform is learner.action_transform)
    return dict(**counts, all_pass=all(checks.values()), checks=checks,
        rule='Detach theta only in direct factual-TD forward where F(s,-1)==F(s,1)',
        scope='Counts of gradient-enabled direct critic predictions; actual production calls once per update',
        feature_paths='Actor/CQL forward_features and separate target critic unchanged',
        proof='Per-row theta derivative is zero by detach; unit tests compare base/action/choice gradients exactly')


def make_learner(seed, *, arm):
    learner, replay, contract, stations, initializer = previous.make_learner(seed, arm=arm)
    attach_support_td(learner)
    return learner, replay, contract, stations, initializer
