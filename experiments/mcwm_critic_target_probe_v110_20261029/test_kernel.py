"""Check the sign and removed-row term of the directional derivative."""
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'experiments/mcwm_bus/runtime/H2Oplus'))
import numpy as np
import torch
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer
from kernel import probe


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2, .4]))

    def forward(self, s, a):
        return (self.weight[0]*s[:, 0]+self.weight[1]*s[:, 0]*a[:, 0])[None]


class Actor:
    def soft_action_components(self, s, deterministic_positive=False):
        return dict(zero_action=torch.full((len(s), 1), -1.),
            positive_action=torch.full((len(s), 1), .5), hold_probability=torch.full((len(s),), .6))


def batch():
    s = torch.arange(128, dtype=torch.float32).reshape(-1, 1)/100
    return dict(observations=s, actions=torch.full_like(s, -.3), next_observations=s+.1,
        rewards=torch.full((128,), -3.), discounts=torch.full((128,), .9), terminals=torch.zeros(128))


class KernelTests(unittest.TestCase):
    def test_direction_matches_analytic_linear_critic_and_preserves_rng(self):
        learner = SimpleNamespace(device=torch.device('cpu'), critic=Critic(), actor=Actor(),
            protocol={'real_td_coefficient': .75}, coefficient=.25,
            config=SimpleNamespace(bootstrap_probability=1., actor_q_reduction='mean-all',
                online_backup='h2oplus', online_fraction=.5, use_ratio_correction=False,
                online_cql_alpha=0., reward_scale=.1, mixed_backup_lambda=.1,
                critic_loss='mse', huber_delta=1.))
        learner.value = lambda s: .3*s[:, 0]
        learner._sac_bootstrap = lambda b: .3*b['next_observations'][:, 0]
        learner._target = MethodType(TransitH2OTrainer._target, learner)
        learner._elementwise_td_loss = MethodType(TransitH2OTrainer._elementwise_td_loss, learner)
        learner._concatenate_batches = lambda a, b: {k: torch.cat((a[k], b[k])) for k in a}
        learner._reduce_actor_q = lambda q: q.mean(dim=0)
        original, source, fresh = batch(), batch(), batch()
        for k, v in dict(observations=1.5, actions=.2, next_observations=1.6, rewards=-8.).items():
            fresh[k][64:] = v
        rows = [dict(sequence=36, s=[2.], raw_s=[2.], feedback_selected=True)]
        rng = torch.get_rng_state().clone()
        params = learner.critic.weight.detach().clone()
        with patch('SimpleSAC.evaluate_transit_v19_completion_safety_reserve.completion_safety_action',
                   side_effect=lambda s, a, variant, c: np.asarray(a)):
            result = probe(learner, original, source, fresh, rows, SimpleNamespace(max_hold_seconds=60))
        gradients = []
        for b in (original, fresh):
            target = learner._target(b, backup='h2oplus')[64:]
            q = learner.critic(b['observations'][64:], b['actions'][64:])
            loss = .1875*(q-target[None]).square().mean()
            gradients.append(torch.autograd.grad(loss, learner.critic.weight)[0])
        expected = -3.*float((gradients[1]-gradients[0])[1])
        self.assertAlmostEqual(result['selected_roots'][0]['proposal']['response']['delta'], expected, places=6)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(params, learner.critic.weight))
        self.assertIsNone(learner.critic.weight.grad)
        self.assertTrue(result['common_targets_exact'])
        learner.config.bootstrap_probability = .8
        self.assertEqual(probe(learner, original, source, fresh, rows, SimpleNamespace())['status'], 'HOLD')


if __name__ == '__main__':
    unittest.main()
