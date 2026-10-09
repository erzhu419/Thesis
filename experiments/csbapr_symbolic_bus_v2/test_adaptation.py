"""Detect coordinate mistakes, broken reference gradients and disabled losses."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from settings import P, use_snapshot
from adaptation import ResidualPolicy, SparseReference, RuleReference, equilibrium_states

use_snapshot()
from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig
from csbapr.losses.jacobian import compute_jacobian_loss


def states(n=32):
    s = torch.rand(n, 15)*.5
    s[:, 0] = torch.arange(n).remainder(12)/12
    s[:, 5:7] += .4
    s[:, 8] = .6
    return s


def reference(objective):
    powers = np.vstack((np.zeros((1, 10)), np.eye(10), np.array([[0]*9+[2]]))).astype(int)
    coefficients = np.zeros((12, 11, len(powers)), np.float32)
    coefficients[:, 0, 10] = .1
    coefficients[:, 9, 1] = .2
    coefficients[:, 9, 10] = .1
    coefficients[:, 9, -1] = .4
    coefficients[:, 10, 10] = .4
    coefficients[:, 10, -1] = .2
    return SparseReference(dict(powers=powers, coefficients=coefficients,
        cost_mean=np.zeros((12, 2)), cost_std=np.ones((12, 2))), objective)


class AdaptationTests(unittest.TestCase):
    def test_context_and_physical_anchor(self):
        s = states()
        projected = equilibrium_states(s)
        retained = [i for i in range(15) if i not in (5, 6)]
        torch.testing.assert_close(s[:, retained], projected[:, retained])
        torch.testing.assert_close(projected[:, 5:7], s[:, 8:9].expand(-1, 2))
        for objective in P["objectives"]:
            ref = reference(objective)
            actor = ResidualPolicy(ref, True)
            torch.testing.assert_close(torch.tanh(actor(projected)[0]), ref(projected), atol=1e-6, rtol=0)
            self.assertTrue(math.isinf(actor.compute_L_eff()))
        no_control = RuleReference("passenger")(projected)
        self.assertLess(float((30*(no_control+1)).max()), .031)

    def test_reference_actions_and_jacobians(self):
        for objective in P["objectives"]:
            s = states().requires_grad_(True)
            ref = reference(objective)
            a = ref(s)
            self.assertEqual(a.shape, (32, 1))
            self.assertTrue(torch.isfinite(torch.autograd.grad(a.sum(), s)[0]).all())
            self.assertTrue((a.abs() <= P["reference_action_clip"]).all())
            actor = ResidualPolicy(ref, True)
            loss = compute_jacobian_loss(actor, ref, s)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_both_costs_reach_symbolic_updates(self):
        torch.set_num_threads(1)
        for objective in P["objectives"]:
            config = CSBAPRConfig(hidden_dim=64, num_critics=3, batch_size=16,
                use_nau_actor=False, actor_type="tanh-mlp", weight_reg=0.,
                jac_weight=.01, weight_sym=.01, track_L_eff=False, device="cpu")
            agent = CSBAPRAgent(15, 1, config)
            ref = reference(objective)
            agent.actor = ResidualPolicy(ref, True)
            agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
            agent.pi_sym_torch = ref
            s = states()
            for row in s.numpy():
                agent.replay_buffer.push(row, np.array([0.], np.float32), -100., row+.001, False)
            result = agent.update()
            self.assertGreater(result["jac_loss"], 0.)
            self.assertGreater(result["sym_penalty"], 0.)
            self.assertTrue(all(np.isfinite(v) for k,v in result.items() if not k.startswith("L_eff")))

    def test_sparse_fit_and_reload(self):
        import fit_reference
        rng = np.random.default_rng(7)
        s = rng.uniform(.1, 1., (12*80, 15)).astype(np.float32)
        s[:, 0] = np.repeat(np.arange(12)/12, 80)
        a = rng.uniform(-1, 1, (len(s), 1)).astype(np.float32)
        ns = s.copy()
        ns[:, 5] += .1*a[:, 0]
        costs = np.column_stack((100+600*ns[:, 5], 500+300*(a[:, 0]+1))).astype(np.float32)
        data = dict(states=s, actions=a, next_states=ns, costs=costs)
        with TemporaryDirectory() as directory, patch.object(fit_reference, "arrays", return_value=data), \
                patch.object(fit_reference, "model_folder", return_value=Path(directory)):
            fit_reference.fit(1)
            with np.load(Path(directory) / "reference.npz") as artifact:
                ref = SparseReference(dict(artifact), "headway")
            predicted = ref.prediction(torch.from_numpy(s), torch.from_numpy(a))
            np.testing.assert_allclose(predicted[:, 0].numpy(), .1*a[:, 0], atol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
