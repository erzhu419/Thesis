"""Focused tests for non-null interventions and unchanged training contracts."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("THESIS_H2O_ROOT", str(ROOT/"mcwm_bus/runtime/H2Oplus"))
sys.path.insert(0, str(ROOT/"mcwm_bus"))
import torch
from model import Normalizer, WIDTH, targets
from decision_model import combined_error, decision_model_class
from shared_model import SharedBusModel, OUTPUTS, select_outputs, mechanism_losses
from source_weight import reliability_weights, source_trainer_class


class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.s = torch.rand(8, 229)
        self.s[:, 1:3] = torch.tensor([1/16, 0])
        slots = self.s[:, 5:].reshape(-1, 16, 14)
        slots[:, :, 0] = 0
        slots[:, 0, 0] = 1
        slots[:, :, 1] = torch.arange(16) % 12
        slots[:, :, 2] = 0
        self.a = torch.linspace(-1, 1, 8).reshape(-1, 1).requires_grad_()

    def test_decision_error_changes_ordering(self):
        errors = combined_error(torch.ones(2), torch.tensor([.1, 2.]), 1., 1., 1.)
        self.assertLess(float(errors[0]), float(errors[1]))
        self.assertLess(float(torch.exp(-errors[1])), float(torch.exp(-errors[0])))

    def test_source_weights_only_downweight_eligible_rows(self):
        weights = reliability_weights(torch.tensor([[0.]*4, [3.]*4, [0.]*4, [3.]*4]),
            torch.ones(4), torch.tensor([True, True, False, False]),
            torch.tensor([True, True, True, False]), .25)
        self.assertEqual(float(weights[0]), 1.)
        self.assertLess(float(weights[1]), .3)
        self.assertEqual(float(weights[2]), .25)
        self.assertEqual(float(weights[3]), 1.)

    def test_td_weight_does_not_leak_to_offline_or_model(self):
        calls = []

        class Parent:
            def _ensemble_td_loss(self, predictions, target, weights=None):
                calls.append(weights)
                return torch.tensor(0.)

        cls = source_trainer_class(SimpleNamespace(UsageTrainer=Parent))
        learner = cls()
        learner.real_td_calls = 0
        learner.current_source_weights = torch.tensor([.25, 1.])
        learner._ensemble_td_loss(torch.ones(2, 2), torch.zeros(2))
        learner._ensemble_td_loss(torch.ones(2, 2), torch.zeros(2))
        self.assertIsNone(calls[0])
        torch.testing.assert_close(calls[1], torch.tensor([.25, 1.]))
        self.assertEqual(learner.real_td_calls, 2)

    def test_shared_encoder_once_and_complete_output_partition(self):
        self.assertEqual(sorted(i for indices in OUTPUTS.values() for i in indices), list(range(WIDTH)))
        model = SharedBusModel()
        calls = []
        hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        out = model(self.s, self.a)
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(out.shape, (8, WIDTH))
        for name, indices in OUTPUTS.items():
            gradient = torch.autograd.grad(out[:, indices].sum(), self.a, retain_graph=True)[0]
            self.assertGreater(float(gradient.abs().sum()), 0, name)

    def test_group_selection_uses_only_chosen_family(self):
        outputs = dict(direct=torch.ones(2, WIDTH), residual=torch.full((2, WIDTH), 3.))
        gates = {name: "residual" if name in ("duration", "cost") else "direct" for name in OUTPUTS}
        chosen = select_outputs(outputs, gates)
        for name, indices in OUTPUTS.items():
            torch.testing.assert_close(chosen[:, indices], outputs[gates[name]][:, indices])

    def test_residual_starts_at_source_and_source_has_no_gradient(self):
        base = SharedBusModel()
        residual = SharedBusModel(base)
        torch.testing.assert_close(base(self.s, self.a), residual(self.s, self.a), rtol=0, atol=0)
        residual(self.s, self.a).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(all(float(head[-1].weight.grad.abs().sum()) > 0 for head in residual.head.values()))

    def test_mechanism_losses_own_their_output_columns(self):
        raw = dict(s=self.s, ns=self.s, a=self.a.detach(), r=-torch.ones(8), dt=torch.ones(8))
        y = targets(raw, torch.full((12,), 46.), "cpu")
        norm = Normalizer(y)
        out = SharedBusModel()(self.s, self.a).detach()
        baseline = mechanism_losses(out, y, norm)
        for owner, indices in OUTPUTS.items():
            changed = out.clone()
            changed[:, indices] += torch.arange(len(indices)).float()+3
            after = mechanism_losses(changed, y, norm)
            for other in OUTPUTS:
                if other != owner:
                    torch.testing.assert_close(after[other], baseline[other], rtol=0, atol=0)

    def test_decision_decode_restores_generator_and_global_rng(self):
        cls = decision_model_class(SimpleNamespace(UsageModel=object))
        model = object.__new__(cls)
        model.device, model.protocol = "cpu", dict(decision_draws=2)
        model.generator = torch.Generator().manual_seed(77)
        original = model.generator
        before = original.get_state().clone()
        global_before = torch.random.get_rng_state().clone()
        model._reference_backup = lambda ns, reward: ns[:, 0]+reward
        model.decode = lambda s, out: (torch.rand(s.shape, generator=model.generator),
            torch.zeros(len(s)), torch.ones(len(s)))
        batch = dict(s=self.s, logged_ns=self.s, logged_reward=torch.zeros(8))
        first = model._decision_error(torch.empty(8, WIDTH), batch, 35)
        second = model._decision_error(torch.empty(8, WIDTH), batch, 35)
        torch.testing.assert_close(first, second)
        self.assertIs(model.generator, original)
        self.assertTrue(torch.equal(before, original.get_state()))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
