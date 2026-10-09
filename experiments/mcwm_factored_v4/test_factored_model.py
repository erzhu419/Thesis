"""Check output ownership, parent restrictions and residual fitting gradients."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("THESIS_H2O_ROOT", str(ROOT/"mcwm_bus/runtime/H2Oplus"))
sys.path.insert(0, str(ROOT/"mcwm_bus"))
import torch
from factored_model import FactoredBusModel, OUTPUTS, WIDTH


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.s = torch.rand(8, 229)
        self.s[:, 1:3] = torch.tensor([1/16, 0])
        slots = self.s[:, 5:].reshape(-1, 16, 14)
        slots[:, :, 0] = 0
        slots[:, 0, 0] = 1
        slots[:, :, 1] = torch.arange(16) % 12
        self.a = torch.linspace(-1, 1, 8).reshape(-1, 1).requires_grad_()

    def test_partition_and_capacity(self):
        self.assertEqual(sorted(k for v in OUTPUTS.values() for k in v), list(range(WIDTH)))
        models = [FactoredBusModel(masked=masked) for masked in (True, False)]
        self.assertEqual(*[sum(p.numel() for p in m.parameters()) for m in models])
        self.assertEqual(models[0](self.s, self.a).shape, (8, WIDTH))

    def test_excluded_inputs_do_not_change_duration_or_headway(self):
        model = FactoredBusModel()
        changed = self.s.clone()
        changed[:, 3:5] += 2
        changed[:, 5:].reshape(-1, 16, 14)[:, :, 7:10] += 3
        indices = OUTPUTS["duration"]+OUTPUTS["headway"]
        torch.testing.assert_close(model(self.s, self.a)[:, indices], model(changed, self.a)[:, indices], rtol=0, atol=0)

    def test_residual_initialization_and_backward(self):
        base = FactoredBusModel()
        model = FactoredBusModel(base)
        torch.testing.assert_close(model(self.s, self.a), base(self.s, self.a), rtol=0, atol=0)
        loss = model(self.s, self.a).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        for head in model.head.values():
            self.assertTrue(torch.isfinite(head[-1].weight.grad).all())
            self.assertGreater(float(head[-1].weight.grad.abs().sum()), 0)

    def test_action_enters_each_mechanism(self):
        model = FactoredBusModel()
        out = model(self.s, self.a)
        for name, indices in OUTPUTS.items():
            grad = torch.autograd.grad(out[:, indices].sum(), self.a, retain_graph=True)[0]
            self.assertGreater(float(grad.abs().sum()), 0, name)


if __name__ == "__main__":
    unittest.main()
