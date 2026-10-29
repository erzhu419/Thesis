"""Synthetic continuation checks using the actual historical model classes."""

from contextlib import nullcontext
from functools import lru_cache
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model_state import capture_augmentation, restore_augmentation


@lru_cache(None)
def load_original():
    folder = Path(__file__).resolve().parents[1] / "mcwm_seed_confirmation_v13"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("v105_test_original", folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Costs:
    def phase(self, name):
        return nullcontext()


def actor(states, deterministic=True):
    return states[:, :1] * .2 - .1, None


def build_fixture(model_class, device="cpu"):
    """Eight synthetic transitions; no dataset, fit or native environment."""
    from model import BusModel, Normalizer, targets, losses
    from control_model import QDeltaBellman
    from mechanisms import TargetSupport
    from confirmation_common import P

    torch.set_num_threads(1)
    device = torch.device(device)
    torch.manual_seed(6251)
    aug = model_class.__new__(model_class)
    aug.device, aug.out, aug.costs = device, Path("unused-fixture"), Costs()
    aug.protocol = dict(P, arm="fixed0250", training_seed=6201, relative_out="unused-fixture")
    aug.p = aug.protocol
    aug.mode, aug.family = "fixed0250", "residual"
    aug.history = [dict(source_events=20000, selected_family="residual",
                        calibration_loss=dict(direct=3., residual=2.), fit_weight_range=[.1, 1.])]
    aug.stations = torch.full((12,), 4., device=device)
    n = 8
    s = np.zeros((n, 229), np.float32)
    s[:, 0], s[:, 1] = .1 + np.arange(n) * .001, 1 / 16
    slots = s[:, 5:].reshape(n, 16, 14)
    slots[:, 0, :13] = [1, 0, .25, 1, 1, .5, .5, .1, .2, .1, .1, 1, 1]
    ns = s.copy()
    dt = np.arange(n, dtype=np.float32) + 5
    ns[:, 0] += dt / 24000
    ns[:, 3] += (np.arange(n) + 1) / 20000
    ns[:, 7] = .5
    raw = dict(s=s, a=np.linspace(-.8, .8, n, dtype=np.float32)[:, None],
               ns=ns, dt=dt, r=-7 * dt)
    aug.target = {split: targets(raw, aug.stations, device)
                  for split in ("fit", "stop", "calibrate")}
    aug.fit = targets(raw, aug.stations, device)
    aug.fit_ns = torch.as_tensor(ns, device=device)
    aug.norm = Normalizer(aug.fit).to(device)
    aug.banks = {family: [] for family in ("source", "direct", "residual")}
    for member in range(3):
        source = BusModel().to(device).eval()
        direct = BusModel().to(device).eval()
        residual = BusModel(source).to(device).eval()
        with torch.no_grad():
            residual.head[-1].bias.fill_(.02 * (member + 1))
        aug.banks["source"].append(source)
        aug.banks["direct"].append(direct)
        aug.banks["residual"].append(residual)
    aug.models = aug.banks[aug.family]
    with torch.no_grad():
        aug.errors = losses(aug.predict(aug.fit["s"], aug.fit["a"]), aug.fit, aug.norm).mean(1)
    aug.support = TargetSupport(aug.target["fit"], aug.target["stop"], n, .95)
    aug.trust = QDeltaBellman(229, 1, gamma=aug.protocol["qdelta_gamma"], device=device)
    aug.rng = np.random.default_rng(6201 + 30000)
    aug.rng.integers(100, size=11)
    aug.generator = torch.Generator(device=device).manual_seed(6201 + 40000)
    torch.rand(13, generator=aug.generator, device=device)
    # Populate real Adam moments and the Polyak target, not an empty optimizer.
    aug.update_trust(actor)
    return aug


class ModelStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = load_original().previous.previous.ContextModel

    def setUp(self):
        self.aug = build_fixture(self.model_class)

    def restored(self, state=None):
        payload = capture_augmentation(self.aug) if state is None else state
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        return restore_augmentation(torch.load(stream, weights_only=False),
                                    model_class=self.model_class, device="cpu",
                                    out="unused-restored", costs=Costs())

    def test_live_refit_banks_prediction_and_next_generation(self):
        # Stand in for a completed refit's changed bank; no pilot file is read.
        with torch.no_grad():
            self.aug.banks["residual"][1].head[-1].bias.add_(.37)
        saved_rng = torch.get_rng_state().clone()
        with patch.object(self.model_class, "__init__", side_effect=AssertionError("must not fit")):
            restored = self.restored()
        self.assertTrue(torch.equal(saved_rng, torch.get_rng_state()))
        self.assertIs(restored.models, restored.banks["residual"])
        for source, residual in zip(restored.banks["source"], restored.banks["residual"]):
            self.assertIs(source, residual.base)
            self.assertTrue(all(not p.requires_grad for p in source.parameters()))
            self.assertTrue(all(p.requires_grad for p in residual.head.parameters()))
        for family in ("source", "direct", "residual"):
            for old, new in zip(self.aug.banks[family], restored.banks[family]):
                torch.testing.assert_close(old(self.aug.fit["s"], self.aug.fit["a"]),
                                           new(restored.fit["s"], restored.fit["a"]), rtol=0, atol=0)
        torch.testing.assert_close(self.aug.support.distance(self.aug.fit["s"], self.aug.fit["a"]),
                                   restored.support.distance(restored.fit["s"], restored.fit["a"]), rtol=0, atol=0)
        expected, weight = self.aug.generate(actor, 5)
        actual, restored_weight = restored.generate(actor, 5)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(weight, restored_weight, rtol=0, atol=0)
        np.testing.assert_array_equal(self.aug.rng.integers(1000, size=20), restored.rng.integers(1000, size=20))

    def test_trust_adam_and_tau_continue_identically(self):
        self.assertTrue(self.aug.trust.optimizer.state)
        restored = self.restored()
        expected, actual = self.aug.update_trust(actor), restored.update_trust(actor)
        self.assertEqual(expected, actual)
        for original, recovered in ((self.aug.trust.net, restored.trust.net),
                                     (self.aug.trust.target, restored.trust.target)):
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, recovered.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(self.aug.trust.weight(self.aug.fit["s"], self.aug.fit["a"]),
                                   restored.trust.weight(restored.fit["s"], restored.fit["a"]), rtol=0, atol=0)

    def test_snapshot_is_independent_and_missing_trust_state_fails(self):
        state = capture_augmentation(self.aug)
        original = state["errors"].clone()
        self.aug.errors.add_(1)
        torch.testing.assert_close(state["errors"], original, rtol=0, atol=0)
        del state["trust"]["optimizer"]
        with self.assertRaises(KeyError):
            self.restored(state)


if __name__ == "__main__":
    unittest.main()
