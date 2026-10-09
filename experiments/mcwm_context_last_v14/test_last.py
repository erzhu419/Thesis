"""Persistent routing and next-update gradient gates stay separate from the main trial."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from last_common import HERE, P, PREVIOUS, module

run = module("last_test_run", HERE / "run.py")
launch = module("last_test_launch", HERE / "launch.py")


def model():
    return dict(classes=["calibrated", "joint50", "joint75"], mean=[0.]*3, scale=[1.]*3,
        centers=[[0.]*3, [1.]*3, [2.]*3], variance=[1.]*3)


class LastTests(unittest.TestCase):
    def test_hysteresis_requires_persistent_evidence_and_uses_hard_weight(self):
        filt = run.HysteresisFilter(model(), P)
        # The filter's original evidence function belongs to its own imported module.
        globals_ = run.HysteresisFilter.__mro__[1].observe.__globals__
        with patch.dict(globals_, evidence=lambda m, f: np.array([-4., -4., 4.])):
            self.assertEqual(filt.observe([0.]*3), .25)
            self.assertEqual(filt.observe([0.]*3), .30)
        filt.reset()
        self.assertEqual(filt.coefficient, .25)
        self.assertFalse(filt.active)
        with patch.dict(globals_, evidence=lambda m, f: np.array([4., -4., -4.])):
            for _ in range(8):
                self.assertEqual(filt.observe([0.]*3), .25)

    def test_branch_roster_has_only_new_interventions_and_explicit_references(self):
        rows = launch.jobs()
        self.assertEqual(len(rows), 1224)
        self.assertEqual(sum("--evaluate" in r["cmd"] for r in rows), 1200)
        self.assertFalse(any("--initialize" in r["cmd"] for r in rows))
        trains = [r for r in rows if r["signature"].endswith("/train")]
        self.assertEqual(len(trains), 24)
        self.assertTrue(all(any(arm in r["cmd"] for arm in P["training_arms"]) for r in trains))
        self.assertTrue(launch.main_launch.remote().endswith("/mcwm_context_last_v14"))
        for row in rows:
            self.assertNotIn("require_node", row)
            self.assertEqual(row["vram"], 0)
        self.assertEqual(P["reference_experiment"], "mcwm_seed_confirmation_v13")

    def test_branch_days_do_not_overlap_main_days(self):
        own = {(b["demand_seed"], b["sumo_seed"]) for role in ("development_blocks", "test_blocks") for b in P[role]}
        main = {(b["demand_seed"], b["sumo_seed"]) for role in ("development_blocks", "test_blocks") for b in PREVIOUS[role]}
        self.assertFalse(own & main)

    def test_probe_measures_conflicting_gradients_without_changing_current_coefficient(self):
        learner = object.__new__(run.LastTrainer)
        parameter = torch.nn.Parameter(torch.tensor(0.))
        learner.critic = torch.nn.ParameterList([parameter])
        learner.probe, learner.probe_losses = True, []
        learner.model_loss = (parameter+1.)**2
        learner.config = SimpleNamespace(online_fraction=.5)
        learner.costs = SimpleNamespace(phase=lambda name: nullcontext())
        learner.protocol, learner.coefficient = P, .25
        parent = run.LastTrainer.__mro__[1]
        with patch.object(parent, "_ensemble_td_loss", return_value=torch.tensor(0.)), \
                patch.object(run.main_run.base.TransitH2OTrainerV7, "_ensemble_td_loss", side_effect=lambda *a, **k:(parameter-1.)**2):
            learner._ensemble_td_loss(None, None)
            learner._ensemble_td_loss(None, None)
        self.assertAlmostEqual(learner.pending_relation["gradient_cosine"], -1.)
        self.assertEqual(learner.coefficient, .25)

    def test_gradient_permission_changes_only_the_next_update(self):
        learner = object.__new__(run.LastTrainer)
        learner.arm, learner.protocol = "gradient_gated", P
        learner.context_filter = SimpleNamespace(coefficient=.30)
        learner.gradient_allowed, learner.gradient_trace, learner.total_updates = False, [], 50048
        used = []
        def update(*args, **kwargs):
            used.append(learner.coefficient)
            learner.total_updates += 1
            learner.pending_relation = dict(gradient_cosine=1., real_gradient_norm=1., model_gradient_norm=1.)
            return {}
        with patch.object(run.LastTrainer.__mro__[1], "update", side_effect=update):
            learner.update(online=True)
            learner.update(online=True)
        self.assertEqual(used, [.25, .30])


if __name__ == "__main__":
    unittest.main()
