"""Check causal feature selection, fallback and the independent campaign roster."""

import ast
import json
import unittest
from types import SimpleNamespace
import numpy as np

import launch
import run
from control_common import P, HERE


def bundle():
    classes = [s["name"] for s in P["scenarios"]]
    def model(size):
        return dict(classes=classes, mean=[0.]*size, scale=[1.]*size,
            centers=[[float(i)]*size for i in range(3)], variance=[1.]*size)
    return dict(recognizer=model(8), physical_recognizer=model(3),
        coefficients=dict(zip(classes, [0., .125, .25])), global_coefficient=.25)


class ControlTests(unittest.TestCase):
    def trainer(self, arm):
        learner = object.__new__(run.AdaptiveTrainer)
        learner.configure_context(dict(P, arm=arm, relative_out=f"smoke/joint75/{arm}_seed6201"), "joint75", bundle())
        return learner

    def test_full_and_physical_models_use_their_own_dimensions(self):
        for arm, dimensions in (("recognized_full",8), ("recognized_physical",3), ("physical_window4_fallback",3)):
            learner = self.trainer(arm)
            self.assertEqual(len(learner.feature_indices), dimensions)
            self.assertEqual(len(learner.context_filter.model["mean"]), dimensions)
            learner.context_filter.observe([.5]*dimensions)
            self.assertTrue(0. <= learner.context_filter.coefficient <= .25)

    def test_fixed_control_is_not_adaptive(self):
        learner = self.trainer("fixed0250")
        self.assertIsNone(learner.context_filter)
        self.assertEqual(learner.coefficient,.25)

    def test_window_low_margin_uses_global_coefficient(self):
        learner = self.trainer("physical_window4_fallback")
        learner.context_filter.observe([.5]*3)
        self.assertEqual(learner.context_filter.decision,(.25,False))

    def test_terminal_resets_window_and_records_episode_index(self):
        learner = self.trainer("physical_window4_fallback")
        learner.blocks.size = 2
        states = np.zeros((5,229))
        states[3:,14:16] = 2.
        durations = np.array([1.,1.,1.,np.expm1(2.),np.expm1(2.)])
        learner.replay_buffer = SimpleNamespace(online_size=5, _online_total_appended=5,
            _online_terminals=np.array([0.,0.,1.,0.,0.]), _online_delta_t_seconds=durations,
            _online_observations=states, _online_actions=np.zeros((5,1)))
        learner.observe_completed()
        self.assertEqual([r["episode_index"] for r in learner.trace],[0,1])
        self.assertEqual(len(learner.context_filter.history),1)
        np.testing.assert_allclose(learner.trace[-1]["belief"],
            run.window.softmax(np.array([-2.,-.5,0.])))

    def test_roster_is_unpinned_and_does_not_wait_for_v8(self):
        items = launch.base.placements()
        self.assertEqual(len(items),24)
        self.assertEqual(len([launch.task(i,c) for i in items for c in range(10,20)]),240)
        for item in items:
            row=launch.task(item)
            self.assertNotIn("require_node",row)
            self.assertNotIn("wait_for_files",row)
            self.assertEqual(row["allowed_nodes"],[f"node{i:03d}" for i in range(1,7)])
            self.assertTrue(row["allow_cpu_training"])
            self.assertEqual(row["vram"],0)
        self.assertTrue(launch.task(launch.base.placements(True)[0])["allow_cpu_training"])

    def test_cpu_loop_only_changes_the_original_device_assignment(self):
        def function(path):
            tree = ast.parse(path.read_text())
            return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
        current = function(HERE / "run.py")
        original = function(HERE.parent / "mcwm_context_v7/run.py")
        self.assertEqual(ast.dump(ast.Module(body=current.body[1:],type_ignores=[])),
            ast.dump(ast.Module(body=original.body[1:],type_ignores=[])))

    def test_test_days_are_disjoint_from_v7_and_v8(self):
        new={(b["demand_seed"],b["sumo_seed"]) for b in P["test_blocks"]}
        for previous in ("mcwm_context_v7","mcwm_td_upper_v8"):
            p=json.loads((HERE.parent / previous / "protocol.json").read_text())
            old={(b["demand_seed"],b["sumo_seed"]) for role in ("development_blocks","test_blocks") for b in p[role]}
            self.assertFalse(new & old)


if __name__ == "__main__":
    unittest.main()
