"""Check the interventions and the causal diagnostic comparisons before launch."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from mapping_common import P, HERE, module, write

launch = module("mapping_test_launch", HERE / "launch.py")
run = module("mapping_test_run", HERE / "run.py")
collector = module("mapping_test_collector", HERE / "collect_results.py")
policy_arm = collector.policy_arm
window, GatedWindowFilter = run.window, run.GatedWindowFilter


def bundle():
    model = dict(classes=["calibrated", "joint50", "joint75"], mean=[0.]*3, scale=[1.]*3,
        centers=[[float(i)]*3 for i in range(3)], variance=[1.]*3)
    return dict(physical_recognizer=model, coefficients=dict(calibrated=.25, joint50=.25, joint75=.125),
        global_coefficient=.25, provider_seed=6202, consumer_seed=6201)


class MappingTests(unittest.TestCase):
    def trainer(self, arm, seed=6201):
        learner = object.__new__(run.MappingTrainer)
        learner.configure_context(dict(P, arm=arm, relative_out=f"smoke/joint75/{arm}_seed{seed}"), "joint75", bundle())
        return learner

    def test_training_loop_is_the_same_function(self):
        self.assertIs(run.train, run.previous.train)
        self.assertIs(run.evaluate, run.previous.previous.evaluate)
        self.assertTrue(launch.remote("node001").endswith("/mcwm_context_mapping_v11"))

    def test_fixed_controls_have_no_recognizer(self):
        for arm, coefficient in P["fixed_coefficients"].items():
            learner = self.trainer(arm)
            self.assertIsNone(learner.context_filter)
            self.assertEqual(learner.coefficient, coefficient)

    def test_shared_mapping_does_not_depend_on_training_seed(self):
        for seed in P["training_seeds"]:
            for arm in ("shared_soft", "shared_gated"):
                learner = self.trainer(arm, seed)
                self.assertEqual(learner.bundle["coefficients"], P["shared_coefficients"])
                self.assertEqual(learner.context_filter.model, bundle()["physical_recognizer"])

    def test_evidence_filter_is_identical_after_routing_intervention(self):
        model = bundle()["physical_recognizer"]
        args = (model, [.25, .25, .375], 4, .25, .15)
        soft = window.WindowFilter(*args)
        gated = GatedWindowFilter(*args, adjusted_context="joint75")
        for features in ([0.]*3, [2.]*3, [1.]*3, [.2]*3, [2.5]*3):
            soft.observe(features)
            gated.observe(features)
            np.testing.assert_array_equal(soft.belief, gated.belief)
        soft.reset()
        gated.reset()
        self.assertEqual(gated.coefficient, .25)
        self.assertEqual(len(gated.history), 0)

    def test_neutral_context_does_not_leak_in_gated_arm(self):
        soft = self.trainer("shared_soft").context_filter
        gated = self.trainer("shared_gated").context_filter
        soft.belief = gated.belief = np.array([.65, .2, .15])
        self.assertTrue(gated.decision[1])
        self.assertGreater(soft.coefficient, .25)
        self.assertEqual(gated.coefficient, .25)
        gated.belief = np.array([.1, .2, .7])
        self.assertAlmostEqual(gated.coefficient, .25+.125*.7)
        gated.belief = np.array([.3, .3, .4])
        self.assertEqual(gated.decision, (.25, False))

    def test_legacy_routing_preserves_the_previous_coefficients(self):
        learner = self.trainer("legacy_window")
        self.assertEqual(learner.bundle["coefficients"], bundle()["coefficients"])
        learner.context_filter.belief = np.array([.1, .2, .7])
        self.assertAlmostEqual(learner.context_filter.coefficient, .25-.125*.7)

    def test_episode_end_resets_the_original_evidence_window(self):
        learner = self.trainer("shared_gated")
        learner.blocks.size = 2
        states = np.zeros((5, 229))
        states[3:, 14:16] = 2.
        learner.replay_buffer = SimpleNamespace(online_size=5, _online_total_appended=5,
            _online_terminals=np.array([0., 0., 1., 0., 0.]),
            _online_delta_t_seconds=np.array([1., 1., 1., np.expm1(2.), np.expm1(2.)]),
            _online_observations=states, _online_actions=np.zeros((5, 1)))
        learner.observe_completed()
        self.assertEqual([r["episode_index"] for r in learner.trace], [0, 1])
        self.assertEqual(len(learner.context_filter.history), 1)
        self.assertEqual(learner.trace[-1]["predicted_context"], "joint75")
        self.assertGreater(learner.coefficient, .25)

    def test_update_usage_counts_gradient_updates_not_blocks(self):
        learner = self.trainer("shared_soft")
        with patch.object(run.previous.AdaptiveTrainer, "update", return_value={}):
            for coefficient in (.25, .375, .3):
                learner.coefficient = coefficient
                learner.update(online=True)
        self.assertEqual(learner.coefficient_count, 3)
        self.assertEqual(learner.changed_updates, 2)
        self.assertAlmostEqual(learner.coefficient_sum/3, (.25+.375+.3)/3)

    def test_oracle_alias_matches_static_context_training_exactly(self):
        for scene, coefficient in P["shared_coefficients"].items():
            self.assertEqual(P["fixed_coefficients"][policy_arm("oracle_shared", scene)], coefficient)

    def test_collector_does_not_count_oracle_aliases_as_new_runs(self):
        with TemporaryDirectory() as temporary, patch.object(collector, "HERE", Path(temporary)), \
                patch.object(collector, "P", dict(P, training_seeds=[6201])):
            for scene in collector.CONTEXTS:
                for index, arm in enumerate(P["arms"]):
                    folder = collector.HERE / "runs" / scene / f"{arm}_seed6201" / "results"
                    write(folder / "ready.json", dict(arm=arm, seed=6201, scenario=scene,
                        continuation_updates=20000, total_policy_updates=70000, total_critic_updates=70000,
                        target_online_training_events=0, real_td_coefficient=.75, source_events=25000,
                        context_provider_seed=6202))
                    write(folder / "runtime.json", dict(device="cpu", gpu_model=None))
                    write(folder / "coefficient_usage.json", dict(updates=20000, update_mean_coefficient=.25))
                    if arm not in P["fixed_coefficients"]:
                        write(folder / "context_trace.json", dict(trace=[dict(predicted_context=scene,
                            context_accepted=True, belief=[.1, .2, .7])]))
                    for case in range(10, 20):
                        write(folder / f"eval_{case:02d}.json", dict(arm=arm, training_seed=6201,
                            scenario=scene, case=case, split="test", horizon_seconds=P["evaluation_horizon_seconds"],
                            deployment_hold_scale=1., sumo_version=[0, "SUMO 1.25.0"],
                            demand_scale=P["evaluation_demand_scales"][(case-10)//5],
                            generalized_seconds_per_departed_passenger=2000.+3*index+case,
                            passengers=dict(departed_passengers=100, completion_rate=.99,
                                accrued_wait_seconds_per_departed_passenger=300.,
                                accrued_in_vehicle_seconds_per_departed_passenger=800.),
                            **P["test_blocks"][case % 5]))
            summary = collector.summarize()
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["training_ready"], 15)
            self.assertEqual(summary["evaluation_ready"], 150)
            for outcome in summary["outcomes"]:
                if outcome["arm"] == "oracle_shared" and outcome["reference"] == "fixed0250":
                    self.assertEqual(outcome["delta_seconds"], 3. if outcome["scenario"] == "joint75" else 0.)

    def test_collection_uses_shared_storage_via_an_actual_execution_node(self):
        with TemporaryDirectory() as temporary, patch.object(launch, "HERE", Path(temporary)), \
                patch.object(launch.infra.scheduler, "run_on") as connection, \
                patch.object(launch.infra.scheduler, "_ssh_rsync_shell_for_node", return_value="ssh") as shell, \
                patch.object(launch.infra.scheduler, "_ssh_target_for_node", return_value="host"), \
                patch.object(launch.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            launch.collect([dict(node="node004", smoke=True)])
            connection.assert_called_once_with("node004", "true", timeout=30, check=True)
            shell.assert_called_once_with("node004")

    def test_cpu_roster_is_unpinned_and_evaluation_is_sharded(self):
        items = launch.base.placements()
        self.assertEqual(len(items), 30)
        self.assertEqual(len([launch.task(i, c) for i in items for c in range(10, 20)]), 300)
        for item in items:
            task = launch.task(item)
            self.assertNotIn("require_node", task)
            self.assertNotIn("wait_for_files", task)
            self.assertEqual(task["allowed_nodes"], P["training_nodes"])
            self.assertEqual(task["vram"], 0)
            self.assertTrue(task["allow_cpu_training"])
            evaluation = launch.task(item, 10)
            self.assertNotIn("require_node", evaluation)
            self.assertEqual(evaluation["vram"], 0)
            self.assertEqual(evaluation["wait_for_files"], [str(launch.base.gate(item))])

    def test_test_days_are_new_and_training_budget_is_unchanged(self):
        new = {(b["demand_seed"], b["sumo_seed"]) for b in P["test_blocks"]}
        for folder in ("mcwm_context_v7", "mcwm_td_upper_v8", "mcwm_context_control_v10"):
            old = json.loads((HERE.parent / folder / "protocol.json").read_text())
            self.assertFalse(new & {(b["demand_seed"], b["sumo_seed"]) for role in
                ("development_blocks", "test_blocks") for b in old.get(role, [])})
        self.assertEqual(P["source_event_budget"], 25000)
        self.assertEqual(P["warmup_events"], 5000)
        self.assertEqual(P["real_td_coefficient"], .75)


if __name__ == "__main__":
    unittest.main()
