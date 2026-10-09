"""Check coefficient isolation, frozen budgets and paired test-day collection."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from midweight_common import HERE, P, PREVIOUS, FIXED, module, write

run = module("midweight_test_run", HERE / "run.py")
launch = module("midweight_test_launch", HERE / "launch.py")
collector = module("midweight_test_collector", HERE / "collect_results.py")


class MidweightTests(unittest.TestCase):
    def test_trainer_and_training_loop_are_unchanged(self):
        self.assertIs(run.MappingTrainer, run.previous.MappingTrainer)
        self.assertIs(run.train, run.previous.train)
        self.assertIs(run.evaluate, run.previous.evaluate)
        self.assertIs(run.previous.previous.ContextTrainer, run.MappingTrainer)
        self.assertEqual(run.train.__globals__["HERE"], HERE)
        self.assertEqual(run.train.__globals__["FIXED"], FIXED)

    def test_new_fixed_coefficients_are_actually_applied(self):
        for arm, coefficient in FIXED.items():
            learner = object.__new__(run.MappingTrainer)
            learner.configure_context(dict(P, arm=arm, relative_out=f"smoke/joint75/{arm}_seed6201"), "joint75", None)
            self.assertIsNone(learner.context_filter)
            self.assertEqual(learner.coefficient, coefficient)
            self.assertEqual(learner.output, HERE / f"smoke/joint75/{arm}_seed6201/results")

    def test_gated_mapping_and_recognizer_are_frozen(self):
        for seed in P["training_seeds"]:
            bundle = json.loads((HERE.parent / "mcwm_context_control_v10/adaptation" / f"seed{seed}.json").read_text())
            learner = object.__new__(run.MappingTrainer)
            learner.configure_context(dict(P, arm="shared_gated", relative_out=f"smoke/joint75/shared_gated_seed{seed}"), "joint75", bundle)
            self.assertIsInstance(learner.context_filter, run.previous.GatedWindowFilter)
            self.assertEqual(learner.bundle["coefficients"], PREVIOUS["shared_coefficients"])
            self.assertEqual(learner.context_filter.model, bundle["physical_recognizer"])
            self.assertEqual(learner.coefficient, .25)

    def test_only_arms_test_days_and_description_changed(self):
        changed = {key for key in P if P[key] != PREVIOUS.get(key)}
        self.assertEqual(changed, {"name", "arms", "fixed_coefficients", "test_blocks", "selection", "interpretation"})
        self.assertEqual(P["source_event_budget"], 25000)
        self.assertEqual(P["warmup_events"], 5000)
        self.assertEqual(P["real_td_coefficient"], .75)

    def test_new_days_have_not_been_used_in_prior_context_studies(self):
        new = {(b["demand_seed"], b["sumo_seed"]) for b in P["test_blocks"]}
        for folder in ("mcwm_context_v7", "mcwm_td_upper_v8", "mcwm_context_control_v10", "mcwm_context_mapping_v11"):
            old = json.loads((HERE.parent / folder / "protocol.json").read_text())
            self.assertFalse(new & {(b["demand_seed"], b["sumo_seed"]) for role in
                ("development_blocks", "test_blocks") for b in old.get(role, [])})

    def test_cpu_jobs_are_unpinned_and_daily_evaluations_are_independent(self):
        items = launch.base.placements()
        self.assertEqual(len(items), 24)
        self.assertEqual(len([launch.task(i, c) for i in items for c in range(10, 20)]), 240)
        for item in items:
            train = launch.task(item)
            evaluation = launch.task(item, 10)
            self.assertNotIn("require_node", train)
            self.assertNotIn("require_node", evaluation)
            self.assertNotIn("wait_for_files", train)
            self.assertEqual(train["allowed_nodes"], P["training_nodes"])
            self.assertEqual(train["vram"], evaluation["vram"])
            self.assertEqual(train["vram"], 0)
            self.assertTrue(train["allow_cpu_training"])
            self.assertEqual(evaluation["wait_for_files"], [str(launch.base.gate(item))])
            self.assertIn(str(HERE), train["cmd"])
        self.assertEqual(launch.remote("node001"), launch.infra.HPC+"/Thesis/experiments/mcwm_context_midweight_v12")

    def fixture(self, folder):
        for scene in collector.CONTEXTS:
            for index, arm in enumerate(P["arms"]):
                out = folder / "runs" / scene / f"{arm}_seed6201" / "results"
                coefficient = FIXED.get(arm, .31)
                write(out / "ready.json", dict(arm=arm, seed=6201, scenario=scene,
                    continuation_updates=20000, total_policy_updates=70000, total_critic_updates=70000,
                    target_online_training_events=0, real_td_coefficient=.75, source_events=25000,
                    context_provider_seed=6202))
                write(out / "runtime.json", dict(device="cpu", gpu_model=None))
                write(out / "coefficient_usage.json", dict(updates=20000, update_mean_coefficient=coefficient,
                    minimum_coefficient=coefficient, maximum_coefficient=coefficient))
                if arm not in FIXED:
                    write(out / "context_trace.json", dict(trace=[dict(predicted_context=scene,
                        context_accepted=True, belief=[.1, .2, .7])]))
                for case in range(10, 20):
                    write(out / f"eval_{case:02d}.json", dict(arm=arm, training_seed=6201,
                        scenario=scene, case=case, split="test", horizon_seconds=P["evaluation_horizon_seconds"],
                        deployment_hold_scale=1., sumo_version=[0, "SUMO 1.25.0"],
                        demand_scale=P["evaluation_demand_scales"][(case-10)//5],
                        generalized_seconds_per_departed_passenger=2000.+3*index+case,
                        passengers=dict(departed_passengers=100, completion_rate=.99,
                            accrued_wait_seconds_per_departed_passenger=300.,
                            accrued_in_vehicle_seconds_per_departed_passenger=800.),
                        **P["test_blocks"][case % 5]))

    def test_collector_reports_paired_fixed_and_gated_differences(self):
        with TemporaryDirectory() as directory, patch.object(collector, "HERE", Path(directory)), \
                patch.object(collector, "P", dict(P, training_seeds=[6201])):
            self.fixture(collector.HERE)
            summary = collector.summarize()
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["training_ready"], 12)
            self.assertEqual(summary["evaluation_ready"], 120)
            for row in summary["outcomes"]:
                if row["arm"] == "shared_gated" and row["reference"] == "fixed03125":
                    self.assertEqual(row["delta_seconds"], 3.)
                    self.assertEqual(row["paired_day_deltas"], [3.]*5)

    def test_collector_rejects_stale_test_days(self):
        with TemporaryDirectory() as directory, patch.object(collector, "HERE", Path(directory)), \
                patch.object(collector, "P", dict(P, training_seeds=[6201])):
            self.fixture(collector.HERE)
            path = collector.HERE / "runs/calibrated/fixed0250_seed6201/results/eval_10.json"
            row = json.loads(path.read_text())
            write(path, dict(row, demand_seed=68601))
            with self.assertRaises(AssertionError):
                collector.summarize()

    def test_collector_does_not_treat_partial_results_as_complete(self):
        with TemporaryDirectory() as directory, patch.object(collector, "HERE", Path(directory)), \
                patch.object(collector, "P", dict(P, training_seeds=[6201])):
            self.fixture(collector.HERE)
            (collector.HERE / "runs/joint75/shared_gated_seed6201/results/eval_19.json").unlink()
            summary = collector.summarize()
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["evaluation_ready"], 119)
            self.assertNotIn("outcomes", summary)


if __name__ == "__main__":
    unittest.main()
