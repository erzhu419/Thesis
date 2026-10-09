"""Fresh prefixes, matched resource requests and development-only model selection."""

import ast
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch

from confirmation_common import HERE, P, PREVIOUS, module, write

run = module("confirmation_test_run", HERE / "run.py")
launch = module("confirmation_test_launch", HERE / "launch.py")
collector = module("confirmation_test_collect", HERE / "collect_results.py")


class ConfirmationTests(unittest.TestCase):
    def test_initialization_does_not_load_pretrained_state(self):
        tree = ast.parse(inspect.getsource(run.initialize))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("load_offline_initializer", calls)
        self.assertIn("save_offline_initializer", calls)
        self.assertEqual(len(P["training_seeds"]), 20)
        self.assertFalse(set(P["training_seeds"]) & {6201, 6202, 6203})
        self.assertEqual(P["offline_updates_in_initializer"], 50000)

    def test_h2o_reference_uses_unscaled_observed_td(self):
        learner = object.__new__(run.NoModelTrainer)
        loss = torch.tensor(3.)
        with patch.object(run.base.TransitH2OTrainerV7, "_ensemble_td_loss", return_value=loss):
            self.assertIs(learner._ensemble_td_loss(None, None), loss)

    def test_fixed_and_gated_configurations_are_preserved(self):
        for arm, coefficient in P["fixed_coefficients"].items():
            learner = object.__new__(run.MappingTrainer)
            learner.configure_context(dict(P, arm=arm, relative_out=f"smoke/joint75/{arm}_seed63001"), "joint75", None)
            self.assertEqual(learner.coefficient, coefficient)
            self.assertIsNone(learner.context_filter)
        for key in ("shared_coefficients", "global_coefficient", "window_blocks", "fallback_score_margin",
                    "source_event_budget", "warmup_events", "actor_taper_event", "model_refresh_events"):
            self.assertEqual(P[key], PREVIOUS[key])

    def test_roster_is_unpinned_and_dependencies_are_per_seed_and_policy(self):
        rows = launch.jobs()
        self.assertEqual(len(rows), 5100)
        self.assertEqual(sum("--initialize" in row["cmd"] for row in rows), 20)
        self.assertEqual(sum("--evaluate" in row["cmd"] for row in rows), 4800)
        for row in rows:
            self.assertNotIn("require_node", row)
            self.assertEqual(row["allowed_nodes"], P["training_nodes"])
            self.assertEqual(row["vram"], 0)
            if "/runs/" in row["signature"] and row["signature"].endswith("/train"):
                self.assertIn("initializer_seed", row["wait_for_files"][0])
            if "--evaluate" in row["cmd"]:
                self.assertEqual(len(row["wait_for_files"]), 1)

    def test_collection_runs_server_analysis_and_pulls_only_two_files(self):
        from types import SimpleNamespace
        calls = []
        def connect(node, *args, **kwargs):
            calls.append(node)
            if node == "node007":
                raise RuntimeError("Connection closed by remote host")
        with patch.object(launch.infra.scheduler, "run_on", side_effect=connect), \
                patch.object(launch.infra.scheduler, "_ssh_rsync_shell_for_node", return_value="ssh"), \
                patch.object(launch.infra.scheduler, "_ssh_target_for_node", return_value="host"), \
                patch.object(launch.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as sync:
            self.assertEqual(launch.collect(), "node004")
            self.assertEqual(calls, ["node007", "node004"])
            args = sync.call_args.args[0]
            sources = [a for a in args if a.startswith("host:")]
            self.assertEqual([Path(a).name for a in sources],
                ["analysis_summary.json", "handoff_status.json"])
            self.assertFalse(any(a.endswith("/") for a in sources))
            self.assertNotIn("--include=*/", args)

    def test_server_export_preserves_negative_seed_results_without_raw_records(self):
        init = dict(seed=63001, fresh_initialization=True, offline_updates=50000, online_events=0,
            config=dict(large="server-only"))
        policy = dict(arm="fixed0250", seed=63001, scenario="joint75", initializer_seed=63001,
            fresh_initialization=True, continuation_updates=20000, model_history=["server-only"])
        outcomes = [dict(scenario="joint75", arm="fixed0300", reference="fixed0250", demand_scale=1.,
            seed=seed, delta_seconds=delta, completion_delta_pp=0., paired_day_deltas=[delta]*5)
            for seed, delta in ((63001, -2.), (63002, 4.))]
        report = dict(complete=True, initialization_ready=1, training_ready=1, reference_ready=0,
            evaluation_ready=20, rule_ready=0, initializers=[init], training=[policy], references=[],
            missing=[], paired_seed_outcomes=outcomes)
        with TemporaryDirectory() as temporary, patch.object(collector, "HERE", Path(temporary)), \
                patch.object(collector, "summarize", return_value=report):
            compact = collector.server_analysis()
            self.assertEqual(compact["analysis_location"], "server")
            self.assertNotIn("training", compact)
            self.assertNotIn("missing", compact)
            group = compact["paired_seed_summary"][0]
            self.assertEqual(group["mean_delta_seconds"], 1.)
            self.assertEqual(group["better_seeds"], 1)
            self.assertEqual([r["delta_seconds"] for r in group["seed_deltas"]], [-2., 4.])
            status = json.loads((collector.HERE / "handoff_status.json").read_text())
            self.assertNotIn("config", status["initializers"][0])
            self.assertNotIn("model_history", status["policies"][0])
            self.assertEqual(json.loads((collector.HERE / "summary.json").read_text()), report)

    def test_handoff_needs_only_compact_status_not_local_ready_files(self):
        with TemporaryDirectory() as temporary, patch.object(launch, "HERE", Path(temporary)/HERE.name):
            row = dict(arm="fixed0250", seed=63001, scenario="joint75", initializer_seed=63001,
                fresh_initialization=True, continuation_updates=64)
            write(launch.HERE / "smoke_handoff_status.json", dict(experiment=P["name"], smoke=True,
                initializers=[dict(seed=63001, fresh_initialization=True, offline_updates=64, online_events=0)],
                policies=[row]))
            launch.handoff(True)
            self.assertTrue(launch.init_gate(63001, True).exists())
            item = next(i for i in launch.placements(True) if i["arm"] == "fixed0250")
            self.assertTrue(launch.policy_gate(item).exists())
            self.assertFalse((launch.HERE / "smoke/joint75/fixed0250_seed63001/results/ready.json").exists())

    def test_development_and_test_days_are_disjoint_from_previous_batches(self):
        dev = {(b["demand_seed"], b["sumo_seed"]) for b in P["development_blocks"]}
        test = {(b["demand_seed"], b["sumo_seed"]) for b in P["test_blocks"]}
        self.assertFalse(dev & test)
        for name in ("mcwm_context_v7", "mcwm_td_upper_v8", "mcwm_context_control_v10", "mcwm_context_mapping_v11", "mcwm_context_midweight_v12"):
            old = json.loads((HERE.parent / name / "protocol.json").read_text())
            used = {(b["demand_seed"], b["sumo_seed"]) for role in
                ("development_blocks", "test_blocks") for b in old.get(role, [])}
            self.assertFalse((dev | test) & used)

    def fixture(self, root, omit=None):
        for arm in ("h2o_matched", "fixed0250"):
            for seed in (63001, 63002):
                for case in range(20):
                    if (arm, seed, case) == omit:
                        continue
                    folder = root / "runs/joint75" / f"{arm}_seed{seed}/results"
                    score = 1000.+(100. if seed == 63002 else 0.) if arm == "h2o_matched" else 2000.-(100. if seed == 63002 else 0.)
                    if case >= 10:
                        score = -score
                    write(folder / f"eval_{case:02d}.json", dict(split="development" if case < 10 else "test",
                        generalized_seconds_per_departed_passenger=score))

    def test_selection_uses_development_only_and_never_reselects(self):
        with TemporaryDirectory() as temporary, patch.object(collector, "HERE", Path(temporary)), \
                patch.object(collector, "CONTEXTS", ["joint75"]), \
                patch.object(collector, "P", dict(P, training_seeds=[63001, 63002], arms=["h2o_matched", "fixed0250"])):
            self.fixture(collector.HERE)
            selection = collector.select_development()
            self.assertEqual([r["seed"] for r in selection["selected"]], [63001, 63002])
            self.assertFalse(selection["test_results_used"])
            write(collector.result_path("joint75", "fixed0250", 63001, 0),
                dict(split="development", generalized_seconds_per_departed_passenger=-99999.))
            self.assertEqual(collector.select_development(), selection)

    def test_missing_development_case_does_not_choose_from_partial_candidates(self):
        with TemporaryDirectory() as temporary, patch.object(collector, "HERE", Path(temporary)), \
                patch.object(collector, "CONTEXTS", ["joint75"]), \
                patch.object(collector, "P", dict(P, training_seeds=[63001, 63002], arms=["h2o_matched", "fixed0250"])):
            self.fixture(collector.HERE, ("fixed0250", 63002, 9))
            self.assertIsNone(collector.select_development())
            self.assertFalse((collector.HERE / "selection.json").exists())


if __name__ == "__main__":
    unittest.main()
