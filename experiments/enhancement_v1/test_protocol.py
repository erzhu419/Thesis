"""Cheap checks for duplicated cells, incorrect budgets, and missing evaluation gates."""

import unittest
from unittest.mock import patch
from study_common import P
from deploy import make_jobs, environment, spec, root, transfer, CPUS
from collect import interval
from rebalance_cs_cpu import queued_groups
from rebalance_gpu import movable_groups


class ProtocolTests(unittest.TestCase):
    def test_budget(self):
        updates = P["adaptation_days"]*P["updates_per_day"]
        self.assertEqual(P["prefix_updates"]+updates, 16144)
        self.assertEqual(P["prefix_updates"]+updates+updates//P["extra_critic_every"], 17680)
        self.assertNotEqual(P["source_fit_day"], P["source_stop_day"])

    def test_matrix(self):
        with patch("deploy.write"):
            jobs = make_jobs()
        self.assertEqual(len(jobs), 862)
        self.assertEqual(len({j["signature"] for j in jobs}), len(jobs))
        tests = [j for j in jobs if " --evaluate " in j["cmd"]]
        self.assertEqual(len(tests), 766)
        self.assertEqual({j["require_node"] for j in tests}, set(CPUS))
        self.assertTrue(all(j["vram"] == 0 and j["wait_for_files"] for j in tests))
        self.assertTrue(all("jtl110cpu" not in j["require_node"] for j in jobs))
        self.assertTrue(all({"runs", "smoke", "runtime"}.issubset(j["stage_excludes"]) for j in jobs))
        self.assertTrue(all("/gates/" in p for j in jobs for p in j.get("wait_for_files", [])))

    def test_evaluation_data(self):
        env = environment("node001", evaluation=True)
        self.assertTrue(env["THESIS_DATASET"].endswith("contract.h5"))
        self.assertEqual(env["SUMO_HOME"], env["THESIS_SUMO_ROOT"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_resource_families_separate_training_from_other_work(self):
        task = spec("mc", "fixed_direct", 22104, "node007")
        sibling = spec("mc", "fixed_direct", 22105, "node007")
        self.assertEqual(task["vram_resource_family"], sibling["vram_resource_family"])
        self.assertEqual(task["vram"], 768)
        self.assertEqual(task["ram_mb"], 9000)
        for other in (spec("legacy", "learned_full", 22104, "node007"),
                      spec("mc", "fixed_direct", 22000, "node007", smoke=True),
                      spec("mc", "fixed_direct", 22104, "node007", case=0),
                      spec("mc", "fixed_direct", 22104, "jtl110gpu2")):
            self.assertNotEqual(task["vram_resource_family"], other["vram_resource_family"])

    def test_seed_unit(self):
        value = interval([1., 1., 1., 1., 1.])
        self.assertEqual(value["ci95"], [1., 1.])
        self.assertEqual(len(value["by_training_seed"]), 5)

    def test_cs_cpu_placement(self):
        task = spec("cs", "nau_bc1", 22101, "node005")
        self.assertEqual(task["vram"], 0)
        self.assertEqual(task["cpu"], 1)
        self.assertTrue(task["allow_cpu_training"])
        self.assertIn("--device cpu", task["cmd"])
        self.assertEqual(root("node005"), root("node007"))
        gpu = spec("mc", "fixed_residual", 22101, "node007")
        self.assertGreater(gpu["vram"], 0)

    def test_shared_storage_handoff(self):
        item = dict(kind="cs", arm="nau_bc1", seed=22101, node="node005", smoke=False)
        with patch("deploy.scheduler.run_on") as check, patch("deploy.subprocess.Popen") as copy, \
                patch("deploy.write"):
            transfer(item)
        check.assert_called_once()
        copy.assert_not_called()

    def test_running_group_is_not_moved(self):
        rows = [dict(signature=f"thesis-enhancement-v1/runs/cs/{a}_seed22101/train", status="queued")
                for a in P["cs_arms"]]
        self.assertEqual(set(queued_groups(rows)), {22101})
        rows[0]["status"] = "running"
        self.assertEqual(queued_groups(rows), {})
        self.assertEqual(queued_groups(rows[1:]), {})

    def test_legacy_resume_is_managed_by_runner(self):
        task = spec("legacy", "learned_full", 22103, "jtl110gpu2")
        self.assertTrue(task["cmd"].endswith(" --resume"))
        self.assertEqual(task["resume_flag"], "")
        self.assertTrue(task["resume_managed_by_cmd"])
        self.assertTrue(task["skip_resume_scan"])

    def test_gpu_rebalance_requires_a_complete_unstarted_group(self):
        tasks = [dict(id=arm, signature=f"thesis-enhancement-v1/runs/mc/{arm}_seed22104/train",
                      status="queued") for arm in P["mc_arms"]]
        self.assertEqual(set(movable_groups(tasks)), {22104})
        self.assertEqual(movable_groups(tasks[:-1]), {})
        tasks[0]["status"] = "running"
        self.assertEqual(movable_groups(tasks), {})


if __name__ == "__main__":
    unittest.main()
