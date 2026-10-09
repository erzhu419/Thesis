"""Placement freedom, migrated result handoff and node-local execution."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import launch
from node_run import command


class LaunchTests(unittest.TestCase):
    def test_training_and_evaluation_are_not_pinned(self):
        for item in launch.placements():
            train = launch.task(item)
            self.assertNotIn("require_node", train)
            self.assertNotIn("require_gpu_idx", train)
            self.assertEqual(train["allowed_nodes"], ["node007", "jtl110gpu", "jtl110gpu2"])
            self.assertNotIn("jtl311linux", train["allowed_nodes"])
            self.assertTrue(train["cmd"].startswith(str(launch.HERE/"node_python")))
            for case in range(10):
                evaluation = launch.task(item, case)
                self.assertNotIn("require_node", evaluation)
                self.assertEqual(evaluation["allowed_nodes"], launch.infra.CPUS)

    def test_migrated_task_uses_actual_node(self):
        item = next(i for i in launch.placements() if i["name"] == "calibrated/shared_gated_seed6202")
        signature = launch.P["name"]+"/runs/"+item["name"]+"/train"
        payload = json.dumps(dict(tasks=[dict(signature=signature, node="jtl110gpu2")]))
        submission = json.dumps(dict(submitted=[dict(id="t106260", signature=signature)]))
        with patch.object(launch.Path, "exists", return_value=True), \
                patch.object(launch.Path, "read_text", return_value=submission), \
                patch.object(launch.subprocess, "check_output", return_value=payload):
            located = next(i for i in launch.located_items() if i["name"] == item["name"])
        self.assertEqual(located["node"], "jtl110gpu2")
        self.assertEqual(launch.remote(located["node"]), str(launch.HERE))

    def test_node_local_runtime_preserves_gpu_visibility_and_relative_output(self):
        config = dict(train=dict(python="/resident/train/python", env=dict(THESIS_DATASET="/resident/data.h5")),
            evaluate=dict(python="/shared/eval/python", env=dict(THESIS_DATASET="/shared/contract.h5")))
        args = ["--out", "runs/calibrated/shared_gated_seed6202"]
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
            argv, env = command(config, Path("/selected/experiment"), args)
        self.assertEqual(argv[:3], ["/resident/train/python", "-u", "/selected/experiment/run.py"])
        self.assertEqual(argv[3:], args)
        self.assertEqual(env["THESIS_DATASET"], "/resident/data.h5")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        argv, env = command(config, Path("/selected/experiment"), args+["--evaluate"])
        self.assertEqual(argv[0], "/shared/eval/python")
        self.assertEqual(env["THESIS_DATASET"], "/shared/contract.h5")


if __name__ == "__main__":
    unittest.main()
