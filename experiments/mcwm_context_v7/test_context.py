"""Check causal blocks, belief updates, selection separation and scheduling."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from context_common import HERE, P, FIXED, CONTEXTS, cases, evaluation_protocol, module
from context import ContextFilter, CompletedBlocks, fit_recognizer
import selection
launch = module("context_test_launcher", HERE / "launch.py")


class ContextTests(unittest.TestCase):
    def test_filter_identifies_distinct_completed_evidence(self):
        labeled = {c: [np.full(8, i), np.full(8, i+.01)] for i, c in enumerate(CONTEXTS)}
        model = fit_recognizer(labeled, CONTEXTS, .05)
        belief = ContextFilter(model, [0., .125, .25], P)
        for _ in range(4):
            belief.observe(np.full(8, 2.))
        self.assertGreater(belief.belief[2], .99)
        self.assertGreater(belief.coefficient, .24)

    def test_nonphysical_rows_and_reset_do_not_cross_blocks(self):
        replay = SimpleNamespace(online_size=5, _online_total_appended=5,
            _online_observations=np.zeros((5,229)), _online_actions=np.zeros((5,1)),
            _online_delta_t_seconds=np.array([1.,0.,1.,1.,1.]), _online_terminals=np.array([0,0,0,1,0]))
        replay._online_observations[-1, 3] = 1000.
        stream = CompletedBlocks(2)
        reset = []
        stream.consume(replay, reset=lambda: reset.append(True))
        self.assertEqual(len(stream.blocks), 1)
        self.assertEqual(stream.blocks[0]["end_source_row"], 3)
        self.assertEqual(stream.blocks[0]["features"][5], 0.)
        self.assertEqual(len(reset), 1)
        stream.consume(replay)
        self.assertEqual(len(stream.blocks), 1)

    def test_final_cases_and_adaptation_gates(self):
        items = launch.base.placements()
        self.assertEqual(len(items), 30)
        self.assertEqual(sum(len(cases(i["arm"])) for i in items), 480)
        for item in launch.base.placements(True):
            row = launch.task(item)
            self.assertEqual(row["vram"], 0)
            self.assertTrue(row["allow_cpu_training"])
            self.assertEqual(row["allowed_nodes"], launch.infra.CPUS)
        for item in items:
            row = launch.task(item)
            self.assertNotIn("require_node", row)
            self.assertNotIn("jtl311linux", row["allowed_nodes"])
            self.assertEqual("wait_for_files" in row, item["arm"] not in FIXED)
        protocol, case = evaluation_protocol(P, 15)
        self.assertEqual(case, 5)
        self.assertEqual(protocol["evaluation_blocks"], P["test_blocks"])
        self.assertTrue(set(b["demand_seed"] for b in P["test_blocks"]).isdisjoint(b["demand_seed"] for b in P["development_blocks"]))

    def test_selection_never_reads_test_days(self):
        with TemporaryDirectory() as directory, patch.object(selection, "HERE", Path(directory)):
            for s in CONTEXTS:
                for a, coefficient in FIXED.items():
                    folder = Path(directory)/"runs"/s/f"{a}_seed6202"/"results"
                    folder.mkdir(parents=True)
                    (folder/"ready.json").write_text(json.dumps(dict(arm=a,seed=6202,scenario=s)))
                    (folder/"context_blocks.json").write_text(json.dumps(dict(blocks=[dict(features=[1.]*8)])))
                    for c in range(10):
                        row = dict(case=c,split="development",arm=a,scenario=s,training_seed=6202,
                            horizon_seconds=24000,demand_scale=P["evaluation_demand_scales"][c//5],
                            generalized_seconds_per_departed_passenger=coefficient+100,
                            **P["development_blocks"][c%5])
                        (folder/f"eval_{c:02d}.json").write_text(json.dumps(row))
                    (folder/"eval_10.json").write_text("invalid test data must not be opened")
            bundle = selection.make_bundle(6202, 6201)
            self.assertEqual(set(bundle["coefficients"].values()), {0.})
            self.assertEqual(bundle["provider_seed"], 6202)
            self.assertFalse(bundle["test_outcomes_used"])
            self.assertFalse(bundle["mapping_has_context_effect"])


if __name__ == "__main__":
    unittest.main()
