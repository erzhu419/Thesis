"""Exercise the actual terminal callback with synthetic learner/reference artifacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from state import compare_states, export_state
from test_state import make_learner
import verify


def callback_fixture(folder):
    p = json.loads((verify.HERE / "protocol.json").read_text())
    root = Path(folder)
    out = root / "out"
    out.mkdir()
    learner, _ = make_learner(out)
    args = SimpleNamespace(out=out, thesis_root=root / "thesis",
        seed=p["training_seed"], initializer=root / "synthetic_initializer.pt")
    reference = args.thesis_root / p["reference_actor"]
    learner.save_policy_artifact(reference, extra=dict(
        training_seed=args.seed, updates=70000, synthetic_test=True))
    # Declared identity metadata only; no historical training/collection is run.
    training = dict(continuation_updates=20000, source_events=p["expected_source_events"],
        demand_scale_episode_counts=p["expected_source_episodes"],
        model_td_coefficient=.25, real_td_coefficient=.75,
        target_online_training_events=0, synthetic_test=True)
    return learner, args, training, reference


class VerifyTests(unittest.TestCase):
    def test_callback_exports_verifies_and_restores_original_endpoint(self):
        with TemporaryDirectory() as folder:
            learner, args, training, _ = callback_fixture(folder)
            endpoint = export_state(learner)
            with patch.object(learner, "update", wraps=learner.update) as updates:
                result = verify.finish_recovery(learner, args, training)
            self.assertEqual(updates.call_count, 2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["decision"], "ONE_HISTORICAL_ENDPOINT_RECOVERED")
            self.assertTrue(all(result["identity"].values()))
            self.assertTrue(result["restoration"]["all_pass"])
            self.assertTrue(result["next_update"]["all_pass"])
            self.assertEqual(result["budget"]["diagnostic_updates"], 2)
            self.assertEqual(result["budget"]["diagnostic_attempts"], 2)
            self.assertEqual(result["budget"]["target_native_seconds"], 0)
            self.assertEqual(result["restored_terminal_updates"], 70000)
            self.assertEqual(result["terminal_updates_at_exit"], 70000)
            self.assertFalse(result["next_stage_authorized"])
            summary = json.loads((args.out / "recovery_summary.json").read_text())
            self.assertEqual(summary, result)
            saved = torch.load(args.out / "full_state.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["total_updates"], 70000)
            self.assertEqual(saved["online_critic_updates"], 20000)
            self.assertGreater(result["checkpoint_bytes"], 0)
            comparison = compare_states(endpoint, export_state(learner))
            self.assertTrue(comparison["all_pass"], comparison)

    def test_changed_reference_preserves_snapshot_and_runs_no_diagnostic_update(self):
        with TemporaryDirectory() as folder:
            learner, args, training, reference = callback_fixture(folder)
            old = torch.load(reference, map_location="cpu", weights_only=False)
            first = next(iter(old["actor"].values()))
            first.view(-1)[0] += .125
            torch.save(old, reference)
            endpoint = export_state(learner)
            with patch.object(learner, "update", wraps=learner.update) as updates:
                result = verify.finish_recovery(learner, args, training)
            self.assertEqual(updates.call_count, 0)
            self.assertEqual(result["status"], "verification_failed")
            self.assertEqual(result["decision"], "HOLD_IDENTITY_MISMATCH_NO_EXPANSION")
            self.assertFalse(result["identity"]["actor_exact"])
            self.assertEqual(result["actor_identity"]["differing_field_count"], 1)
            self.assertEqual(result["budget"]["diagnostic_updates"], 0)
            self.assertEqual(result["budget"]["diagnostic_attempts"], 0)
            self.assertEqual(result["restoration"], {})
            self.assertEqual(result["next_update"], {})
            self.assertEqual(result["terminal_updates_at_exit"], 70000)
            self.assertEqual(json.loads((args.out / "recovery_summary.json").read_text()), result)
            saved = torch.load(args.out / "full_state.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["total_updates"], 70000)
            self.assertTrue(compare_states(saved["actor"], learner.actor.state_dict())["all_pass"])
            comparison = compare_states(endpoint, export_state(learner))
            self.assertTrue(comparison["all_pass"], comparison)


if __name__ == "__main__":
    unittest.main()
