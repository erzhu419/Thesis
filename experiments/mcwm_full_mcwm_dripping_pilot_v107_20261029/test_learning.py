"""Exercise production replay sampling with small arrays; no learning or simulation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/mcwm_bus/runtime/H2Oplus"))
from SimpleSAC.transit_v3.replay_buffer import TransitReplayBuffer
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
from learning import TargetReplayAdapter


P = dict(fresh_pool_capacity=285, fresh_rows=64, private_seed=606201,
         feedback_records_per_packet=5)


def replay_fixture():
    replay = TransitReplayBuffer.__new__(TransitReplayBuffer)
    replay.device = torch.device("cpu")
    replay.offline_size, replay._online_size = 400, 300
    replay._online_pointer, replay._online_total_appended = 300, 300
    replay._offline_device_cache = replay._online_device_cache = None
    replay._online_cache_size = -1
    for prefix, size, offset in (("", 400, -1000), ("_online_", 300, -2000)):
        ids = np.arange(size, dtype=np.float32) + offset
        fields = dict(observations=np.repeat(ids[:, None], 229, axis=1),
            actions=ids[:, None] / 10000, rewards=ids * 2,
            next_observations=np.repeat((ids + .5)[:, None], 229, axis=1),
            discounts=np.ones(size, np.float32), terminals=np.zeros(size, np.float32),
            delta_t_seconds=np.full(size, 3, np.float32))
        for key, value in fields.items():
            setattr(replay, prefix + key, value)
    return replay


def packet(start):
    return [SimpleNamespace(s=np.full(229, i, np.float32),
        a=np.asarray([i / 10000], np.float32), r=-2.0 * i,
        ns=np.full(229, i + .5, np.float32), dt=i / 10,
        terminal=bool(i % 2)) for i in range(start, start + 5)]


class LearningTests(unittest.TestCase):
    def assert_batch_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_no_fresh_is_original_batch_and_rng_without_reading_outcomes(self):
        original, adapted = replay_fixture(), replay_fixture()
        adapter = TargetReplayAdapter(adapted, "no_fresh", P, TransitMDPConfigV7())
        # These objects have no transition fields: no_fresh must not inspect them.
        adapter.add_target([object() for _ in range(5)])
        expected_rng, actual_rng = np.random.default_rng(91), np.random.default_rng(91)
        for _ in range(3):
            expected = original.sample_mixed(256, .5, rng=expected_rng)
            actual = adapted.sample_mixed(256, .5, rng=actual_rng)
            for left, right in zip(expected, actual):
                self.assert_batch_equal(left, right)
            self.assertEqual(expected_rng.bit_generator.state, actual_rng.bit_generator.state)
        self.assertEqual(adapter.summary()["target_outcome_labels_received"], 0)
        self.assertEqual(adapter.target_rows, [])
        self.assertEqual(adapter.summary()["replaced_rows"], 0)
        self.assertEqual(adapter.summary()["private_draws"], 192)
        self.assertTrue(adapter.summary()["main_rng_unchanged"])
        self.assertTrue(adapter.summary()["source_unchanged"])

    def test_fresh_replaces_only_last64_with_complete_consistent_rows(self):
        original, adapted = replay_fixture(), replay_fixture()
        adapter = TargetReplayAdapter(adapted, "fresh", P, TransitMDPConfigV7())
        rows = packet(1000)
        adapter.add_target(rows)
        expected_rng, actual_rng = np.random.default_rng(92), np.random.default_rng(92)
        expected_offline, expected_source = original.sample_mixed(256, .5, rng=expected_rng)
        offline, source = adapted.sample_mixed(256, .5, rng=actual_rng)
        self.assert_batch_equal(source, expected_source)
        for key in offline:
            self.assertTrue(torch.equal(offline[key][:64], expected_offline[key][:64]), key)
        selected = np.random.default_rng(P["private_seed"]).integers(0, 5, size=64)
        for position, index in enumerate(selected, start=64):
            row = rows[index]
            expected = dict(observations=row.s, actions=row.a, rewards=row.r,
                next_observations=row.ns, delta_t_seconds=row.dt,
                terminals=float(row.terminal), discounts=0.0 if row.terminal else 1.0)
            for key, value in expected.items():
                self.assertTrue(torch.equal(offline[key][position],
                    torch.as_tensor(value, dtype=offline[key].dtype)), key)
        self.assertEqual(expected_rng.bit_generator.state, actual_rng.bit_generator.state)
        self.assertEqual(adapter.summary()["replaced_rows"], 64)
        self.assertTrue(adapter.summary()["main_rng_unchanged"])
        self.assertTrue(adapter.summary()["source_unchanged"])

    def test_bounded_latest_pool_and_private_rng_match_across_arms(self):
        fresh_replay, nofresh_replay = replay_fixture(), replay_fixture()
        fresh = TargetReplayAdapter(fresh_replay, "fresh", P, TransitMDPConfigV7())
        nofresh = TargetReplayAdapter(nofresh_replay, "no_fresh", P, TransitMDPConfigV7())
        for start in range(0, 300, 5):
            fresh.add_target(packet(start))
            nofresh.add_target([object() for _ in range(5)])
            fresh_replay.sample_mixed(256, .5, rng=np.random.default_rng(start))
            nofresh_replay.sample_mixed(256, .5, rng=np.random.default_rng(start))
            self.assertLessEqual(len(fresh.target_rows), 285)
            self.assertEqual(fresh.private_rng.bit_generator.state,
                             nofresh.private_rng.bit_generator.state)
        self.assertEqual(len(fresh.target_rows), 285)
        self.assertEqual(float(fresh.target_rows[0]["observations"][0]), 15)
        self.assertEqual(float(fresh.target_rows[-1]["observations"][0]), 299)
        self.assertEqual(fresh.summary()["target_outcome_labels_received"], 300)
        self.assertEqual(nofresh.summary()["target_outcome_labels_received"], 0)
        self.assertEqual(nofresh.target_rows, [])


if __name__ == "__main__":
    unittest.main()
