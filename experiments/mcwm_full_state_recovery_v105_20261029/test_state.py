"""Actual MappingTrainer updates on a tiny synthetic replay; zero environment steps."""
from dataclasses import asdict
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
import torch

from state import compare_states, export_state, load_final, save_final
from test_model_state import build_fixture, load_original, Costs


def make_learner(folder):
    implementation = load_original()
    base = implementation.base
    model_class = implementation.previous.previous.ContextModel
    aug = build_fixture(model_class)
    p = dict(aug.protocol, relative_out=str(folder), context_window_physical_rows=8,
             model_batch_size=8, trust_update_every=1, qdelta_updates_per_extra_step=1)
    aug.protocol = aug.p = p
    n = 32
    states = aug.fit["s"].cpu().numpy().repeat(4, axis=0)
    next_states = aug.fit_ns.cpu().numpy().repeat(4, axis=0)
    path = Path(folder) / "synthetic.h5"
    from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION
    data = dict(observations=states, next_observations=next_states,
        actions=np.linspace(-.8, .8, n, dtype=np.float32)[:, None],
        rewards=-np.arange(1, n+1, dtype=np.float32),
        discounts=np.full(n, .99, np.float32), terminals=np.zeros(n, np.float32),
        delta_t_seconds=np.full(n, 30., np.float32),
        split_id=np.zeros(n, np.int8), policy_id=np.zeros(n, np.int8),
        policy_seed=np.zeros(n, np.int32), sumo_seed=np.arange(n, dtype=np.int32)//4,
        episode_id=np.arange(n, dtype=np.int64)//4,
        trajectory_id=np.arange(n, dtype=np.int64)//4)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema_version=SCHEMA_VERSION,
            config=json.dumps(asdict(base.TransitMDPConfigV7())),
            station_max_by_line=json.dumps([4.]*12), policy_names=json.dumps(["synthetic"]))
        for key, value in data.items():
            handle.create_dataset(key, data=value)
    replay = base.TransitReplayBufferV7(path, device="cpu", online_capacity=32)
    for i in range(13):
        replay.append(states[i], data["actions"][i], float(data["rewards"][i]),
            next_states[i], .99, terminal=float(i == 5), episode_id=i//6,
            trajectory_id=i//6, delta_t_seconds=30.)
    config = base.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=8,
        target_subset_size=2, use_ratio_correction=False, reward_scale=1e-4,
        offline_objective="iql", online_backup="h2oplus", mixed_backup_lambda=.1,
        online_update_order="actor-before-critic")
    learner = implementation.MappingTrainer(replay, config=config, device="cpu", seed=6201)
    learner.configure_context(p, "calibrated", None)
    learner.augmentation, learner.costs = aug, Costs()
    learner.model_fraction, learner.model_seed = .25, 66201
    learner.observe_completed()
    # These are fixture counters, not a claim that 69999 training updates ran.
    learner.total_updates, learner.online_critic_updates = 69999, 19999
    learner.coefficient_count, learner.coefficient_sum = 19999, 4999.75
    learner.coefficient_min = learner.coefficient_max = .25
    for group in learner.actor_optimizer.param_groups:
        group["lr"] *= .25
    learner.update(online=True, collect_metrics=False)
    return learner, model_class


class StateTests(unittest.TestCase):
    def test_complete_roundtrip_and_next_real_update_are_exact(self):
        with TemporaryDirectory() as folder:
            learner, model_class = make_learner(folder)
            checkpoint = Path(folder) / "final.pt"
            before = export_state(learner)
            save_final(learner, checkpoint, extra={"synthetic_test": True})
            learner.update(online=True, collect_metrics=False)
            expected = export_state(learner)
            self.assertTrue(learner.actor_optimizer.state)
            self.assertTrue(learner.critic_optimizer.state)
            self.assertTrue(learner.value_optimizer.state)
            self.assertTrue(learner.alpha_optimizer.state)
            self.assertTrue(learner.augmentation.trust.optimizer.state)
            random.random(); np.random.random(); torch.rand(3)
            extra = load_final(learner, checkpoint, model_class=model_class, out=folder)
            self.assertEqual(extra, {"synthetic_test": True})
            result = compare_states(before, export_state(learner))
            self.assertTrue(result["all_pass"], result)
            self.assertEqual(learner.actor_optimizer.param_groups[0]["lr"], 7.5e-5)
            self.assertEqual(learner.blocks.cursor, 13)
            self.assertEqual(len(learner.blocks.pending), 7)
            learner.update(online=True, collect_metrics=False)
            result = compare_states(expected, export_state(learner))
            self.assertTrue(result["all_pass"], result)
            self.assertEqual(result["max_abs_difference"], 0.)
            self.assertEqual(learner.total_updates, 70001)

    def test_endpoint_and_offline_binding_fail_before_continuation(self):
        with TemporaryDirectory() as folder:
            learner, model_class = make_learner(folder)
            checkpoint = Path(folder) / "final.pt"
            learner.total_updates = 69999
            with self.assertRaisesRegex(ValueError, "50000"):
                save_final(learner, checkpoint)
            learner.total_updates = 70000
            save_final(learner, checkpoint)
            learner.replay_buffer.dataset_path = Path(folder) / "different.h5"
            with self.assertRaisesRegex(ValueError, "replay binding"):
                load_final(learner, checkpoint, model_class=model_class)


if __name__ == "__main__":
    unittest.main()
