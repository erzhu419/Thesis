"""Actual MC-WM update on synthetic rows; no real labels or native steps."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from learning import TargetReplayAdapter, install_mechanism, mechanism_summary

sys.path.insert(0, str(ROOT / "experiments/mcwm_full_state_recovery_v105_20261029"))
from test_model_state import load_original
from test_state import make_learner

IMPLEMENTATION = load_original()
from experiments.mcwm_full_mcwm_dripping_pilot_v107_20261029.test_learning import replay_fixture

P = dict(fresh_pool_capacity=285, fresh_rows=64, private_seed=606201,
         feedback_records_per_packet=5)


def fresh_packet(learner):
    rows = []
    for index in range(5):
        s = learner.augmentation.fit["s"][index].cpu().numpy().copy()
        s[0] = .6 + index * .01
        ns = s.copy()
        ns[0] += (5 + index) / 24000
        rows.append(SimpleNamespace(s=s, a=np.array([.2 + index * .1], np.float32),
            ns=ns, r=-50. - index, dt=5. + index, terminal=False))
    return rows


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer_steps = 0
        original_step = torch.optim.Adam.step

        def counted(optimizer, *args, **kwargs):
            cls.optimizer_steps += 1
            return original_step(optimizer, *args, **kwargs)

        cls.counter = patch.object(torch.optim.Adam, "step", counted)
        cls.counter.start()

    @classmethod
    def tearDownClass(cls):
        cls.counter.stop()
        print(json.dumps(dict(synthetic_optimizer_steps=cls.optimizer_steps,
            real_data_optimizer_steps=0, native_seconds=0)))

    def assert_batch_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)

    def test_actual_update_keeps_v107_critic_batch_and_original_auxiliary_inputs(self):
        with TemporaryDirectory() as folder:
            learner, _ = make_learner(folder)
            learner.config = replace(learner.config, batch_size=256)
            learner.protocol.update(model_batch_size=128, trust_update_every=4)
            contract = IMPLEMENTATION.base.TransitMDPConfigV7(**learner.replay_buffer.contract_config)
            raw, reference = deepcopy(learner.replay_buffer), deepcopy(learner.replay_buffer)
            expected_original, expected_source = raw.sample_mixed(256, .5, rng=deepcopy(learner.rng))
            rows = fresh_packet(learner)
            prior = TargetReplayAdapter(reference, "fresh", P, contract)
            prior.add_target(rows)
            expected_critic, expected_critic_source = reference.sample_mixed(256, .5, rng=deepcopy(learner.rng))
            adapter = TargetReplayAdapter(learner.replay_buffer, "fresh", P, contract)
            adapter.add_target(rows)
            install_mechanism(learner, adapter)

            targets, value_inputs, actor_inputs = [], [], []
            target = learner._target

            def record_target(batch, *, backup):
                if len(batch["observations"]) == 256:
                    targets.append({key: batch[key].clone() for key in expected_critic})
                return target(batch, backup=backup)

            learner._target = record_target
            handle = learner.target_critic.register_forward_pre_hook(
                lambda module, args: value_inputs.append((args[0].clone(), args[1].clone()))
                if len(args[0]) == 128 else None)
            update_actor = learner._update_sac_actor

            def record_actor(states, **kwargs):
                actor_inputs.append((states.clone(), kwargs["temperature_states"].clone()))
                return update_actor(states, **kwargs)

            learner._update_sac_actor = record_actor
            try:
                learner.update(online=True, collect_metrics=False)
            finally:
                handle.remove()

            self.assertEqual(len(targets), 1)
            self.assert_batch_equal(targets[0], {key: torch.cat((expected_critic[key], expected_critic_source[key]))
                                                  for key in expected_critic})
            self.assertEqual(len(value_inputs), 1)
            torch.testing.assert_close(value_inputs[0][0], expected_original["observations"], rtol=0, atol=0)
            torch.testing.assert_close(value_inputs[0][1], expected_original["actions"], rtol=0, atol=0)
            self.assertEqual(len(actor_inputs), 1)
            expected_states = torch.cat((expected_original["observations"], expected_source["observations"]))
            for states in actor_inputs[0]:
                torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
            mechanism = adapter._original_auxiliary_batch
            self.assert_batch_equal(mechanism.offline, expected_original)
            self.assertTrue(all(mechanism.offline[key].data_ptr() != mechanism.critic_offline[key].data_ptr()
                                for key in mechanism.offline))
            evidence = mechanism_summary(learner, adapter)
            self.assertEqual(evidence, adapter.summary()["mechanism"])
            self.assertEqual(evidence["auxiliary_captures"], 1)
            self.assertEqual(evidence["original_batch_used"], dict(value=1, actor=1, temperature=1))
            self.assertEqual([evidence[key] for key in ("value_calls", "actor_calls", "temperature_calls")], [1, 1, 1])
            self.assertEqual(evidence["critic_complete_replacements"], 64)
            self.assertEqual(learner.total_updates, 70001)

    def test_no_fresh_keeps_v107_sampler_and_rng_without_learning_hooks(self):
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7

        prior_replay, replay = replay_fixture(), replay_fixture()
        prior = TargetReplayAdapter(prior_replay, "no_fresh", P, TransitMDPConfigV7())
        adapter = TargetReplayAdapter(replay, "no_fresh", P, TransitMDPConfigV7())
        learner = IMPLEMENTATION.MappingTrainer.__new__(IMPLEMENTATION.MappingTrainer)
        original_sampler = adapter._original_sample_mixed
        install_mechanism(learner, adapter)
        self.assertIs(adapter._original_sample_mixed, original_sampler)
        for name in ("_iql_value_terms", "_actor_states", "_temperature_states"):
            self.assertNotIn(name, learner.__dict__)
        prior.add_target([object() for _ in range(5)])
        adapter.add_target([object() for _ in range(5)])
        old_rng, new_rng = np.random.default_rng(91), np.random.default_rng(91)
        for _ in range(3):
            old = prior_replay.sample_mixed(256, .5, rng=old_rng)
            new = replay.sample_mixed(256, .5, rng=new_rng)
            for expected, actual in zip(old, new):
                self.assert_batch_equal(expected, actual)
            self.assertEqual(old_rng.bit_generator.state, new_rng.bit_generator.state)
            self.assertEqual(prior.private_rng.bit_generator.state, adapter.private_rng.bit_generator.state)
        evidence = mechanism_summary(learner, adapter)
        self.assertEqual(evidence["mode"], "original_no_fresh")
        self.assertEqual(evidence["auxiliary_captures"], 0)
        self.assertEqual(evidence["original_batch_used"], dict(value=0, actor=0, temperature=0))
        self.assertEqual(evidence["critic_complete_replacements"], 0)


if __name__ == "__main__":
    unittest.main()
