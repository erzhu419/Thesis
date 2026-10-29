"""Synthetic production-method tests; no real labels or native calls."""

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
from learning import TargetReplayAdapter, install_mechanism, mechanism_summary, original

sys.path.insert(0, str(ROOT / "experiments/mcwm_full_state_recovery_v105_20261029"))
from test_model_state import load_original
from test_state import make_learner

IMPLEMENTATION = load_original()
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer as Base
from experiments.mcwm_full_mcwm_dripping_pilot_v107_20261029.test_learning import replay_fixture

P = dict(fresh_pool_capacity=285, fresh_rows=64, private_seed=606201,
         feedback_records_per_packet=5)


def packet(learner):
    rows = []
    for index in range(5):
        s = learner.augmentation.fit["s"][index].cpu().numpy().copy()
        s[0] = .6 + index*.01
        ns = s.copy()
        ns[0] += (5+index)/24000
        rows.append(SimpleNamespace(s=s, a=np.array([.2+index*.1], np.float32),
            ns=ns, r=-50.-index, dt=5.+index, terminal=False))
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

    def test_production_gradient_rng_and_auxiliary_data(self):
        with TemporaryDirectory() as folder:
            learner, _ = make_learner(folder)
            learner.config = replace(learner.config, batch_size=256,
                bootstrap_probability=1., critic_loss="huber")
            learner.protocol.update(model_batch_size=128, trust_update_every=4)
            baseline = deepcopy(learner)
            contract = IMPLEMENTATION.base.TransitMDPConfigV7(**learner.replay_buffer.contract_config)
            expected_offline, expected_source = deepcopy(learner.replay_buffer).sample_mixed(
                256, .5, rng=deepcopy(learner.rng))
            baseline_replay = TargetReplayAdapter(baseline.replay_buffer, "fresh", P, contract)
            baseline_replay.add_target(packet(baseline))
            original.install_mechanism(baseline, baseline_replay)
            baseline_targets = []
            baseline_target = baseline._target

            def record_baseline(batch, *, backup):
                result = baseline_target(batch, backup=backup)
                if len(result) == 256:
                    baseline_targets.append(result.clone())
                return result

            baseline._target = record_baseline
            rng_before = torch.get_rng_state()
            baseline.update(online=True, collect_metrics=False)
            baseline_rng = torch.get_rng_state()

            replay = TargetReplayAdapter(learner.replay_buffer, "fresh", P, contract)
            replay.add_target(packet(learner))
            targets, real_losses, expected_gradients, actual_gradients = [], [], [], []
            target, td_loss = learner._target, learner._ensemble_td_loss

            def record_target(batch, *, backup):
                result = target(batch, backup=backup)
                if len(result) == 256:
                    targets.append(result.clone())
                return result

            def record_loss(predictions, target_values, weights=None):
                real_losses.append((predictions, target_values, weights))
                result = td_loss(predictions, target_values, weights)
                if len(real_losses) == 2:
                    auxiliary = replay._original_auxiliary_batch
                    historical, fresh = auxiliary.offline, auxiliary.critic_offline
                    historical_q = learner.critic(historical["observations"], historical["actions"])
                    fresh_q = learner.critic(fresh["observations"][64:], fresh["actions"][64:])
                    historical_loss = Base._ensemble_td_loss(learner, historical_q, targets[1][:128])
                    fresh_loss = Base._ensemble_td_loss(learner, fresh_q, targets[0][64:128])
                    source_loss = Base._ensemble_td_loss(learner, predictions, target_values)
                    expected = (.1875*historical_loss + .1875*fresh_loss
                                + .375*source_loss + .25*learner.model_loss)
                    expected_gradients.extend(torch.autograd.grad(expected,
                        tuple(learner.critic.parameters()), retain_graph=True))
                return result

            learner._target, learner._ensemble_td_loss = record_target, record_loss
            install_mechanism(learner, replay)
            value_inputs, actor_inputs = [], []
            handle = learner.target_critic.register_forward_pre_hook(
                lambda module, args: value_inputs.append((args[0].clone(), args[1].clone()))
                if len(args[0]) == 128 else None)
            update_actor = learner._update_sac_actor

            def record_actor(states, **kwargs):
                actor_inputs.append((states.clone(), kwargs["temperature_states"].clone()))
                return update_actor(states, **kwargs)

            learner._update_sac_actor = record_actor
            critic_step = learner.critic_optimizer.step

            def record_step(*args, **kwargs):
                actual_gradients.extend(p.grad.clone() for p in learner.critic.parameters())
                return critic_step(*args, **kwargs)

            learner.critic_optimizer.step = record_step
            torch.set_rng_state(rng_before)
            try:
                learner.update(online=True, collect_metrics=False)
            finally:
                handle.remove()
            self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng))
            self.assertEqual(learner.rng.bit_generator.state, baseline.rng.bit_generator.state)
            self.assertEqual(replay.private_rng.bit_generator.state, baseline_replay.private_rng.bit_generator.state)
            self.assertEqual(len(targets), 2)
            torch.testing.assert_close(targets[0], baseline_targets[0], rtol=0, atol=0)
            self.assertTrue(expected_gradients)
            for actual, expected in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(real_losses[0][0].shape[1], 192)
            torch.testing.assert_close(real_losses[0][2], torch.cat((torch.full((128,), .75),
                                                                   torch.full((64,), 1.5))))
            self.assertEqual(len(value_inputs), 1)
            torch.testing.assert_close(value_inputs[0][0], expected_offline["observations"], rtol=0, atol=0)
            torch.testing.assert_close(value_inputs[0][1], expected_offline["actions"], rtol=0, atol=0)
            self.assertEqual(len(actor_inputs), 1)
            expected_states = torch.cat((expected_offline["observations"], expected_source["observations"]))
            for states in actor_inputs[0]:
                torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
            evidence = mechanism_summary(learner, replay)
            self.assertEqual(evidence, replay.summary()["mechanism"])
            self.assertEqual(evidence["history_coverage_updates"], 1)
            self.assertEqual(evidence["extra_critic_rows"], 64)
            self.assertEqual(evidence["extra_bootstrap_rows"], 256)
            self.assertEqual(evidence["extra_bootstrap_calls"], 1)
            self.assertTrue(evidence["main_rng_preserved"] and evidence["common_targets_exact"])
            self.assertEqual(evidence["original_batch_used"], dict(value=1, actor=1, temperature=1))

    def test_no_fresh_keeps_original_methods_batches_and_rng(self):
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        old_replay, new_replay = replay_fixture(), replay_fixture()
        old = TargetReplayAdapter(old_replay, "no_fresh", P, TransitMDPConfigV7())
        new = TargetReplayAdapter(new_replay, "no_fresh", P, TransitMDPConfigV7())
        learner = IMPLEMENTATION.MappingTrainer.__new__(IMPLEMENTATION.MappingTrainer)
        install_mechanism(learner, new)
        for name in ("_target", "_ensemble_td_loss", "_iql_value_terms", "_actor_states", "_temperature_states"):
            self.assertNotIn(name, learner.__dict__)
        old.add_target([object() for _ in range(5)])
        new.add_target([object() for _ in range(5)])
        old_rng, new_rng = np.random.default_rng(91), np.random.default_rng(91)
        for _ in range(3):
            before = torch.get_rng_state()
            prior, current = old_replay.sample_mixed(256, .5, rng=old_rng), new_replay.sample_mixed(256, .5, rng=new_rng)
            for left, right in zip(prior, current):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(old_rng.bit_generator.state, new_rng.bit_generator.state)
            self.assertEqual(old.private_rng.bit_generator.state, new.private_rng.bit_generator.state)
        evidence = mechanism_summary(learner, new)
        self.assertEqual(evidence["mode"], "original_no_fresh")
        for name in ("history_coverage_updates", "history_critic_rows_per_update", "extra_critic_rows",
                     "extra_bootstrap_rows", "extra_bootstrap_calls", "auxiliary_captures"):
            self.assertEqual(evidence[name], 0)


if __name__ == "__main__":
    unittest.main()
