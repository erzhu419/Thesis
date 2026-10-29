from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_cost_advantage_v85_20261027 import native


@dataclass
class FixtureMDP:
    max_hold_seconds: float = 60.
    scheduled_headway_scale_seconds: float = 600.
    horizon_seconds: float = 24000.


class FixtureActor(nn.Module):
    def __init__(self, hidden_dims, *, layer_norm, activation):
        super().__init__()
        self.architecture = (hidden_dims, layer_norm, activation)
        self.linear = nn.Linear(229, 1)
        self.dropout = nn.Dropout(.5)
        self.last_deterministic = None

    def forward(self, states, *, deterministic=False):
        self.last_deterministic = deterministic
        value = self.linear(self.dropout(states))
        if not deterministic:
            value = value+torch.randn_like(value)
        return torch.tanh(value), value.new_zeros(len(states))


class NativeReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.v84, self.v75, self.inputs = (self.root/name for name in ('v84', 'v75', 'inputs'))
        self.inputs.mkdir()
        (self.inputs/'history_metadata.json').write_text(json.dumps(dict(
            contract=dict(max_hold_seconds=60., scheduled_headway_scale_seconds=600., horizon_seconds=24000.),
            station_max_by_line=[35, 40, 29])))
        self.source = dict(source='v75', source_arm='fresh', seed=6207, scenario='local_speed_drop',
            packets=37, covariates=185, complete_outcomes=185, counts=dict(native.COUNTS),
            data_deadline=12000, max_available_at=12000., published_clock=12001.,
            future_training_used=False)
        self.saved_actor = FixtureActor((8, 8), layer_norm=False, activation='relu')
        with torch.no_grad():
            self.saved_actor.linear.weight.fill_(.001)
            self.saved_actor.linear.bias.fill_(-.1)
        destination = self.v84/'runs/local_speed_drop/reference_actor.pt'
        destination.parent.mkdir(parents=True)
        torch.save(dict(metadata=self.source, actor=self.saved_actor.state_dict()), destination)
        config = self.v75/'runs/executed_cql/validation/seed6207/local_speed_drop/fresh/protocol.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps(dict(policy_config=dict(hidden_dim=8, actor_layer_norm=False, activation='relu'))))
        self.patches = [patch.object(native, 'V84', self.v84), patch.object(native, 'V75', self.v75),
            patch.object(native, '_runtime', return_value=(FixtureActor, FixtureMDP, ExecutionAction, self.inputs))]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_actor_only_source_clock_belongs_to_prior_episode(self):
        actual_load = torch.load
        with patch.object(native.np, 'load', side_effect=AssertionError('Replay files must not be opened')):
            with patch.object(native.torch, 'load', wraps=actual_load) as actor_load:
                controller, mdp, stations, metadata = native.load_reference('local_speed_drop')
        self.assertEqual(actor_load.call_count, 1)
        self.assertEqual(actor_load.call_args.args[0], self.v84/'runs/local_speed_drop/reference_actor.pt')
        self.assertEqual(controller.actor.architecture, ((8, 8), False, 'relu'))
        self.assertFalse(controller.actor.training)
        self.assertTrue(all(not value.requires_grad for value in controller.actor.parameters()))
        self.assertEqual(metadata['source_snapshot_metadata'], self.source)
        self.assertEqual(metadata['source_snapshot_metadata']['published_clock'], 12001)
        self.assertEqual(metadata['heldout_available_at'], 0)
        self.assertTrue(metadata['available_before_heldout_episode'])
        self.assertEqual(metadata['source_total_updates'], 50296)
        self.assertEqual(metadata['new_optimizer_updates'], 0)
        self.assertEqual(metadata['heldout_external_seed'], 216207)
        self.assertEqual(stations.dtype, np.float32)
        self.assertEqual(mdp.horizon_seconds, 24000)
        self.assertFalse(hasattr(controller, 'replay_buffer'))
        self.assertFalse(hasattr(controller, 'actor_optimizer'))

    def test_loading_and_policy_do_not_consume_rng_or_change_parameters(self):
        torch.manual_seed(731)
        random.seed(19)
        np.random.seed(23)
        before_torch = torch.get_rng_state().clone()
        before_python, before_numpy = random.getstate(), deepcopy(np.random.get_state())
        controller, _, _, _ = native.load_reference('local_speed_drop')
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        state = np.zeros(229, np.float32)
        state[0] = .5
        before = controller.snapshot_state()
        first = controller(state)
        second = controller(state)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(controller.actor.last_deterministic)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertTrue(all(torch.equal(value, controller.snapshot_state()[key]) for key,value in before.items()))
        expected = controller.action_transform.numpy(state, controller.last_proposal)
        self.assertTrue(np.array_equal(first, expected))

    def test_snapshot_is_independent_and_wrong_scenario_is_rejected(self):
        controller, _, _, _ = native.load_reference('local_speed_drop')
        snapshot = controller.snapshot_state()
        snapshot['linear.weight'].zero_()
        self.assertFalse(torch.equal(snapshot['linear.weight'], controller.actor.linear.weight))
        payload = self.v84/'runs/local_speed_drop/reference_actor.pt'
        value = torch.load(payload, map_location='cpu', weights_only=False)
        value['metadata']['scenario'] = 'stable'
        torch.save(value, payload)
        with self.assertRaisesRegex(ValueError, 'matching completed v84'):
            native.load_reference('local_speed_drop')


if __name__ == '__main__':
    unittest.main()
