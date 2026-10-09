"""Check action authority, same-vehicle labels and both learner dimensions."""

import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import h5py
import torch

from station_common import P, physical_action, neutral_action, observation, state_headway_reward
from environments import controlled_travel_steps, StationSampler
from networks import bindings
from run import StationTrainer, config
from bus_h2o.passenger_time import PassengerTimeSnapshot


class MemoryReplay:
    def __init__(self):
        self.rows = []

    def append(self, *values):
        self.rows.append(values)


class InterleavedEnv:
    sequence = ((0, 1, 100), (1, 1, 110), (0, 2, 180), (1, 2, 190), (0, 3, 260))

    def reset(self):
        self.index, self.done, self.commands = 0, False, []
        self.buses = [SimpleNamespace(bus_id=i, trip_id=1) for i in range(2)]
        self.line_map = {"7X": SimpleNamespace(bus_all=self.buses,
            passenger_time_ledger=SimpleNamespace(snapshot=lambda: self.snapshot()))}
        self.refresh()

    def snapshot(self):
        t = self.sequence[self.index][2]
        return PassengerTimeSnapshot(t, 2, 3, 2*t, 3*t)

    def refresh(self):
        bus, station, time = self.sequence[self.index]
        self.state = {("7X", bus): [[0, station, time, 320+station, 400-station,
            360, 360, 2, 3, .1, 8, 1, 1, 50, 2, 3]]}

    def step_to_event(self, commands):
        if commands:
            self.commands.append(commands)
            self.index += 1
            self.refresh()


class FakeReplay:
    online_size = 32

    def __init__(self, dim):
        states = torch.zeros(32, 15+dim)
        states[:, 1:3] = .1
        states[:, 3:5] = 1.
        states[:, 5:7] = .5
        states[:, 11:13] = 1.
        self.batch = dict(observations=states, actions=torch.zeros(32, dim), next_observations=states+.001,
                          rewards=torch.zeros(32), discounts=torch.full((32,), .99),
                          terminals=torch.zeros(32), delta_t_seconds=torch.full((32,), 30.))

    def sample(self, size, **kwargs):
        return {k:v[np.arange(size) % 32] for k,v in self.batch.items()}

    def sample_mixed(self, size, fraction, **kwargs):
        return self.sample(size), self.sample(size) if fraction else None


class ControlTests(unittest.TestCase):
    def test_block_read_preserves_selected_rows_and_rewards(self):
        from dataset import Replay
        stations = np.full(12, 40, np.float32)
        indices = np.array([0, 2, 31, 128, 280])
        for mode, dim in (("holding", 1), ("dual", 2)):
            n = 300
            columns = {"observations": np.arange(n*(15+dim), dtype=np.float32).reshape(n, 15+dim),
                       "actions": np.arange(n*dim, dtype=np.float32).reshape(n, dim),
                       "next_observations": np.arange(n*(15+dim), dtype=np.float32).reshape(n, 15+dim)+1,
                       "delta_t_seconds": np.arange(n, dtype=np.float32)+1,
                       "role": np.arange(n, dtype=np.float32) % 10,
                       "headway_rewards": -np.arange(n, dtype=np.float32),
                       "passenger_rewards": -3*np.arange(n, dtype=np.float32)}
            with TemporaryDirectory() as directory:
                with h5py.File(Path(directory) / "dataset.h5", "w") as data:
                    data.attrs["stations"] = json.dumps(stations.tolist())
                    for key, value in columns.items():
                        data.create_dataset(key, data=value, chunks=True, compression="lzf")
                for objective in ("headway", "passenger"):
                    with patch("dataset.data_folder", return_value=Path(directory)), \
                            patch("dataset.np.linspace", return_value=indices):
                        replay = Replay(mode, objective)
                    for key in ("observations", "actions", "next_observations", "delta_t_seconds", "role"):
                        np.testing.assert_array_equal(replay.offline[key], columns[key][indices])
                    raw = columns[objective+"_rewards"][indices]
                    np.testing.assert_array_equal(replay.raw_rewards, raw)
                    np.testing.assert_array_equal(replay.offline["rewards"], (raw-raw.mean())*(10/raw.std()))
                    np.testing.assert_array_equal(replay.stations, stations)
                    self.assertEqual(replay.offline_size, len(indices))

    def test_action_and_speed_authority(self):
        self.assertEqual(physical_action([-1., -1.]), [0., .8])
        self.assertEqual(physical_action([1., 1.]), [60., 1.2])
        signals = dict(n_signals=10, cycle=60, green_frac=.5)
        neutral = controlled_travel_steps(100, 4, 1, signals, 20)
        slow = controlled_travel_steps(100, 4, .8, signals, 20)
        fast = controlled_travel_steps(100, 4, 1.2, signals, 20)
        self.assertEqual(neutral, 96)
        self.assertGreater(slow, neutral)
        self.assertLess(fast, neutral)

    def test_interleaved_settlement(self):
        replay, env = MemoryReplay(), InterleavedEnv()
        sampler = StationSampler(env, replay, "dual", np.full(12, 40, np.float32))
        action = np.array([-.5, .8], np.float32)
        sampler.collect(lambda s, deterministic: np.repeat(action[None], len(s), 0), 4)
        self.assertEqual(len(replay.rows), 2)
        for s, a, ns, rh, rp, dt in replay.rows:
            np.testing.assert_array_equal(ns[-2:], a)
            self.assertAlmostEqual(float(ns[1]-s[1]), 1/40, places=6)
            self.assertAlmostEqual(rh, float(state_headway_reward(ns[None])[0]), places=5)
            self.assertAlmostEqual(rp, -7*80.)
            self.assertEqual(dt, 80.)
        np.testing.assert_allclose(env.commands[0][("7X", 0)], [15., 1.16])

    def test_both_dimensions_train(self):
        torch.set_num_threads(1)
        for mode, dim in (("holding", 1), ("dual", 2)):
            learner = StationTrainer(FakeReplay(dim), config=config(mode), seed=1, **bindings(mode))
            learner.seed = 1
            for online in (False, True):
                result = learner.update(online=online)
                self.assertTrue(all(np.isfinite(v) for v in result.values()))
            action = learner.act(learner.replay_buffer.batch["observations"].numpy(), deterministic=True)
            self.assertEqual(action.shape, (32, dim))

    def test_model_and_bellman_update(self):
        from world_model import StationModels
        torch.set_num_threads(1)
        for mode, objective, dim in (("holding", "headway", 1), ("dual", "passenger", 2)):
            replay = FakeReplay(dim)
            replay.offline = {k:np.tile(v.numpy(), (4,)+(1,)*(v.ndim-1)) for k,v in replay.batch.items()}
            replay.offline["role"] = np.arange(128) % 10
            replay.online = {k:v.copy() for k,v in replay.offline.items() if k != "role"}
            replay.offline_size, replay.objective = 128, objective
            replay.mean, replay.std = -20., 5.
            replay.normalize = lambda r: (np.asarray(r)+20.)*2.
            learner = StationTrainer(replay, config=config(mode), seed=2, **bindings(mode))
            learner.seed = 2
            with TemporaryDirectory() as tmp, patch.dict(P, model_updates=8, model_refit_updates=8):
                models = StationModels(replay, 2, Path(tmp))
                models.update_trust(learner.actor)
                batch, weights = models.generate(learner.actor, 8)
                np.testing.assert_array_equal(batch["next_observations"][:, 15:].numpy(), batch["actions"].numpy())
                if objective == "headway":
                    np.testing.assert_allclose(batch["rewards"].numpy(), replay.normalize(state_headway_reward(batch["next_observations"].numpy())), rtol=1e-5)
                self.assertTrue(torch.isfinite(weights).all())
                models.refit(10000)
                learner.models = models
                self.assertTrue(all(np.isfinite(v) for v in learner.update(online=True).values()))


if __name__ == "__main__":
    unittest.main()
