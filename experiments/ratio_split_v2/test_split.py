"""Regression for renewed samplers assigning every source day to trajectory zero."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from run import base


class Sampler:
    def __init__(self, env, replay, **kwargs):
        self.env, self.replay = env, replay
        self.episodes = self.next_trajectory_id = 0

    def collect(self, policy, count):
        replay, i = self.replay, self.replay.online_size
        replay._online_episode_ids[i] = self.episodes
        replay._online_trajectory_ids[i] = self.next_trajectory_id
        replay._online_ratio_validation[i] = base.TransitReplayBufferV7._online_trajectory_validation_mask(
            np.array([self.next_trajectory_id]))[0]
        replay._online_terminals[i] = 1
        replay.online_size += 1
        self.env.done = True


class SplitTest(unittest.TestCase):
    def test_new_sampler_inserts_disjoint_complete_days(self):
        replay = SimpleNamespace(online_size=0, online_capacity=10, station_max_by_line=np.ones(12),
                                 contract_config={}, _online_episode_ids=np.zeros(10, dtype=int),
                                 _online_trajectory_ids=np.zeros(10, dtype=int),
                                 _online_ratio_validation=np.zeros(10, dtype=bool),
                                 _online_terminals=np.zeros(10))
        with patch.object(base, "AllLinesSimEnv", side_effect=lambda *a, **k: SimpleNamespace(done=False)), \
             patch.object(base, "AllLinesTransitSamplerV7", Sampler), \
             patch.object(base, "load_protocol", return_value={"source_env": "unused"}):
            for day in range(5):
                base.collect_day(replay, SimpleNamespace(act=None), dict(source_day_seed_base=24100),
                                 day, SimpleNamespace(phase=lambda name: nullcontext()))
        np.testing.assert_array_equal(replay._online_episode_ids[:5], np.arange(5))
        np.testing.assert_array_equal(replay._online_trajectory_ids[:5], np.arange(5))
        np.testing.assert_array_equal(replay._online_ratio_validation[:5], [False, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
