"""Check speed execution and both action coordinates before releasing training."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from actions import physical_action
from common import HERE, P, write


def resac_contract():
    from resac_runner import DualEnv, core, sim
    original = sim.Bus.__bases__[0]
    route = SimpleNamespace(speed_limit=10., distance=10000.)

    def moving(cls, speed):
        bus = cls.__new__(cls)
        bus.__dict__.update(last_station=SimpleNamespace(station_name="a"),
            next_station=SimpleNamespace(station_name="b"), route_index={("a", "b"):route},
            current_speed=10., in_station=False, holding=False, dwelling=False,
            next_station_dis=10000., last_station_dis=0., absolute_distance=0.,
            direction=1, acceleration=3., deceleration=5., control_speed_ratio=speed)
        return bus

    before, neutral, faster = moving(original, 1.), moving(sim.Bus, 1.), moving(sim.Bus, 1.2)
    original.drive(before, 0., 0., [], False)
    neutral.drive(0., np.array([0., 30.]), [], False)
    faster.drive(0., np.array([0., 60.]), [], False)
    assert before.absolute_distance == neutral.absolute_distance
    assert faster.absolute_distance > neutral.absolute_distance
    assert route.speed_limit == 10.
    stationary = moving(sim.Bus, 1.)
    stationary.__dict__.update(in_station=True, holding=True, dwelling=False,
        held=True, holding_time=0., trip_id=5)
    stationary.drive(0., np.array([10., 60.]), [], False)
    assert stationary.control_speed_ratio == float(np.float32(1.2))
    assert stationary.dwelling_time == 10.
    DualEnv.smoke, DualEnv.mode = True, "dual"
    core.env_bus = DualEnv
    cfg = core.TrainConfig(seed=64001, max_episodes=1, role="specialist", fixed_mode="normal", batch_size=32, replay_capacity=256)
    env = core.make_environment(cfg)
    learner = core.BusSAC(env, cfg, torch.device("cpu"))
    replay = core.ReplayBuffer(256)
    rng = np.random.default_rng(71)
    for i in range(96):
        state = rng.uniform(1, 20, env.state_dim).astype(np.float32)
        state[:4] = (i % 15, i % 15, i % 5, i % 2)
        action = rng.uniform(0, 60, 2).astype(np.float32)
        replay.push(state, action, -float(i+1), state.copy(), False)
    metrics = learner.update(replay, 0)
    assert all(np.isfinite(v) for v in metrics.values())
    assert learner.policy.get_action(state).shape == (2,)
    assert replay.actions.shape[-1] == 2
    return dict(neutral_distance=neutral.absolute_distance, fast_distance=faster.absolute_distance,
                critic_loss=metrics["critic_loss"], policy_loss=metrics["policy_loss"], action_dim=2)


def cs_contract(kind):
    from cs_runner import build_agent, episode, setup
    runtime = setup(kind)
    metrics_by_arm = []
    for arm in P["arms"][kind]:
        args = SimpleNamespace(kind=kind, mode="dual", objective="passenger", arm=arm, seed=64001)
        agent = build_agent(args, runtime)
        agent.config.batch_size = 16
        rng = np.random.default_rng(72)
        states, actions, rewards = [], [], []
        for i in range(64):
            state = rng.uniform(.1, .9, 15).astype(np.float32)
            state[0] = (i % 12)/12
            action = rng.uniform(-.8, .8, 2).astype(np.float32)
            reward = -float(i+1)*1000
            agent.replay_buffer.push(state, action, reward, state.copy(), 0.)
            states.append(state); actions.append(action); rewards.append(reward)
        metrics = agent.update(recent_rollout=dict(state=states, action=actions, reward=rewards))
        assert all(np.isfinite(v) for k,v in metrics.items() if not k.startswith("L_eff"))
        assert agent.select_action(state, deterministic=True).shape == (2,)
        metrics_by_arm.append(dict(arm=arm, q_loss=metrics["q_loss"], alpha=metrics["alpha"]))
    from sim_core.bus import Bus
    def moving(speed):
        return SimpleNamespace(current_route=SimpleNamespace(speed_limit=10., distance=10000.),
            _signal_wait_remaining=0., _signal_stopped=False, current_speed=10., _speed_ratio=speed,
            acceleration=3., deceleration=5., direction_int=1, last_station_dis=0.,
            next_station_dis=10000., absolute_distance=0., line_id_str=None, _segment_effective_speed=10.)
    neutral, fast = moving(1.), moving(1.2)
    fast.trip_id = 5
    Bus._start_dwelling(fast, 10., 1.2)
    assert fast._speed_ratio == 1.2
    Bus._advance_on_route(neutral)
    Bus._advance_on_route(fast)
    assert fast.absolute_distance > neutral.absolute_distance
    received, stored = [], []
    observation = np.zeros(15, np.float32)
    observation[2] = 1.

    class FakeEnv:
        line_map = {"line":SimpleNamespace(max_agent_num=1)}
        def reset(self):
            return None
        def initialize_state(self):
            return {"line":{0:[observation.copy()]}}, {}, False
        def step_to_event(self, commands):
            received.append(commands["line"][0].copy())
            next_state = observation.copy(); next_state[2] = 2.
            return {"line":{0:[next_state]}}, {"line":{0:-10.}}, True
    fake_agent = SimpleNamespace(select_action=lambda *args, **kwargs: np.array([-.5, .75], np.float32),
        replay_buffer=SimpleNamespace(push=lambda *values: stored.append(values), size=0))
    episode(FakeEnv(), fake_agent, lambda x:x/55., kind, train=True, deterministic=True,
            config=SimpleNamespace(surprise_window=16, batch_size=32))
    assert len(stored) == 1 and stored[0][1].shape == (2,)
    assert np.allclose(stored[0][1], [-.5, .75])
    assert np.allclose(received[0], [15., 1.15])
    return dict(arms=metrics_by_arm, neutral_distance=neutral.absolute_distance,
        fast_distance=fast.absolute_distance, replay_action=list(map(float, stored[0][1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=P["arms"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert np.allclose(physical_action([0., 30.], "resac"), [0., 1.])
    assert np.allclose(physical_action([-1., 0.], "cs_native"), [0., 1.])
    result = resac_contract() if args.kind == "resac" else cs_contract(args.kind)
    write(HERE / "contracts" / (args.kind+".json"), dict(kind=args.kind, passed=True, **result))
    print("DONE dual action contracts", args.kind, result, flush=True)
