"""
AntWallBroken: controlled actuator-failure environment inspired by the
Ant-Broken constraint-transfer setting of Malik et al. (2021, Section 4.2).

This is a project-specific Gymnasium-v5 variant, not a reproduction of that
paper's environment or learning protocol.

Setup:
  sim:  AntWall  — full Ant (8 working legs) + spatial wall x > -3
  target: four actuator channels disabled (action[4:]=0) + same wall

Why this is useful for dynamics adaptation:
  - Dynamics gap  : 4 of 8 actuators dead in real → LARGE action→dynamics shift
  - Spatial       : wall at x=-3 exists in BOTH sim and target domains
  - QΔ learns dynamics difference (broken legs → different next-state distribution)
  - The soft-cap variant adds a dense, shared transition predicate

Project-specific reward:
  r = distance_from_origin + healthy_reward - ctrl_cost
  (NOT the standard forward_velocity reward)

Obs: standard Ant-v5 (27-dim, excludes x,y positions from obs).
     x-position tracked via unwrapped.data.qpos[0] for wall checking.
"""

import numpy as np
import gymnasium as gym


class AntWallBrokenEnv(gym.Env):
    """
    Ant with wall termination and target-domain actuator pairs 4--7 disabled.

    mode="sim":  full actuators, wall at x=-3 terminates
    mode="real": actuator channels 4--7 are zeroed, wall at x=-3 terminates
    """

    OBS_DIM = 27
    ACT_DIM = 8
    WALL_X = -3.0
    HEALTHY_REWARD = 1.0

    def __init__(self, mode: str = "sim", seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._env = gym.make("Ant-v5", include_cfrc_ext_in_observation=False,
                             terminate_when_unhealthy=True)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def _x_position(self):
        return float(self._env.unwrapped.data.qpos[0])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.mode == "real":
            action[4:] = 0.0  # two actuator pairs are disabled

        obs, _, terminated, truncated, info = self._env.step(action)

        # Project-specific distance reward, chosen to preserve radial motion.
        x = self._x_position()
        y = float(self._env.unwrapped.data.qpos[1])
        dist = float(np.sqrt(x * x + y * y))
        ctrl_cost = 0.5 * float(np.sum(action ** 2))
        reward = dist + self.HEALTHY_REWARD - ctrl_cost

        # Wall constraint: x < -3 → terminate with zero reward
        if x < self.WALL_X:
            terminated = True
            reward = 0.0
            info["wall_hit"] = True

        info["x_position"] = x
        info["distance_from_origin"] = dist
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()


class AntWallBrokenSoftCeilingEnv(AntWallBrokenEnv):
    """
    AntWallBroken + soft velocity cap |vx| > V_CAP.

    Keeps the hard wall termination at x=-3 and the four-actuator dynamics gap
    (action[4:]=0 in the target); adds a continuous quadratic penalty when the
    magnitude of forward velocity exceeds V_CAP. The cap is dense enough to
    provide a measurable transition-predicate ablation.

    obs[13] = vx per Ant-v5 layout.
    """

    V_CAP = 1.5
    PENALTY_SCALE = 1.0

    @classmethod
    def violation_mask_from_observation(cls, observations):
        observations = np.asarray(observations)
        return np.abs(observations[..., 13]) > cls.V_CAP

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        vx = float(obs[13])
        excess = max(0.0, abs(vx) - self.V_CAP)
        if excess > 0:
            reward -= self.PENALTY_SCALE * excess * excess
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info
