"""Walker2d with a controlled sim-to-target ground-friction shift."""

import numpy as np
import gymnasium as gym


class FrictionWalkerEnv(gym.Env):
    """
    Walker2d with configurable ground friction.

    mode="sim": friction_factor applied (default 0.3 = 30% of normal friction)
    mode="real": standard friction (1x)
    """

    OBS_DIM = 17
    ACT_DIM = 6
    FRICTION_FACTOR = 0.3

    def __init__(self, mode: str = "sim", friction_factor: float = None, seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._ff = self.FRICTION_FACTOR if friction_factor is None else friction_factor

        self._env = gym.make("Walker2d-v5")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._modified = False

    def _modify_friction(self):
        """Modify MuJoCo geom friction in-place."""
        if self.mode == "sim" and not self._modified:
            model = self._env.unwrapped.model
            # geom_friction shape: (n_geom, 3) — [sliding, torsional, rolling]
            model.geom_friction[:] *= self._ff
            self._modified = True

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        obs, info = self._env.reset(**kwargs)
        self._modify_friction()
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()


class FrictionWalkerSoftCeilingEnv(FrictionWalkerEnv):
    """
    FrictionWalker + a shared soft root-height threshold z > 1.25.

    Walker2d's z_root ~1.20 standing, ~1.27 when moving. Ceiling at 1.25
    binds often (agent must stay low), with continuous penalty 10*(z-1.25)².

    The threshold is applied identically in the simulator and target members;
    only the ground-friction factor changes between domains.
    """

    Z_CEILING = 1.25  # Walker normal z ≈ 1.20; ceiling at 1.25 = "stay low"
    PENALTY_SCALE = 10.0

    @classmethod
    def violation_mask_from_observation(cls, observations):
        observations = np.asarray(observations)
        return observations[..., 0] > cls.Z_CEILING

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        z = float(obs[0])
        excess = max(0.0, z - self.Z_CEILING)
        if excess > 0:
            reward -= self.PENALTY_SCALE * excess * excess
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info
