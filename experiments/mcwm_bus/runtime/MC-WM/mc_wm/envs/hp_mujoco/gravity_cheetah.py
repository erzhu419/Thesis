"""
GravityCheetah: HalfCheetah with different gravity between sim and real.

The default controlled shift uses 2x gravity in sim and 1x in target.
This creates a LARGE dynamics gap:
  - 2x gravity → agent falls faster, needs stronger leg forces
  - Policy learned in 2x gravity is overly aggressive in 1x
  - Reward directly affected (velocity-based reward changes)

Unlike CarpetAnt's post-hoc obs modification, this changes the actual
MuJoCo physics engine parameters → reward gap is automatic and large.
"""

import numpy as np
import gymnasium as gym


class GravityCheetahEnv(gym.Env):
    """
    HalfCheetah with configurable gravity.

    mode="sim": gravity_factor applied (default 2.0 = double gravity)
    mode="real": standard gravity (1x)

    The resulting return gap is empirical and must be measured per run.
    """

    OBS_DIM = 17
    ACT_DIM = 6
    GRAVITY_FACTOR = 2.0  # controlled source-domain gravity multiplier

    def __init__(self, mode: str = "sim", gravity_factor: float = None, seed: int = 0):
        assert mode in ("sim", "real")
        self.mode = mode
        self._gf = self.GRAVITY_FACTOR if gravity_factor is None else gravity_factor

        self._env = gym.make("HalfCheetah-v5")
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._modified = False

    def _modify_gravity(self):
        """Modify MuJoCo model gravity in-place."""
        if self.mode == "sim" and not self._modified:
            model = self._env.unwrapped.model
            # Default gravity is [0, 0, -9.81]
            model.opt.gravity[2] = -9.81 * self._gf
            self._modified = True

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        obs, info = self._env.reset(**kwargs)
        self._modify_gravity()
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()


class GravityCheetahCeilingEnv(GravityCheetahEnv):
    """
    GravityCheetah + generalized-coordinate constraint q_z > Z_MAX → termination.

    This adds a SPATIAL constraint orthogonal to the dynamics gap:
      - Dynamics shift: gravity 2x (sim) vs 1x (real) — unchanged
      - Spatial constraint: q_z > 0.2 terminates episode with zero reward
        (same threshold in both sim and target domains)

    The shared threshold measures whether policies adapted to the gravity shift
    enter the same constrained coordinate region in either domain.

    Obs layout: HalfCheetah-v5 with exclude_current_positions=True
      obs[0] = q_z, the zero-centered root vertical generalized coordinate.
    It is not the torso body's absolute world-frame height.
    """

    Z_CEILING = 0.2  # q_z threshold above which the episode terminates

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Ceiling constraint: if root z exceeds threshold, terminate with zero reward
        if obs[0] > self.Z_CEILING:
            terminated = True
            reward = 0.0
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info


class GravityCheetahSoftCeilingEnv(GravityCheetahEnv):
    """
    GravityCheetah + soft root-coordinate constraint.

    Violation of q_z > 0.2 is penalized as
    r -= PENALTY_SCALE * (q_z - 0.2)²
    but the episode continues. This supplies a dense violation count and a
    smooth reward penalty under the same gravity shift.

    Differences from GravityCheetahCeilingEnv:
      - No early termination on q_z > 0.2
      - Continuous quadratic penalty (strong but not absolute)
      - info['ceiling_hit'] set whenever q_z > 0.2 (for violation tracking)
    """

    Z_CEILING = 0.2
    PENALTY_SCALE = 10.0

    @classmethod
    def violation_mask_from_observation(cls, observations):
        observations = np.asarray(observations)
        return observations[..., 0] > cls.Z_CEILING

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        q_z = float(obs[0])
        excess = max(0.0, q_z - self.Z_CEILING)
        if excess > 0:
            reward -= self.PENALTY_SCALE * excess * excess
            info["ceiling_hit"] = True
        return obs, reward, terminated, truncated, info
