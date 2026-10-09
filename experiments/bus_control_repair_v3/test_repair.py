"""Regression checks for the concrete reference and loss failures."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from repair_common import HERE, P, module
from symbolic import SmoothReference, SoftResidualPolicy, equilibrium_states, normalized_jacobian_loss


def sample_states(n=12):
    states = torch.rand(n, 15)
    states[:, 0] = torch.arange(n) / 12
    states[:, 8] = .6
    return states


def fitted_reference():
    powers = np.vstack((np.zeros((1, 10)), np.eye(10))).astype(int)
    coeffs = np.zeros((12, 11, 11), np.float32)
    coeffs[:, 9, 10] = .2
    coeffs[:, 10, 10] = .2
    return SmoothReference(dict(powers=powers, coefficients=coeffs,
        cost_mean=np.zeros((12, 2)), cost_std=np.ones((12, 2))), "headway")


class LinearReference(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, states):
        return self.scale * states[:, :1]


class RepairTests(unittest.TestCase):
    def test_failed_dynamics_do_not_change_single_station_reference(self):
        ref, states = fitted_reference(), sample_states()
        expected = ref(states)
        ref.coefficients[:, :9] = 1e6
        torch.testing.assert_close(expected, ref(states))
        self.assertTrue(torch.isfinite(ref(states)).all())

    def test_soft_mean_can_leave_incorrect_reference_equilibrium(self):
        actor = SoftResidualPolicy()
        projected = equilibrium_states(sample_states())
        initial = actor(projected)[0]
        torch.testing.assert_close(initial, torch.zeros_like(initial), atol=1e-7, rtol=0)
        initial.sum().backward()
        self.assertGreater(float(actor.equilibrium_mean.grad.abs()), 0)
        with torch.no_grad():
            actor.equilibrium_mean.fill_(-2.)
        torch.testing.assert_close(actor(projected)[0], torch.full_like(initial, -2.), atol=1e-6, rtol=0)

    def test_reference_derivative_scale_cannot_blow_up_initial_penalty(self):
        states = sample_states()
        zero = LinearReference(0.)
        for scale in (1., 1e3):
            loss = normalized_jacobian_loss(zero, LinearReference(scale), states)
            self.assertAlmostEqual(float(loss), scale**2 / (1 + scale**2), places=6)
        self.assertEqual(float(normalized_jacobian_loss(zero, zero, states)), 0.)

    def test_normalized_penalty_still_trains_policy(self):
        actor = SoftResidualPolicy()
        loss = normalized_jacobian_loss(actor, LinearReference(.1), sample_states())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_actor_and_critic_share_repaired_penalty_and_alpha_cap(self):
        import cs
        from types import SimpleNamespace
        import csbapr.agent as agent_module
        import csbapr.losses.q_loss as q_module
        self.assertIs(agent_module.compute_jacobian_loss, normalized_jacobian_loss)
        self.assertIs(q_module.compute_jacobian_loss, normalized_jacobian_loss)
        with patch.object(cs, "reference", return_value=fitted_reference()):
            agent = cs.build_agent(SimpleNamespace(arm="smooth_soft", seed=64001, objective="headway"))
        self.assertEqual(agent.config.alpha_cap, P["alpha_cap"])
        self.assertEqual(agent.config.jac_weight, .01)
        self.assertEqual(agent.config.weight_sym, .01)

    def test_same_guard_for_single_and_dual_actions_preserves_speed(self):
        mc = module("repair_mc_test", HERE / "mc.py")
        from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_DIM, RAW_EVENT_STATE_INDEX, TransitMDPConfigV7
        raw = np.zeros((1, RAW_EVENT_STATE_DIM), np.float32)
        values = dict(sim_time_seconds=6000, forward_headway_seconds=600,
            backward_headway_seconds=600, target_forward_headway_seconds=600,
            target_backward_headway_seconds=600, onboard_passengers=5,
            eligible_arrival_rate_per_second=.1, bus_capacity=60,
            forward_bus_present=1, backward_bus_present=0)
        for key, value in values.items():
            raw[0, RAW_EVENT_STATE_INDEX[key]] = value
        config = TransitMDPConfigV7()
        for proposed in (np.array([-1.], np.float32), np.array([-1., .5], np.float32)):
            prior = np.array([-1.], np.float32)
            direct = mc.execute_hold(raw, 0, prior, proposed, "direct", config)
            np.testing.assert_array_equal(direct, proposed)
            guarded = mc.execute_hold(raw, 0, prior, proposed, "guarded", config)
            self.assertEqual(float(guarded[0]), 1.)
            self.assertEqual(float(prior[0]), 1.)
            if len(proposed) == 2:
                self.assertEqual(float(guarded[1]), float(proposed[1]))


if __name__ == "__main__":
    unittest.main()
