"""Test the failure modes that motivated the new dual-control branch."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from aligned_common import CONFIG,state,pack_event_batch_v7,ridge_of_state
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_INDEX,RAW_EVENT_STATE_DIM
from chronological import close_pending
from dual_networks import Actor,Critic


def example_state():
    raw = np.zeros((1,RAW_EVENT_STATE_DIM),np.float32)
    fields = dict(line_id=0,station_index=2,sim_time_seconds=900,forward_headway_seconds=360,backward_headway_seconds=360,
        target_forward_headway_seconds=360,target_backward_headway_seconds=360,bus_capacity=50,onboard_passengers=10,
        forward_bus_present=1,backward_bus_present=1)
    for name,value in fields.items():
        raw[0,RAW_EVENT_STATE_INDEX[name]] = value
    return state(pack_event_batch_v7(raw,0,np.array([-1.],np.float32),CONFIG),np.zeros(16,np.float32),np.full(12,46,np.float32))


class ContractTests(unittest.TestCase):
    def test_zero_holding_and_neutral_speed_are_representable(self):
        actor = Actor((32,32))
        with torch.no_grad():
            actor.network.network[-1].weight.zero_()
            actor.network.network[-1].bias[0] = -10.
        s = torch.from_numpy(example_state()[None])
        action,_ = actor(s,deterministic=True)
        self.assertTrue(torch.equal(action,torch.tensor([[-1.,0.]])))
        self.assertTrue(torch.isfinite(actor.action_log_prob(s,action)).all())

    def test_speed_has_gradient_in_both_holding_branches(self):
        s = torch.from_numpy(example_state()[None])
        for branch in ("zero_action","positive_action"):
            actor = Actor((32,32))
            components = actor.soft_action_components(s)
            components[branch][:,1].sum().backward()
            self.assertGreater(float(actor.network.network[-1].weight.grad[3].abs().sum()),0.)
        critic = Critic(2,(32,32))
        self.assertEqual(critic(s,components["zero_action"].detach()).shape,(2,1))

    def test_sample_and_density_agree_for_both_holding_branches(self):
        actor = Actor((32,32))
        s = torch.from_numpy(example_state()[None])
        with torch.no_grad():
            actor.network.network[-1].weight.zero_()
            for gate in (-10.,10.):
                actor.network.network[-1].bias[0] = gate
                action,logp = actor(s,deterministic=True)
                self.assertTrue(torch.allclose(logp,actor.action_log_prob(s,action),atol=1e-5))

    def test_passenger_returns_telescope_including_zero_time_and_terminal(self):
        snapshots = [SimpleNamespace(time_seconds=t,accrued_wait_seconds=w,accrued_in_vehicle_seconds=v)
                     for t,w,v in ((900,10,20),(900,10,20),(910,40,60),(920,50,80))]
        s = example_state()
        rewards = []
        for index in range(3):
            pending = dict(state=s,action=np.array([-1.,0.]),snapshot=snapshots[index])
            row = close_pending(pending,s,snapshots[index+1],terminal=index==2)
            rewards.append(row[4])
        self.assertEqual(rewards,[0.,-100.,-40.])
        self.assertEqual(sum(rewards),-(2*(50-10)+80-20))
        self.assertEqual(row[3],0.)

    def test_headway_reward_matches_the_existing_station_function(self):
        self.assertAlmostEqual(float(ridge_of_state(example_state()[None])[0]),0.,places=5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
