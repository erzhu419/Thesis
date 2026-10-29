"""Verify Huber clipping, factual theta pressure and isolated target evaluation."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.pressure import pressure,td_theta_terms
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.test_learning import initial,packet
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import attach_context


def test_analytic_clipped_theta_pressure_matches_autograd_with_exposure_weights():
    theta = torch.tensor([.4,-.2],dtype=torch.double,requires_grad=True)
    feature = torch.tensor([-.8,0.,.5],dtype=torch.double)
    base = torch.tensor([[.1,5.,-8.],[2.,-5.,3.]],dtype=torch.double)
    q = base+theta[:,None]*feature[None]
    target = torch.tensor([0.,-4.,1.],dtype=torch.double)
    exposure = torch.tensor([1.,3.,2.],dtype=torch.double)
    values = td_theta_terms(q,target,feature,weights=exposure)
    error = q-target[None]
    loss = .25*(torch.nn.functional.huber_loss(q,target[None].expand_as(q),reduction='none')
        *(exposure/exposure.sum())[None]).sum()/2
    actual, = torch.autograd.grad(loss,theta)
    torch.testing.assert_close(actual,values['exposure_weighted_gradient'])
    assert values['huber_clipped'].sum() > 0
    assert torch.equal(values['gradient'][:,1],torch.zeros(2,dtype=torch.double))
    assert not torch.equal(values['gradient'],values['raw_mse_gradient'])


def fixture():
    learner = attach_context(initial())
    learner.config = replace(learner.config,critic_loss='huber',huber_delta=1.,
        reward_scale=1e-4,bootstrap_probability=1.,target_subset_size=2,
        target_subset_replacement=False,backup_entropy=True)
    rows=[]
    for index,row in enumerate(packet()):
        values = vars(row).copy()
        values['s'] = np.asarray(values['s']).copy()
        # Genuine F completion-floor condition: full support collapses to60s.
        values['s'][0] = .5
        current = round(float(values['s'][2])*15)
        event = values['s'][5:].reshape(16,14)[current]
        event[11],event[12],event[9],event[8] = 1.,0.,.2,.1
        values['a'] = np.array([1.],np.float32)
        values['ctx_s'] = dict(observed=index != 1,z=None if index == 1 else -.6)
        values['ctx_ns'] = dict(observed=True,z=-.4)
        values['r'] = -100000.
        values['terminal'] = False
        rows.append(values)
    return learner,rows


def test_forced60_feedback_has_theta_pressure_but_missing_context_does_not():
    learner,rows=fixture()
    rng = torch.get_rng_state().clone()
    values=pressure(learner,rows,[-1.5,-1.,-.5,-.1,.1,.5,1.,1.5])
    assert torch.equal(rng,torch.get_rng_state())
    assert values['forced_60'].all() and values['single_support'].all()
    assert values['gradient'][:,0].abs().sum() > 0
    assert not values['gradient'][:,1].any()
    assert values['error_reconstruction_error'] < 1e-5
    assert values['samples'] == 8 and not values['gradient_component_decomposition_valid']
    assert learner.online_critic_updates == 0
    assert not learner.critic_optimizer.state
    assert all(parameter.grad is None for parameter in learner.critic.parameters())


def test_production_loss_contract_cannot_silently_be_replaced_by_MSE():
    learner,rows=fixture()
    learner.config=replace(learner.config,critic_loss='mse')
    with pytest.raises(ValueError,match='actual v81 Huber'):
        pressure(learner,rows,[0.]*8)
