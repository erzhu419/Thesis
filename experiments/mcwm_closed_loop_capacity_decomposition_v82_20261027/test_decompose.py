"""Catch head-selection misattribution, F aliases and gradient-accounting errors."""
from types import SimpleNamespace

import torch
from torch import nn

from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.decompose import (
    actor_gradient_decomposition, audit_saved_gradients, executed_grid,
    explicit_components, q_decomposition)
from SimpleSAC.transit_v4.networks import HurdleTransitActor


class Base(nn.Module):
    ensemble_size = 2

    def forward_features(self, features, actions):
        return torch.stack((actions[:,0]+features[:,0],
                            -.6*actions[:,0]+features[:,0]+.12))


class Critic(nn.Module):
    ensemble_size = 2

    def __init__(self):
        super().__init__()
        self.base = Base()
        self.action_weight = nn.Parameter(torch.tensor([.8,-.5],dtype=torch.float64))

    def encode_states(self, states):
        return torch.cat((states[:,:1],states[:,-2:]),1)

    def forward_features(self, features, actions):
        z = torch.where(features[:,-1].bool(),features[:,-2],torch.zeros_like(features[:,-2]))
        return self.base.forward_features(features[:,:-2],actions)+self.action_weight[:,None]*z[None]*(actions[:,0][None]+1.)*.5


class Transform:
    max_hold = 60.

    def __call__(self,states,actions):
        return torch.where((states[:,1] > .5)[:,None],torch.ones_like(actions),actions)

    def hold_seconds(self,actions):
        return (actions+1.)*30.


class Actor(nn.Module):
    _positive_transform = staticmethod(HurdleTransitActor._positive_transform)
    _positive_log_det = staticmethod(HurdleTransitActor._positive_log_det)

    def __init__(self):
        super().__init__()
        self.output = nn.Parameter(torch.tensor([.2,-.4,-.7],dtype=torch.float64))

    def distribution_parameters(self,states):
        return tuple(self.output[index].expand(len(states)) for index in range(3))


def fixture():
    states = torch.zeros((2,229),dtype=torch.float64)
    states[1,1] = 1.
    context = torch.tensor([[.7,1.],[-.8,1.]],dtype=torch.float64)
    learner = SimpleNamespace(critic=Critic(),actor=Actor(),
        action_transform=Transform(),alpha=torch.tensor(.1,dtype=torch.float64))
    return learner,states,context


def test_minimum_head_switch_and_action_advantages_close_separately():
    learner,states,context=fixture()
    actions = torch.tensor([[-1.,-.5,0.,.5,1.],[-1.,-.5,0.,.5,1.]],dtype=torch.float64)
    result=q_decomposition(learner.critic,states,context,actions)
    assert result['head_changed'].any() and result['head_switch'].max() > 0
    assert result['min_reconstruction_error'].max() < 1e-14
    assert result['advantage_reconstruction_error'].max() < 1e-14
    assert torch.allclose(result['full_advantage'],result['base_advantage']+
        result['direct_advantage']+result['head_switch_advantage'],atol=1e-14,rtol=0)
    # This catches reporting min(Q_base+term)-min(Q_base) as term alone.
    assert not torch.allclose(result['full_advantage']-result['base_advantage'],
                              result['direct_advantage'])


def test_grid_aliases_do_not_create_actions_or_reapply_F():
    learner,states,context=fixture()
    grid=executed_grid(learner.action_transform,states)
    assert grid['unique_count'].tolist()==[5,1]
    assert grid['executed_hold_seconds'][1].tolist()==[60.]*5
    result=q_decomposition(learner.critic,states,context,grid['executed_actions'])
    assert result['unique_support_count'].tolist()==[5,1]
    assert not result['preference_changed'][1]
    assert not result['full_advantage'][1].any()
    assert not result['base_advantage'][1].any()


def test_gradient_decomposition_noise_entropy_and_deployed_action_are_frozen():
    learner,states,context=fixture()
    before=torch.get_rng_state().clone()
    parameters=learner.actor.output.detach().clone()
    noise=torch.tensor([-1.5,-1.,-.5,0.,.2,.5,1.,1.5],dtype=torch.float64)
    sample=actor_gradient_decomposition(learner,states,context,noise)
    deterministic=actor_gradient_decomposition(learner,states,context)
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(parameters,learner.actor.output.detach())
    assert learner.actor.output.grad is None and learner.critic.action_weight.grad is None
    assert sample['gradient_reconstruction_error'].max() < 1e-14
    assert sample['SAC_gradient_reconstruction_error'].max() < 1e-14
    assert torch.equal(sample['deployed_holding_seconds'],deterministic['deployed_holding_seconds'])
    assert sample['deployed_holding_seconds'][1] == 60.
    for name in ('base_Q','direct_capacity','head_switch','full_Q'):
        assert sample['gradients'][name][1].abs().max() < 1e-14
    assert sample['gradients']['entropy'][1].abs().max() > 0
    assert sample['gradients']['direct_capacity'][0].abs().max() > 0


def test_missing_context_is_not_observed_zero_or_nan_arithmetic():
    learner,states,context=fixture()
    context[:,0],context[:,1]=torch.nan,0.
    actions=torch.tensor([[-1.,0.,1.],[-1.,0.,1.]],dtype=torch.float64)
    result=q_decomposition(learner.critic,states,context,actions)
    assert torch.isfinite(result['full_heads']).all()
    assert torch.equal(result['full_min'],result['base_min'])
    assert not result['direct'].any() and not result['head_switch'].any()


def test_explicit_components_match_original_deterministic_hurdle():
    learner,states,_=fixture()
    actor=learner.actor
    gate,mean,std=actor.distribution_parameters(states)
    actual=explicit_components(actor,gate,mean,std)
    original=HurdleTransitActor.soft_action_components_from_parameters(actor,
        gate,mean,std,deterministic_positive=True)
    for name in ('hold_probability','zero_probability','zero_log_prob'):
        assert torch.equal(actual[name],original[name])
    for name in ('positive_action','zero_action'):
        assert torch.equal(actual[name],original[name])
    for name in ('positive_conditional_log_prob','positive_log_prob'):
        assert torch.equal(actual[name][:,0],original[name])


def test_large_common_Q_level_requires_uncancelled_gate_scale_and_detects_bad_term():
    class Offset(Base):
        def forward_features(self, features, actions):
            return super().forward_features(features,actions)+300.
    learner,states,context=fixture()
    learner.critic.base=Offset()
    learner.critic.float()
    learner.actor.float()
    learner.alpha=learner.alpha.float()
    states,context=states.float(),context.float()
    value=actor_gradient_decomposition(learner,states,context,
        torch.tensor([-1.5,-1.,-.5,0.,.2,.5,1.,1.5]))
    # Adding a common level does not change ideal action gradients. Native
    # FP32 reverse products can nevertheless lose several micro-units at gate.
    assert value['gradient_reconstruction_error'].max() > 16*torch.finfo(torch.float32).eps
    audit=audit_saved_gradients(value['q'],value['gate_probability'],value['gradients'])
    assert audit['all_pass'], audit['checks']
    assert audit['Q_closure_error'][:,1:].max() < 1e-6
    assert audit['analytic_gate_gradients']['full_Q'][1] == 0.
    corrupted={name:gradient.clone() for name,gradient in value['gradients'].items()}
    corrupted['direct_capacity'][0,0] += .01
    rejected=audit_saved_gradients(value['q'],value['gate_probability'],corrupted)
    assert not rejected['all_pass']
    assert not rejected['checks']['independent_gate_reference']
