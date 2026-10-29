"""Verify actual F support, isolated gradient routing and no-fresh replay parity."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_support_td_v83_20261027.learning import (
    attach_support_td, singleton_td_audit, SupportTDQEnsemble)
from experiments.mcwm_closed_loop_context_wiring_v80_20261027 import learning as previous
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.test_learning import initial,packet
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_action_contract import state
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    OPTIMIZERS, _equal, _rng, _restore, _state)


def paired(arm='fresh'):
    old=previous.attach_context(initial(arm))
    old.config=replace(old.config,critic_loss='huber',huber_delta=1.)
    return old,attach_support_td(deepcopy(old))


def states():
    return torch.tensor(np.stack([
        state(time=9000.,fwd_present=1.,bwd_present=0.,load=5.)[1],
        state(time=9000.,forward=360.,backward=400.,load=20.)[1],
        state(time=19000.,load=5.,bwd_present=0.)[1]]))


def test_actual_F_singleton60_partial_reserve_and_tail_keep_their_support():
    old,new=paired()
    canonical=states()
    low=old.action_transform(canonical,torch.full((3,1),-1.))
    high=old.action_transform(canonical,torch.ones(3,1))
    assert ((low==high).reshape(-1)).tolist()==[True,False,False]
    assert old.action_transform.hold_seconds(low).reshape(-1).tolist()==[60.,0.,0.]
    assert float(old.action_transform.hold_seconds(high)[-1])==7.5
    theta=old.critic.action_weight
    parameter_ids=[id(p) for p in new.critic.parameters()]
    assert type(new.critic) is SupportTDQEnsemble
    assert len(parameter_ids)==len(set(parameter_ids))
    assert new.critic._support_transform is new.action_transform
    assert old.target_critic.__class__ is new.target_critic.__class__


def test_factual_TD_values_base_and_action_gradients_exact_only_singleton_theta_stops():
    old,new=paired()
    canonical=states()
    context=torch.tensor([[-.8,1.],[.4,1.],[-.6,1.]])
    packed=previous.pack_states(canonical,context)
    with torch.no_grad():
        old.critic.action_weight.copy_(torch.tensor([.3,-.2]))
        new.critic.action_weight.copy_(old.critic.action_weight)
    action=old.action_transform(canonical,torch.full((3,1),.3)).detach()
    a,b=action.clone().requires_grad_(),action.clone().requires_grad_()
    left,right=old.critic(packed,a),new.critic(packed,b)
    assert torch.equal(left,right)
    target=left.detach().min(0).values+.4
    loss0=old._ensemble_td_loss(left,target)
    loss1=new._ensemble_td_loss(right,target)
    grad0=torch.autograd.grad(loss0,(*old.critic.base.parameters(),old.critic.action_weight,a))
    grad1=torch.autograd.grad(loss1,(*new.critic.base.parameters(),new.critic.action_weight,b))
    assert _equal(grad0[:-2],grad1[:-2])
    assert torch.equal(grad0[-1],grad1[-1])
    feature=context[:,0]*(action[:,0]+1.)*.5
    derivative=(left.detach()-target[None]).clamp(-old.config.huber_delta,old.config.huber_delta)
    expected=(derivative[:,1:]*feature[None,1:]).sum(1)/left.numel()
    torch.testing.assert_close(grad1[-2],expected,rtol=0,atol=1e-8)
    assert not torch.equal(grad0[-2],grad1[-2])
    # The singleton record still trains base Q; its theta contribution alone is zero.
    only0=new.critic(packed[:1],action[:1])
    values=torch.autograd.grad(only0.sum(),(*new.critic.base.parameters(),new.critic.action_weight))
    assert any(bool(g.any()) for g in values[:-1])
    assert not values[-1].any()


def test_choice_and_missing_context_theta_gradients_preserve_original_behavior():
    old,new=paired()
    canonical=states()[1:]
    context=torch.tensor([[.4,1.],[float('nan'),0.]])
    packed=previous.pack_states(canonical,context)
    action=torch.full((2,1),-.5)
    q0,q1=old.critic(packed,action),new.critic(packed,action)
    assert torch.equal(q0,q1)
    assert torch.equal(torch.autograd.grad(q0.sum(),old.critic.action_weight)[0],
        torch.autograd.grad(q1.sum(),new.critic.action_weight)[0])


@pytest.mark.parametrize('forced',[False,True])
def test_actor_CQL_target_and_value_paths_are_unchanged_at_nonzero_theta(forced):
    old,new=paired()
    rows=packet()
    if forced:
        for row in rows:
            row.s=states()[0].numpy().copy()
    for model in (old,new):
        model.replay_buffer.add_target(rows,packet(5))
        with torch.no_grad():
            model.critic.action_weight.copy_(torch.tensor([.3,-.2]))
            model.target_critic.action_weight.copy_(torch.tensor([.1,-.05]))
    batches0=old.replay_buffer.sample_mixed(8,.5,rng=old.rng)
    batches1=new.replay_buffer.sample_mixed(8,.5,rng=new.rng)
    batch0={key:torch.cat((batches0[0][key],batches0[1][key])) for key in batches0[0]}
    batch1={key:torch.cat((batches1[0][key],batches1[1][key])) for key in batches1[0]}
    rng=_rng()
    target=old._target(batch0,backup='h2oplus');after=_rng()
    _restore(rng)
    assert torch.equal(target,new._target(batch1,backup='h2oplus')) and _equal(after,_rng())
    assert _equal(old._iql_value_terms(batches0[0]),new._iql_value_terms(batches1[0]))
    rng=_rng()
    left=old.context_actor_terms(old._actor_states(*batches0))[0]
    gradient0=torch.autograd.grad(left,tuple(old.actor.parameters()));after=_rng()
    _restore(rng)
    right=new.context_actor_terms(new._actor_states(*batches1))[0]
    gradient1=torch.autograd.grad(right,tuple(new.actor.parameters()))
    assert torch.equal(left,right) and _equal(gradient0,gradient1) and _equal(after,_rng())
    rng=_rng()
    left=old._cql_penalty(batches0[0],None)
    gradient0=torch.autograd.grad(left,tuple(old.critic.parameters()));after=_rng()
    _restore(rng)
    right=new._cql_penalty(batches1[0],None)
    gradient1=torch.autograd.grad(right,tuple(new.critic.parameters()))
    assert torch.equal(left,right) and _equal(gradient0,gradient1) and _equal(after,_rng())


def test_no_fresh_actual_eight_updates_preserve_original_modules_Adam_and_RNG():
    old=previous.attach_context(initial('no_fresh'))
    old.total_updates=50000
    for name in OPTIMIZERS:
        optimizer=getattr(old,name)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter is old.critic.action_weight:
                    continue
                optimizer.state[parameter].update(step=torch.tensor(50000.),
                    exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.ones_like(parameter))
    new=attach_support_td(deepcopy(old))
    rows=packet()
    for row in rows:
        row.s=states()[0].numpy().copy()
    for model in (old,new):
        model.replay_buffer.add_target(rows,packet(5))
    start=_rng()
    for model in (old,new):
        _restore(start)
        model.begin_actor_packet()
        for index in range(8):
            model.continue_step(collect_diagnostics=index==0)
        if model is old:
            after=_rng()
    assert _equal(after,_rng())
    assert _equal(_state(old),_state(new))
    assert _equal(old.replay_buffer.outcome_rng.bit_generator.state,
        new.replay_buffer.outcome_rng.bit_generator.state)
    audit=singleton_td_audit(new)
    assert audit['all_pass'] and audit['calls']==8 and audit['rows']==64
    assert audit['observed_rows']==audit['singleton_observed_rows']==0
    assert new.total_updates==50008 and new.online_critic_updates==8
