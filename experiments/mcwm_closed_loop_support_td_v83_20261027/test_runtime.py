"""Run the real tiny-network stack to catch extra TD calls or Adam resets."""
import pytest
import torch

from experiments.mcwm_closed_loop_context_wiring_v80_20261027.test_learning import initial,packet
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import attach_context
from experiments.mcwm_closed_loop_support_td_v83_20261027.learning import attach_support_td
from experiments.mcwm_closed_loop_support_td_v83_20261027.context_audit import FirstPacketAudit
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import OPTIMIZERS


@pytest.mark.parametrize('arm',['fresh','no_fresh'])
def test_formal_first_packet_counts_actual_td_gate_without_extra_steps(tmp_path,arm):
    learner=attach_support_td(attach_context(initial(arm)))
    learner.total_updates=50000
    for name in OPTIMIZERS:
        optimizer=getattr(learner,name)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter is learner.critic.action_weight:
                    continue
                optimizer.state[parameter].update(step=torch.tensor(50000.),
                    exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.ones_like(parameter))
    learner.replay_buffer.add_target(packet(),packet(5))
    audit=FirstPacketAudit(learner)
    learner.begin_actor_packet()
    for index in range(8):
        learner.continue_step(collect_diagnostics=index==0)
        audit.record_update()
    result=audit.finish(tmp_path)
    assert result['all_pass'],result['checks']
    gate=result['td_gradient_gate']
    assert gate['all_pass'],gate['checks']
    assert gate['calls']==8 and gate['rows']==8*learner.config.batch_size
    assert gate['observed_rows']==(16 if arm=='fresh' else 0)
    assert learner.total_updates==50008
    assert learner.actor_schedule_audit()['packet_critic_steps']==[8]
    assert learner.actor_schedule_audit()['packet_actor_steps']==[8]
    assert result['no_extra_steps'] and result['actual_updates']==8
    from experiments.mcwm_closed_loop_support_td_v83_20261027.control import save_final_learner
    save_final_learner(learner,tmp_path/'final_learner.pt',cell={'arm':arm},
        clock=18000.,counts={'critic':8})
    stored=torch.load(tmp_path/'final_learner.pt',weights_only=False)
    assert stored['state']['updates']==50008 and not stored['replay_stored']
    assert stored['td_gradient_gate']['calls']==8
    assert stored['learning_source'].endswith('support_td_v83_20261027')
