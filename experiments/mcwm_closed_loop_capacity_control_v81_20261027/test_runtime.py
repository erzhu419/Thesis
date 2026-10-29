"""Exercise the real context update stack and new causal trace accounting."""
from types import SimpleNamespace
import json

import pytest
import torch

from experiments.mcwm_closed_loop_capacity_control_v81_20261027.context_audit import (
    ContextAudit, FirstPacketAudit)
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.test_learning import (
    initial, packet)
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import attach_context
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import OPTIMIZERS


@pytest.mark.parametrize('arm',['fresh','no_fresh'])
def test_reused_eight_updates_reach_real_trainer_and_keep_Adam_steps(tmp_path, arm):
    learner = attach_context(initial(arm))
    learner.total_updates = 50000
    # Tiny synthetic fixture: retain nonempty restored Adam state while the new
    # context coefficient has not previously received any optimizer step.
    for name in OPTIMIZERS:
        optimizer = getattr(learner,name)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter is learner.critic.action_weight:
                    continue
                optimizer.state[parameter].update(step=torch.tensor(50000.),
                    exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.ones_like(parameter))
    learner.replay_buffer.add_target(packet(),packet(5))
    audit = FirstPacketAudit(learner)
    learner.begin_actor_packet()
    for index in range(8):
        learner.continue_step(collect_diagnostics=index == 0)
        audit.record_update()
    result = audit.finish(tmp_path)
    assert result['all_pass'], result['checks']
    assert result['actual_updates'] == 8 and result['reused_main_day']
    assert learner.total_updates == 50008
    assert learner.actor_schedule_audit()['optimizer_order_exact']
    assert learner.actor_schedule_audit()['packet_critic_steps'] == [8]
    assert learner.actor_schedule_audit()['packet_actor_steps'] == [8]
    assert torch.load(tmp_path/'first_packet_rng.pt', weights_only=False)['before']
    from experiments.mcwm_closed_loop_capacity_control_v81_20261027.control import save_final_learner
    save_final_learner(learner,tmp_path/'final_learner.pt',cell={'arm':arm},clock=18000.,counts={'critic':8})
    stored = torch.load(tmp_path/'final_learner.pt',weights_only=False)
    assert stored['state']['updates'] == 50008 and not stored['replay_stored']
    assert stored['state']['modules']['critic']['action_weight'].shape == (2,)
    assert torch.equal(stored['state']['modules']['critic']['action_weight'],
        learner.critic.action_weight)


def context(sequence, vehicle='bus', clock=10.):
    return dict(observed=True,z=.2,reasons=[],
        clocks=dict(issue=clock,event=clock-1,obs_start=clock,obs_end=clock),
        counts=dict(onboard=12,next_queue=8,release_next=3),
        binding=dict(sequence=sequence,episode_id=0,trip_id=vehicle,
            vehicle_id=vehicle,line_id='L',station_id='S',issue_clock=clock))


def test_sidecar_serialization_preserves_cross_vehicle_zero_duration_endpoints(tmp_path):
    audit = ContextAudit(tmp_path,'fresh')
    current, successor = context(1,'bus1'),context(2,'bus2')
    issue = dict(current['binding'],started_at=10.,ctx_s=current)
    audit.observe_issue(issue)
    row = SimpleNamespace(sequence=1,ctx_s=current,ctx_ns=successor,started_at=10.,
        completed_at=10.,dt=0.,terminal=False,next_vehicle_id='bus2',episode_id=0)
    audit.record_packet([row],[row],packet=1,processed_at=40.,replay=None)
    audit.close()
    saved = json.loads((tmp_path/'context_selected.jsonl').read_text())
    assert saved['outcomes'][0]['ctx_s']['binding']['vehicle_id'] == 'bus1'
    assert saved['outcomes'][0]['ctx_ns']['binding']['vehicle_id'] == 'bus2'
    assert saved['outcomes'][0]['ctx_ns']['binding']['issue_clock'] == 10.
    assert audit.selected['zero_duration_outcomes'] == 1
    assert all(audit.checks.values())


def test_no_fresh_selected_logging_does_not_read_complete_outcomes(tmp_path):
    audit = ContextAudit(tmp_path,'no_fresh')
    row = SimpleNamespace(sequence=0,started_at=10.,ctx_s=context(0))
    class Forbidden:
        def __getattr__(self,name):
            raise AssertionError('no_fresh read outcome '+name)
    audit.record_packet([row],[Forbidden()],packet=1,processed_at=40.,
        replay=SimpleNamespace(context_outcomes=[]))
    audit.close()
    record = json.loads((tmp_path/'context_selected.jsonl').read_text())
    assert len(record['covariates']) == 1 and record['outcomes'] == []
