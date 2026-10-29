"""Reuse native context/Adam receipts and count the actual singleton TD path."""
from experiments.mcwm_closed_loop_capacity_control_v81_20261027.context_audit import (
    ContextAudit as OriginalContextAudit, FirstPacketAudit as OriginalFirstPacketAudit,
    native_preflight as original_preflight)
from experiments.mcwm_closed_loop_support_td_v83_20261027.learning import singleton_td_audit


def td_gate(learner, *, expected_calls):
    """Counters observe the paid forward path; no extra forward/backward/update."""
    value=singleton_td_audit(learner)
    checks=dict(value['checks'], exact_formal_TD_calls=value['calls']==expected_calls,
        exact_formal_TD_rows=value['rows']==expected_calls*learner.config.batch_size,
        no_fresh_TD_context_isolation=learner.replay_buffer.arm=='fresh' or value['observed_rows']==0,
        fresh_TD_context_quota=value['observed_rows']<=expected_calls*(learner.config.batch_size//4))
    return dict(value,all_pass=all(checks.values()),checks=checks,
        scope='Actual paid grad-enabled TD forwards only; no extra group backward or optimizer step')


class FirstPacketAudit(OriginalFirstPacketAudit):
    def finish(self,out):
        result=super().finish(out)
        gate=td_gate(self.learner,expected_calls=8)
        result['td_gradient_gate']=gate
        result['checks']['singleton_TD_theta_gate']=gate['all_pass']
        result['all_pass']=all(result['checks'].values())
        return result


class ContextAudit(OriginalContextAudit):
    def summary(self,replay,issued_count,*,learner):
        result=super().summary(replay,issued_count)
        gate=td_gate(learner,expected_calls=456)
        result['td_gradient_gate']=gate
        result['integrity']['checks']['singleton_TD_theta_gate']=gate['all_pass']
        result['integrity']['all_pass']=all(result['integrity']['checks'].values())
        return result


def native_preflight(*args,context,first_packet,**kwargs):
    result=original_preflight(*args,context=context,first_packet=first_packet,**kwargs)
    result['td_gradient_gate']=first_packet['td_gradient_gate']
    result['checks']['singleton_TD_theta_gate']=result['td_gradient_gate']['all_pass']
    result['technical_gate_pass']=all(result['checks'].values())
    return result
