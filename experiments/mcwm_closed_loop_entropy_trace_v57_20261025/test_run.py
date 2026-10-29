"""The actual native M8 dispatcher must remain unchanged at lambda zero."""
from copy import deepcopy

from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.test_target import ready
from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import install_sync_target,target_state
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal,_state,_rng
from experiments.mcwm_closed_loop_entropy_trace_v57_20261025.run import lambda_zero_gate


def test_real_native_root_gate_retains_parameters_Adam_and_all_rng():
    learner,evaluator,rows=ready()
    learner=install_sync_target(learner,evaluator,rows,[1.]*270,lambda_trace=1.)
    state,private,rng=deepcopy(_state(learner)),target_state(learner),_rng()
    proof=lambda_zero_gate(learner,576202)
    assert proof['all_pass']
    assert _equal(state,_state(learner))
    assert _equal(private,target_state(learner))
    assert _equal(rng,_rng())
