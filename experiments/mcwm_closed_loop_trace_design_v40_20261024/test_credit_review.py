import numpy as np
from experiments.mcwm_closed_loop_trace_design_v40_20261024.credit_review import reward_visibility


def test_positive_credit_can_leave_float32_root_unchanged():
    result = reward_visibility([-40., -20.], [-20.], [1.e-14], 1, -1., -2., -19.)
    assert result['native_witness_target_changed']
    assert result['coefficient'] > 0 and result['ideal_linear_reward_delta'] != 0
    assert not result['fixed_weight_root_changed']
    assert result['new_training_labels'] == 0


def test_resolved_credit_changes_root_without_changing_bootstrap_or_ratio():
    result = reward_visibility([-40., -20.], [-20.], [1.], 1, -1., -2., -19.)
    assert result['fixed_weight_root_changed']
    assert np.isclose(result['root_delta'], -.9)
