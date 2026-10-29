"""Group contributions must preserve the actual selected-record weights."""
import numpy as np

from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.aggregate import summarize_pressure


def test_forced_pressure_is_a_weighted_contribution_not_a_group_mean():
    payload=dict(gradient=np.array([[2.,-1.],[4.,-2.]]),
        forced_60=np.array([True,False]),single_support=np.array([True,False]),
        huber_clipped=np.zeros((2,2),bool),error_components={})
    metadata=[dict(line_id='A',started_at=1000),dict(line_id='B',started_at=9500)]
    result=summarize_pressure(payload,metadata,np.array([1,3]))
    weighted=result['exposure_weighted']
    np.testing.assert_allclose(weighted['forced60']['signed_theta_partial'],[.5,1.])
    np.testing.assert_allclose(weighted['choice']['signed_theta_partial'],[-.75,-1.5])
    np.testing.assert_allclose(weighted['all']['signed_theta_partial'],[-.25,-.5])
    assert weighted['forced60']['weight_share']==.25
    assert result['exposure_draws']==4 and not result['historical_gradient_reconstruction']
