import pytest
from experiments.mcwm_closed_loop_service_shift_v21_20261022.shift_model import (
    ServiceShiftReadout, shift_targets,
)


def record(identifier='p',history=True):
    return dict(id=identifier,role='train',selected_for_training=True,
        available_at=20.,admitted_at=21.,line_id='7X',features=[1.,0.,0.,.1],
        service_history=dict(status='supported' if history else 'history_missing',
            mean_residual_scaled=.2 if history else None),
        service_targets=dict(arrival=[-1.,None,-2.],departure=[-1.,-2.,-3.]))


def test_shift_targets_preserve_independent_endpoint_censoring_and_hold_reference():
    low=dict(stations=['a','b','c'],arrival=[50.,100.,None],departure=[55.,105.,180.])
    high=dict(stations=['a','b','c'],arrival=[57.,104.,170.],departure=[60.,109.,None])
    target=shift_targets(low,high)
    assert target['arrival']==[-.5,-3.5,None]
    assert target['departure']==[-2.5,-3.5,None]


def test_missing_endpoint_does_not_drop_other_supervised_service_clocks():
    r=record(); model=ServiceShiftReadout(use_history=True)
    model.accept([r],21.); model.refit()
    p=model.predict(r['features'],r['service_history'])
    assert p['status']=='partially_supported' and p['arrival'][1] is None
    assert p['arrival'][0] is not None and p['departure'][2] is not None
    assert p['support_pairs']['arrival']==[1,0,1]
    assert model.predict([0.,1.,0.,.1],r['service_history'])['status']=='unsupported'


def test_fitting_rejects_heldout_future_release_and_duplicate_pairs():
    model=ServiceShiftReadout(); r=record()
    with pytest.raises(ValueError):model.accept([r],20.)
    r['role']='heldout'
    with pytest.raises(ValueError):model.accept([r],22.)
    r['role']='train'; model.accept([r],22.)
    with pytest.raises(ValueError):model.accept([r],22.)


def test_history_absence_is_unsupported_while_shared_baseline_still_uses_pair():
    r=record(history=False)
    models=[ServiceShiftReadout(use_history=x) for x in (False,True)]
    for model in models:model.accept([r],22.);model.refit()
    assert models[0].predict(r['features'],r['service_history'])['departure'][0] is not None
    assert models[1].predict(r['features'],r['service_history'])['status']=='unsupported'
