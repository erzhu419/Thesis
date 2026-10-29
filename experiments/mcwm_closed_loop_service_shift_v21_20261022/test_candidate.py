"""History support and budget admission must control the new response fit."""
import numpy as np
import pytest

from experiments.mcwm_closed_loop_service_shift_v21_20261022.candidate import HistoryResponseReadout
from experiments.mcwm_closed_loop_state_response_v18_20261022.difference import DirectDifferenceReadout


def records():
    return [dict(id=f'train/{line}',line_id=line,role='train',selected_for_training=True,
        available_at=330.,admitted_at=600.,features=[float(i==j) for j in range(3)]+[.2*i],
        service_history=dict(status='supported',mean_residual_scaled=(i-1)*.5),
        outcomes={arm:dict(truth=dict(status='complete',dt=300.,wait=w,ride=r,cost=2*w+r))
            for arm,w,r in (('low',1000.+20*i,2000.-10*i),('high',1003.,1992.))})
        for i,line in enumerate(('7X','102X','705X'))]


def test_history5_uses_same_admitted_pairs_and_reconstructs_physical_cost():
    rows=records();model=HistoryResponseReadout();model.accept(rows,600.);model.refit()
    assert model.summary()['pairs']==model.summary()['fitted_pairs']==3
    assert model.summary()['outcome_labels']==6
    assert len(model.summary()['feature_names'])==5
    for row in rows:
        pred=model.predict(row['features'],row['service_history'])
        assert pred['status']=='supported' and pred['support_pairs']==1
        assert pred['gain']['cost']==2*pred['gain']['wait']+pred['gain']['ride']
        assert pred['gain']['dt']==0 and pred['fixed300_clock']==dict(low=300.,high=300.)


def test_missing_history_is_no_fitted_row_and_no_zero_covariate():
    rows=records();rows[0]['service_history']=dict(status='history_missing',mean_residual_scaled=None)
    model=HistoryResponseReadout();model.accept(rows,600.);model.refit()
    assert len(model.records)==3 and model.summary()['fitted_pairs']==2
    assert model.predict(rows[0]['features'],dict(status='supported',mean_residual_scaled=0.))['status']=='unsupported'
    pred=model.predict(rows[1]['features'],rows[0]['service_history'])
    assert pred['status']=='unsupported' and pred['features'] is None and pred['gain'] is None


@pytest.mark.parametrize('change',[dict(role='heldout'),dict(selected_for_training=False),dict(available_at=601.)])
def test_history_candidate_rejects_unadmitted_or_unreleased_feedback(change):
    row=records()[0];row.update(change)
    with pytest.raises(ValueError):HistoryResponseReadout().accept([row],600.)


def test_new_feature_does_not_change_existing_baseline_predictions():
    rows=records()
    baseline=DirectDifferenceReadout();baseline.accept(rows,600.);baseline.refit()
    before=[baseline.predict(r['features']) for r in rows]
    candidate=HistoryResponseReadout();candidate.accept(rows,600.);candidate.refit()
    assert before==[baseline.predict(r['features']) for r in rows]
    # Zero history is an available measured value; with identical columns' new
    # column all zero the old solve is retained, rather than treated missing.
    for r in rows:r['service_history']['mean_residual_scaled']=0.
    candidate=HistoryResponseReadout();candidate.accept(rows,600.);candidate.refit()
    for r,old in zip(rows,before):
        np.testing.assert_allclose(list(candidate.predict(r['features'],r['service_history'])['gain'].values()),
            list(old['gain'].values()),rtol=0,atol=1e-12)
