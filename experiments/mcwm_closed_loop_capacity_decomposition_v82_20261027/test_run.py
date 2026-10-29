"""The common bank is fixed by source identity and time, never by values/cost."""
import numpy as np

from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.run import common_bank


def row(sequence,clock,line='L'):
    return dict(sequence=sequence,started_at=clock,line_id=line,s=np.zeros(229,np.float32))


def test_common_bank_keeps_equal_state_cross_arm_origins_and_first_real_tail_issue():
    a=row(1,1200.)
    runs={arm:dict(covariates=[a.copy()],issues=[a.copy(),row(2,18300.),
        row(3,18301.),row(4,18600.),row(5,24000.)]) for arm in ('fresh','no_fresh')}
    bank,metadata=common_bank(runs,[18300,24000],300)
    assert len(bank)==6
    assert [(r['origin_arm'],r['sequence'],r['bank']) for r in metadata]==[
        ('fresh',1,'covariate'),('fresh',2,'tail'),('fresh',4,'tail'),
        ('no_fresh',1,'covariate'),('no_fresh',2,'tail'),('no_fresh',4,'tail')]


def test_tail_bins_are_line_specific_and_empty_bins_are_not_fabricated():
    runs={arm:dict(covariates=[],issues=[row(2,18300.,'L'),row(3,18301.,'M'),
        row(4,18900.,'M')]) for arm in ('fresh','no_fresh')}
    bank,_=common_bank(runs,[18300,24000],300)
    assert len(bank)==6
    assert all(r['started_at'] != 18600 for r in bank)
