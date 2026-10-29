"""No-fresh outcome isolation and correct historical/current memory alignment."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_matched_response_v59_20261025.learning import MatchedOutcomeReplay
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7


class CovariatesOnly:
    s = np.zeros(229, np.float32)
    a = np.asarray([-.75], np.float32)
    memory_s = dict(vector=[.75]*84)

    def __getattr__(self, name):
        raise AssertionError('No-fresh accessed a fresh outcome: '+name)


def replay(arm):
    return MatchedOutcomeReplay(table(.1), table(.2), TransitMDPConfigV7(), [3], arm=arm)


def test_no_fresh_never_reads_or_stores_outcomes():
    value = replay('no_fresh')
    value.add_target([CovariatesOnly()])
    assert set(value.recent[0]) == {'s','a','memory_s'}
    historical, source = value.sample_mixed(256,.5,rng=np.random.default_rng(42))
    assert historical['observations'].shape == source['observations'].shape == (128,313)
    assert torch.count_nonzero(historical['observations'][:,229:]) == 0
    assert torch.count_nonzero(historical['next_observations'][:,229:]) == 0
    assert torch.all(value.last_covariates['memory_s'][64:] == .75)


def test_fresh_memory_follows_its_own_transition_and_sampler_draws_match():
    fresh, control = replay('fresh'), replay('no_fresh')
    row=SimpleNamespace(s=np.zeros(229,np.float32), a=np.asarray([-.75],np.float32),
        ns=np.ones(229,np.float32), r=-4.,dt=1.,terminal=False,
        memory_s=dict(vector=[.25]*84),memory_ns=dict(vector=[.5]*84))
    fresh.add_target([row]);control.add_target([row])
    fr, fs=fresh.sample_mixed(256,.5,rng=np.random.default_rng(42))
    nr, ns=control.sample_mixed(256,.5,rng=np.random.default_rng(42))
    assert fresh.last_sampling == control.last_sampling
    assert torch.all(fr['observations'][64:,229:] == .25)
    assert torch.all(fr['next_observations'][64:,229:] == .5)
    assert torch.equal(fs['observations'], ns['observations'])
    assert torch.count_nonzero(nr['next_observations'][:,229:]) == 0
