"""Catch stale candidate weights and incorrect admission-based coverage."""
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import make_trainer, table
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.probes import ZeroInitProbe, coverage


def test_parity_uses_current_packet_models_without_advancing_optimizer_or_rng():
    learner = make_trainer('fresh')
    data = table(.72, count=2)
    rows = [SimpleNamespace(s=data['s'][i], ns=data['ns'][i], a=data['a'][i],
        memory_s={'vector': [1., 0., .5]}, memory_ns={'vector': [1., 1., .25]}) for i in range(2)]
    probes = ZeroInitProbe(learner, 3)
    # Simulate main learning after construction. A stale initializer copy
    # would now disagree; the probe must use this packet's exact parameters.
    with torch.no_grad():
        next(learner.critic.parameters()).add_(.25)
        next(learner.target_critic.parameters()).add_(.125)
    before = learner.total_updates
    rng = torch.get_rng_state().clone()
    probes.observe(learner, rows)
    assert probes.summary()['all_pass']
    assert probes.summary()['full_head_row_comparisons'] == 8
    assert learner.total_updates == before
    assert torch.equal(rng, torch.get_rng_state())


def test_coverage_retains_missing_context_and_counts_only_admitted_rows():
    # A whole native day may contain many more records; only this passed-in
    # packet contributes. A missing context must not be counted as available.
    context = {'line_ids': ['102X', '7X'], 'vector': [0.]*7+[1., .02, 1., 1., 0., .1, 0.]}
    next_context = {'line_ids': ['102X', '7X'], 'vector': [0.]*7+[1., .02, 1., 0., 1., .2, .05]}
    good = SimpleNamespace(line_id='7X', started_at=9500, memory_s=context, memory_ns=next_context)
    missing = SimpleNamespace(line_id='102X', started_at=1000, memory_s=None, memory_ns=None)
    result = coverage([good, missing])
    assert result['admitted_rows'] == 2
    assert result['rows_with_both_contexts'] == 1
    assert result['by_line'] == {'7X': [1, 1, 1, 1, 1, 0, 1, 1]}
    assert result['by_line_window'] == {'7X/9000-12000': [1, 1, 1, 1, 1, 0, 1, 1]}
