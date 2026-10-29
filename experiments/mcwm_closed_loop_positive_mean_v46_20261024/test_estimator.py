"""Detect native-parity drift, accidental estimator changes and frozen mutation."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.test_frozen import learner
from experiments.mcwm_closed_loop_positive_mean_v46_20261024.estimator import (
    PART_FIELDS, score_row_block,
)
from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_probe import chain
from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import _trace_row
from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


SEEDS = tuple(range(336202, 336210))


@pytest.mark.parametrize('seed', (336202, 336203, 336204))
def test_m1_and_lambda_zero_preserve_all_old_native_components_and_frozen_state(seed):
    model, (first, second) = learner(), chain()
    parameter = next(iter(model.critic.parameters()))
    model.critic_optimizer.state[parameter] = dict(step=torch.tensor(2.),
        exp_avg=torch.full_like(parameter, .2), exp_avg_sq=torch.full_like(parameter, .3))
    saved, rng = deepcopy(_state(model)), _rng()
    reference = _trace_row(model, first, second, seed)
    result = score_row_block(model, first, second,
        draw_seeds=tuple(range(seed, seed+8)), cached_m1=reference)
    assert result['integrity']['all_pass'] and result['first_m1_native_exact']
    assert result['new_training_updates'] == result['new_native_branches'] == 0
    for field in PART_FIELDS:
        assert result['m1'][field] == reference[field]
    for single in result['singles']:
        native = _trace_row(model, first, second, single['draw_seed'])
        for field in PART_FIELDS:
            # Both heads are always selected; fixing their order only changes
            # the descriptive list, never the scalar target or matching Q.
            if field != 'target_subset':
                assert single[field] == native[field]
        trace = decompose_trace([single], [], 0.)
        assert trace['lambda_zero_native_exact']
        assert np.float32(trace['targets'][0]).view(np.uint32) == np.float32(native['native_target']).view(np.uint32)
    assert _equal(saved, _state(model)) and _equal(rng, _rng())


def test_m8_changes_only_positive_joint_reduction_and_keeps_legal_trace_baseline():
    model, (first, second) = learner(), chain()
    result = score_row_block(model, first, second, draw_seeds=SEEDS)
    m1, m8, singles = result['m1'], result['m8'], result['singles']
    joint = torch.tensor([row['positive_joint'] for row in singles], dtype=torch.float32).mean()
    assert m8['positive_joint'] == float(joint)
    assert len({row['positive_joint'] for row in singles}) > 1
    for field in ('next_value', 'reward_scaled', 'value_contribution', 'target_subset',
                  'recorded_next_target_q', 'zero_joint', 'zero_q',
                  'zero_probability', 'hold_probability'):
        assert all(row[field] == m1[field] for row in singles+[m8])
    cfg = result['target_config']
    assert cfg['matching_q_slot'] == 0 and cfg['native_two_action_batch']
    assert m1['recorded_next_target_q'] == min(cfg['matching_q_heads'])
    assert cfg['F_unchanged'] and cfg['analytic_gate_unchanged']
    assert cfg['first_zero_branch_unchanged'] and cfg['joint_positive_mean']
    for row, detail in zip(singles, result['draw_details']):
        expected = np.float32(np.float32(detail['positive_q'])
            -np.float32(float(model.alpha))*np.float32(detail['positive_log_prob']))
        assert np.float32(row['positive_joint']).view(np.uint32) == expected.view(np.uint32)
    ending = score_row_block(model, second, None, draw_seeds=SEEDS)
    assert ending['m8']['recorded_next_target_q'] is None
    assert ending['m8']['baseline_missing_reason'] == 'no_next_real_recorded_action'
    for arm in ('m1', 'm8'):
        trace = decompose_trace([result[arm], ending[arm]], [.7], 1.)
        assert trace['root']['trace_target'] is not None
    assert m8['native_target_exact'] is False


def test_true_terminal_cannot_gain_bootstrap_or_successor_q_from_mean():
    model, (first, second) = learner(), chain()
    first['terminal'] = True
    result = score_row_block(model, first, second, draw_seeds=SEEDS)
    for row in result['singles']+[result['m8']]:
        assert row['bootstrap_contribution'] == 0.
        assert row['native_target'] == row['reward_scaled']
        assert row['recorded_next_target_q'] is None
        assert row['baseline_missing_reason'] == 'terminal_transition'


def test_bad_reference_or_noncausal_memory_stops_before_mutation():
    model, (first, second) = learner(), chain()
    reference = _trace_row(model, first, second, SEEDS[0])
    reference['native_target'] += .01
    saved, rng = deepcopy(_state(model)), _rng()
    with pytest.raises(RuntimeError, match='resident v44'):
        score_row_block(model, first, second, draw_seeds=SEEDS, cached_m1=reference)
    assert _equal(saved, _state(model)) and _equal(rng, _rng())
    second['memory_s']['vector'][0] += .01
    with pytest.raises(RuntimeError, match='real contiguous'):
        score_row_block(model, first, second, draw_seeds=SEEDS)
