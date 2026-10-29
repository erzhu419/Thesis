"""Native diagonal parity and isolation of the two frozen sampling factors."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.test_frozen import learner
from experiments.mcwm_closed_loop_trace_support_v39_20261024.test_probe import chain
from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import _trace_row
from experiments.mcwm_closed_loop_trace_target_v44_20261024.trace import decompose_trace
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state
from experiments.mcwm_closed_loop_trace_components_v45_20261024.factor import SEEDS, factorize_row


def test_three_native_diagonals_match_v44_and_preserve_model_adam_rng():
    model, (first, second) = learner(), chain()
    # Populated Adam moments must survive the frozen read just as empty state does.
    parameter = next(iter(model.critic.parameters()))
    model.critic_optimizer.state[parameter] = dict(step=torch.tensor(2.),
        exp_avg=torch.full_like(parameter, .2), exp_avg_sq=torch.full_like(parameter, .3))
    saved, rng = deepcopy(_state(model)), _rng()
    references = {str(seed): _trace_row(model, first, second, seed) for seed in SEEDS}
    result = factorize_row(model, first, second, cached_diagonal=references)
    assert result['diagonal_exact'] and result['integrity']['all_pass']
    assert result['new_training_updates'] == result['new_native_branches'] == 0
    for seed in SEEDS:
        actual = result['cells'][f'{seed}/{seed}']
        expected = references[str(seed)]
        for field in ('native_target', 'recorded_next_target_q', 'target_subset',
                      'next_value', 'sac_q_without_entropy', 'sac_entropy_bonus'):
            assert actual[field] == expected[field]
        detail = actual['factor_detail']
        assert detail['matching_q_slot'] == 0 and detail['native_two_action_batch']
        assert actual['recorded_next_target_q'] == min(detail['matching_q_heads'])
        assert detail['matching_q_heads'] == expected['subset_statistics']['selected_heads']
    assert _equal(saved, _state(model)) and _equal(rng, _rng())


def test_cartesian_actor_draw_and_native_subset_are_independent_factors():
    model, (first, second) = learner(), chain()
    result = factorize_row(model, first, second)
    for actor_seed in SEEDS:
        rows = [result['cells'][f'{actor_seed}/{seed}'] for seed in SEEDS]
        assert all(row['factor_detail']['actor_components'] == rows[0]['factor_detail']['actor_components']
                   for row in rows)
        assert all(row['factor_detail']['executed_target_actions'] == rows[0]['factor_detail']['executed_target_actions']
                   for row in rows)
    for head_seed in SEEDS:
        rows = [result['cells'][f'{seed}/{head_seed}'] for seed in SEEDS]
        assert all(row['target_subset'] == rows[0]['target_subset'] for row in rows)
        assert all(row['recorded_next_target_q'] == rows[0]['recorded_next_target_q'] for row in rows)
    cells = list(result['cells'].values())
    assert len({row['next_value'] for row in cells}) == 1
    assert len({row['factor_detail']['actor_components']['positive_action'][0][0] for row in cells}) > 1
    for row in cells:
        detail = row['factor_detail']
        reconstructed = (row['value_contribution']+detail['sac_q_contribution']
                         +detail['entropy_contribution']+detail['bootstrap_component_rounding'])
        assert reconstructed == pytest.approx(row['bootstrap_contribution'], abs=1e-12)


def test_factor_rows_remain_legal_trace_inputs_and_lambda_zero_is_native_exact():
    model, (first, second) = learner(), chain()
    beginning = factorize_row(model, first, second)
    ending = factorize_row(model, second, None)
    for key, initial in beginning['cells'].items():
        final = ending['cells'][key]
        assert final['recorded_next_target_q'] is None
        assert final['baseline_missing_reason'] == 'no_next_real_recorded_action'
        assert decompose_trace([initial, final], [.7], 0.)['lambda_zero_native_exact']
        assert decompose_trace([initial, final], [.7], 1.)['root']['trace_target'] is not None


def test_bad_cached_diagonal_or_noncausal_memory_is_reported_without_mutation():
    model, (first, second) = learner(), chain()
    cached = {str(seed): _trace_row(model, first, second, seed) for seed in SEEDS}
    cached[str(SEEDS[0])]['native_target'] += .01
    saved, rng = deepcopy(_state(model)), _rng()
    with pytest.raises(RuntimeError, match='resident v44'):
        factorize_row(model, first, second, cached_diagonal=cached)
    assert _equal(saved, _state(model)) and _equal(rng, _rng())
    second['memory_s']['vector'][0] += .01
    with pytest.raises(RuntimeError, match='real contiguous'):
        factorize_row(model, first, second)
