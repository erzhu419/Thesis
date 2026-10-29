"""Real frozen target parity, paired CRN and resident-model reuse without SGD."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024 import frozen
from experiments.mcwm_closed_loop_paired_recovery_v37r1_20261024.test_run import heldout, transition
from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import attach_shared_level
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.test_learning import initial
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import freeze
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _state


def learner():
    value = attach_shared_level(initial())
    with torch.no_grad():
        value.critic.level_weight.fill_(.01)
        value.critic.action_weight[0].fill_(.3)
        value.target_critic.level_weight.fill_(.02)
        value.target_critic.action_weight[1].fill_(.2)
    return freeze(value)


def test_real_chain_scores_actual_action_delayed_reward_target_and_keeps_model_rng():
    model = learner()
    first = transition(1, 18400., arm='low')
    second = deepcopy(first)
    second.update(sequence=2, a=[-.9375], started_at=18400., completed_at=18523.,
                  available_at=18553., dt=123., r=-70.)
    second['memory_ns']['observed_at'] = 18523.
    second['ns'][0] = 18523./24000.
    before, rng = deepcopy(_state(model)), _rng()
    result = frozen.score_chain(model, [dict(transition=first), dict(transition=second)])
    low, high = result['rows']
    assert result['integrity']['all_pass'] and result['synthetic300_state'] is False
    assert len(result['rows']) == 2 and high['completed_at'] == 18523.
    assert low['dt'] == 0. and not low['terminal'] and low['reward_scaled'] == 0.
    assert high['holding_seconds'] == 1.875  # Saved actual action is not passed through F again.
    assert high['reward_scaled'] == pytest.approx(-.007)
    for score in result['rows']:
        assert score['target'] == pytest.approx(score['reward_scaled']+score['bootstrap_contribution'])
        assert score['bootstrap_contribution'] == pytest.approx(score['value_contribution']+score['sac_contribution'])
        assert score['td_min'] == score['target']-score['q_min']
        assert score['target_lag_min'] == score['target_q_min']-score['q_min']
        assert len(score['q_heads']) == len(score['target_q_heads']) == 2
    assert _equal(before, _state(model)) and _equal(rng, _rng())


def test_single_row_crn_target_equal_for_identical_next_inputs_and_parameter_mutation_detected():
    model = learner()
    low = transition(1, 18400., arm='low')
    high = deepcopy(low)
    high.update(sequence=2, a=[-.9375])
    scores = frozen.score_chain(model, [low, high])['rows']
    assert scores[0]['target'] == scores[1]['target']
    assert scores[0]['target_subset'] == scores[1]['target_subset']
    before, rng = frozen.frozen_state(model), _rng()
    with torch.no_grad():
        model.critic.action_weight.add_(.01)
    with pytest.raises(RuntimeError, match='frozen_parameters_and_versions'):
        frozen.unchanged(model, before, rng)


def test_old_four_issue_gate_compares_full_q_variants_and_common_random_targets():
    model, inputs = learner(), heldout()
    saved = [dict(slot=value['slot'], **frozen.pair_prediction(model,
        {arm: frozen.row_from_record(row) for arm, row in value['paired_transitions'].items()}, seed=336202))
        for value in inputs]
    proof = frozen.verify_old(model, inputs, saved, seed=336202)
    assert proof['old_issues'] == 4 and proof['common_random_number_targets_exact']
    damaged = deepcopy(saved)
    damaged[0]['bellman_by_action']['high']['target'] += .01
    with pytest.raises(RuntimeError, match='bellman_by_action'):
        frozen.verify_old(model, inputs, damaged, seed=336202)


def test_load_frozen_uses_saved_three_models_and_never_reconstructs_updates(tmp_path, monkeypatch):
    model_root, source_root = tmp_path/'models', tmp_path/'source'
    inputs = heldout()
    native = source_root/'runs'/'collection'/'stable'
    native.mkdir(parents=True)
    (native/'heldout_prediction_inputs.json').write_text(json.dumps(inputs))
    for arm in frozen.ARMS:
        model = learner()
        model.memory_arm = arm
        model.total_updates, model.replay_buffer.target_received = 50456, 285
        root = model_root/'stable'/arm
        root.mkdir(parents=True)
        torch.save(dict(learner=model, seed=6202, scenario='stable', arm=arm,
            contract=dict(horizon_seconds=24000), stations=[1, 2], factual_proof=dict(all_pass=True)),
            root/'final_model.pt')
        (root/'summary.json').write_text(json.dumps(dict(integrity=dict(all_pass=True),
            budget=dict(feedback_slots=285, shadow_updates=456))))
        saved = [dict(slot=value['slot'], **frozen.pair_prediction(model,
            {name: frozen.row_from_record(row) for name, row in value['paired_transitions'].items()}, seed=336202))
            for value in inputs]
        (root/'heldout.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in saved))

    def forbidden_update(*args, **kwargs):
        raise AssertionError('Frozen reuse cannot execute SGD')

    monkeypatch.setattr(type(model), 'continue_step', forbidden_update)
    before = _rng()
    models, contract, stations, proof = frozen.load_frozen('stable', model_root, source_root)
    assert set(models) == set(frozen.ARMS)
    assert contract['horizon_seconds'] == 24000 and stations == [1, 2]
    assert proof['all_pass'] and proof['new_updates'] == proof['new_feedback'] == 0
    assert proof['checkpoints_downloaded'] == 0 and _equal(before, _rng())
