"""Detect new-seed initializer misrouting or a change to the frozen v73 runtime."""
import ast
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026 import learning as frozen
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import learning as v
from experiments.mcwm_closed_loop_frozen_validation_v75_20261027 import preflight
from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026 import preflight as old_preflight
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.test_learning import table
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    shared, isolated = tmp_path/'shared', tmp_path/'v75'
    shared.mkdir()
    (isolated/'inputs').mkdir(parents=True)
    metadata = dict(contract=asdict(TransitMDPConfigV7()), station_max_by_line=[40]*12)
    (shared/'history_metadata.json').write_text(json.dumps(metadata))
    np.savez(shared/'history.npz', **table(.1))
    np.savez(shared/'source_9101.npz', **table(.4))
    config = v.TransitTrainerConfig(hidden_dim=8, ensemble_size=2, batch_size=256,
        cql_n_actions=10, online_backup='h2oplus', mixed_backup_lambda=.1)
    payload = dict(total_updates=50000, online_critic_updates=0,
        extra=dict(source_training_seed=6206), source_config=asdict(config))
    path = isolated/'inputs/h2oplus_seed6206.pt'
    torch.save(payload, path)
    monkeypatch.setattr(v, 'SHARED_INPUTS', shared)
    monkeypatch.setattr(v, 'HERE', isolated)
    return shared, isolated, path, payload


def test_frozen_implementation_objects_are_reused_without_math_changes():
    assert v.DecoupledReplay is frozen.DecoupledReplay
    assert v.FreshOutcomeReplay is frozen.FreshOutcomeReplay
    assert v.ExecutedCQLTrainer is frozen.ExecutedCQLTrainer
    assert v.ExecutionAction is frozen.ExecutionAction
    assert (v.cql_checks, v.recency_checks, v.target_checks) == (
        frozen.cql_checks, frozen.recency_checks, frozen.target_checks)
    assert preflight.native_preflight is old_preflight.native_preflight


def test_both_arms_load_same_new_seed_archive_and_preserve_shared_global(inputs, monkeypatch):
    shared, isolated, initializer, payload = inputs
    resident_path = v.native.resident.INPUTS
    calls = []

    class Learner:
        def __init__(self, replay, **kwargs):
            self.replay_buffer = replay
            self.config, self.seed = kwargs['config'], kwargs['seed']
            self.original_weights = replay.history_weights.copy()
            calls.append(('construct', replay.arm, self.seed))

        def load_offline_initializer(self, path):
            calls.append(('restore', self.replay_buffer.arm, path))

        def activate_executed_cql(self):
            calls.append(('cql', self.replay_buffer.arm))

    monkeypatch.setattr(v, 'ExecutedCQLTrainer', Learner)
    monkeypatch.setattr(v, 'set_cpu_optimizer_runtime', lambda learner:
        calls.append(('cpu', learner.replay_buffer.arm)))
    configs = []
    for arm in ('fresh', 'no_fresh'):
        learner, replay, contract, stations, path = v.load_learner(6206, arm=arm)
        assert path == initializer
        assert type(replay) is frozen.DecoupledReplay
        assert learner.replay_buffer is replay
        assert np.array_equal(learner.original_weights, replay.history_weights)
        assert replay.outcome_seed == 606206
        configs.append(asdict(learner.config))
    assert configs[0] == configs[1] == dict(payload['source_config'], use_ratio_correction=False)
    assert calls == [(what, arm, *extra) for arm in ('fresh', 'no_fresh')
        for what, extra in [('construct', [6206]), ('restore', [initializer]), ('cpu', []), ('cql', [])]]
    assert v.native.resident.INPUTS == resident_path


@pytest.mark.parametrize('field,value', [('total_updates', 49999),
    ('online_critic_updates', 1), ('source_training_seed', 6207)])
def test_partial_online_or_wrong_seed_initializer_is_rejected(inputs, monkeypatch, field, value):
    _, _, path, payload = inputs
    (payload['extra'] if field == 'source_training_seed' else payload)[field] = value
    torch.save(payload, path)
    monkeypatch.setattr(v, 'ExecutedCQLTrainer', lambda *a, **kw:
        pytest.fail('Invalid archive reached trainer construction'))
    with pytest.raises(ValueError, match='matching new50000'):
        v.load_learner(6206, arm='fresh')


def test_native_control_loop_is_frozen_v73():
    here = Path(__file__).parent
    old = here.parent/'mcwm_closed_loop_decoupled_feedback_v73_20261026/control.py'
    functions = []
    for path in (old, here/'control.py'):
        function = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'run')
        # Only the version name in the terminal integrity error changed.
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace('v73 decoupled-feedback', 'v75 frozen-v73')
        functions.append(ast.dump(function, include_attributes=False))
    assert functions[0] == functions[1]


def test_generated_protocol_keeps_frozen_method_and_separate_initialization_budget():
    from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.control import protocol
    from experiments.mcwm_closed_loop_decoupled_feedback_v73_20261026.control import protocol as old_protocol
    p, old = protocol(), old_protocol()
    assert p['training_seeds'] == [6206, 6207, 6208]
    for key in ('runtime_contract', 'actor_schedule', 'feedback_selection', 'recipe',
                'fresh_target', 'data_permissions', 'data_rights'):
        assert p[key] == old[key]
    compute = p['compute_budget']
    assert compute['new_main_days'] == 12
    assert compute['fresh_main_days'] == compute['no_fresh_main_days'] == 6
    assert compute['new_main_sim_seconds'] == 288000
    assert compute['covariate_slots'] == 3420
    assert compute['fresh_supervised_outcomes'] == 1710
    assert compute['updates_per_optimizer'] == 5472
    assert compute['initializer_retraining_updates'] == 150000
    assert compute['initializer_updates_per_seed'] == p['initializer_updates'] == 50000
    assert compute['cached_comparison_days'] == 0
    assert 'no_fresh_regression_days' not in compute
    assert p['initializer_directory'].endswith('mcwm_closed_loop_frozen_validation_v75_20261027/inputs')
    assert p['old_v73_gate_conclusion'] == 'closed: two stable tails failed'
