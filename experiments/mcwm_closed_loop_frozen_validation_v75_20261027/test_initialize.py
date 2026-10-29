"""The new seeds must retain the original all-data IQL recipe and budget."""
from copy import deepcopy

import pytest

from experiments.mcwm_closed_loop_frozen_validation_v75_20261027.initialize import (
    SOURCE_CONFIG, checkpoint_progress, validate_template)


def template():
    return dict(artifact_schema='transit-offline-initializer-v7-v1',
        total_updates=50000, online_critic_updates=0,
        source_config=deepcopy(SOURCE_CONFIG), extra=dict(source_training_seed=6201))


def test_template_retains_all_original_hyperparameters_without_mutating_reference():
    value = template()
    result = validate_template(value)
    assert result == SOURCE_CONFIG and result is not value['source_config']
    for field, wrong in [('offline_objective', 'cql'), ('ensemble_size', 4),
                         ('reward_scale', 1.), ('offline_update_order', 'actor-before-critic')]:
        bad = template()
        bad['source_config'][field] = wrong
        with pytest.raises(ValueError, match='original50000-step'):
            validate_template(bad)


def test_template_cannot_use_continued_or_different_seed_checkpoint():
    for name, wrong in [('total_updates', 50456), ('online_critic_updates', 456)]:
        bad = template()
        bad[name] = wrong
        with pytest.raises(ValueError):
            validate_template(bad)
    bad = template()
    bad['extra']['source_training_seed'] = 6203
    with pytest.raises(ValueError):
        validate_template(bad)


def test_resume_budget_reaches_original_checkpoint_boundaries_without_extra_updates():
    assert checkpoint_progress(0) == 10000
    assert checkpoint_progress(10000) == 10000
    assert checkpoint_progress(49999) == 1
    assert checkpoint_progress(50000) == 0
    for invalid in [-1, 50001, 10000.5]:
        with pytest.raises(ValueError):
            checkpoint_progress(invalid)
