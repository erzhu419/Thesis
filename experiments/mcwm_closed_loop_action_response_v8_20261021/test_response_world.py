"""Detect extra labels, future leakage and passive-sidecar changes to legacy training."""

from pathlib import Path
import copy
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))

from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalIntervalCalibration
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_local_world import source_rows
from experiments.mcwm_closed_loop_action_local_v3_20261020.test_probes import assert_same
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import _learner_state, rng_state
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.test_world import issue_and_row
from experiments.mcwm_closed_loop_paired_world_v6_20261020.test_paired_world import paired_learner
from experiments.mcwm_closed_loop_action_response_v8_20261021.response_world import ResponseWorld
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7


@pytest.fixture
def sidecar():
    learner = paired_learner()
    physical = LocalIntervalCalibration(source_rows(), TransitMDPConfigV7())
    return learner, ResponseWorld(learner.world, physical)


def issue(sequence=7, *, vehicle=None):
    event, row = issue_and_row(sequence)
    event.update(vehicle_id=vehicle or f"bus{sequence}", trip_id=f"trip{sequence}",
                 event=np.asarray(source_rows()[0]["event"], dtype=np.float32))
    return event, row


def register(response, event):
    response.memory.observe_event(event)
    context = response.context(event)
    response.register(event, context)
    return context


def test_only_selected_packet_labels_fit_while_all_observed_commands_supply_memory(sidecar):
    learner, response = sidecar
    events_and_rows = [issue(sequence) for sequence in (7, 8, 9)]
    for event, row in events_and_rows:
        register(response, event)
    replay_received = learner.replay_buffer.target_received
    response.accept([events_and_rows[0][1]], [8], 9031.)
    assert response.labels == len(response.records) == 1
    assert response.records[0]["row"].sequence == 7
    assert set(response.pending) == {9}
    assert len(response.memory.pending) == 3
    assert learner.replay_buffer.target_received == replay_received
    response.refit()
    assert response.refits == 1 and response.labels == 1
    assert any(torch.count_nonzero(weights) for weights in response.readouts.values())
    assert all(not torch.count_nonzero(weights) for weights in response.frozen_readouts.values())


def test_unreleased_selected_label_cannot_enter_fit_or_consume_its_saved_issue(sidecar):
    _learner, response = sidecar
    event, row = issue()
    register(response, event)
    with pytest.raises(ValueError, match="before its feedback release"):
        response.accept([row], [], row.available_at-1)
    assert response.labels == 0 and not response.records
    assert row.sequence in response.pending
    assert all(not torch.count_nonzero(weights) for weights in response.readouts.values())
    response.accept([row], [], row.available_at)
    assert response.labels == 1


def test_selected_label_without_issue_or_with_changed_executed_action_is_rejected(sidecar):
    _learner, response = sidecar
    event, row = issue()
    with pytest.raises(ValueError, match="lacks its issue-time memory"):
        response.accept([row], [], row.available_at)
    register(response, event)
    changed = copy.deepcopy(row)
    changed.a[0] += .1
    with pytest.raises(ValueError, match="saved executed issue"):
        response.accept([changed], [], changed.available_at)
    assert response.labels == 0 and not response.records


def test_fitting_nonzero_candidate_preserves_legacy_models_optimizers_and_all_rng(sidecar):
    learner, response = sidecar
    event, row = issue()
    register(response, event)
    response.accept([row], [], row.available_at)
    before = copy.deepcopy(_learner_state(learner))
    global_before = rng_state()
    frozen_before = {name: copy.deepcopy(model.state_dict())
                     for name, model in learner.world.frozen_models.items()}
    twin_before = copy.deepcopy(learner.world.twin_trust.net.state_dict())
    physical_before = (response.physical.online_weights.copy(),
                       copy.deepcopy(response.physical.context), response.physical.parameter_version)
    response.refit()
    assert any(torch.count_nonzero(weights) for weights in response.readouts.values())
    assert_same(_learner_state(learner), before)
    assert_same(rng_state(), global_before)
    for name, model in learner.world.frozen_models.items():
        assert_same(model.state_dict(), frozen_before[name])
    assert_same(learner.world.twin_trust.net.state_dict(), twin_before)
    np.testing.assert_array_equal(response.physical.online_weights, physical_before[0])
    assert response.physical.context == physical_before[1]
    assert response.physical.parameter_version == physical_before[2]


def test_issue_features_and_nonholding_prediction_are_not_rewritten_by_later_observations(sidecar):
    _learner, response = sidecar
    prior, _ = issue(6)
    register(response, prior)
    event, row = issue()
    context = register(response, event)
    original = context["memory_features"].copy()
    nonholding = context["nonholding_seconds"]
    assert original.any() and nonholding > 0
    response.physical.online_weights += 30.
    response.physical.context["7X"] += 100.
    response.memory.observe_event(prior)
    response.memory.observe_event(event)
    assert not np.array_equal(response.memory.features(event["started_at"]), original)
    assert response.context(event)["nonholding_seconds"] != nonholding
    response.accept([row], [], row.available_at)
    record = response.records[0]
    np.testing.assert_array_equal(record["memory_features"], original)
    assert record["nonholding_seconds"] == nonholding
    response.refit()
    np.testing.assert_array_equal(record["memory_features"], original)
    assert record["nonholding_seconds"] == nonholding
