"""Detect loss of lambda-zero identity, native recurrence or future-only credit."""
from copy import deepcopy
from unittest.mock import patch

import torch

from experiments.mcwm_closed_loop_entropy_trace_v57_20261025.target import (
    ETA, frozen_block, physical_trace, trace_target,
)


def test_lambda_zero_returns_root_object_without_evaluating_future():
    root = torch.tensor([3., -4.])
    def forbidden():
        raise AssertionError('lambda=0 must not construct future arrays')
    for beta in (0., 1.):
        assert trace_target(root, forbidden, lambda_trace=0., beta=beta) is root


def test_beta_zero_is_bitwise_original_native_recurrence():
    torch.manual_seed(99)
    native = torch.randn(271)*41.
    q = torch.randn(270)*31.
    coefficients = torch.rand(270)
    coefficients[7] = 0.
    entropy = torch.randn(271)
    old = native.clone()
    for index in range(269, -1, -1):
        if coefficients[index] != 0:
            old[index] = native[index]+coefficients[index]*(old[index+1]-q[index])
    actual = physical_trace(native, q, coefficients, entropy, beta=0.)
    assert torch.equal(actual, old)
    expected = native+native.new_tensor(ETA)*(old-native)
    future = dict(native=native, matching_q=q, coefficients=coefficients, entropy=entropy)
    assert torch.equal(trace_target(native, future, beta=0.), expected)


def test_future_only_entropy_keeps_root_and_last_bootstrap_and_breaks_at_zero_link():
    native = torch.tensor([10., 20., 30., 40.])
    q = torch.tensor([3., 4., 5.])
    coefficients = torch.tensor([1., 0., .5])
    entropy = torch.tensor([999., 2., 3., 7.])
    actual = physical_trace(native, q, coefficients, entropy, beta=1.)
    assert torch.equal(actual, torch.tensor([25., 20., 44., 40.]))
    changed = entropy.clone()
    changed[0] = -999.
    assert torch.equal(actual, physical_trace(native, q, coefficients, changed, beta=1.))
    assert actual[-1] == native[-1]
    assert actual[1] == native[1]


def test_frozen_native_m8_arrays_keep_rng_private_state_and_correct_entropy_accounting():
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.test_target import ready
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import (
        ReferenceSyncMean8Trainer, install_sync_target, target_state,
    )
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _state
    model, evaluator, rows = ready()
    model = install_sync_target(model, evaluator, rows, [1.]*270, lambda_trace=1.)
    state, private, rng = deepcopy(_state(model)), target_state(model), torch.get_rng_state().clone()
    block = frozen_block(model, range(576202, 576210))
    assert _equal(state, _state(model))
    assert _equal(private, target_state(model))
    assert torch.equal(rng, torch.get_rng_state())
    expected = block['reward']+block['discount']*(.1*block['value']+.9*block['sac'])
    assert torch.equal(block['native'], expected)
    # Independent original dispatcher on the same eight frozen actor draws:
    # this catches changed first-zero/gate retention or Q-minus-logpi averaging.
    reference = deepcopy(model)
    parts = []
    try:
        for seed in range(576202, 576210):
            torch.manual_seed(seed)
            parts.append(reference.sync_actor.soft_action_components(
                reference.physical_batch['next_observations'][:, :229]))
        reference._future_subset = torch.tensor([0, 1])
        with patch.object(reference.sync_actor, 'soft_action_components', side_effect=parts):
            native_reference = ReferenceSyncMean8Trainer._target(reference,
                reference.physical_batch, backup='h2oplus')
    finally:
        torch.set_rng_state(rng)
    assert torch.equal(block['native'], native_reference)
    assert torch.equal(block['beta0'], trace_target(block['native'], dict(
        native=block['native'], matching_q=block['matching_q'],
        coefficients=block['coefficients'], entropy=block['entropy']), beta=0.))
    assert torch.equal(block['beta0'][-1], block['native'][-1])
    assert torch.equal(block['beta1'][-1], block['native'][-1])
    correction = block['accounting64']['entropy_removal']
    assert correction[-1] == 0.
    assert torch.isfinite(correction).all()
    actual = (block['beta1']-block['beta0']).double()
    assert torch.equal(actual-correction, block['rounding64']['beta1_minus_beta0'])
    assert block['matching_heads'].shape == (2,270)
    assert block['positive_heads'].shape == (8,271)
    assert float(block['future_mass'].max()) <= 9.+1.e-6
