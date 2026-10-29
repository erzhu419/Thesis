"""Detect lost pending commands, future covariates and fabricated same-batch time."""

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    ActionResponseMemory, FEATURE_NAMES, decode_event_step, fit_response_readout,
    response_features,
)


def event(*, vehicle="bus", trip="trip", line="7X", time=100.):
    return dict(vehicle_id=vehicle, trip_id=trip, line_id=line, started_at=time)


def state(*, line=0, count=1, current=0):
    value = torch.zeros(1, 229)
    value[:, 0], value[:, 1], value[:, 2] = .5, count/16, current/15
    slots = value[:, 5:].reshape(1, 16, 14)
    slots[:, :count, 0] = 1
    slots[:, :count, 1] = line
    slots[:, :count, 3:5] = 1
    slots[:, :, 13] = -1
    return value


def test_pending_command_survives_other_bus_events_and_hold_expiry_until_same_vehicle():
    memory = ActionResponseMemory()
    memory.register(event(), hold_seconds=7.5, onboard_passengers=20, nonholding_seconds=100)
    memory.observe_event(event(vehicle="other", time=101))
    values = memory.features(103).reshape(3, 6)
    np.testing.assert_allclose(values[0], [1/16, 4.5/60, 20*4.5/6000, 3/300, 104.5/300, 1/16])
    assert memory.features(110)[1] == 0
    assert len(memory.pending) == 1
    memory.observe_event(event(time=150))
    np.testing.assert_array_equal(memory.features(150), np.zeros(18))


def test_saved_features_stay_saved_and_new_trip_cannot_retain_previous_trip_memory():
    memory = ActionResponseMemory()
    memory.register(event(), hold_seconds=10, onboard_passengers=5)
    saved = memory.features(100)
    memory.observe_event(event(trip="next_trip", time=500))
    np.testing.assert_array_equal(memory.features(500), np.zeros(18))
    assert saved[0] == 1/16 and saved[1] == pytest.approx(10/60)
    assert saved[4] == 0 and saved[5] == 0  # no fabricated physical prediction


def test_response_features_use_current_slot_and_actual_capped_action_once():
    s = state(line=2, count=3, current=2)
    s[:, 5+2*14+8] = .4
    memory = np.zeros(18, np.float32)
    x = response_features(s, torch.tensor([[-.75]]), memory, 120., [0, 1, 2])
    assert x.shape == (1, len(FEATURE_NAMES))
    assert float(x[0, :18].sum()) == 0
    assert float(x[0, 18:42].sum()) == 0
    assert x[0, FEATURE_NAMES.index("705X_hold")].item() == .125
    assert x[0, FEATURE_NAMES.index("705X_hold_onboard")].item() == pytest.approx(.05)
    assert x[0, FEATURE_NAMES.index("705X_hold_nonholding_interval")].item() == pytest.approx(.05)
    zero = response_features(s, [[-1.]], memory, 120., [0, 1, 2])
    assert zero[0, FEATURE_NAMES.index("705X_hold")].item() == 0
    assert zero[0, FEATURE_NAMES.index("705X_nonholding_interval")].item() == pytest.approx(.4)
    missing = response_features(s, [[-1.]], memory, None, [0, 1, 2])
    assert float(missing.sum()) == 0
    observed_zero = response_features(s, [[-1.]], memory, 0., [0, 1, 2])
    assert observed_zero[0, FEATURE_NAMES.index("705X_nonholding_available")].item() == 1


def test_same_batch_is_exact_and_does_not_call_or_consume_learned_decoder():
    s = state(count=3, current=0)
    called = []
    def decode(states, output):
        called.append(1)
        return states+1, states.new_ones(len(states)), states.new_ones(len(states))
    ns, reward, dt = decode_event_step(s, torch.tensor([[-.75]]), torch.zeros(1, 4), decode)
    expected = s.clone()
    expected[:, 2], expected[:, 18] = 1/15, -.75
    torch.testing.assert_close(ns, expected, rtol=0, atol=0)
    assert not called and reward.item() == 0 and dt.item() == 0
    boundary = state(count=3, current=2)
    _, reward, dt = decode_event_step(boundary, [[-1.]], torch.zeros(1, 4), decode)
    assert called and reward.item() == 1 and dt.item() == 1


def test_mixed_batch_analytic_and_learned_dispatch_keep_row_order():
    s = torch.cat((state(count=2), state(count=1), state(count=3, current=1)))
    actions = torch.tensor([[-.9], [-.8], [-.7]])
    calls = []
    def decode(states, output):
        calls.append(len(states))
        return states+10, states.new_full((len(states),), -12), states.new_full((len(states),), 4)
    ns, reward, dt = decode_event_step(s, actions, torch.zeros(3, 4), decode)
    assert calls == [1]
    torch.testing.assert_close(ns[1], s[1]+10)
    torch.testing.assert_close(reward, torch.tensor([0., -12., 0.]))
    torch.testing.assert_close(dt, torch.tensor([0., 4., 0.]))
    assert ns[2, 5+14+13].item() == pytest.approx(-.7)


def test_zero_prior_fit_uses_four_global_residuals_without_interval_substitution():
    x = torch.zeros(6, len(FEATURE_NAMES))
    y = torch.tensor([[.2, -.4, .1, -.1]]).expand(6, -1)
    torch.testing.assert_close(fit_response_readout(x, y), torch.zeros(len(FEATURE_NAMES), 4))
    x[:, 18] = 1
    weights = fit_response_readout(x, y)
    torch.testing.assert_close(weights[18], y[0]*6/7)
    assert weights[19:].count_nonzero().item() == 0
