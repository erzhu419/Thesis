"""Tests for changes that could invalidate the new comparisons."""

import unittest
from types import SimpleNamespace
import numpy as np
import torch
from mcwm import combined_loss, gradient_relation, timed_actor
from guard import PersonLedger
from response import Responses, I


class DiagnosticsTest(unittest.TestCase):
    def test_inference_timer_preserves_actor_output(self):
        actor = torch.nn.Linear(2, 1)
        x = torch.ones(1, 2)
        samples = []
        torch.testing.assert_close(timed_actor(actor, samples)(x), actor(x))
        self.assertEqual(len(samples), 1)
        self.assertGreater(samples[0], 0.)

    def test_scaled_control_has_no_model_and_exact_coefficient(self):
        x = torch.tensor(2., requires_grad=True)
        loss = combined_loss(x.square(), None, .25, "scaled_nomodel")
        loss.backward()
        self.assertEqual(loss.item(), 3.)
        self.assertEqual(x.grad.item(), 3.)
        self.assertEqual(combined_loss(x.square(), None, .25, "h2o_matched").item(), 4.)

    def test_gradient_diagnostic_does_not_change_rng_or_accumulate_gradients(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        real, model = x.square().sum(), -x.square().sum()
        state = torch.get_rng_state().clone()
        row = gradient_relation(real, model, (x,), .25)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIsNone(x.grad)
        self.assertAlmostEqual(row["gradient_cosine"], -1., places=6)
        combined_loss(real, model, .25, "mcwm").backward()
        torch.testing.assert_close(x.grad, x.detach())

    def test_passenger_wait_ride_transfer_and_window_censoring(self):
        ledger = PersonLedger(SimpleNamespace(current_time=0))
        ledger.advance(1, {"wait": {"a", "b"}, "ride": set()})
        ledger.advance(4, {"wait": {"b"}, "ride": {"a"}})
        ledger.advance(7, {"wait": {"a", "b"}, "ride": set()})
        ledger.advance(9, {"wait": {"b"}, "ride": {"a"}})
        ledger.advance(11, {"wait": {"b"}, "ride": set()})
        totals = ledger.finalize()
        self.assertEqual(totals["wait"], {"a": 5., "b": 10.})
        self.assertEqual(totals["ride"], {"a": 5.})

    def test_response_does_not_connect_reused_vehicle_across_trips(self):
        response = Responses()
        raw = np.zeros(len(I))
        raw[I["bus_capacity"]] = 50
        raw[I["target_forward_headway_seconds"]] = 360
        for station, instant in [(1, 10), (2, 40), (1, 70), (2, 100)]:
            raw[I["station_index"]] = station
            raw[I["sim_time_seconds"]] = instant
            response.observe("bus", raw, 20)
        self.assertEqual(len(response.rows), 2)
        self.assertEqual(response.summary()["all"]["next_event_seconds"]["mean"], 30.)


if __name__ == "__main__":
    unittest.main()
