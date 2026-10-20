"""Identify wrong interval provenance, premature learning, and optimistic scoring."""

from types import SimpleNamespace
import unittest

import numpy as np

from interval import (AFFECTED_LINES, DOWNSTREAM_STOP, EI, IntervalDiagnostic,
                      MODELS, SEMANTICS, serialize)
from scenarios import UPSTREAM_STOP
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7, RAW_EVENT_STATE_DIM


def raw():
    event = np.zeros(RAW_EVENT_STATE_DIM)
    for name in ("forward_headway_seconds", "backward_headway_seconds",
                 "target_forward_headway_seconds", "target_backward_headway_seconds"):
        event[EI[name]] = 360.
    event[EI["forward_bus_present"]] = event[EI["backward_bus_present"]] = 1.
    event[EI["bus_capacity"]] = 50.
    event[EI["eligible_stranded_passengers"]] = 5.
    event[EI["onboard_passengers"]] = 20.
    event[EI["base_dwell_seconds"]] = 10.
    return event.tolist()


def physical(sequence, *, line="7X", started=7000., duration=120., holding=0., domain="target"):
    return dict(domain=domain, episode_id=0, sequence=sequence,
        vehicle_id=f"{line}:fleet", trip_id=f"{line}:trip:{sequence}", line_id=line,
        station_id=UPSTREAM_STOP, next_station_id=DOWNSTREAM_STOP,
        started_at=started, completed_at=started+duration, available_at=started+duration+30.,
        dt=duration, a=[holding/30.-1.], event=raw(), semantics=SEMANTICS)


def calibration():
    return [physical(index*10+j, line=line, started=j*360., duration=100.+j,
                     holding=float(j), domain="source")
            for index, line in enumerate(AFFECTED_LINES) for j in range(5)]


class IntervalTests(unittest.TestCase):
    def make(self):
        return IntervalDiagnostic(calibration(), TransitMDPConfigV7())

    def test_source_requires_real_local_labels_and_three_line_coverage(self):
        rows = calibration()
        rows[0]["domain"] = "target"
        with self.assertRaisesRegex(ValueError, "source calibration"):
            IntervalDiagnostic(rows, TransitMDPConfigV7())
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            IntervalDiagnostic(calibration()[:-1], TransitMDPConfigV7())
        rows = calibration()
        rows[0]["semantics"] = "global_control_transition"
        with self.assertRaisesRegex(ValueError, "same-trip"):
            IntervalDiagnostic(rows, TransitMDPConfigV7())
        native = SimpleNamespace(**calibration()[0])
        native.a, native.event = np.asarray(native.a), np.asarray(native.event)
        self.assertEqual(serialize(native), calibration()[0])

    def test_saved_executed_action_predictions_are_scored_before_batch_learning(self):
        diagnostic = self.make()
        initial = diagnostic.frozen_weights.copy()
        first = physical(101, holding=30., duration=180.)
        second = physical(102, started=7010., holding=0., duration=150.)
        saved_first = diagnostic.register(first)
        saved_second = diagnostic.register(second)
        self.assertEqual(set(saved_first["predicted_seconds"]), set(MODELS))
        self.assertAlmostEqual(saved_first["predicted_seconds"]["online_linear"]
                               - saved_second["predicted_seconds"]["online_linear"], 30.)
        logs = diagnostic.observe([first, second], now=7210.)
        online = [row for row in logs if row["model"] == "online_linear"]
        self.assertEqual([row["predicted_seconds"] for row in online],
            [saved_first["predicted_seconds"]["online_linear"], saved_second["predicted_seconds"]["online_linear"]])
        self.assertTrue(all(row["parameter_version"] == 0 for row in logs))
        self.assertTrue(np.array_equal(diagnostic.frozen_weights, initial))
        self.assertFalse(np.array_equal(diagnostic.online_weights, initial))
        self.assertEqual(diagnostic.summary()["coverage"]["learned_labels"], 2)
        self.assertNotEqual(diagnostic.context["7X"], 0.)
        future = diagnostic.register(physical(103, started=7300., duration=150.))["predicted_seconds"]
        self.assertEqual(future["frozen_source_prior"], saved_second["predicted_seconds"]["frozen_source_prior"])
        self.assertGreater(future["online_linear"], future["frozen_source_prior"])
        self.assertGreater(future["local_context"], future["online_linear"])

    def test_delay_and_available_time_budget_crossing_and_missing_labels(self):
        diagnostic = self.make()
        crossing = physical(201, started=8950., duration=120.)
        diagnostic.register(crossing)
        with self.assertRaisesRegex(ValueError, "delay"):
            diagnostic.observe([crossing], now=9099.)
        diagnostic.observe([crossing], now=9100.)
        self.assertEqual(diagnostic.summary()["models"]["online_linear"]["by_period"]["crossing"]["n"], 1)
        self.assertEqual(diagnostic.summary()["models"]["online_linear"]["by_period"]["before"]["n"], 0)
        on_time = physical(202, started=17850., duration=120.)
        diagnostic.register(on_time)
        diagnostic.observe([on_time], now=18000.)
        self.assertEqual(diagnostic.summary()["coverage"]["learned_labels"], 2)
        too_late = physical(203, started=17850., duration=140.)
        diagnostic.register(too_late)
        before = diagnostic.online_weights.copy()
        logs = diagnostic.observe([too_late], now=18020.)
        self.assertTrue(all(not row["learning_eligible"] for row in logs))
        self.assertTrue(np.array_equal(diagnostic.online_weights, before))
        diagnostic.observe([physical(204, started=18500.)], now=18700.)
        summary = diagnostic.summary()
        self.assertEqual(summary["coverage"]["missing_saved_prediction"], 1)
        self.assertEqual(summary["labels"]["overall"]["n"], 4)
        self.assertEqual(summary["models"]["online_linear"]["overall"]["n"], 3)
        self.assertEqual(summary["coverage"]["learned_by_period"]["crossing"], 1)
        self.assertEqual(summary["coverage"]["learned_by_period"]["late"], 1)
        self.assertEqual(summary["coverage"]["by_line"]["7X"]["learned_by_period"]["late"], 1)
        self.assertEqual(summary["models"]["online_linear"]["by_period"]["late"]["n"], 2)
        self.assertEqual(summary["labels"]["by_period"]["crossing"]["mean_nonholding_seconds"], 120.)
        self.assertIsNone(summary["models"]["online_linear"]["by_period"]["early"]["mae_seconds"])

    def test_same_trip_identity_and_saved_action_must_match_physical_label(self):
        for changed in ({"trip_id": "other-trip"}, {"a": [1.]}):
            diagnostic = self.make()
            row = physical(301)
            diagnostic.register(row)
            with self.assertRaises(ValueError):
                diagnostic.observe([dict(row, **changed)], now=7200.)


if __name__ == "__main__":
    unittest.main()
