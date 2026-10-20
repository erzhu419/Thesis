"""Detect a fabricated time target, future covariate leakage and a disconnected readout."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/"mcwm_closed_loop_diagnostic_v2_20261020"))
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import (
    AFFECTED_LINES, DiagnosticWorld, EI, FEATURE_NAMES, LocalIntervalCalibration,
    LocalWorld, adapter_features, fit_readout,
)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.interval import SEMANTICS
from scenarios import UPSTREAM_STOP, DOWNSTREAM_STOP
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7, RAW_EVENT_STATE_DIM


def source_rows():
    rows = []
    for line_id, line in enumerate(AFFECTED_LINES):
        for index in range(5):
            raw = np.zeros(RAW_EVENT_STATE_DIM)
            raw[EI["line_id"]] = line_id
            raw[EI["bus_capacity"]] = 50.
            for direction in ("forward", "backward"):
                raw[EI[f"{direction}_bus_present"]] = 1.
                raw[EI[f"{direction}_headway_seconds"]] = 360.
                raw[EI[f"target_{direction}_headway_seconds"]] = 360.
            rows.append(dict(domain="source", episode_id=0, sequence=line_id*10+index,
                vehicle_id=f"bus{line_id}", trip_id=f"trip{index}", line_id=line,
                station_id=UPSTREAM_STOP, next_station_id=DOWNSTREAM_STOP,
                started_at=360.*index, completed_at=360.*index+120.+index,
                available_at=360.*index+150.+index, dt=120.+index,
                a=[-1.], event=raw.tolist(), semantics=SEMANTICS))
    return rows


def state(line=0, current=0):
    values = torch.zeros(1, 229)
    values[0, 0] = .3
    values[0, 1] = (current+1)/16
    values[0, 2] = current/15
    slots = values[:, 5:].reshape(1, 16, 14)
    slots[0, :current+1, 0] = 1.
    slots[0, current, 1] = line
    return values


class ConstantModel(torch.nn.Module):
    def forward(self, states, actions):
        result = states.new_zeros(len(states), 1124)
        result[:, :4] = states.new_tensor([1., 2., 0., 0.])
        return result


def fake_world():
    world = LocalWorld.__new__(LocalWorld)
    world.device = torch.device("cpu")
    world.models = {"direct": ConstantModel()}
    world.family, world.version, world.context_enabled = "direct", 0, False
    world.norm = SimpleNamespace(g_mean=torch.zeros(4), g_std=torch.ones(4))
    world.stations = torch.full((12,), 46.)
    world.physical = SimpleNamespace(context_delta_seconds=lambda: np.array([6., 0., 0.]),
                                     parameter_version=1)
    world.numeric_lines, world.local_enabled, world.local_ridge = [0, 1, 2], True, 1.
    world.local_readouts = {"direct": torch.ones(len(FEATURE_NAMES), 4)}
    world.pending = {}
    world.local_fit_records, world.local_prediction_records = [], []
    world.local_labels = world.local_refits = 0
    from collections import Counter
    world.local_rows_by_line = Counter()
    return world


class LocalWorldTests(unittest.TestCase):
    def test_reference_context_starts_zero_and_only_tracks_available_local_fit(self):
        physical = LocalIntervalCalibration(source_rows(), TransitMDPConfigV7())
        np.testing.assert_array_equal(physical.context_delta_seconds(), np.zeros(3))
        physical.context["7X"] = 28.
        np.testing.assert_array_equal(physical.context_delta_seconds(), [28., 0., 0.])
        self.assertEqual(physical.numeric_line_by_name, {"7X": 0, "102X": 1, "705X": 2})

    def test_route_features_use_current_slot_and_executed_holding(self):
        values = state(line=2, current=3)
        features = adapter_features(values, torch.tensor([[0.]]), [6., 12., 18.], [0, 1, 2])
        torch.testing.assert_close(features, torch.tensor([[.1, .2, .3, 0., 0., .3, 0., 0., .15]]))
        zero = adapter_features(values, torch.tensor([[0.]]), [0., 0., 0.], [0, 1, 2])
        self.assertEqual(float(zero.abs().sum()), 0.)

    def test_zero_prior_and_actual_global_targets_determine_readout(self):
        x = torch.zeros(5, 9)
        x[:, 0] = 1.
        # E.g. log-global-dt residual .2, not the physical 28 second shift.
        residual = torch.tensor([[.2, -.4, 1., -2.]]).expand(5, -1)
        weights = fit_readout(x, residual)
        torch.testing.assert_close(weights[0], residual[0]*5/6)
        self.assertEqual(float(weights[1:].abs().sum()), 0.)
        empty_context = fit_readout(torch.zeros_like(x), residual)
        self.assertEqual(float(empty_context.abs().sum()), 0.)

    def test_calibration_changes_only_four_global_heads_and_off_is_identity(self):
        world = fake_world()
        s, a = state(), torch.tensor([[-.5]])
        base = world.predict(s, a, calibrated=False)
        calibrated = world.predict(s, a, calibrated=True)
        self.assertGreater(float((calibrated[:, :4]-base[:, :4]).abs().sum()), 0.)
        torch.testing.assert_close(calibrated[:, 4:], base[:, 4:], rtol=0, atol=0)
        world.local_enabled = False
        torch.testing.assert_close(world.predict(s, a), base, rtol=0, atol=0)
        world.physical.context_delta_seconds = lambda: np.zeros(3)
        torch.testing.assert_close(world.predict(s, a, calibrated=True), base, rtol=0, atol=0)

    def test_saved_issue_covariate_and_global_clock_survive_later_physical_fit(self):
        world = fake_world()
        s, a = state(), np.array([-.5], np.float32)
        issue = dict(s=s[0].numpy(), a=a, started_at=100., sequence=7)
        world.register(issue)
        saved = world.pending[7]["local_covariates"]["features"].clone()
        world.physical.context_delta_seconds = lambda: np.array([60., 0., 0.])
        ns = state()[0].numpy().copy()
        ns[0] = .300125
        row = SimpleNamespace(s=s[0].numpy(), a=a, ns=ns, r=-8., dt=3., terminal=False,
            line_id="7X", sequence=7, started_at=100., completed_at=103., available_at=133.)
        with patch.object(DiagnosticWorld, "observe", return_value=[dict(sequence=7)]):
            logs = world.observe([row], 133.)
        torch.testing.assert_close(world.local_fit_records[0]["features"], saved, rtol=0, atol=0)
        self.assertAlmostEqual(float(world.local_fit_records[0]["standardized_target"][0, 0]), np.log1p(3.), places=6)
        self.assertEqual(logs[0]["local_adapter"]["physical_parameter_version_at_issue"], 1)
        self.assertAlmostEqual(logs[0]["local_adapter"]["context_abs_mean_seconds"], 2.)

    def test_global_readout_fit_has_nonzero_action_response_without_changing_labels(self):
        world = fake_world()
        records = []
        for action in (-1., -.5, 0., .5, 1.):
            s, a = state(), torch.tensor([[action]])
            x = world.local_features(s, a)
            raw = world.models["direct"](s, a)[:, :4]
            # Observed action-dependent global cost residual, entirely distinct
            # from the physical interval label used to construct x.
            y = raw+torch.tensor([[0., float((action+1)/2), 0., 0.]])
            records.append(dict(s=s, a=a, features=x, standardized_target=y))
        world.local_fit_records = records
        world.refit_local()
        low = world.predict(state(), torch.tensor([[-1.]]), calibrated=True)
        high = world.predict(state(), torch.tensor([[1.]]), calibrated=True)
        self.assertGreater(float(high[0, 1]-low[0, 1]), 0.)
        self.assertEqual(world.local_refits, 1)


if __name__ == "__main__":
    unittest.main()
