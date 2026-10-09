"""Check attribution semantics before spending complete SUMO service days."""

import json
import unittest
import numpy as np
import run
from bus_h2o.mdp_contract_v7 import RAW_EVENT_STATE_DIM, RAW_EVENT_STATE_INDEX


class ActionsTest(unittest.TestCase):
    def setUp(self):
        self.config, _ = run.contract()

    def raw(self, progress, onboard=0, backward=0, arrival=1):
        local = np.zeros((1, RAW_EVENT_STATE_DIM), np.float32)
        values = dict(sim_time_seconds=progress * self.config.horizon_seconds,
                      target_forward_headway_seconds=300, target_backward_headway_seconds=300,
                      bus_capacity=80, onboard_passengers=onboard,
                      forward_bus_present=1, backward_bus_present=backward,
                      eligible_arrival_rate_per_second=arrival)
        for name, value in values.items():
            local[0, RAW_EVENT_STATE_INDEX[name]] = value
        return run.common.pack_event_batch_v7(local, 0, config=self.config)

    def holds(self, raw, layer, proposed=-1):
        action, parent = run.execute(raw, np.array([proposed], np.float32),
                                     layer, self.config, 0.25)
        return tuple(float(run.common.policy_action_to_hold_seconds_v7(x[0], self.config))
                     for x in (action, parent))

    def test_floor_windows(self):
        for time, expected in ((0.1, (0, 0, 0)), (0.2, (0, 0, 60)),
                               (0.6, (0, 20, 60)), (0.8, (0, 0, 0))):
            for layer, hold in zip(("no_floor", "late20", "early60"), expected):
                actual, parent = self.holds(self.raw(time), layer)
                self.assertAlmostEqual(actual, hold, places=4)
                self.assertAlmostEqual(parent, 0, places=4)

    def test_ineligible_has_no_floor(self):
        for changes in (dict(onboard=40), dict(backward=1), dict(arrival=0)):
            hold, parent = self.holds(self.raw(0.6, **changes), "early60", proposed=1)
            self.assertAlmostEqual(hold, parent, places=4)
            self.assertLessEqual(hold, 15.0001)

    def test_native_rule_is_unscaled(self):
        hold, parent = self.holds(self.raw(0.2), "native", proposed=1)
        self.assertAlmostEqual(hold, 60, places=4)
        self.assertEqual(hold, parent)

    def test_matrix_and_shards(self):
        p = json.loads((run.HERE / "protocol.json").read_text())
        rows = run.cases(p)
        self.assertEqual(len(rows), p["expected_cases"])
        self.assertEqual(len({r["case_id"] for r in rows}), len(rows))
        self.assertTrue(all(len(rows[i::p["shards"]]) == 10 for i in range(p["shards"])))

    def test_deployed_actors(self):
        import libsumo
        self.assertEqual(libsumo.getVersion()[1], "SUMO 1.25.0")
        p = json.loads((run.HERE / "protocol.json").read_text())
        _, stations = run.contract()
        raw = self.raw(0.2)
        state = run.common.raw_state_to_canonical_v7(raw, stations, self.config)
        for method in p["policy_exports"]:
            actor, _ = run.common.load_actor_checkpoint_v7(
                run.HERE / "inputs" / f"{method}.pt", "cpu", require_policy_artifact=True)
            with run.torch.inference_mode():
                action, _ = actor(run.torch.from_numpy(state[None]), deterministic=True)
            self.assertEqual(tuple(action.shape), (1, 1))
            self.assertTrue(run.torch.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
