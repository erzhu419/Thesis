"""Small accounting/binding tests; no simulation or actor artifacts loaded."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('historical_run', HERE/'run.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


def point(clock, wait, ride):
    return dict(time_seconds=clock, accrued_wait_seconds=wait, accrued_in_vehicle_seconds=ride)


class RunnerTests(unittest.TestCase):
    def test_exact_full_and_window_costs_keep_the_whole_day_denominator(self):
        rows = [point(9000, 100, 200), point(12000, 140, 260), point(18300, 200, 400)]
        result = run.periods(rows, point(24000, 230, 420), 10)
        self.assertEqual(result['full_day']['cost'], 880)
        self.assertEqual(result['early']['cost'], 140)
        self.assertEqual(result['post_budget']['cost'], 80)
        self.assertEqual(result['post_budget']['cost_per_passenger'], 8)

    def test_shifted_boundary_is_missing_not_nearest_or_zero(self):
        result = run.periods([point(18301, 200, 400)], point(24000, 230, 420), 10)
        self.assertEqual(result['post_budget']['status'], 'missing')
        self.assertIsNone(result['post_budget']['cost'])
        self.assertEqual(result['full_day']['status'], 'complete')

    def test_initialization_or_wrong_seed_artifact_is_not_a_final_actor(self):
        extra = dict(training_seed=6201, contract_config={'x':1}, arm='fixed0250',
            scenario='formal_calibrated_three_demand', updates=70000, smoke=False)
        self.assertTrue(all(run.artifact_checks(extra, 6201, {'x':1}).values()))
        self.assertFalse(run.artifact_checks(extra, 6202, {'x':1})['training_seed'])
        extra['updates'] = 50000
        self.assertFalse(run.artifact_checks(extra, 6201, {'x':1})['finalized_historical_actor'])

    def test_failure_keeps_observed_native_time_and_unknown_unavailable_time(self):
        ledger = SimpleNamespace(snapshot=lambda: SimpleNamespace(time_seconds=1234.))
        env = SimpleNamespace(_bridge=SimpleNamespace(passenger_time_ledger=ledger))
        self.assertEqual(run.consumed(env, True), (1234., True))
        self.assertEqual(run.consumed(None, True), (None, True))
        self.assertEqual(run.consumed(None, False), (0., False))

    def test_shock_requires_exact_native_clock_and_all_declared_lanes(self):
        p = dict(shock_at_seconds=9000, speed_factor=.65)
        scene = dict(name='local_speed_drop', scheduled_shock_at_seconds=9000,
            actual_shock_at_seconds=9000, speed_factor=.65,
            lane_caps=[dict(lane_id=x) for x in ['a','b']])
        self.assertTrue(all(run.scenario_checks(scene, 'local_speed_drop', p, ['a','b']).values()))
        scene['actual_shock_at_seconds'] = 9001
        self.assertFalse(run.scenario_checks(scene, 'local_speed_drop', p, ['a','b'])['native_shock_clock'])
        scene['lane_caps'].pop()
        self.assertFalse(run.scenario_checks(scene, 'local_speed_drop', p, ['a','b'])['affected_lanes'])


if __name__ == '__main__': unittest.main()
