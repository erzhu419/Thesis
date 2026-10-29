from copy import deepcopy
import unittest

from experiments.mcwm_closed_loop_issue_conditions_v91_20261028.conditions import (
    CorridorHistory, DOWNSTREAM, LINES, UPSTREAM,
)

EI = dict(station_index=0, sim_time_seconds=1)


def add_interval(history, line, trip, end, duration=120., hold=30., sequence=0, adjacent=True):
    history.observe(dict(line_id=line, trip_id=trip, station_id=UPSTREAM, sequence=sequence,
        started_at=end-duration, event=[10., end-duration-1.], a=[2*hold/60.-1.]))
    history.observe(dict(line_id=line, trip_id=trip, station_id=DOWNSTREAM, sequence=sequence+1,
        started_at=end, event=[11. if adjacent else 12., end-1.], a=[1.]))


def history_with_baselines():
    history = CorridorHistory(EI)
    for index, line in enumerate(LINES):
        for j, nonholding in enumerate((80., 90., 100.)):
            add_interval(history, line, line+'_'+str(j), 6000.+index*100+j*200,
                duration=nonholding+30, hold=30, sequence=10*j)
    return history


class CorridorHistoryTest(unittest.TestCase):
    def test_executed_hold_subtracted_once_and_snapshot_is_read_only(self):
        history = history_with_baselines()
        add_interval(history, '7X', 'latest', 11900., duration=180., hold=30.)
        before = deepcopy(history.__dict__)
        value = history.snapshot(12000.)
        self.assertEqual(history.__dict__, before)
        self.assertEqual(value['status'], 'supported')
        self.assertEqual(value['value'], 1.)
        self.assertTrue(value['historical_only'])
        source = value['latest_sources'][0]
        self.assertEqual(source['commanded_hold_seconds'], 30.)
        self.assertEqual(source['nonholding_ready_seconds'], 150.)
        self.assertEqual(source['baseline_nonholding_seconds'], 90.)

    def test_same_clock_new_completion_and_future_ingestion_are_excluded(self):
        history = history_with_baselines()
        add_interval(history, '7X', 'past', 11900., duration=150., hold=30.)
        add_interval(history, '102X', 'same_clock', 12000., duration=300., hold=0.)
        add_interval(history, '705X', 'future', 12100., duration=400., hold=0.)
        value = history.snapshot(12000.)
        self.assertEqual(value['latest_completed_at'], 11900.)
        self.assertEqual(value['value'], .5)
        self.assertEqual(len(value['latest_sources']), 1)
        self.assertTrue(value['historical_only'])

    def test_latest_ties_are_averaged_after_line_baseline_normalization(self):
        history = CorridorHistory(EI)
        for line, center in zip(LINES, (90., 130., 200.)):
            for i, offset in enumerate((-10., 0., 10.)):
                add_interval(history, line, line+str(i), 6000.+200*i,
                    duration=center+offset+30., hold=30.)
        add_interval(history, '7X', 'latest7', 11900., duration=180., hold=30.)
        add_interval(history, '102X', 'latest102', 11900., duration=160., hold=30.)
        add_interval(history, '705X', 'older', 11899., duration=600., hold=0.)
        value = history.snapshot(12000.)
        self.assertEqual(value['value'], .5)
        self.assertEqual([row['line_id'] for row in value['latest_sources']], ['102X', '7X'])
        self.assertEqual(value['baseline_counts'], {line: 3 for line in LINES})

    def test_baseline_cutoff_missing_coverage_and_stale_history_are_not_zero(self):
        history = CorridorHistory(EI)
        for line in LINES:
            for i in range(2):
                add_interval(history, line, line+str(i), 8000.+i*100, sequence=i*10)
            add_interval(history, line, line+'cutoff', 9000., sequence=50)
        add_interval(history, '7X', 'recent', 11999.)
        value = history.snapshot(12000.)
        self.assertEqual(value['baseline_counts'], {line: 2 for line in LINES})
        self.assertIsNone(value['value'])
        self.assertEqual(value['reason'], 'baseline_coverage_missing')
        history = history_with_baselines()
        add_interval(history, '7X', 'latest', 11099.)
        self.assertIsNone(history.snapshot(12000.)['value'])
        self.assertEqual(history.snapshot(12000.)['reason'], 'latest_corridor_interval_stale')
        add_interval(history, '7X', 'age_boundary', 11100.)
        self.assertEqual(history.snapshot(12000.)['status'], 'supported')

    def test_only_same_trip_adjacent_exact_corridor_is_accepted(self):
        history = CorridorHistory(EI)
        add_interval(history, '7X', 'skip', 8000., adjacent=False)
        history.observe(dict(line_id='7X', trip_id='one', station_id=UPSTREAM, sequence=1,
            started_at=8000., event=[10., 7999.], a=[0.]))
        history.observe(dict(line_id='7X', trip_id='two', station_id=DOWNSTREAM, sequence=2,
            started_at=8100., event=[11., 8099.], a=[0.]))
        self.assertEqual(history.summary()['counts'].get('corridor_intervals', 0), 0)
        self.assertEqual(history.summary()['counts']['nonadjacent_corridor_issues'], 1)


if __name__ == '__main__':
    unittest.main()
