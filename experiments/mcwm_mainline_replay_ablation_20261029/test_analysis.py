from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.mcwm_mainline_replay_ablation_20261029.analysis import (
    analyze,load_day,metric_windows,PERIODS)


def fixture():
    records,refs,plan=[],[],[]
    for seed in (6206,6207,6208):
        for scene in ('stable','local_speed_drop'):
            point=dict(wait=10.,ride=10.,cost=30.,departed=15079,cost_per_passenger=30./15079)
            records.append(dict(seed=seed,scenario=scene,status='complete',checks=dict(synthetic=True),
                periods={period:dict(point,cost=45.,ride=25.) for period in PERIODS},
                budget=dict(training_fresh_outcomes=285,**{name+'_updates':456 for name in ('critic','value','actor','alpha')}),
                native_seconds=24000))
            refs.append(dict(identity=seed,scenario=scene,periods={period:dict(arms=dict(
                fresh=deepcopy(point),no_fresh=dict(point,cost=60.,ride=40.))) for period in PERIODS}))
            plan.append(dict(seed=seed,scenario=scene))
    return records,dict(cells=refs),dict(native_plan=plan,development_seeds=[6206,6207,6208],scenarios=['stable','local_speed_drop'])


class AnalysisTest(unittest.TestCase):
    def test_primary_direction_seed_units_and_fixed_reference_label(self):
        records,reference,protocol=fixture()
        out=analyze(records,reference,protocol)
        self.assertTrue(out['decision']['component_mean_majority_pass'])
        self.assertEqual(out['aggregates']['decoupled_minus_coupled']['full_day']['overall']['wins'],6)
        self.assertEqual(out['seed_readout']['complete_seeds'],3)
        self.assertEqual(out['budget']['known_new_native_seconds'],144000)
        self.assertEqual(out['budget']['new_complete_outcome_labels'],1710)
        self.assertIn('not the coupled method own matched',out['interpretation']['secondary'])

    def test_missing_day_is_not_filled_with_zero_cost(self):
        records,reference,protocol=fixture()
        records[-1]=dict(seed=6208,scenario='local_speed_drop',status='failed',
            checks=dict(complete=False),periods={},budget=None,native_seconds=None,failure=dict(message='failure'))
        out=analyze(records,reference,protocol)
        self.assertEqual(out['status'],'hold')
        self.assertFalse(out['decision']['component_mean_majority_pass'])
        self.assertIsNone(out['cells'][-1]['comparisons']['decoupled_minus_coupled']['full_day'])
        self.assertFalse(out['budget']['native_spend_complete'])
        self.assertEqual(out['aggregates']['decoupled_minus_coupled']['full_day']['overall']['available'],5)

    def test_failed_post_training_check_cannot_pass_despite_favorable_cost(self):
        records,reference,protocol=fixture()
        records[0]['checks']['frozen_after_budget']=False
        out=analyze(records,reference,protocol)
        self.assertEqual(out['status'],'hold')
        self.assertFalse(out['decision']['component_mean_majority_pass'])
        self.assertIn('frozen_after_budget',out['missing_or_failed'][0]['failed_checks'])

    def test_metric_endpoint_missing_has_no_interpolated_window(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            rows=[dict(time_seconds=t,accrued_wait_seconds=t,accrued_in_vehicle_seconds=2*t)
                  for t in range(0,24001,300) if t!=18300]
            (directory/'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            summary=dict(cost=dict(departed_passengers=15079,accrued_wait_seconds=24000,
                accrued_in_vehicle_seconds=48000,generalized_passenger_seconds=96000))
            windows,checks=metric_windows(directory,summary)
            self.assertIsNone(windows['post_budget'])
            self.assertFalse(checks['complete_windows'])
            self.assertTrue(checks['final_ledger'])

    def test_missing_summary_retains_failed_work(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            (directory/'failure.json').write_text(json.dumps(dict(native_time=1211,
                consumed_budget=dict(critic_updates=8),message='halt')))
            row=load_day(directory,directory,dict(seed=6206,scenario='stable'))
            self.assertEqual(row['native_seconds'],1211)
            self.assertEqual(row['budget']['critic_updates'],8)
            self.assertEqual(row['status'],'failed')


if __name__=='__main__':
    unittest.main()
