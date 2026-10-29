"""Synthetic four-block calibration contracts; never read collected T/C labels."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028 import calibrate as module
from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.analysis import SEEDS
from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.test_analysis import calibration_fixture
from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import predict


class CalibrationTest(unittest.TestCase):
    def test_four_block_scores_once_and_fixed_same_domain_comparator(self):
        args=calibration_fixture()
        frozen=deepcopy(args[2])
        with patch.object(module,'seed_score',wraps=module.seed_score) as scores, \
                patch.object(module,'calibrated_margin',wraps=module.calibrated_margin) as margins:
            result=module.calibrate(*args)
        self.assertEqual(scores.call_count,4)
        self.assertEqual(margins.call_count,2)  # one candidate and its predeclared comparator
        scale=predict(args[2]['bias_model'],.1,-1.)['support_scale']
        expected=[(1.+(-40.+10*index)/15079)/scale for index in range(4)]
        for block,value in zip(result['block_scores'],expected):
            self.assertAlmostEqual(block['score'],value)
        self.assertAlmostEqual(result['C'],max(expected))
        self.assertAlmostEqual(result['state_independent']['C'],1.-10./15079)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['ready_for_validation'])
        self.assertEqual(result['observed_C_feasibility']['accepted_roots'],24)
        self.assertEqual(result['observed_C_feasibility']['selected_scenes'],8)
        self.assertTrue(result['state_independent']['comparison_only'])
        self.assertFalse(result['state_independent']['selected_for_deployment'])
        self.assertEqual(result['frozen_model'],frozen)
        self.assertEqual(args[2],frozen)
        self.assertEqual(result['budget']['bias_refits'],0)
        self.assertLess(len(json.dumps(result,allow_nan=False).encode()),131072)

    def test_missing_required_truth_keeps_four_blocks_and_never_calibrates(self):
        for positive_m in (False,True):
            with self.subTest(positive_m=positive_m):
                args=calibration_fixture()
                pair=args[0]['pairs'][-1]
                pair['label_delta_per_passenger']['cost']=None
                if positive_m:
                    pair['predictions']['service_history6']=.1
                with patch.object(module,'seed_score',side_effect=AssertionError('incomplete C scored')), \
                        patch.object(module,'calibrated_margin',side_effect=AssertionError('incomplete C calibrated')):
                    result=module.calibrate(*args)
                self.assertEqual(result['status'],'hold')
                self.assertFalse(result['ready_for_validation'])
                self.assertIsNone(result['C'])
                self.assertEqual([row['external_seed'] for row in result['block_scores']],list(SEEDS))
                self.assertTrue(all(row['score'] is None for row in result['block_scores']))
                self.assertEqual(result['budget']['candidate_calibrations'],0)

    def test_missing_whole_seed_is_not_silently_calibrated_on_three_blocks(self):
        args=calibration_fixture()
        args[0]['pairs']=[pair for pair in args[0]['pairs'] if pair['external_seed'] != SEEDS[-1]]
        result=module.calibrate(*args)
        self.assertFalse(result['integrity']['checks']['four_complete_declared_blocks'])
        self.assertIsNone(result['C'])
        self.assertEqual(len(result['block_scores']),4)

    def test_fit_readiness_and_exact_frozen_bias_binding_are_required(self):
        args=calibration_fixture()
        args[3]['decision']['ready_for_calibration']=False
        result=module.calibrate(*args)
        self.assertFalse(result['integrity']['checks']['fixed_fit_ready'])
        self.assertIsNone(result['C'])
        args=calibration_fixture()
        args[2]['bias_model']['coef'][0]=1.
        result=module.calibrate(*args)
        self.assertFalse(result['integrity']['checks']['fixed_bias_binding'])
        self.assertEqual(result['budget']['candidate_calibrations'],0)

    def test_all_rejected_stops_only_on_observed_C_feasibility(self):
        args=calibration_fixture()
        for pair in args[0]['pairs']:
            pair['label_delta_per_passenger']=dict(wait=0.,ride=10./15079,cost=10./15079)
        result=module.calibrate(*args)
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['ready_for_validation'])
        self.assertEqual(result['decision']['reason'],'STOP_ALL_C_ROOTS_REJECTED')
        self.assertEqual(result['observed_C_feasibility']['accepted_roots'],0)
        self.assertTrue(result['decision']['V_kept_sealed'])
        self.assertIn('Observed C roots only',result['interpretation']['feasibility_scope'])

    def test_noop_later_root_is_not_removed_from_the_block_maximum(self):
        args=calibration_fixture()
        row=args[0]['pairs'][-1]
        row['physical_noop']=True
        row['checks']['physical_separation']=False
        row['label_delta_per_passenger']=dict(wait=0.,ride=100./15079,cost=100./15079)
        result=module.calibrate(*args)
        scale=predict(args[2]['bias_model'],.1,-1.)['support_scale']
        self.assertAlmostEqual(result['C'],(1.+100./15079)/scale)
        self.assertEqual(result['block_scores'][-1]['score'],result['C'])
        self.assertFalse(result['ready_for_validation'])

    def test_future_tail_boundary_never_changes_calibration_or_selection(self):
        args=calibration_fixture()
        before=module.calibrate(*args)
        for pair in args[0]['pairs']:
            pair['tail_boundary']=dict(reference=None,candidate=dict(time_seconds=18300,accrued_wait_seconds=999999))
            pair['tail']=dict(observed=True,delta_per_passenger=dict(wait=123456,ride=-999999,cost=0))
        after=module.calibrate(*args)
        self.assertEqual(after,before)

    def test_cli_writes_once_and_does_not_repeat_a_fixed_calibration(self):
        args=calibration_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=('summary','protocol','fit-model','fit-summary')
            command=[]
            for name,value in zip(names,args):
                path=root/(name+'.json')
                path.write_text(json.dumps(value))
                command.extend(('--'+name,str(path)))
            output=root/'calibration.json'
            command.extend(('--out',str(output)))
            with redirect_stdout(io.StringIO()):
                result=module.main(command)
            self.assertEqual(json.loads(output.read_text())['C'],result['C'])
            with patch.object(module,'calibrate',side_effect=AssertionError('repeated calibration')):
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    module.main(command)


if __name__=='__main__':
    unittest.main()
