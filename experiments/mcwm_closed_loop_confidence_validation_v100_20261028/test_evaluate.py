"""Synthetic reserved-V decisions; no native simulation or collected labels."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_confidence_validation_v100_20261028 import evaluate as module
from experiments.mcwm_closed_loop_confidence_validation_v100_20261028.analysis import analyze
from experiments.mcwm_closed_loop_confidence_validation_v100_20261028.test_analysis import fixture, evaluation_fixture


class EvaluationTest(unittest.TestCase):
    def test_first_accept_and_reference_plus_one_suffix_not_sum_or_best_query(self):
        args=evaluation_fixture()
        args[0]['pairs'].reverse()  # storage order must not determine policy order
        result=module.evaluate(*args)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['passed'])
        self.assertEqual(result['independent_unit']['external_seeds'],1)
        for scene in ('stable','local_speed_drop'):
            methods=result['by_scene'][scene]['methods']
            candidate=methods['calibrated_candidate']
            self.assertEqual(candidate['selected_line'],'102X')
            self.assertEqual(candidate['interventions'],1)
            self.assertAlmostEqual(candidate['full_day']['cost'],(70000.-20.)/15079)
            self.assertAlmostEqual(candidate['full_day']['wait'],(20000.-5.)/15079)
            self.assertAlmostEqual(candidate['full_day']['ride'],(30000.-10.)/15079)
            self.assertEqual(candidate['delta_vs_reference']['full_day']['cost'],-20./15079)
            self.assertEqual(methods['raw_history6']['selected_line'],'7X')
            self.assertEqual(methods['line3']['selected_line'],'7X')
            self.assertEqual(methods['service5']['selected_line'],'7X')
            self.assertEqual(methods['state_independent']['selected_line'],'102X')
            self.assertEqual(methods['reference']['delta_vs_reference']['full_day']['cost'],0.)
        self.assertAlmostEqual(result['aggregate']['calibrated_candidate']['mean_delta_vs_raw_history6']['cost'],-50./15079)
        self.assertLess(len(json.dumps(result,allow_nan=False).encode()),65536)

    def test_all_reference_is_zero_delta_and_fails_nonzero_and_improvement_gates(self):
        result=module.evaluate(*evaluation_fixture(candidate_margin=10.))
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['real_nonzero_intervention'])
        self.assertFalse(result['gates']['mean_cost_better_than_reference'])
        for scene in ('stable','local_speed_drop'):
            candidate=result['by_scene'][scene]['methods']['calibrated_candidate']
            self.assertIsNone(candidate['selected_root'])
            self.assertEqual(candidate['delta_vs_reference']['full_day']['cost'],0.)
            self.assertEqual(candidate['delta_vs_reference']['tail']['cost'],0.)

    def test_nonzero_measured_noop_delta_cannot_substitute_for_real_departure_dose(self):
        c,p,a,m,fit,calibration=fixture()
        for scene in c:
            pair=scene['pairs'][1]
            pair['physical']['candidate']['departure_clock']=pair['physical']['reference']['departure_clock']
            pair['checks']['physical_departure_separation']=False
        result=module.evaluate(analyze(c,p,a,m),p,fit,calibration)
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['gates']['real_nonzero_intervention'])
        self.assertTrue(result['gates']['mean_cost_better_than_reference'])
        self.assertTrue(result['gates']['mean_cost_better_than_raw_history6'])
        self.assertEqual(result['aggregate']['calibrated_candidate']['mean_delta_vs_reference']['cost'],-20./15079)
        self.assertFalse(result['passed'])

    def test_missing_tail_on_an_unselected_root_keeps_the_whole_V_incomplete(self):
        c,p,a,m,fit,calibration=fixture()
        c[0]['pairs'][2]['tail_boundary']['candidate']=None
        result=module.evaluate(analyze(c,p,a,m),p,fit,calibration)
        self.assertEqual(result['status'],'hold')
        self.assertIsNone(result['passed'])
        self.assertIsNone(result['aggregate'])
        self.assertFalse(result['integrity']['checks']['all_exact_tail_boundaries'])
        self.assertEqual(set(result['by_scene']),{'stable','local_speed_drop'})

    def test_reference_full_and_tail_ledgers_must_match_across_all_three_roots(self):
        for field in ('full','tail'):
            with self.subTest(field=field):
                c,p,a,m,fit,calibration=fixture()
                pair=c[0]['pairs'][2]
                if field == 'full':
                    pair['reference_full_endpoint']['wait']+=1.
                    pair['reference_full_endpoint']['cost']+=2.
                else:
                    pair['tail_boundary']['reference']['accrued_wait_seconds']+=1.
                result=module.evaluate(analyze(c,p,a,m),p,fit,calibration)
                self.assertEqual(result['status'],'hold')
                self.assertFalse(result['integrity']['checks']['same_reference_ledger_across_roots'])

    def test_improved_mean_does_not_hide_a_worse_scenario(self):
        c,p,a,m,fit,calibration=fixture()
        pair=c[1]['pairs'][1]
        pair['remaining']['candidate'].update(wait=1000.,ride=2010.,cost=4010.)
        pair['delta']=dict(wait=0.,ride=10.,cost=10.)
        result=module.evaluate(analyze(c,p,a,m),p,fit,calibration)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['gates']['mean_cost_better_than_reference'])
        self.assertFalse(result['gates']['no_scene_worse_than_reference'])
        self.assertEqual(result['aggregate']['calibrated_candidate']['worst_scene'],'local_speed_drop')
        self.assertEqual(result['aggregate']['calibrated_candidate']['worst_scene_delta_vs_reference'],10./15079)
        self.assertFalse(result['passed'])

    def test_full_day_gain_with_stable_tail_harm_fails_tail_gate(self):
        c,p,a,m,fit,calibration=fixture()
        c[0]['pairs'][1]['tail_boundary']['candidate']['accrued_in_vehicle_seconds']-=30.
        result=module.evaluate(analyze(c,p,a,m),p,fit,calibration)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['gates']['mean_cost_better_than_reference'])
        self.assertTrue(result['gates']['no_scene_worse_than_reference'])
        self.assertFalse(result['gates']['stable_tail_nonpositive'])
        tail=result['by_scene']['stable']['methods']['calibrated_candidate']['delta_vs_reference']['tail']
        self.assertEqual(tail['cost'],10./15079)
        self.assertFalse(result['passed'])

    def test_missing_label_or_reference_endpoint_holds_instead_of_zero_filling(self):
        for field in ('label_delta_per_passenger','reference_full_endpoint'):
            with self.subTest(field=field):
                args=evaluation_fixture()
                args[0]['pairs'][0][field]=None
                result=module.evaluate(*args)
                self.assertEqual(result['status'],'hold')
                self.assertIsNone(result['aggregate'])

    def test_changed_calibration_snapshot_or_issue_score_is_not_silently_used(self):
        args=evaluation_fixture()
        args[3]['C']+=.1
        result=module.evaluate(*args)
        self.assertEqual(result['status'],'hold')
        self.assertFalse(result['integrity']['checks']['frozen_calibration_binding'])
        args=evaluation_fixture()
        args[0]['pairs'][0]['policy_scores']['calibrated_candidate']['accept']=True
        result=module.evaluate(*args)
        self.assertEqual(result['status'],'hold')
        self.assertFalse(result['integrity']['checks']['issue_only_policy_scores'])

    def test_cli_keeps_the_reserved_evaluation_write_once(self):
        args=evaluation_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            command=[]
            for name,value in zip(('summary','protocol','fit-model','calibration'),args):
                path=root/(name+'.json')
                path.write_text(json.dumps(value))
                command.extend(('--'+name,str(path)))
            output=root/'evaluation.json'
            command.extend(('--out',str(output)))
            with redirect_stdout(io.StringIO()):
                module.main(command)
            self.assertTrue(json.loads(output.read_text())['passed'])
            with patch.object(module,'evaluate',side_effect=AssertionError('reserved evaluation repeated')):
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    module.main(command)


if __name__=='__main__':
    unittest.main()
