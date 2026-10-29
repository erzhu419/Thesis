"""Synthetic V receipts and frozen policy snapshots; no collected T/C/V data."""
from copy import deepcopy
import json
import unittest

from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.test_analysis import (
    fixture as calibration_fixture, set_q)
from experiments.mcwm_closed_loop_confidence_validation_v100_20261028.analysis import (
    SEEDS, analyze, policy_scores)
from experiments.mcwm_closed_loop_confidence_validation_v100_20261028.evaluate import METHODS


def fixture(candidate_margin=.05):
    original,p,a,m=calibration_fixture()
    c=deepcopy(original[:2])
    p.update(external_seeds=list(SEEDS),source_calibration='/frozen/v99/calibration.json',
             validation=dict(comparators=list(METHODS)))
    p['budget'].update(total_cap=169200,expected_collectors=2,expected_pairs=6,expected_physical_labels=12)
    p['retrieval']['final_summary_bytes']=65536
    binding=deepcopy(c[0]['frozen_bias'])
    binding['bias_model']['coef']=[0.,.5]
    fit_model={key:deepcopy(binding[key]) for key in
        ('primary_model','bias_model','source_summary','source_protocol')}
    fit_model.update(candidate_deployment_domain=deepcopy(p['candidate_deployment_domain']),
        integrity=dict(all_pass=True,checks=dict(T_integrity=True,fixed_fit_numerical=True)),
        budget=dict(real_bias_fits=1,native_seconds=0,new_labels=0,optimizer_updates=0,new_initialization_updates=0))
    calibration=dict(status='complete',ready_for_validation=True,alpha=.2,C=candidate_margin,
        state_independent=dict(C=.21),frozen_model=deepcopy(fit_model),
        integrity=dict(all_pass=True,checks=dict(four_complete_C_blocks=True)),
        source_identity=dict(source_fit=p['source_fit']))
    calibration_binding=dict(source_calibration=p['source_calibration'],source_fit=p['source_fit'],
        alpha=.2,C=candidate_margin,state_independent_C=.21,ready_for_validation=True)
    for scene in c:
        scene.update(external_seed=SEEDS[0],frozen_bias=deepcopy(binding),frozen_calibration=deepcopy(calibration_binding))
        scene['reference_snapshot']['heldout_external_seed']=SEEDS[0]
        for pair,q,(wait,ride) in zip(scene['pairs'],(.5,.01,.1),((10.,10.),(-5.,-10.),(-100.,0.))):
            pair.update(external_seed=SEEDS[0],source_calibration=p['source_calibration'],
                reference_full_endpoint=dict(wait=20000.,ride=30000.,cost=70000.))
            set_q(pair,m,q)
            pair['remaining']['candidate'].update(wait=1000.+wait,ride=2000.+ride,cost=4000.+2*wait+ride)
            pair['delta']=dict(wait=wait,ride=ride,cost=2*wait+ride)
            saved=policy_scores(pair['predictions']['service_history6'],q,binding,calibration_binding)
            pair['policy_scores']=saved
            scene['coverage'][pair['line_id']]['policy_scores']=saved
            pair['checks'].update(same_frozen_calibration=True,same_issue_policy_scores=True)
            pair['tail_boundary']={role:dict(time_seconds=18300.,scheduled_time_seconds=18300.,
                accrued_wait_seconds=19000.,accrued_in_vehicle_seconds=28000.,
                waiting_passengers=100,in_vehicle_passengers=200,completed_passengers=14000)
                for role in ('reference','candidate')}
            pair['tail_boundary_status']={role:'observed_exact' for role in ('reference','candidate')}
    first=c[0]['pairs'][0]
    gate=dict(all_pass=True,checks=deepcopy(first['checks']),root=first['root'],external_seed=SEEDS[0],
        scenario='stable',new_pair_in_formal_cohort=True,labels_used_for_gate=False,cost_sign_used_for_gate=False)
    for scene in c:
        scene['first_pair_gate']=deepcopy(gate)
    return c,p,a,m,fit_model,calibration


def evaluation_fixture(**kwargs):
    c,p,a,m,fit,calibration=fixture(**kwargs)
    return analyze(c,p,a,m),p,fit,calibration


class AnalysisTest(unittest.TestCase):
    def test_one_V_seed_two_scenes_six_pairs_and_twelve_labels(self):
        report,*_=evaluation_fixture()
        self.assertEqual(report['status'],'complete')
        self.assertEqual(len(report['pairs']),6)
        self.assertEqual(report['budget']['labels_observed'],12)
        self.assertEqual(report['interpretation']['independent_external_seeds'],1)
        self.assertTrue(all(row['tail']['observed'] for row in report['pairs']))
        self.assertLess(len(json.dumps(report,allow_nan=False).encode()),65536)

    def test_saved_policy_acceptance_must_match_issue_only_frozen_forward(self):
        c,p,a,m,*_=fixture()
        c[0]['pairs'][0]['policy_scores']['calibrated_candidate']['accept']=True
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertFalse(report['pairs'][0]['checks']['saved_policy_scores_exact'])

    def test_missing_tail_keeps_valid_full_day_labels_for_later_evaluation_hold(self):
        c,p,a,m,*_=fixture()
        c[0]['pairs'][2]['tail_boundary']['candidate']=None
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'complete')
        self.assertTrue(report['pairs'][2]['valid'])
        self.assertFalse(report['pairs'][2]['tail']['observed'])
        self.assertEqual(report['budget']['labels_observed'],12)

    def test_complete_noop_retains_cost_and_fixed_policy_selection(self):
        c,p,a,m,*_=fixture()
        pair=c[0]['pairs'][1]
        pair['physical']['candidate']['departure_clock']=pair['physical']['reference']['departure_clock']
        pair['checks']['physical_departure_separation']=False
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'complete')
        self.assertTrue(report['pairs'][1]['physical_noop'])
        self.assertEqual(report['pairs'][1]['label_delta_per_passenger']['cost'],-20./15079)
        self.assertTrue(report['pairs'][1]['policy_scores']['calibrated_candidate']['accept'])

    def test_scene_calibration_binding_drift_cannot_pass(self):
        c,p,a,m,*_=fixture()
        c[1]['frozen_calibration']['C']+=.1
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertFalse(report['scene_checks'][str(SEEDS[0])+'/local_speed_drop']['frozen_calibration_matches'])


if __name__=='__main__':
    unittest.main()
