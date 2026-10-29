"""Synthetic C receipts only: no new T/C labels, native queries or fitting."""
from copy import deepcopy
import json
import unittest

from experiments.mcwm_closed_loop_collection_domain_v97_20261028.test_analysis import fixture as training_fixture
from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.analysis import (
    LINES, SCENES, SEEDS, analyze, predictions)


def fixture():
    original, protocol, audit, model = training_fixture()
    protocol.update(external_seeds=list(SEEDS),source_fit='/frozen/v98/model.json',
        source_fit_summary='/frozen/v98/summary.json',source_model='source_model.json',
        candidate_deployment_domain=dict(finite_historical_q=True),
        calibration=dict(alpha=.2,blocks=4,V=[5696843]),retrieval=dict(final_summary_bytes=131072))
    protocol['budget'].update(total_cap=676800,scenario_cap=84600,
        expected_collectors=8,expected_pairs=24,expected_physical_labels=48)
    bias=dict(q_range=protocol['q_range_for_scaling_only'],coef=[0.,0.],
        gram_inverse=[[.1,0.],[0.,.1]],n=12,rank=2,ridge=1.,linear_solves=1)
    binding=dict(source_fit=protocol['source_fit'],primary_model='service_history6',bias_model=bias,
        source_summary='/frozen/v97/summary.json',source_protocol='/frozen/v97/protocol.json')
    collectors=[]
    for seed in SEEDS:
        for index,scene in enumerate(SCENES):
            collector=deepcopy(original[index])
            collector.update(external_seed=seed,frozen_bias=deepcopy(binding))
            collector['reference_snapshot']['heldout_external_seed']=seed
            for pair in collector['pairs']:
                pair.update(external_seed=seed,source_fit=protocol['source_fit'])
                pair['checks']['same_frozen_bias']=True
            collectors.append(collector)
    first=collectors[0]['pairs'][0]
    gate=dict(all_pass=True,checks=deepcopy(first['checks']),root=first['root'],external_seed=SEEDS[0],
        scenario='stable',new_pair_in_formal_cohort=True,labels_used_for_gate=False,cost_sign_used_for_gate=False)
    for collector in collectors:
        collector['first_pair_gate']=deepcopy(gate)
    return collectors,protocol,audit,model


def set_q(pair, model, q):
    condition=pair['condition']
    condition['value']=q
    condition['latest_sources'][0]['normalized_slowdown']=q
    condition['latest_sources'][0]['nonholding_ready_seconds']=100.+60*q
    pair['predictions'].update(predictions(model,pair['line_id'],pair['service']['features'],q,3.5,7.5))


def calibration_fixture():
    collectors,protocol,audit,model=fixture()
    model['models']['service_history6']['coef']=[-1.875]*3+[0.]*3
    for collector in collectors:
        delta=-40.+10*SEEDS.index(collector['external_seed'])
        for pair in collector['pairs']:
            set_q(pair,model,.1)
            pair['remaining']['candidate'].update(wait=1000.,ride=2000.+delta,cost=4000.+delta)
            pair['delta']=dict(wait=0.,ride=delta,cost=delta)
    summary=analyze(collectors,protocol,audit,model)
    binding=summary['frozen_bias']
    fit_model={key:deepcopy(binding[key]) for key in
        ('primary_model','bias_model','source_summary','source_protocol')}
    fit_model.update(candidate_deployment_domain=deepcopy(protocol['candidate_deployment_domain']),
        integrity=dict(all_pass=True,checks=dict(T_integrity=True,fixed_fit_numerical=True)),
        budget=dict(real_bias_fits=1,native_seconds=0,new_labels=0,optimizer_updates=0,new_initialization_updates=0))
    fit_summary=dict(status='complete',decision=dict(ready_for_calibration=True),
        validation=dict(all_pass=True),numerical=dict(all_pass=True))
    return summary,protocol,fit_model,fit_summary


class AnalysisTest(unittest.TestCase):
    def test_eight_collectors_twenty_four_pairs_preserve_small_full_summary(self):
        report=analyze(*fixture())
        self.assertEqual(report['status'],'complete')
        self.assertEqual(len(report['pairs']),24)
        self.assertEqual(report['budget']['labels_observed'],48)
        self.assertEqual(report['interpretation']['model_fits'],0)
        size=len(json.dumps(report,separators=(',',':'),allow_nan=False).encode())
        self.assertGreater(size,65536)
        self.assertLess(size,131072)

    def test_C_rank_one_is_descriptive_and_does_not_prevent_calibration(self):
        report,*_=calibration_fixture()
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['covariate_design']['rank'],1)
        self.assertFalse(report['covariate_design']['rank_is_admission_gate'])

    def test_outside_old_range_and_complete_noop_keep_the_measured_label(self):
        c,p,a,m=fixture()
        pair=c[0]['pairs'][0]
        set_q(pair,m,-.0013855844736099243)
        pair['physical']['candidate']['departure_clock']=pair['physical']['reference']['departure_clock']
        pair['checks']['physical_departure_separation']=False
        for collector in c:
            collector['first_pair_gate']['checks']=deepcopy(pair['checks'])
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'complete')
        row=report['pairs'][0]
        self.assertTrue(row['physical_noop'])
        self.assertTrue(row['training_support']['extrapolation'])
        self.assertEqual(row['label_delta_per_passenger']['cost'],10./15079)

    def test_tail_boundary_only_scores_and_missing_boundary_does_not_delete_C_truth(self):
        c,p,a,m=fixture()
        pair=c[0]['pairs'][0]
        pair['tail_boundary']=dict(reference=dict(time_seconds=18300.,accrued_wait_seconds=1000.,accrued_in_vehicle_seconds=2000.),
            candidate=dict(time_seconds=18300.,accrued_wait_seconds=1002.,accrued_in_vehicle_seconds=2001.))
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'complete')
        row=report['pairs'][0]
        self.assertEqual(row['tail']['delta_per_passenger']['cost'],5./15079)
        self.assertEqual(row['tail_boundary'],pair['tail_boundary'])
        pair['tail_boundary']['candidate']=None
        missing=analyze(c,p,a,m)
        self.assertEqual(missing['status'],'complete')
        self.assertIsNone(missing['pairs'][0]['tail']['delta_per_passenger'])
        self.assertEqual(missing['pairs'][0]['predictions'],row['predictions'])
        self.assertEqual(missing['pairs'][0]['label_delta_per_passenger'],row['label_delta_per_passenger'])

    def test_missing_collector_preserves_the_missing_block_without_zero_fill(self):
        c,p,a,m=fixture()
        c.pop()
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertEqual(len(report['pairs']),21)
        self.assertEqual(len(report['exceptions']['missing']),3)
        self.assertEqual(report['budget']['labels_observed'],42)

    def test_held_native_error_is_reported_without_erasing_actual_pairs_or_actor(self):
        c,p,a,m=fixture()
        c[0].update(status='held',error=dict(type='RuntimeError',message='fixture'))
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertEqual(len(report['pairs']),24)
        self.assertEqual(len(report['exceptions']['native_failures']),1)
        self.assertTrue(report['scene_checks'][str(SEEDS[0])+'/stable']['original_actor'])

    def test_all_collectors_must_bind_the_same_frozen_bias(self):
        c,p,a,m=fixture()
        c[-1]['frozen_bias']['bias_model']['coef'][0]=1.
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertFalse(report['scene_checks'][str(SEEDS[-1])+'/'+SCENES[-1]]['frozen_bias_matches'])


if __name__=='__main__':
    unittest.main()
