"""Synthetic collection receipts only; no native query or target fitting."""
from copy import deepcopy
import json
import unittest

from experiments.mcwm_closed_loop_confidence_collection_v96_20261028.analysis import (
    BUDGET, COSTS, LINES, MODELS, SCENES, SEEDS, ZERO, analyze, predictions,
)
from experiments.mcwm_closed_loop_service_response_v88_20261028.service import service_state


def fixture():
    support = dict(q_training_ranges={'7X':[.01,.47],'102X':[.01,.47],'705X':[.01,.50]})
    protocol = dict(source_initializer=6207,external_seeds=list(SEEDS),scenarios=list(SCENES),
        frozen_common_support=support,q_range_for_scaling_only=[.01,.50],budget=dict(total_cap=338400))
    model = dict(common_support=deepcopy(support), design=dict(line_centers={line:[.2,-.76,.1] for line in LINES}),
        models={name:dict(coef=[-.5,-.4,-.3]+([.1,.2] if name != 'line3' else [])
                          +([.3] if name == 'service_history6' else [])) for name in MODELS})
    paths = {scene:'/resident/'+scene+'/actor.pt' for scene in SCENES}
    audit = dict(all_pass=True,checks=dict(original_actor=True),source_initializer=6207,source_reference_paths=paths,
        new_fits=0,new_neural_updates=0,new_suffixes=0,new_labels=0,v87_truth_used=0,native_seconds=0)
    collectors=[]
    for seed in SEEDS:
        for scene in SCENES:
            q = .1 if scene == 'stable' else .4
            pairs,coverage,selection=[],{},{}
            for index,line in enumerate(LINES):
                now = 12001.+index
                root = dict(started_at=now,sequence=100+index,trip_id=line+'_1',station_id=line+'10')
                ctx = dict(observed=True,reasons=[],capacity=50.,
                    counts=dict(onboard=10,release_next=2,compatible_next_queue=4,seats_after_release=42),
                    clocks=dict(issue=now,event=now-1,obs_start=now,obs_end=now,atomic_at_issue=True))
                service=service_state(ctx)
                condition=dict(status='supported',reason=None,value=q,historical_only=True,issue_clock=now,
                    baseline_before=9000,baseline_counts={line:3 for line in LINES},
                    baseline_medians_seconds={line:100. for line in LINES},
                    baseline_latest_completed_at={line:8000. for line in LINES},
                    baseline_latest_observed_at={line:8000. for line in LINES},
                    latest_completed_at=11900.,latest_observed_at=11900.,latest_age_seconds=now-11900.,
                    latest_sources=[dict(line_id=line,trip_id=line+'_old',completed_at=11900.,observed_at=11900.,
                        nonholding_ready_seconds=100.+60*q,baseline_nonholding_seconds=100.,normalized_slowdown=q)])
                saved=predictions(model,line,service['features'],q,3.5,7.5)
                slot=dict(root=root,context=ctx,service=service,condition=condition,predictions=saved,
                    observation_checks=dict(clock_unchanged=True,ledger_unchanged=True,rng_unchanged=True),status='supported')
                coverage[line]=slot
                selection[line]=dict(selected=root,status='complete',reasons={})
                ref=dict(wait=1000.,ride=2000.,cost=4000.,dt=24000-now,observed_at=24000,available_at=24030)
                cand=dict(ref,wait=1005.,cost=4010.)
                checks={key:True for key in ('same_complete_root','same_fixed_reference','same_external_seed',
                    'native_fork','zero_suffix_training','zero_truth_feedback','actual_parallel')}
                pair=dict(status='complete',all_pass=True,checks=checks,line_id=line,scenario=scene,external_seed=seed,
                    root=root,service=service,context=ctx,condition=condition,predictions=saved,reference_hold_seconds=3.5,
                    candidate_hold_seconds=7.5,sim_step_seconds=1.,action_step_seconds=1.,overlap_seconds=30.,
                    physical={role:dict(departure_clock=now+hold,next_arrival_clock=now+90+offset)
                        for role,hold,offset in (('reference',4,0),('candidate',8,1))},
                    remaining=dict(reference=ref,candidate=cand),delta={key:cand[key]-ref[key] for key in COSTS},
                    before_suffixes=True,prediction_recorded_at=now)
                pairs.append(pair)
            old=dict(source='v75',source_arm='fresh',seed=6207,scenario=scene,packets=37,covariates=185,
                complete_outcomes=185,counts=dict(critic=296,value=296,actor=296,alpha=296),
                max_available_at=12000,data_deadline=12000,deterministic=True,actor_frozen=True)
            reference=dict(prior_run_reference=True,available_before_heldout_episode=True,source_snapshot_metadata=old,
                           source_actor_path=paths[scene],heldout_external_seed=seed)
            suffix=sum(2*pair['remaining']['reference']['dt'] for pair in pairs)
            budget={key:0 for key in BUDGET}
            budget.update(prefixes=1,suffixes=6,native_prefix_seconds=12600,suffix_native_seconds=suffix,
                          total_native_seconds=12600+suffix,labels_observed=6)
            collectors.append(dict(status='complete',seed=6207,scenario=scene,external_seed=seed,
                integrity=dict(all_pass=True,checks=dict(actor_frozen=True)),reference_snapshot=reference,
                pairs=pairs,coverage=coverage,selection=selection,budget=budget,
                labels_used_for_selection=False,predictions_used_for_selection=False,training_started=False,models_fitted=False))
    gate=dict(all_pass=True,checks=deepcopy(collectors[0]['pairs'][0]['checks']),root=collectors[0]['pairs'][0]['root'],external_seed=SEEDS[0],
        scenario='stable',new_pair_in_formal_cohort=True,labels_used_for_gate=False,cost_sign_used_for_gate=False)
    for scene in collectors:scene['first_pair_gate']=deepcopy(gate)
    return collectors,protocol,audit,model


class AnalysisTest(unittest.TestCase):
    def test_complete_collection_needs_no_target_fit(self):
        report=analyze(*fixture())
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['future_bias_design']['rank'],2)
        self.assertEqual(report['interpretation']['model_fits'],0)
        self.assertLess(len(json.dumps(report).encode()),64000)

    def test_same_clock_history_is_not_future_supervision(self):
        c,p,a,m=fixture()
        condition=c[0]['pairs'][0]['condition']
        condition['latest_sources'][0]['completed_at']=c[0]['pairs'][0]['root']['started_at']
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertFalse(report['pairs'][0]['checks']['strict_historical_q'])

    def test_original_per_line_q_range_is_not_global_scaling_range(self):
        c,p,a,m=fixture()
        condition=c[0]['pairs'][0]['condition']
        condition['value']=.49
        condition['latest_sources'][0]['normalized_slowdown']=.49
        report=analyze(c,p,a,m)
        self.assertFalse(report['pairs'][0]['checks']['original_line_q_range'])
        self.assertIsNone(report['pairs'][0]['label_delta_per_passenger'])

    def test_nonseparable_cost_is_preserved_but_not_admitted(self):
        c,p,a,m=fixture()
        pair=c[0]['pairs'][0]
        pair['physical']['candidate']['departure_clock']=pair['physical']['reference']['departure_clock']
        report=analyze(c,p,a,m)
        self.assertEqual(len(report['exceptions']['nonseparable']),1)
        self.assertEqual(report['pairs'][0]['observed_delta']['cost'],10.)
        self.assertIsNone(report['pairs'][0]['label_delta_per_passenger'])

    def test_blocked_collectors_produce_hold_and_missing_without_zero_labels(self):
        c,p,a,m=fixture()
        for scene in c[1:]:
            scene.update(status='not_started',pairs=[],coverage={})
            scene['selection']={line:dict(selected=None,status='missing',reasons={}) for line in LINES}
            scene['budget']={key:0 for key in BUDGET}
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertEqual(len(report['exceptions']['missing']),9)
        self.assertEqual(report['budget']['labels_observed'],6)
        self.assertEqual(len(report['pairs']),3)

    def test_missing_pair_does_not_become_an_unchanged_outcome(self):
        c,p,a,m=fixture()
        c[0]['pairs'].pop()
        report=analyze(c,p,a,m)
        self.assertEqual(len(report['exceptions']['missing']),1)
        self.assertEqual(len(report['pairs']),11)

    def test_saved_prediction_is_recomputed_even_when_runtime_flags_pass(self):
        c,p,a,m=fixture()
        c[0]['pairs'][0]['predictions']['service_history6']+=1.
        report=analyze(c,p,a,m)
        self.assertFalse(report['pairs'][0]['checks']['saved_prediction_exact'])

    def test_runtime_receipt_must_cover_prefix_rng_and_training(self):
        c,p,a,m=fixture()
        del c[0]['pairs'][0]['checks']['same_external_seed']
        report=analyze(c,p,a,m)
        self.assertFalse(report['pairs'][0]['checks']['same_prefix_and_rng'])

    def test_preflight_cannot_be_a_later_selected_root(self):
        c,p,a,m=fixture()
        for scene in c:scene['first_pair_gate']['root']=c[0]['pairs'][1]['root']
        report=analyze(c,p,a,m)
        self.assertFalse(report['integrity']['checks']['exact_first_physical_gate'])

    def test_budget_and_training_are_separate_gates(self):
        c,p,a,m=fixture()
        c[0]['budget']['total_native_seconds']=338401
        c[0]['budget']['new_fits']=1
        report=analyze(c,p,a,m)
        self.assertFalse(report['integrity']['checks']['within_budget'])
        self.assertFalse(report['integrity']['checks']['zero_training_fitting_or_feedback'])

    def test_truncated_endpoint_and_duplicate_query_are_visible(self):
        c,p,a,m=fixture()
        c[0]['pairs'][0]['remaining']['candidate']['observed_at']=23999
        report=analyze(c,p,a,m)
        self.assertEqual(len(report['exceptions']['truncated']),1)
        c[0]['pairs'].append(c[0]['pairs'][0])
        with self.assertRaises(ValueError):analyze(c,p,a,m)

    def test_failure_only_receipt_preserves_known_cost_or_marks_unknown_lower_bound(self):
        c,p,a,m=fixture()
        rest=sum(scene['budget']['total_native_seconds'] for scene in c[:-1])
        c[-1]=dict(external_seed=SEEDS[-1],scenario=SCENES[-1],status='failed',
            reason='native failure without collector summary',failure=dict(type='RuntimeError',message='fixture'),native_seconds=None)
        report=analyze(c,p,a,m)
        self.assertEqual(report['status'],'hold')
        self.assertEqual(len(report['exceptions']['missing']),3)
        self.assertEqual(report['budget']['total_native_seconds'],rest)
        self.assertTrue(report['budget']['total_native_seconds_is_lower_bound'])
        self.assertEqual(report['budget']['unknown_suffix_costs'],1)
        self.assertFalse(report['integrity']['checks']['within_budget'])
        spent={key:0 for key in BUDGET}
        spent.update(prefixes=1,native_prefix_seconds=5120,total_native_seconds=5120)
        c[-1]['failure']['consumed_budget']=spent
        known=analyze(c,p,a,m)
        self.assertEqual(known['budget']['total_native_seconds'],rest+5120)
        self.assertFalse(known['budget']['total_native_seconds_is_lower_bound'])
        self.assertEqual(known['interpretation']['model_fits'],0)


if __name__=='__main__':unittest.main()
