from copy import deepcopy
import json
import unittest

from experiments.mcwm_closed_loop_cost_advantage_v85_20261027.analysis import analyze


def artifact():
    return dict(gate=dict(all_pass=True),fits=2,ridge=1.,denominator=15079.,hold_scale=7.5,
        models=dict(line=dict(design_rank=3),state=dict(design_rank=4)),
        training_scope=dict(source_scope='v84_train',pairs=6,physical_outcome_labels=12,
            source_seed=6207,external_seed=106207,label_available_at=24030,
            cases=[scene+'/'+line for scene in ('stable','local_speed_drop')
                   for line in ('7X','102X','705X')],heldout_labels_used=0),
        actor_updates=0,critic_updates=0,value_updates=0,B_updates=0,world_updates=0,trust_updates=0)


def scenes():
    result=[]
    for scenario,targets in (('stable',(-30.,20.,-10.)),('local_speed_drop',(-5.,10.,-2.))):
        pairs=[];selection={}
        for index,(line,delta) in enumerate(zip(('7X','102X','705X'),targets)):
            root=dict(started_at=12005.+index,sequence=100+index,trip_id=line+'_9',station_id=line+'15')
            reference=dict(wait=100.,ride=800.,cost=1000.,dt=24000-root['started_at'],
                           observed_at=24000,available_at=24030)
            candidate=dict(reference,ride=800.+delta,cost=1000.+delta)
            prediction=dict(supported=True,reason=None,z=.2,models=dict(
                line=dict(predicted_delta=1.,choice='reference'),
                state=dict(predicted_delta=-1. if delta<0 else 1.,
                           choice='candidate' if delta<0 else 'reference')))
            pair=dict(status='complete',all_pass=True,checks=dict(native_fork=True),line_id=line,
                root=root,reference_hold_seconds=3.3,candidate_hold_seconds=7.5,
                prediction=prediction,prediction_recorded_at=root['started_at'],before_suffixes=True,
                overlap_seconds=20.,remaining=dict(reference=reference,candidate=candidate),
                delta=dict(wait=0.,ride=delta,cost=delta),
                physical=dict(reference=dict(departure_clock=12006.,next_arrival_clock=12090.,missing={}),
                              candidate=dict(departure_clock=12011.,next_arrival_clock=12092.,missing={})),
                reference_full_endpoint=dict(wait=10000.,ride=30000.,cost=50000.))
            pairs.append(pair)
            selection[line]=dict(selected=root,attempted=1,reasons={},status='selected')
        old=dict(source='v75',source_arm='fresh',seed=6207,scenario=scenario,
            packets=37,covariates=185,complete_outcomes=185,
            counts=dict(critic=296,value=296,actor=296,alpha=296),data_deadline=12000,
            max_available_at=12000.,published_clock=12001.,deterministic=True,actor_frozen=True)
        reference_snapshot=dict(prior_run_reference=True,available_before_heldout_episode=True,
                                source_snapshot_metadata=old)
        result.append(dict(status='complete',scenario=scenario,seed=6207,external_seed=216207,
            prefix_id='v85/'+scenario,departed_passengers=15079,
            integrity=dict(all_pass=True,checks={}),reference_snapshot=reference_snapshot,
            budget=dict(native_prefix_seconds=12020.,suffix_native_seconds=71950.,
                total_native_seconds=83970.,prefixes=1,suffixes=6,labels_observed=6,
                new_initialization_updates=0,prefix_training_updates=0,suffix_updates=0,labels_fed_back=0),
            pairs=pairs,selection=selection))
    return result


class AnalysisTest(unittest.TestCase):
    def test_sealed_choices_cost_regret_and_primary_gate(self):
        report=analyze(scenes(),artifact())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertTrue(report['promotion']['passed'])
        state=report['overall']['models']['state']
        self.assertEqual(state['sum_selected_minus_reference']['cost'],-47.)
        self.assertEqual(state['sum_two_action_regret'],0.)
        self.assertEqual(report['overall']['models']['line']['sum_two_action_regret'],47.)
        self.assertEqual(state['candidate_choices'],4)
        self.assertLess(len(json.dumps(report).encode()),48000)

    def test_ood_is_explicit_reference_fallback_not_zero_prediction(self):
        values=scenes();pair=values[0]['pairs'][0]
        pair['reference_hold_seconds'],pair['candidate_hold_seconds']=4.,0.
        pair['prediction']=dict(supported=False,reason='nonpositive_holding_difference',z=.2,
            models={name:dict(predicted_delta=None,choice='reference') for name in ('line','state')})
        report=analyze(values,artifact())
        self.assertTrue(report['integrity']['all_pass'])
        row=report['pairs'][0]
        self.assertIsNone(row['models']['state']['predicted_delta'])
        self.assertEqual(row['models']['state']['selected_minus_reference']['cost'],0.)
        self.assertEqual(row['models']['state']['two_action_regret'],30.)
        self.assertEqual(report['overall']['models']['state']['prediction_count'],5)
        self.assertEqual(len(report['exceptions']['outside_action_support']),1)
        pair['prediction']['models']['state']['predicted_delta']=0.
        self.assertFalse(analyze(values,artifact())['integrity']['all_pass'])

    def test_missing_is_not_imputed_and_prevents_promotion(self):
        values=scenes();values[1]['pairs'].pop()
        values[1]['selection']['705X']['selected']=None
        values[1]['selection']['705X']['status']='missing'
        report=analyze(values,artifact())
        self.assertEqual(len(report['pairs']),5)
        self.assertEqual(len(report['exceptions']['missing']),1)
        self.assertFalse(report['promotion']['passed'])
        self.assertFalse(report['reference_endpoint_audit']['local_speed_drop']['all_equal'])

    def test_future_prediction_or_heldout_label_in_fit_fails(self):
        values=scenes();values[0]['pairs'][0]['before_suffixes']=False
        self.assertFalse(analyze(values,artifact())['integrity']['all_pass'])
        fitted=artifact();fitted['training_scope']['heldout_labels_used']=1
        self.assertFalse(analyze(scenes(),fitted)['training_checks']['no_heldout_truth_in_fit'])

    def test_reference_endpoint_or_budget_break_blocks_promotion(self):
        values=scenes();values[0]['pairs'][1]['reference_full_endpoint']['wait']+=1
        report=analyze(values,artifact())
        self.assertFalse(report['integrity']['checks']['reference_full_day_endpoints_equal'])
        self.assertFalse(report['promotion']['passed'])
        values=scenes();values[0]['budget']['total_native_seconds']+=2000
        values[0]['budget']['native_prefix_seconds']+=2000
        self.assertFalse(analyze(values,artifact())['integrity']['checks']['fixed_native_budget'])

    def test_matching_line_baseline_does_not_promote_state_or_all_abstention(self):
        values=scenes()
        for scene in values:
            for pair in scene['pairs']:
                pair['prediction']['models']['line']=deepcopy(pair['prediction']['models']['state'])
        report=analyze(values,artifact())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertFalse(report['promotion']['checks']['state_beats_line_baseline'])
        for scene in values:
            for pair in scene['pairs']:
                pair['prediction']['models']['state']=dict(predicted_delta=1.,choice='reference')
        report=analyze(values,artifact())
        self.assertFalse(report['promotion']['checks']['state_actually_changes_an_action'])
        self.assertFalse(report['promotion']['passed'])

    def test_single_intervention_takes_first_negative_not_sum_of_later_gains(self):
        values=scenes()
        first=values[0]['pairs'][0]
        first['remaining']['candidate']['ride']=815.
        first['remaining']['candidate']['cost']=1015.
        first['delta']=dict(wait=0.,ride=15.,cost=15.)
        later=values[0]['pairs'][2]
        later['remaining']['candidate']['ride']=700.
        later['remaining']['candidate']['cost']=900.
        later['delta']=dict(wait=0.,ride=-100.,cost=-100.)
        report=analyze(values,artifact())
        self.assertLess(report['by_scenario']['stable']['models']['state']['sum_selected_minus_reference']['cost'],0.)
        single=report['single_intervention_shadow']['by_scenario']['stable']
        state=single['models']['state']
        self.assertEqual(state['selected_root']['line_id'],'7X')
        self.assertEqual(state['interventions'],1)
        self.assertEqual(state['selected_minus_reference']['cost'],15.)
        self.assertEqual(state['full_selected_endpoint']['cost'],50015.)
        self.assertEqual(state['single_opportunity_oracle_regret'],115.)
        self.assertEqual(single['state_minus_line_selected_cost'],15.)
        self.assertFalse(report['promotion']['checks']['single_state_non_degradation'])
        self.assertFalse(report['promotion']['passed'])

    def test_single_intervention_skips_ood_and_no_negative_means_baseline(self):
        values=scenes();first=values[0]['pairs'][0]
        first['reference_hold_seconds'],first['candidate_hold_seconds']=4.,0.
        first['prediction']=dict(supported=False,reason='nonpositive_holding_difference',z=.2,
            models={name:dict(predicted_delta=None,choice='reference') for name in ('line','state')})
        report=analyze(values,artifact())
        single=report['single_intervention_shadow']['by_scenario']['stable']
        self.assertEqual(single['models']['state']['selected_root']['line_id'],'705X')
        self.assertEqual(single['models']['state']['selected_minus_reference']['cost'],-10.)
        self.assertIsNone(single['models']['line']['selected_root'])
        self.assertEqual(single['models']['line']['interventions'],0)
        self.assertEqual(single['models']['line']['selected_minus_reference']['cost'],0.)
        self.assertEqual(single['models']['line']['full_selected_endpoint'],dict(wait=10000.,ride=30000.,cost=50000.))


if __name__=='__main__':
    unittest.main()
