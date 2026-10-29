from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.mcwm_closed_loop_fresh_response_v86_20261027.test_analysis import (
    artifact as source_artifact, scenes as source_scenes, set_delta, set_prediction,
)
from experiments.mcwm_closed_loop_frozen_response_confirmation_v87_20261028.analysis import (
    POLICY, SEEDS, analyze,
)


def model():
    result = source_artifact()
    result['feature_names'] = ['line_7X', 'line_102X', 'line_705X', 'headway_response']
    result['source_reference_paths'] = {scene: '/resident/v84/'+scene+'/reference_actor.pt'
        for scene in ('stable', 'local_speed_drop')}
    for arm in result['arms'].values():
        arm['models']['line']['coef'] = [.1, .2, .3]
        arm['models']['state']['coef'] = [.1, .2, .3, .4]
    return result


def audit():
    source = model()
    frozen = {key: deepcopy(source[key]) for key in ('action_support', 'legacy_action_support',
        'ridge', 'denominator', 'hold_scale', 'feature_names')}
    frozen.update(decision_threshold=0, policy=POLICY,
        coefficients={arm: {name: source['arms'][arm]['models'][name]['coef']
            for name in ('line', 'state')} for arm in ('stale', 'fresh')})
    return dict(all_pass=True, checks=dict(source_model_exact=True, original_reference_roots=True),
        source_model_path='/resident/v86/model.json', source_initializer=6207,
        confirmation_seeds=list(SEEDS), original_training_fits=4, new_fits=0,
        new_neural_updates=0, heldout_truth_used=0,
        source_reference_paths=source['source_reference_paths'], frozen=frozen)


def scenes():
    values = []
    for seed in SEEDS:
        for scene in source_scenes():
            scene['external_seed'] = seed
            scene['prefix_id'] = 'v87/external'+str(seed)+'/'+scene['scenario']
            scene['reference_snapshot']['heldout_external_seed'] = seed
            scene['frozen_source'] = dict(model_path='/resident/v86/model.json',
                freeze_audit_path='/resident/v87/freeze_audit.json', new_fits=0, source_model_exact=True)
            scene['budget']['new_fits'] = 0
            values.append(scene)
    return values


class ConfirmationAnalysisTest(unittest.TestCase):
    def test_six_new_cells_pass_without_pooling_source_result(self):
        inputs = scenes()
        before = deepcopy(inputs)
        report = analyze(inputs, audit(), model())
        self.assertEqual(inputs, before)
        self.assertTrue(report['integrity']['all_pass'])
        self.assertTrue(report['promotion']['passed'])
        main = report['primary']['overall']
        self.assertEqual(main['cells'], 6)
        self.assertEqual(main['models']['fresh']['interventions'], 6)
        self.assertAlmostEqual(main['comparisons']['fresh_minus_reference']['mean']['cost'], -17.5/15079)
        self.assertEqual(main['comparisons']['fresh_minus_stale']['wins'], 6)
        self.assertEqual(main['comparisons']['fresh_minus_stale']['loss_rate'], 0.)
        self.assertEqual(main['comparisons']['stale_minus_legacy_narrow']['ties'], 6)
        self.assertEqual(report['budget']['labels_observed'], 36)
        self.assertEqual(report['budget']['new_fits'], 0)
        self.assertTrue(report['interpretation']['old_v86_not_pooled'])
        self.assertEqual(len(report['pairs']), 18)
        self.assertLess(len(json.dumps(report).encode()), 128000)
        self.assertNotIn('runtime_checks', report['pairs'][0])

    def test_first_wrong_intervention_cannot_be_hidden_by_later_query_gains(self):
        values = scenes()
        set_delta(values[0]['pairs'][0], 15.)
        set_delta(values[0]['pairs'][2], -100.)
        report = analyze(values, audit(), model())
        self.assertTrue(report['promotion']['checks']['fresh_mean_beats_reference'])
        self.assertLess(report['query_metrics']['overall']['models']['fresh']['sum_selected_minus_reference']['cost'], 0.)
        cell = report['primary']['by_cell']['416207/stable']
        self.assertEqual(cell['models']['fresh']['selected_root']['line_id'], '7X')
        self.assertEqual(cell['models']['fresh']['selected_minus_reference']['cost'], 15.)
        self.assertEqual(cell['models']['fresh']['full_selected_endpoint']['cost'], 50015.)
        self.assertEqual(report['primary']['overall']['comparisons']['fresh_minus_reference']['worst_cell'], 15/15079)
        self.assertFalse(report['promotion']['checks']['every_cell_not_worse_than_reference'])
        self.assertFalse(report['promotion']['passed'])

    def test_each_scenario_mean_and_each_cell_are_required(self):
        values = scenes()
        for scene in values:
            if scene['scenario'] == 'local_speed_drop':
                set_delta(scene['pairs'][0], 1.)
        report = analyze(values, audit(), model())
        self.assertTrue(report['promotion']['checks']['fresh_mean_beats_reference'])
        self.assertFalse(report['promotion']['checks']['local_speed_drop_mean_not_worse_than_reference'])
        self.assertFalse(report['promotion']['passed'])

    def test_no_false_fresh_gain_if_stale_matches_fresh_or_everyone_abstains(self):
        values = scenes()
        for scene in values:
            for pair in scene['pairs']:
                pair['prediction']['models']['stale'] = deepcopy(pair['prediction']['models']['fresh'])
                pair['prediction']['models']['legacy_narrow'] = deepcopy(pair['prediction']['models']['fresh'])
        report = analyze(values, audit(), model())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertFalse(report['promotion']['checks']['fresh_mean_beats_stale'])
        for scene in values:
            for pair in scene['pairs']:
                for name in ('fresh', 'stale', 'legacy_narrow'):
                    set_prediction(pair, name, 1.)
        report = analyze(values, audit(), model())
        self.assertFalse(report['promotion']['checks']['fresh_actually_intervenes'])
        self.assertFalse(report['promotion']['passed'])

    def test_real_new_seed_is_audited_and_old_seed_not_relabelled(self):
        values = scenes()
        values[0]['external_seed'] = 316207
        with self.assertRaisesRegex(ValueError, 'new declared external seeds'):
            analyze(values, audit(), model())
        self.assertEqual(values[0]['external_seed'], 316207)
        values = scenes()
        values[0]['reference_snapshot']['heldout_external_seed'] = 316207
        report = analyze(values, audit(), model())
        self.assertFalse(report['seed_integrity']['416207']['integrity']['checks']['reference_metadata_actual_seed'])
        self.assertFalse(report['promotion']['passed'])

    def test_missing_scene_or_state_is_none_in_primary_and_explicit_exception(self):
        values = scenes()[:-1]
        report = analyze(values, audit(), model())
        self.assertEqual(len(report['exceptions']['missing']), 3)
        self.assertIsNone(report['primary']['overall']['models']['fresh']['mean_full_day_cost_per_passenger'])
        self.assertIsNone(report['primary']['overall']['comparisons']['fresh_minus_stale']['wins'])
        self.assertIsNone(report['primary']['by_cell']['616207/local_speed_drop']['models']['fresh']['full_selected_endpoint'])
        self.assertFalse(report['promotion']['passed'])
        values = scenes()
        values[0]['pairs'].pop()
        report = analyze(values, audit(), model())
        self.assertEqual(len(report['exceptions']['missing']), 1)
        self.assertFalse(report['promotion']['passed'])

    def test_ood_retained_as_none_and_guard_effect_separate(self):
        values = scenes()
        first = values[0]['pairs'][0]
        first['reference_hold_seconds'] = 3.7
        set_prediction(first, 'stale', -1.)
        set_prediction(first, 'legacy_narrow', None, False, 'reference_holding_outside_3_to_3_6')
        report = analyze(values, audit(), model())
        self.assertTrue(report['integrity']['all_pass'])
        self.assertEqual(len(report['exceptions']['outside_action_support']), 1)
        self.assertIsNone(report['pairs'][0]['models']['legacy_narrow']['predicted_delta'])
        cell = report['primary']['by_cell']['416207/stable']
        self.assertEqual(cell['comparisons_per_passenger']['stale_minus_legacy_narrow']['cost'], -30/15079)
        self.assertEqual(cell['comparisons_per_passenger']['fresh_minus_stale']['cost'], 0.)

    def test_frozen_coefficients_threshold_and_new_fit_are_hard_gates(self):
        frozen = audit()
        frozen['frozen']['coefficients']['fresh']['state'][0] += .01
        self.assertFalse(analyze(scenes(), frozen, model())['freeze_checks']['exact_frozen_model_and_rule'])
        frozen = audit()
        frozen['frozen']['decision_threshold'] = -.1
        self.assertFalse(analyze(scenes(), frozen, model())['promotion']['passed'])
        frozen = audit()
        frozen['new_fits'] = 4
        self.assertFalse(analyze(scenes(), frozen, model())['freeze_checks']['zero_new_fits'])
        values = scenes()
        values[0]['budget']['new_fits'] = 1
        report = analyze(values, audit(), model())
        self.assertEqual(report['budget']['new_fits'], 1)
        self.assertFalse(report['seed_integrity']['416207']['integrity']['checks']['no_new_fits'])
        self.assertFalse(report['promotion']['passed'])

    def test_prefix_suffix_training_and_budget_break_fail(self):
        values = scenes()
        values[0]['budget']['suffix_updates'] = 1
        self.assertFalse(analyze(values, audit(), model())['promotion']['passed'])
        values = scenes()
        values[0]['budget']['native_prefix_seconds'] += 4000
        values[0]['budget']['total_native_seconds'] += 4000
        report = analyze(values, audit(), model())
        self.assertFalse(report['integrity']['checks']['native_budget'])
        values = scenes()
        values[0]['frozen_source']['source_model_exact'] = False
        self.assertFalse(analyze(values, audit(), model())['promotion']['passed'])

    def test_script_imports_from_its_remote_style_cwd(self):
        script = Path(__file__).with_name('analysis.py').resolve()
        process = subprocess.run([sys.executable, str(script), '--help'], cwd=script.parent,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--audit', process.stdout)


if __name__ == '__main__':
    unittest.main()
