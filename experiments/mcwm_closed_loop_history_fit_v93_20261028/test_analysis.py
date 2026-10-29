import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mcwm_closed_loop_history_fit_v93_20261028.analysis import ABLATION, DIMENSIONS, LINES, MODELS, analyze
from experiments.mcwm_closed_loop_history_fit_v93_20261028.source import load_training


def package(rows, centers, matrices, *, zero=False, source_checks=None):
    coefficients = dict(line3=[-1., -1.5, -2.], service5=[-1., -1.5, -2., .5, -.2],
        service_history6=[-1., -1.5, -2., .2, -.1, 8.])
    models = {}
    for index, name in enumerate(MODELS, 1):
        matrix = np.asarray(matrices[name])
        coef = [0.]*DIMENSIONS[name] if zero else coefficients[name]
        models[name] = dict(coef=coef, feature_names=['feature'+str(j) for j in range(len(coef))],
            prediction=(matrix@coef).tolist(), design_rank=int(np.linalg.matrix_rank(matrix)), solve_index=index,
            normal_equation_residual_max=0.)
    return dict(models=models, training_rows=rows,
        design=dict(line_centers=centers, matrices={name: np.asarray(value).tolist() for name, value in matrices.items()}),
        source_audit=dict(all_pass=True, checks=source_checks or dict(fixed_source=True), source_summary='test-only source'),
        integrity=dict(all_pass=True, checks=dict(synthetic_readout_fixture=True)),
        ridge=1., denominator=15079., hold_scale=7.5,
        budget=dict(model_solves=3, new_native_seconds=0, new_labels=0, initializer_updates=0,
            optimizer_updates=0, v87_truth_used=0, training_pairs=12, training_physical_labels=24,
            historical_native_seconds=349310))


def artifact():
    rows = []
    x = {'7X': ((.1, -.94), (.2, -.8)), '102X': ((.34, -.64), (.56, -.5)),
         '705X': ((.2, -.96), (.86, .12))}
    for scene in ('stable', 'local_speed_drop'):
        for line in LINES:
            for j, group in enumerate(('lighter', 'heavier')):
                target = -2. if scene == 'stable' else 3.
                index = len(rows)
                rows.append(dict(id=scene+'/'+line+'/'+group, scenario=scene, line_id=line, load_bin=group,
                    external_seed=716207, root=dict(sequence=index, started_at=12000.+index),
                    features=[*x[line][j], 0. if scene == 'stable' else 1.], a=.52+.001*index,
                    target=target, raw_delta_person_seconds=target*15079))
    centers = {}
    for line in LINES:
        selected = [row for row in rows if row['line_id'] == line]
        denominator = sum(row['a']**2 for row in selected)
        centers[line] = [sum(row['a']**2*row['features'][j] for row in selected)/denominator for j in range(3)]
    matrix = np.asarray([np.r_[row['a']*np.asarray([float(row['line_id'] == line) for line in LINES]),
        row['a']*(np.asarray(row['features'])-centers[row['line_id']])] for row in rows])
    return package(rows, centers, {name: matrix[:, :DIMENSIONS[name]] for name in MODELS})


class TrainingAnalysisTest(unittest.TestCase):
    def test_training_only_numerical_gate_and_json_report_without_any_solve(self):
        with patch('numpy.linalg.solve', side_effect=AssertionError('Readout must not fit')):
            with patch('numpy.linalg.lstsq', side_effect=AssertionError('Readout must not fit')):
                report = analyze(artifact())
        self.assertTrue(report['numerical_gate']['passed'])
        self.assertEqual(report['training']['overall']['service_history6']['direction_correct'], 12)
        self.assertEqual(report['training']['overall']['line3']['candidate_choices'], 12)
        self.assertFalse(report['interpretation']['actual_control_performance_evaluated'])
        self.assertFalse(report['interpretation']['independent_holdout_evaluated'])
        self.assertNotIn('promotion', report)
        self.assertEqual(report['budget']['model_solves'], 3)
        self.assertLess(len(json.dumps(report).encode()), 64000)

    def test_net_history_increment_includes_service_refitting_and_deletion_keeps_weights(self):
        report = analyze(artifact())
        nonzero_refit = False
        for row in report['training']['rows']:
            parts = row['six_minus_five']
            self.assertAlmostEqual(parts['line_delta'], 0.)
            self.assertAlmostEqual(parts['total_6_minus_5'], sum(parts[key]
                for key in ('line_delta', 'load_refit_delta', 'margin_refit_delta', 'q_direct')))
            self.assertAlmostEqual(row['models']['service_history6']['prediction']-row['models'][ABLATION]['prediction'],
                parts['q_direct'])
            nonzero_refit |= abs(parts['load_refit_delta']+parts['margin_refit_delta']) > 1e-6
        self.assertTrue(nonzero_refit)
        self.assertGreater(report['fixed_weight_q_deletion']['changed_training_choices'], 0)
        self.assertEqual(report['fixed_weight_q_deletion']['new_solves'], 0)

    def test_old_local_feature_aliases_are_scored_without_new_promotion_gate(self):
        report = analyze(artifact())
        for pair in report['training']['exact_local_aliases']:
            self.assertTrue(pair['equal_local_service_features'])
            self.assertTrue(pair['models']['service_history6']['predicted_signs_differ'])
            self.assertTrue(pair['models']['service_history6']['both_training_directions_correct'])
            self.assertFalse(pair['models']['service5']['predicted_signs_differ'])
        source = artifact()
        source['training_rows'][0].update(target=2., raw_delta_person_seconds=30158.)
        report = analyze(source)
        self.assertTrue(report['numerical_gate']['passed'])
        self.assertGreater(report['training']['by_scenario']['stable']['service_history6']['mean_two_action_regret_per_passenger'], 0.)

    def test_line_coefficient_drift_or_saved_prediction_corruption_fails_numerical_contract(self):
        source = artifact()
        model = source['models']['service_history6']
        model['coef'][0] += .1
        model['prediction'] = (np.asarray(source['design']['matrices']['service_history6'])@model['coef']).tolist()
        report = analyze(source)
        self.assertFalse(report['numerical_gate']['checks']['shared_line_coefficients'])
        source = artifact()
        source['models']['service5']['prediction'][0] += 1.
        self.assertFalse(analyze(source)['model_numerical_checks']['service5']['saved_prediction_exact'])

    def test_three_solve_budget_target_scalar_and_reserved_seed_are_checked(self):
        source = artifact()
        source['budget']['model_solves'] = 4
        self.assertFalse(analyze(source)['numerical_gate']['passed'])
        source = artifact()
        source['training_rows'][0]['raw_delta_person_seconds'] += 1.
        self.assertFalse(analyze(source)['numerical_gate']['checks']['target_units'])
        source = artifact()
        source['training_rows'][0]['external_seed'] = 816207
        self.assertFalse(analyze(source)['numerical_gate']['checks']['no_heldout_training_data'])

    def test_real_source_load_training_schema_with_zero_synthetic_coefficients_is_json_serializable(self):
        # This reconstructs real source rows/matrices only. It never fits them.
        root = Path(__file__).resolve().parent
        protocol = json.loads((root/'protocol.json').read_text())
        with patch('numpy.linalg.solve', side_effect=AssertionError('No real-data fit in tests')):
            with patch('numpy.linalg.lstsq', side_effect=AssertionError('No real-data fit in tests')):
                source = load_training(protocol['source_summary'], protocol['frozen_protocol'])
                value = package(source['rows'], source['line_centers'], source['matrices'],
                    zero=True, source_checks=source['checks'])
                report = analyze(value)
        self.assertTrue(report['numerical_gate']['passed'])
        self.assertIsInstance(source['rows'][0]['raw_delta_person_seconds'], float)
        encoded = json.dumps(report, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        self.assertLess(len(encoded.encode()), 64000)
        self.assertEqual(report['training']['overall']['service_history6']['candidate_choices'], 0)


if __name__ == '__main__':
    unittest.main()
