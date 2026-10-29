"""Frozen-policy forward and receipts, using synthetic inputs and no native simulator."""
import ast
from copy import deepcopy
import json
from math import isfinite, sqrt
from pathlib import Path
import random
import tempfile
import time
import unittest

import numpy as np

from experiments.mcwm_closed_loop_state_confidence_v95_20261028.confidence import predict


HERE = Path(__file__).resolve().parent
TREE = ast.parse((HERE / 'run.py').read_text())


def function(name, namespace):
    node = next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)
    node = ast.parse(ast.unparse(node)).body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(HERE / 'run.py'), 'exec'), namespace)
    return namespace[name]


class ValidationCollectorTest(unittest.TestCase):
    def test_issue_forward_uses_frozen_margins_without_mutation_or_rng(self):
        score = function('calibrated_policy_scores', dict(confidence_forward=predict))
        predictions = dict(service_history6=-2., line3=3., service5=4.)
        bias = dict(q_range=[0., 1.], coef=[-.5, .25], gram_inverse=[[.25, 0.], [0., .5]])
        calibration = dict(C=.5, state_independent_C=2.)
        inputs = deepcopy((predictions, bias, calibration))
        rng_before, numpy_before = random.getstate(), np.random.get_state()
        scores = score(predictions, .25, bias, calibration)
        candidate = scores['calibrated_candidate']
        self.assertEqual(candidate['b'], -.625)
        self.assertAlmostEqual(candidate['u'], sqrt(1.375))
        self.assertAlmostEqual(candidate['U'], -2.625+.5*sqrt(1.375))
        self.assertTrue(candidate['accept'])
        self.assertEqual(scores['state_independent']['U'], 0.)
        self.assertFalse(scores['state_independent']['accept'])
        self.assertEqual((predictions, bias, calibration), inputs)
        self.assertEqual(random.getstate(), rng_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_original_nonnegative_m_still_rejects_and_old_q_range_is_not_a_gate(self):
        score = function('calibrated_policy_scores', dict(confidence_forward=predict))
        bias = dict(q_range=[0., 1.], coef=[-10., 0.], gram_inverse=[[1., 0.], [0., 1.]])
        calibration = dict(C=0., state_independent_C=0.)
        at_zero = score(dict(service_history6=0.), -1., bias, calibration)
        self.assertLess(at_zero['calibrated_candidate']['U'], 0.)
        self.assertFalse(at_zero['calibrated_candidate']['accept'])
        self.assertFalse(at_zero['state_independent']['accept'])
        outside = score(dict(service_history6=-1.), -1., bias, calibration)
        self.assertTrue(outside['calibrated_candidate']['accept'])
        self.assertAlmostEqual(outside['calibrated_candidate']['u'], sqrt(11.))

    def test_only_exact_actual_tail_boundary_is_persisted(self):
        select = function('exact_tail_boundary', dict(deepcopy=deepcopy))
        original = dict(time_seconds=18300., scheduled_time_seconds=18300.,
            accrued_wait_seconds=123.25, accrued_in_vehicle_seconds=456.5, waiting_passengers=4)
        result = select([original])
        self.assertEqual(result, original)
        self.assertIsNot(result, original)
        self.assertIsNone(select([dict(original, time_seconds=18301.)]))

    def fixtures(self):
        fit = dict(primary_model='service_history6', source_summary='/frozen/v97/summary.json',
            source_protocol='/frozen/v97/protocol.json', candidate_deployment_domain={},
            bias_model=dict(q_range=[0., 1.], coef=[1.25, -.5], gram_inverse=[[.2, .1], [.1, .3]],
                            n=12, rank=2, ridge=1., linear_solves=1),
            integrity=dict(all_pass=True, checks=dict(T_integrity=True, fixed_fit_numerical=True)))
        return {
            'protocol.json': dict(source_initializer=6207, cohort='V', external_seeds=[7],
                scenarios=['stable', 'local_speed_drop'], budget=dict(scenario_cap=84600),
                q_range_for_scaling_only=[0., 1.], candidate_deployment_domain={},
                source_fit='/frozen/v98/model.json', source_calibration='/frozen/v99/calibration.json',
                calibration=dict(alpha=.2)),
            'source_protocol.json': dict(frozen_reference=dict(initializer_seed=6207),
                independent_cohorts=dict(T=[10, 20], C=[1, 2, 3, 4], V=[7]), frozen_common_support={}),
            'source_model.json': dict(integrity=dict(all_pass=True), common_support={}),
            'source_fit.json': fit,
            'source_calibration.json': dict(status='complete', ready_for_validation=True,
                integrity=dict(all_pass=True, checks=dict(complete_C=True)), frozen_model=deepcopy(fit),
                source_identity=dict(source_fit='/frozen/v98/model.json'), alpha=.2, C=.25,
                state_independent=dict(C=.75)),
            'freeze_audit.json': dict(all_pass=True, checks=dict(frozen=True)),
            'first_pair_gate.json': dict(all_pass=False)}

    def invoke_without_native(self, fixtures):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            for name, value in fixtures.items():
                (here / name).write_text(json.dumps(value))
            namespace = dict(Path=Path, HERE=here, json=json, time=time, deepcopy=deepcopy,
                             isfinite=isfinite, LINES=('7X', '102X', '705X'))
            namespace['write'] = function('write', dict(Path=Path, json=json))
            return function('run', namespace)(here / 'run', 'local_speed_drop', 7)

    def test_V_receipt_binds_ready_calibration_without_looking_at_labels(self):
        result = self.invoke_without_native(self.fixtures())
        self.assertEqual(result['status'], 'not_started')
        self.assertEqual(result['prefix_id'], 'v100/7/local_speed_drop')
        self.assertEqual(result['frozen_calibration'], dict(
            source_calibration='/frozen/v99/calibration.json', source_fit='/frozen/v98/model.json',
            alpha=.2, C=.25, state_independent_C=.75, ready_for_validation=True))
        self.assertIs(result['labels_used_for_selection'], False)
        self.assertEqual(result['budget']['total_native_seconds'], 0)

    def test_unready_calibration_or_different_bias_cannot_start_native_validation(self):
        for change in ('not_ready', 'different_bias', 'nonfinite_margin'):
            fixtures = self.fixtures()
            calibration = fixtures['source_calibration.json']
            if change == 'not_ready':
                calibration['ready_for_validation'] = False
            elif change == 'different_bias':
                calibration['frozen_model']['bias_model']['coef'][0] += 1.
            else:
                calibration['C'] = None
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'ready v99 calibration'):
                self.invoke_without_native(fixtures)


if __name__ == '__main__':
    unittest.main()
