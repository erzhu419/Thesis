"""Small receipt/boundary checks; no native environment or real fit is loaded."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
TREE = ast.parse((HERE / 'run.py').read_text())


def function(name, namespace):
    node = next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)
    node = ast.parse(ast.unparse(node)).body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(HERE / 'run.py'), 'exec'), namespace)
    return namespace[name]


class CalibrationCollectorTest(unittest.TestCase):
    def test_exact_boundary_preserves_original_ledger_fields(self):
        boundary = dict(time_seconds=18300., scheduled_time_seconds=18300.,
            waiting_passengers=321, in_vehicle_passengers=456,
            accrued_wait_seconds=8123456.125, accrued_in_vehicle_seconds=16345123.75,
            accrued_generalized_passenger_seconds=32592036.)
        metrics = [dict(boundary, time_seconds=18000.), boundary,
                   dict(boundary, time_seconds=18600.)]
        before = deepcopy(metrics)
        selected = function('exact_tail_boundary', dict(deepcopy=deepcopy))(metrics)
        self.assertEqual(selected, boundary)
        self.assertIsNot(selected, boundary)
        self.assertEqual(metrics, before)

    def test_scheduled_or_nearest_boundary_is_not_substituted(self):
        select = function('exact_tail_boundary', dict(deepcopy=deepcopy))
        self.assertIsNone(select([
            dict(time_seconds=18299., scheduled_time_seconds=18000.),
            dict(time_seconds=18301., scheduled_time_seconds=18300.),
            dict(time_seconds=18300.000001, scheduled_time_seconds=18300.)]))
        self.assertIsNone(select([]))

    def fixtures(self):
        fit = dict(primary_model='service_history6', source_summary='/frozen/v97/summary.json',
            source_protocol='/frozen/v97/protocol.json', candidate_deployment_domain={},
            bias_model=dict(q_range=[0., 1.], coef=[1.25, -0.5], gram_inverse=[[.2, .1], [.1, .3]],
                            n=12, rank=2, ridge=1., linear_solves=1),
            integrity=dict(all_pass=True, checks=dict(T_integrity=True, fixed_fit_numerical=True)))
        return {
            'protocol.json': dict(source_initializer=6207, cohort='C', external_seeds=[1, 2, 3, 4],
                scenarios=['stable'], budget=dict(scenario_cap=84600), q_range_for_scaling_only=[0., 1.],
                candidate_deployment_domain={}, source_fit='/frozen/v98/model.json'),
            'source_protocol.json': dict(frozen_reference=dict(initializer_seed=6207),
                independent_cohorts=dict(T=[10, 20], C=[1, 2, 3, 4]), frozen_common_support={}),
            'source_model.json': dict(integrity=dict(all_pass=True), common_support={}),
            'source_fit.json': fit,
            'freeze_audit.json': dict(all_pass=True, checks=dict(frozen=True)),
            'first_pair_gate.json': dict(all_pass=False)}

    def invoke_without_native(self, fixtures):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            for name, value in fixtures.items():
                (here / name).write_text(json.dumps(value))
            namespace = dict(Path=Path, HERE=here, json=json, time=time, deepcopy=deepcopy,
                             LINES=('7X', '102X', '705X'))
            namespace['write'] = function('write', dict(Path=Path, json=json))
            return function('run', namespace)(here / 'run', 'stable', 2)

    def test_C_receipt_binds_exact_frozen_bias_without_using_it_for_selection(self):
        fixtures = self.fixtures()
        result = self.invoke_without_native(fixtures)
        self.assertEqual(result['status'], 'not_started')
        self.assertEqual(result['prefix_id'], 'v99/2/stable')
        expected = {key: fixtures['source_fit.json'][key]
                    for key in ('primary_model', 'bias_model', 'source_summary', 'source_protocol')}
        expected['source_fit'] = fixtures['protocol.json']['source_fit']
        self.assertEqual(result['frozen_bias'], expected)
        self.assertIs(result['predictions_used_for_selection'], False)
        self.assertIs(result['labels_used_for_selection'], False)
        self.assertEqual(result['budget']['total_native_seconds'], 0)

    def test_wrong_cohort_or_bias_scale_stops_before_native_initialization(self):
        for change in ('cohort', 'scale'):
            fixtures = self.fixtures()
            if change == 'cohort':
                fixtures['protocol.json']['cohort'] = 'T'
            else:
                fixtures['source_fit.json']['bias_model']['q_range'] = [-1., 1.]
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'v95 C cohort'):
                self.invoke_without_native(fixtures)


if __name__ == '__main__':
    unittest.main()
