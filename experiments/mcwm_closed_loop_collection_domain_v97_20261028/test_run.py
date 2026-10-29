"""Exercise collector receipts without importing or starting the native environment."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
TREE = ast.parse((HERE / 'run.py').read_text())


def function(name, namespace):
    node = next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)
    node = ast.parse(ast.unparse(node)).body[0]
    node.body = [ast.Global(names=statement.names) if isinstance(statement, ast.Nonlocal) else statement
                 for statement in node.body]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(HERE / 'run.py'), 'exec'), namespace)
    return namespace[name]


class CollectorReceiptsTest(unittest.TestCase):
    def test_physical_noop_keeps_technical_integrity_but_other_failures_do_not(self):
        check = function('pair_integrity', {})
        checks = dict(actual_departures_observed=True, physical_departure_separation=False,
                      both_real_terminal=True, original_actor_root_exact=True)
        self.assertTrue(check(checks))
        for key in ('actual_departures_observed', 'both_real_terminal', 'original_actor_root_exact'):
            self.assertFalse(check(dict(checks, **{key: False})))

    def test_gate_reuse_requires_an_actual_pair(self):
        writes = []
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(local_first_pair_gate=None, external_seed=1, scenario='stable',
                root_out=Path(directory), gate_path=Path(directory) / 'shared.json', producer=True,
                write=lambda path, value: writes.append((path, value)))
            declare = function('declare_gate', namespace)
            held = declare(reason='no_supported_first_pair')
            self.assertFalse(held['first_pair_reused'])
            self.assertFalse(held['new_pair_in_formal_cohort'])
            namespace['local_first_pair_gate'] = None
            measured = declare(record=dict(all_pass=True, checks=dict(physical_departure_separation=False),
                                           line_id='7X', root={'sequence': 4}))
            self.assertTrue(measured['first_pair_reused'])
            self.assertTrue(measured['all_pass'])

    def test_not_started_receipt_declares_no_selection_from_labels_or_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            fixtures = {
                'protocol.json': dict(source_initializer=6207, external_seeds=[1, 2],
                    scenarios=['stable'], budget=dict(scenario_cap=84600)),
                'source_protocol.json': dict(frozen_reference=dict(initializer_seed=6207),
                    independent_cohorts=dict(T=[1, 2]), frozen_common_support={}),
                'source_model.json': dict(integrity=dict(all_pass=True), common_support={}),
                'freeze_audit.json': dict(all_pass=True, checks=dict(frozen=True)),
                'first_pair_gate.json': dict(all_pass=False)}
            for name, value in fixtures.items():
                (here / name).write_text(json.dumps(value))
            namespace = dict(Path=Path, HERE=here, json=json, time=time, LINES=('7X', '102X', '705X'))
            namespace['write'] = function('write', dict(Path=Path, json=json))
            result = function('run', namespace)(here / 'run', 'stable', 2)
            self.assertEqual(result['status'], 'not_started')
            self.assertIs(result['labels_used_for_selection'], False)
            self.assertIs(result['predictions_used_for_selection'], False)
            self.assertEqual(result['budget']['total_native_seconds'], 0)
            self.assertEqual(result['pairs'], [])

    def test_error_receipt_retains_completed_pairs_observations_and_spend(self):
        with tempfile.TemporaryDirectory() as directory:
            gate_path = Path(directory) / 'first_pair_gate.json'
            gate = dict(all_pass=True)
            gate_path.write_text(json.dumps(gate))
            pairs = [dict(all_pass=True, delta=dict(wait=2, ride=3, cost=7))]
            coverage = {'7X': dict(status='supported', training_support=dict(extrapolation=True))}
            spend = dict(total_native_seconds=31000, unknown_suffix_costs=0)
            namespace = dict(spent_budget=lambda: spend, local_first_pair_gate=gate,
                coverage=coverage, pairs=pairs, controller=None, reference_reproduction={},
                audit=dict(all_pass=True), observation_checks=dict(clock_unchanged=True),
                stream=SimpleNamespace(now=12002, prefix_limit_reached=False, issued_count=1),
                trace=SimpleNamespace(observed_issues=1), scenario_cap=84600, clock_exact=True,
                scenario='stable', seed=6207, external_seed=1, selection={},
                observation_counts=dict(reads=1, observed=1, admissible=1, unsupported=0),
                collection_halted=False, gate_path=gate_path, json=json, history=None,
                root_out=Path(directory), write=lambda path, value: None, progress=lambda phase: None)
            error = dict(type='RuntimeError', message='second native pair failed')
            result = function('finish_parent', namespace)(error=error)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error'], error)
            self.assertEqual(result['failure'], error)
            self.assertFalse(result['integrity']['checks']['error_free'])
            self.assertEqual(result['pairs'], pairs)
            self.assertEqual(result['coverage'], coverage)
            self.assertEqual(result['budget'], spend)
            self.assertEqual(result['observation_counts']['admissible'], 1)


if __name__ == '__main__':
    unittest.main()
