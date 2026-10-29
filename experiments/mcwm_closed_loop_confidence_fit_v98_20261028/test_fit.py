"""Synthetic-only regression tests; never read the T or sealed outcome files."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mcwm_closed_loop_confidence_fit_v98_20261028 import fit as module

HERE = Path(__file__).resolve().parent


def synthetic_input():
    protocol = json.loads((HERE/'protocol.json').read_text())
    pairs = []
    for seed_index, seed in enumerate(protocol['T']):
        for scene_index, scene in enumerate(module.SCENES):
            q = -.02+.1*seed_index+.3*scene_index
            for line_index, line in enumerate(module.LINES):
                m = (-2., .2, -.5)[line_index]
                delta = m+.2+.4*q
                noop = seed_index == scene_index == line_index == 0
                if noop:
                    delta = 3.  # A complete positive physical-noop label must remain in T.
                pairs.append(dict(external_seed=seed, scenario=scene, line_id=line,
                    physical_root_id=[seed, scene, line_index+1],
                    root=dict(started_at=12000.+line_index, sequence=line_index+1),
                    condition=dict(value=q), predictions=dict(service_history6=m),
                    label_delta_per_passenger=dict(cost=delta, wait=delta/2., ride=0.),
                    physical_noop=noop, valid=True,
                    checks=dict(true_terminal=True, physical_separation=not noop)))
    summary = dict(status='complete', collection_gate=dict(passed=True),
        integrity=dict(all_pass=True, checks=dict(complete=True)), pairs=pairs,
        budget=dict(labels_observed=24), future_bias_design=dict(target_solves=0),
        interpretation=dict(model_fits=0))
    return summary, protocol


class FitTest(unittest.TestCase):
    def test_all_twelve_rows_reach_the_single_solver(self):
        source, protocol = synthetic_input()
        with patch.object(module, 'fit_bias', wraps=module.fit_bias) as solver:
            result = module.analyze(source, protocol)
        solver.assert_called_once()
        passed_rows = solver.call_args.args[0]
        self.assertEqual(len(passed_rows), 12)
        self.assertEqual(passed_rows[0]['delta'], 3.)
        self.assertTrue(any(row['prediction'] >= 0. for row in passed_rows))
        self.assertTrue(any(row['delta'] > 0. for row in passed_rows))
        self.assertTrue(any(row['q'] < protocol['q_range_for_scaling_only'][0] for row in passed_rows))
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['numerical']['all_pass'])
        self.assertEqual(result['model']['bias_model']['n'], 12)
        self.assertEqual(result['budget']['real_bias_fits'], 1)
        self.assertEqual(result['model']['candidate_deployment_domain'], protocol['candidate_deployment_domain'])
        self.assertFalse(result['interpretation']['independent_control_evidence'])
        self.assertTrue(all(set(('q', 'm', 'D', 'b', 'u', 'center')) <= set(row) for row in result['pairs']))
        json.dumps(result, allow_nan=False)

    def test_missing_failed_duplicate_or_short_cohort_never_solves(self):
        source, protocol = synthetic_input()
        cases = []
        missing = deepcopy(source)
        missing['pairs'].pop()
        cases.append(missing)
        failed = deepcopy(source)
        failed['status'] = 'hold'
        failed['integrity']['all_pass'] = False
        cases.append(failed)
        duplicate = deepcopy(source)
        duplicate['pairs'][-1] = deepcopy(duplicate['pairs'][0])
        cases.append(duplicate)
        short = deepcopy(source)
        short['budget']['labels_observed'] = 23
        cases.append(short)
        for index, summary in enumerate(cases):
            with self.subTest(case=index), patch.object(module, 'fit_bias') as solver:
                result = module.analyze(summary, protocol)
                solver.assert_not_called()
                self.assertEqual(result['status'], 'hold')
                self.assertIsNone(result['model'])
                self.assertEqual(result['budget']['real_bias_fits'], 0)

    def test_missing_or_nonfinite_training_fields_do_not_solve(self):
        for key in ('q', 'm', 'label'):
            source, protocol = synthetic_input()
            if key == 'q':
                source['pairs'][0]['condition']['value'] = float('nan')
            elif key == 'm':
                source['pairs'][0]['predictions']['service_history6'] = float('inf')
            else:
                source['pairs'][0]['label_delta_per_passenger'] = None
            with self.subTest(field=key), patch.object(module, 'fit_bias') as solver:
                result = module.analyze(source, protocol)
                solver.assert_not_called()
                self.assertFalse(result['validation']['checks']['finite_rows'])
                json.dumps(result, allow_nan=False)

    def test_rank_one_is_hold_even_though_ridge_could_solve_it(self):
        source, protocol = synthetic_input()
        for row in source['pairs']:
            row['condition']['value'] = -.01
        with patch.object(module, 'fit_bias') as solver:
            result = module.analyze(source, protocol)
        solver.assert_not_called()
        self.assertEqual(result['validation']['rank'], 1)
        self.assertEqual(result['status'], 'hold')

    def test_normal_equations_are_checked_without_a_second_solve(self):
        source, protocol = synthetic_input()
        original = module.fit_bias
        def corrupt(rows, q_range):
            model = original(rows, q_range)
            model['coef'][0] += 1.
            return model
        with patch.object(module, 'fit_bias', side_effect=corrupt) as solver:
            result = module.analyze(source, protocol)
        solver.assert_called_once()
        self.assertEqual(result['status'], 'hold')
        self.assertFalse(result['numerical']['checks']['normal_equation'])
        self.assertIsNone(result['model'])
        self.assertFalse(result['decision']['ready_for_calibration'])

    def test_solver_failure_is_recorded_without_retry(self):
        source, protocol = synthetic_input()
        with patch.object(module, 'fit_bias', side_effect=np.linalg.LinAlgError('synthetic failure')) as solver:
            result = module.analyze(source, protocol)
        solver.assert_called_once()
        self.assertEqual(result['decision']['reason'], 'FIXED_FIT_FAILED_NO_RETRY')
        self.assertEqual(result['budget']['real_bias_fits'], 1)
        self.assertIsNone(result['model'])

    def test_c0_resource_gate_can_close_a_valid_fit(self):
        source, protocol = synthetic_input()
        for row in source['pairs']:
            row['predictions']['service_history6'] = 2.
            row['label_delta_per_passenger'] = dict(cost=2., wait=1., ride=0.)
        result = module.analyze(source, protocol)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['model']['integrity']['all_pass'])
        self.assertFalse(result['decision']['ready_for_calibration'])
        self.assertEqual(result['decision']['C0_eligible_roots'], 0)
        self.assertEqual(result['decision']['reason'], 'CLOSED_NO_T_ENTRY_AT_C0')

    def test_training_readout_selects_one_root_without_outcome_filtering(self):
        source, protocol = synthetic_input()
        result = module.analyze(source, protocol)
        first = source['pairs'][0]
        key = str(first['external_seed'])+'/'+first['scenario']
        selected = result['training_single_selection']['by_seed_scene'][key]['raw_m']
        self.assertEqual(selected['selected_root'], first['physical_root_id'])
        self.assertEqual(selected['delta'], 3.)
        self.assertTrue(selected['physical_noop'])
        errors = result['frozen_m_errors']['by_seed_scene'][key]
        expected = [row['predictions']['service_history6']-row['label_delta_per_passenger']['cost']
                    for row in source['pairs'][:3]]
        self.assertAlmostEqual(errors['mae'], sum(abs(value) for value in expected)/3.)
        self.assertEqual(errors['n'], 3)
        self.assertTrue(result['decision']['ready_for_calibration'])

    def test_cli_writes_once_and_rejects_a_second_attempt_before_solving(self):
        source, protocol = synthetic_input()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            protocol['source_summary'] = str(path/'synthetic_source.json')
            (path/'synthetic_source.json').write_text(json.dumps(source))
            (path/'protocol.json').write_text(json.dumps(protocol))
            arguments = ['--protocol', str(path/'protocol.json'), '--out-dir', str(path/'out')]
            with redirect_stdout(io.StringIO()):
                module.main(arguments)
            self.assertTrue(all((path/'out'/name).is_file() for name in
                                ('fit_attempt.json', 'model.json', 'summary.json')))
            saved = (path/'out'/'model.json').read_text()
            with patch.object(module, 'fit_bias') as solver, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main(arguments)
                solver.assert_not_called()
            self.assertEqual((path/'out'/'model.json').read_text(), saved)

    def test_cli_hold_writes_no_model_and_does_not_solve(self):
        source, protocol = synthetic_input()
        source['pairs'].pop()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            protocol['source_summary'] = str(path/'synthetic_source.json')
            (path/'synthetic_source.json').write_text(json.dumps(source))
            (path/'protocol.json').write_text(json.dumps(protocol))
            with patch.object(module, 'fit_bias') as solver, redirect_stdout(io.StringIO()):
                module.main(['--protocol', str(path/'protocol.json'), '--out-dir', str(path/'out')])
                solver.assert_not_called()
            self.assertFalse((path/'out'/'model.json').exists())
            self.assertEqual(json.loads((path/'out'/'summary.json').read_text())['status'], 'hold')


if __name__ == '__main__':
    unittest.main()
