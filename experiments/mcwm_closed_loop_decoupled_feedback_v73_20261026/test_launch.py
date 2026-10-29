"""Prevent baseline drift, duplicate days and accidental eight-day expansion."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('v73_launch_tested', HERE / 'launch.py')
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module('v73_test_readonly', L.V60 / 'supervisor.py')
PROTOCOL = dict(name='v73-test', development_seeds=[6201, 6203], scenarios=['stable', 'local_speed_drop'],
    native_preflight=dict(seed=6201, scenario='stable', reuse_full_day=True, feedback_slots=5,
        policy_updates=8, critic_updates=8, value_updates=8, actor_updates=8, alpha_updates=8),
    summary_outputs=['results/development.json', 'results/policy_drift.json',
                     'results/recency.json', 'results/reference_summary.json'])


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.put(self.root / 'protocol.json', PROTOCOL)
        self.put(self.root / 'execution_ready.json', dict(ready=True))
        self.outputs, self.submissions = {}, []
        self.infra = SimpleNamespace(remote=str, task=self.task,
            presence=mock.Mock(side_effect=lambda paths: {p: p in self.outputs for p in paths}),
            collect=mock.Mock(side_effect=lambda p: self.outputs[p]),
            common_prefix=mock.Mock(return_value=dict(all_pass=True, rows_compared=45)),
            status=mock.Mock(side_effect=lambda ids: [dict(id=i, status='running') for i in ids]),
            submit=mock.Mock(side_effect=self.submit))
        for patcher in (mock.patch.object(L, 'HERE', self.root), mock.patch.object(L, 'module',
            side_effect=lambda name, path: OBSERVER if path.name == 'supervisor.py' else self.infra)):
            patcher.start(); self.addCleanup(patcher.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def task(self, label, runner, argv):
        return dict(signature=PROTOCOL['name']+'/'+label, runner=str(runner), argv=argv, priority='high')

    def submit(self, jobs, batch):
        path = self.root / (batch+'_submission.json')
        if path.exists(): return json.loads(path.read_text())
        value = dict(submitted=[dict(id=f't_{len(self.submissions)}_{index}', signature=row['signature'])
                                for index, row in enumerate(jobs)])
        self.submissions.append(dict(batch=batch, jobs=jobs))
        self.put(path, value)
        return value

    def preflight_ready(self):
        for cell in L.cells(PROTOCOL)[:2]:
            self.outputs[cell['relative']+'/preflight.json'] = dict(technical_gate_pass=True,
                seed=cell['seed'], scenario=cell['scenario'], arm=cell['arm'], feedback_slots=5,
                policy_updates=8, critic_updates=8, value_updates=8, actor_updates=8, alpha_updates=8,
                packet_schedule=[8]*4, initializer=dict(updates=50000), first_feedback=dict(ids=[1,2,3,4,5]),
                execution_contract=dict(F='same'), prefix_before_first_update=dict(
                    trace_path='resident_trace', rows=45, clock=1200., initial_updates=50000))

    def complete(self, cell):
        self.outputs[cell['relative']+'/gate.json'] = dict(status='complete', integrity=dict(all_pass=True))

    def regression_ready(self, *, passed=True):
        self.preflight_ready()
        self.complete(L.cells(PROTOCOL)[1])
        self.outputs['regression_gate.json'] = dict(status='complete', all_pass=passed,
            integrity=dict(all_pass=passed))

    def finish(self, *, passed=True, integrity=True):
        self.regression_ready()
        for cell in L.cells(PROTOCOL): self.complete(cell)
        self.outputs[PROTOCOL['summary_outputs'][0]] = dict(status='complete', integrity=dict(all_pass=integrity),
            decision=dict(scientific_gate_pass=passed, advance_to_confirmation=False))
        for path in PROTOCOL['summary_outputs'][1:]: self.outputs[path] = dict(status='complete')

    def native_jobs(self):
        return [row for batch in self.submissions for row in batch['jobs'] if Path(row['runner']).name == 'control.py']

    def test_two_first_full_days_are_reused_and_never_wait_for_cost(self):
        current = L.advance(); L.advance()
        self.assertEqual(current['stage'], 'decoupled_feedback_native_preflight')
        self.assertEqual(len(self.native_jobs()), 2)
        self.assertEqual({row['argv'][5] for row in self.native_jobs()}, {'fresh', 'no_fresh'})
        self.assertTrue(all('--stop' not in row['argv'] for row in self.native_jobs()))
        self.preflight_ready()
        current = L.advance()
        self.assertEqual(current['stage'], 'decoupled_feedback_full_nofresh_regression')
        self.assertEqual(len(self.submissions), 1)

    def test_full_regression_required_before_remaining_three_fresh_days(self):
        self.preflight_ready()
        self.complete(L.cells(PROTOCOL)[0])
        L.advance()
        self.assertEqual(len(self.native_jobs()), 2)
        self.complete(L.cells(PROTOCOL)[1])
        current = L.advance()
        self.assertEqual(current['stage'], 'decoupled_feedback_regression_comparison')
        self.assertEqual(len(self.native_jobs()), 2)
        self.assertEqual(self.submissions[-1]['jobs'][0]['argv'],
            ['--regression-gate', '--out', str(self.root/'regression_gate.json')])
        self.outputs['regression_gate.json'] = dict(status='complete', all_pass=True, integrity=dict(all_pass=True))
        current = L.advance(); L.advance()
        jobs = self.native_jobs()
        self.assertEqual(len(jobs), 5)
        self.assertEqual(len({row['signature'] for row in jobs}), 5)
        self.assertEqual([row['argv'][5] for row in jobs].count('no_fresh'), 1)
        self.assertEqual(len(self.submissions[-1]['jobs']), 3)
        self.assertTrue(all(row['argv'][5] == 'fresh' for row in self.submissions[-1]['jobs']))
        self.assertTrue(all(row['priority'] == 'high' for row in jobs))

    def test_baseline_can_release_others_while_first_fresh_day_continues(self):
        self.regression_ready()
        current = L.advance()
        self.assertFalse(current['done'])
        self.assertEqual(current['completed'], 1)
        self.assertEqual(len(current['pending']), 4)
        self.assertEqual(len(self.native_jobs()), 5)

    def test_failed_full_baseline_reproduction_stops_without_extra_native_runs(self):
        self.regression_ready(passed=False)
        current = L.advance(); L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(current['stage'], 'decoupled_feedback_regression_attention')
        self.assertEqual(len(self.native_jobs()), 2)
        self.assertTrue(current['native_tasks_may_still_run'])
        self.assertEqual(current['performance_verdict'], 'not_evaluated_incomplete')

    def test_bad_or_missing_technical_prefix_never_opens_remaining_batch(self):
        self.preflight_ready()
        self.infra.common_prefix.return_value = dict(all_pass=False)
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(len(self.native_jobs()), 2)
        self.assertEqual(current['preflight']['failed_checks'], ['full_pre_update_prefix_exact'])

    def test_summary_waits_for_all_five_and_collects_all_four_outputs_only_once(self):
        self.finish(passed=False)
        missing = PROTOCOL['summary_outputs'][-1]
        self.outputs.pop(missing)
        current = L.advance(); L.advance()
        self.assertEqual(current['stage'], 'decoupled_feedback_development_summary')
        self.assertEqual(current['completed'], 5)
        self.assertNotIn(mock.call(PROTOCOL['summary_outputs'][0]), self.infra.collect.call_args_list)
        self.outputs[missing] = dict(status='complete')
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertFalse(current['candidate_passed'])
        self.assertEqual(len(self.submissions), 4)
        self.assertEqual(len(self.native_jobs()), 5)
        for path in PROTOCOL['summary_outputs']: self.infra.collect.assert_any_call(path)

    def test_passing_or_failing_science_stops_for_root_review(self):
        for passed in (True, False):
            self.finish(passed=passed)
            current = L.advance()
            self.assertTrue(current['done'])
            self.assertEqual(current['candidate_passed'], passed)
            self.assertFalse(current['goal_achieved'])
            self.assertTrue(current['requires_agent_review'])
            self.assertFalse(current['decision']['advance_to_confirmation'])
        self.assertEqual(len(self.native_jobs()), 5)

    def test_native_monitor_reads_lean_gates_and_not_resident_summary(self):
        self.finish()
        for cell in L.cells(PROTOCOL):
            self.outputs[cell['relative']+'/summary.json'] = dict(receipts='large resident record')
        L.advance()
        self.assertTrue(all(not row.args[0].endswith('/summary.json') for row in self.infra.collect.call_args_list))

    def test_terminal_regression_failure_does_not_resubmit_or_start_other_days(self):
        self.preflight_ready(); self.complete(L.cells(PROTOCOL)[1])
        self.infra.status.side_effect = lambda ids: [dict(id=i, status='failed' if i.startswith('t_1') else 'running') for i in ids]
        current = L.advance(); L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(current['stage'], 'decoupled_feedback_regression_attention')
        self.assertEqual(len(self.native_jobs()), 2)
        self.assertEqual(len(self.submissions), 2)

    def test_missing_native_gate_after_terminal_cannot_trigger_full_summary_download(self):
        self.regression_ready()
        for cell in L.cells(PROTOCOL): self.complete(cell)
        missing = L.cells(PROTOCOL)[-1]['relative']
        self.outputs.pop(missing+'/gate.json')
        self.outputs[missing+'/summary.json'] = dict(status='complete')
        self.infra.status.side_effect = lambda ids: [dict(id=i, status='done') for i in ids]
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(current['stage'], 'decoupled_feedback_native_attention')
        self.assertEqual(len(self.submissions), 3)
        self.assertNotIn(mock.call(missing+'/summary.json'), self.infra.collect.call_args_list)

    def test_integrity_failure_is_not_a_scientific_verdict(self):
        self.finish(integrity=False)
        current = L.advance()
        self.assertEqual(current['performance_verdict'], 'not_evaluated_incomplete')
        self.assertNotIn('candidate_passed', current)

    def test_transient_read_preserves_known_submission_and_recovers(self):
        self.infra.status.side_effect = RuntimeError('ssh: connect to host node001: Connection timed out')
        self.assertEqual(L.advance()['stage'], 'waiting_transient_development_read')
        self.infra.status.side_effect = lambda ids: [dict(id=i, status='running') for i in ids]
        self.assertEqual(L.advance()['stage'], 'decoupled_feedback_native_preflight')
        self.assertEqual(len(self.native_jobs()), 2)


class InfraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.infra = L.module('v73_infra_test', HERE/'infra.py')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(self.infra, 'HERE', self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_existing_receipt_does_not_submit_again(self):
        value = dict(submitted=[dict(id='t_old', signature='v73/first')])
        (self.root/'first_submission.json').write_text(json.dumps(value))
        with mock.patch.object(self.infra.subprocess, 'run') as run:
            self.assertEqual(self.infra.submit([], 'first'), value)
        run.assert_not_called()

    def test_uncertain_submission_is_not_retried(self):
        (self.root/'first_submission_attempt.json').write_text('{}')
        with mock.patch.object(self.infra.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Unresolved submission'):
                self.infra.submit([], 'first')
        run.assert_not_called()


if __name__ == '__main__': unittest.main()
