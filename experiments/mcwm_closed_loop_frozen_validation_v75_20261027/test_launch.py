"""One fixed campaign, first pair reused, no cost-dependent seed selection."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('v75_launch_tested', HERE/'launch.py')
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
OBSERVER = L.module('v75_test_readonly', L.V60/'supervisor.py')
PROTOCOL = dict(name='v75-test', development_seeds=[6206, 6207, 6208],
    scenarios=['stable', 'local_speed_drop'], native_preflight=dict(seed=6206, scenario='stable',
        reuse_full_day=True, feedback_slots=5, policy_updates=8, critic_updates=8, value_updates=8,
        actor_updates=8, alpha_updates=8), summary_outputs=['results/validation.json',
        'results/evidence.json', 'results/pools.json', 'results/training.json'])


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.put(self.root/'protocol.json', PROTOCOL)
        self.put(self.root/'execution_ready.json', dict(ready=True))
        self.outputs, self.submissions = {}, []
        self.infra = SimpleNamespace(remote=str, task=self.task,
            promote_initializers=mock.Mock(return_value=dict(promoted=[])),
            dispatch_pending_initializers=mock.Mock(return_value=dict(dispatched=['i2'])),
            presence=mock.Mock(side_effect=lambda paths: {p: p in self.outputs for p in paths}),
            collect=mock.Mock(side_effect=lambda p: self.outputs[p]),
            common_prefix=mock.Mock(return_value=dict(all_pass=True, rows_compared=45)),
            status=mock.Mock(side_effect=lambda ids: [dict(id=i, status='running') for i in ids]),
            submit=mock.Mock(side_effect=self.submit))
        for patcher in (mock.patch.object(L, 'HERE', self.root), mock.patch.object(L, 'module',
                side_effect=lambda name, path: OBSERVER if path.name == 'supervisor.py' else self.infra)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def task(self, label, runner, argv):
        return dict(signature=PROTOCOL['name']+'/'+label, runner=str(runner), argv=argv, priority='high')

    def submit(self, jobs, batch):
        path = self.root/(batch+'_submission.json')
        if path.exists():
            return json.loads(path.read_text())
        value = dict(submitted=[dict(id=f't_{len(self.submissions)}_{i}', signature=row['signature'])
            for i, row in enumerate(jobs)])
        self.submissions.append(dict(batch=batch, jobs=jobs))
        self.put(path, value)
        return value

    def initializers_ready(self):
        for seed in PROTOCOL['development_seeds']:
            self.outputs[f'inputs/seed{seed}/initializer_gate.json'] = dict(status='complete', all_pass=True,
                seed=seed, offline_updates=50000, online_updates=0, recipe_exact=True,
                initializer_path=f'inputs/h2oplus_seed{seed}.pt')

    def preflight_ready(self):
        self.initializers_ready()
        for cell in L.cells(PROTOCOL)[:2]:
            self.outputs[cell['relative']+'/preflight.json'] = dict(technical_gate_pass=True,
                seed=cell['seed'], scenario=cell['scenario'], arm=cell['arm'], feedback_slots=5,
                policy_updates=8, critic_updates=8, value_updates=8, actor_updates=8, alpha_updates=8,
                packet_schedule=[8]*4, initializer=dict(updates=50000), first_feedback=dict(ids=[1,2,3,4,5]),
                execution_contract=dict(F='same'), prefix_before_first_update=dict(trace_path='resident',
                    rows=45, clock=1205., initial_updates=50000))

    def complete(self, cell):
        self.outputs[cell['relative']+'/gate.json'] = dict(status='complete', integrity=dict(all_pass=True))

    def finish(self, passed):
        self.preflight_ready()
        for cell in L.cells(PROTOCOL):
            self.complete(cell)
        for name in PROTOCOL['summary_outputs']:
            self.outputs[name] = dict(status='complete')
        self.outputs[PROTOCOL['summary_outputs'][0]].update(integrity=dict(all_pass=True),
            decision=dict(scientific_gate_pass=passed, old_v73_gate_reopened=False))

    def native_jobs(self):
        return [row for batch in self.submissions for row in batch['jobs'] if Path(row['runner']).name == 'control.py']

    def test_missing_initializers_do_not_allocate_any_native_day(self):
        current = L.advance()
        self.assertEqual(current['stage'], 'waiting_original_recipe_initializers')
        self.assertEqual(self.native_jobs(), [])

    def test_incomplete_or_wrong_recipe_initializer_stops_before_native(self):
        self.initializers_ready()
        self.outputs['inputs/seed6207/initializer_gate.json']['offline_updates'] = 40000
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(current['stage'], 'initializer_attention')
        self.assertEqual(self.native_jobs(), [])

    def test_terminal_initializer_without_gate_is_not_silently_retrained(self):
        self.put(self.root/'initializers_submission.json', dict(submitted=[dict(id=f'i{n}') for n in range(3)]))
        self.infra.status.side_effect = lambda ids: [dict(id=i, status='done') for i in ids]
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(self.native_jobs(), [])
        self.infra.submit.assert_not_called()

    def test_queued_third_initializer_redispatches_only_existing_receipt_id(self):
        self.put(self.root/'initializers_submission.json', dict(submitted=[dict(id=f'i{n}') for n in range(3)]))
        statuses = [dict(id='i0', status='running'), dict(id='i1', status='done'),
                    dict(id='i2', status='queued')]
        self.infra.status.side_effect = lambda ids: statuses
        current = L.advance()
        self.assertEqual(current['stage'], 'waiting_original_recipe_initializers')
        self.infra.dispatch_pending_initializers.assert_called_once_with([statuses[2]])
        self.assertEqual(current['initialization']['pending_initializer_dispatch'], dict(dispatched=['i2']))
        self.infra.submit.assert_not_called()
        self.assertEqual(self.native_jobs(), [])

    def test_first_two_full_days_are_reused_then_exactly_ten_remaining(self):
        self.initializers_ready()
        current = L.advance()
        L.advance()
        self.assertEqual(current['stage'], 'validation_native_preflight')
        self.assertEqual(len(self.native_jobs()), 2)
        self.preflight_ready()
        current = L.advance()
        L.advance()
        jobs = self.native_jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({row['signature'] for row in jobs}), 12)
        self.assertEqual(len(self.submissions[-1]['jobs']), 10)
        self.assertEqual([row['argv'][5] for row in jobs].count('fresh'), 6)
        self.assertEqual([row['argv'][5] for row in jobs].count('no_fresh'), 6)
        self.assertTrue(all('--stop' not in row['argv'] and row['priority'] == 'high' for row in jobs))
        self.assertFalse(current['done'])

    def test_bad_native_prefix_stops_before_remaining_pairs(self):
        self.preflight_ready()
        self.infra.common_prefix.return_value = dict(all_pass=False)
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(len(self.native_jobs()), 2)
        self.assertEqual(current['performance_verdict'], 'not_evaluated_incomplete')

    def test_first_full_day_cost_sign_is_never_read_to_release_remaining(self):
        self.preflight_ready()
        first = L.cells(PROTOCOL)[0]
        self.complete(first)
        self.outputs[first['relative']+'/summary.json'] = dict(cost='catastrophically_bad')
        L.advance()
        self.assertEqual(len(self.native_jobs()), 12)
        self.assertNotIn(mock.call(first['relative']+'/summary.json'), self.infra.collect.call_args_list)

    def test_missing_one_native_day_prevents_summary_submission(self):
        self.preflight_ready()
        for cell in L.cells(PROTOCOL)[:-1]:
            self.complete(cell)
        current = L.advance()
        self.assertEqual(current['stage'], 'frozen_v73_validation_native')
        self.assertEqual(current['completed'], 11)
        self.assertTrue(all(Path(row['runner']).name != 'analyze.py'
            for batch in self.submissions for row in batch['jobs']))

    def test_complete_validation_collects_four_small_summaries_and_stops(self):
        for passed in (True, False):
            with self.subTest(passed=passed):
                self.finish(passed)
                current = L.advance()
                self.assertEqual(current['stage'], 'fixed_validation_complete')
                self.assertTrue(current['done'])
                self.assertEqual(current['candidate_passed'], passed)
                self.assertFalse(current['old_v73_gate_reopened'])
                self.assertFalse(current['goal_achieved'])
                self.assertEqual(len(self.native_jobs()), 12)

    def test_summary_is_not_read_before_all_four_outputs_exist(self):
        self.finish(False)
        self.outputs.pop(PROTOCOL['summary_outputs'][-1])
        current = L.advance()
        self.assertEqual(current['stage'], 'frozen_v73_validation_summary')
        self.assertNotIn(mock.call(PROTOCOL['summary_outputs'][0]), self.infra.collect.call_args_list)

    def test_terminal_native_failure_retains_spent_work_and_has_no_replacement(self):
        self.preflight_ready()
        cell = L.cells(PROTOCOL)[2]
        self.outputs[cell['relative']+'/failure.json'] = dict(status='failed', consumed_budget=dict(native_seconds=5000))
        current = L.advance()
        self.assertTrue(current['done'])
        self.assertEqual(current['stage'], 'validation_native_attention')
        self.assertEqual(len(self.native_jobs()), 12)
        self.assertEqual(current['failures'][0]['failure']['consumed_budget']['native_seconds'], 5000)


if __name__ == '__main__':
    unittest.main()
