"""v59 scheduling and restart contracts; no scheduler or SSH operations."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path, imports=None):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, imports or {}):
        spec.loader.exec_module(value)
    return value


INFRA = load('v59_infra_tested', HERE/'infra.py')
LAUNCH = load('v59_launch_tested', HERE/'launch.py', {'infra': INFRA})
CONTINUE = load('v59_continue_tested', HERE/'continue_batch.py', {'infra': INFRA, 'launch': LAUNCH})
P = dict(name='v59-test', scenarios=['stable', 'local_speed_drop'], training_seed=6202)


def receipt(labels):
    return dict(submitted=[dict(id='t'+str(i), signature=P['name']+'/'+name)
        for i, name in enumerate(labels)])


class InfraTests(unittest.TestCase):
    def test_high_priority_parallel_cpu_resources(self):
        rows = []
        with mock.patch.object(LAUNCH, 'protocol', return_value=P), mock.patch.object(INFRA,
                'submit', side_effect=lambda batch, name: rows.extend(batch)):
            LAUNCH.native()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['cpu'] for row in rows}, {2})
        for row in rows:
            self.assertEqual(row['priority'], 'high')
            self.assertEqual(row['allowed_nodes'], INFRA.CPUS)
            self.assertTrue(row['no_auto_requeue'])
            self.assertNotIn('--preflight', row['cmd'])
            self.assertIn('collect.py', row['cmd'])
        self.assertNotEqual(rows[0]['signature'], rows[1]['signature'])

    def test_complete_case_launches_two_independent_learning_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'runs/collection/stable/summary.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='complete')))
            with mock.patch.object(LAUNCH, 'HERE', root), mock.patch.object(LAUNCH,
                    'protocol', return_value=P), mock.patch.object(INFRA, 'remote', side_effect=str), \
                    mock.patch.object(INFRA, 'submit', return_value={}) as submit:
                LAUNCH.learning('stable')
            rows, batch = submit.call_args.args
            self.assertEqual(batch, 'learning_stable')
            self.assertEqual(len(rows), 2)
            self.assertEqual({row['cpu'] for row in rows}, {1})
            self.assertEqual({row['signature'].split('/')[-1] for row in rows}, {'fresh', 'no_fresh'})
            self.assertTrue(all('run.py' in row['cmd'] and row['priority'] == 'high' for row in rows))
            self.assertTrue(all('depends_on' not in row for row in rows))

    def test_incomplete_collection_cannot_start_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'runs/collection/stable/summary.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='integrity_failed')))
            with mock.patch.object(LAUNCH, 'HERE', Path(tmp)), mock.patch.object(LAUNCH,
                    'protocol', return_value=P), mock.patch.object(INFRA, 'submit') as submit:
                with self.assertRaises(RuntimeError):
                    LAUNCH.learning('stable')
                submit.assert_not_called()

    def test_unresolved_submission_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'native_submission_attempt.json').write_text('{}')
            with mock.patch.object(INFRA, 'HERE', root), mock.patch.object(INFRA.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'Unresolved submission'):
                    INFRA.submit([dict(signature='v59/native')], 'native')
                run.assert_not_called()

    def test_existing_receipt_reuses_task_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = receipt(['collection/stable', 'collection/local_speed_drop'])
            (root/'native_submission.json').write_text(json.dumps(existing))
            with mock.patch.object(INFRA, 'HERE', root), mock.patch.object(INFRA.subprocess, 'run') as run:
                self.assertEqual(INFRA.submit([], 'native'), existing)
                run.assert_not_called()

    def test_collection_rejects_large_artifact_types(self):
        for path in ('runs/data.csv', 'runs/model.ckpt', '../old/summary.json'):
            with self.assertRaises(ValueError):
                INFRA.collect(path)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.native = receipt(['collection/stable', 'collection/local_speed_drop'])
        self.native_status = {row['id']: dict(id=row['id'], status='running', node='node001')
            for row in self.native['submitted']}

    def test_case_starts_while_other_case_still_collecting(self):
        found = {'runs/collection/stable/summary.json': True}
        statuses = self.native_status
        learning = dict(submitted=[dict(id='tf', signature=P['name']+'/learning/stable/fresh'),
            dict(id='tn', signature=P['name']+'/learning/stable/no_fresh')])
        with mock.patch.object(LAUNCH, 'protocol', return_value=P), \
                mock.patch.object(CONTINUE, 'read_receipt', side_effect=lambda name: self.native if name == 'native' else None), \
                mock.patch.object(CONTINUE, 'snapshot', return_value=(statuses, found)), \
                mock.patch.object(INFRA, 'collect', return_value=dict(status='complete')), \
                mock.patch.object(LAUNCH, 'learning', return_value=learning) as learn, \
                mock.patch.object(LAUNCH, 'finalize') as finalize:
            value = CONTINUE.step()
        learn.assert_called_once_with('stable')
        finalize.assert_not_called()
        self.assertEqual(value['submitted_ids'], ['tf', 'tn'])
        self.assertFalse(value['done'])
        self.assertFalse(value['all_terminal'])
        self.assertEqual(value['cases']['local_speed_drop']['collection']['state'], 'pending')

    def test_failed_native_skips_learning_and_allows_partial_summary(self):
        statuses = {key: dict(row, status='failed') for key, row in self.native_status.items()}
        final = dict(submitted=[dict(id='ts', signature=P['name']+'/small-summary')])
        with mock.patch.object(LAUNCH, 'protocol', return_value=P), \
                mock.patch.object(CONTINUE, 'read_receipt', side_effect=lambda name: self.native if name == 'native' else None), \
                mock.patch.object(CONTINUE, 'snapshot', return_value=(statuses, {})), \
                mock.patch.object(INFRA, 'publish_state') as publish, \
                mock.patch.object(LAUNCH, 'learning') as learn, \
                mock.patch.object(LAUNCH, 'finalize', return_value=final) as finalize:
            value = CONTINUE.step()
        learn.assert_not_called()
        finalize.assert_called_once()
        self.assertTrue(publish.call_args.args[0]['all_terminal'])
        self.assertTrue(value['all_terminal'])
        for case in value['cases'].values():
            self.assertEqual(case['collection']['state'], 'failed')
            self.assertEqual({row['state'] for row in case['learning'].values()}, {'skipped'})

    def test_failed_worker_is_terminal_without_new_worker(self):
        value = CONTINUE.outcome('tf', 'runs/learning/stable/fresh',
            {'tf': dict(status='failed', node='node002')}, {})
        self.assertEqual(value['state'], 'failed')
        self.assertIn('without declared output', value['message'])

    def test_snapshot_uses_only_receipted_ids(self):
        expected_ids = [row['id'] for row in self.native['submitted']]
        with mock.patch.object(INFRA, 'status', return_value=list(self.native_status.values())) as status, \
                mock.patch.object(INFRA, 'presence', return_value={}):
            CONTINUE.snapshot([self.native], [])
        status.assert_called_once_with(expected_ids)

    def test_unknown_task_is_not_resubmitted(self):
        with mock.patch.object(INFRA, 'status', return_value=[]), \
                mock.patch.object(INFRA, 'presence') as presence:
            with self.assertRaisesRegex(RuntimeError, 'no task will be resubmitted'):
                CONTINUE.snapshot([self.native], [])
            presence.assert_not_called()


if __name__ == '__main__':
    unittest.main()
