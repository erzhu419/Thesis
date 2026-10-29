"""Batch checks for all three models, real missing states, and small exports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.mcwm_closed_loop_multievent_credit_v38_20261024 import analyze as batch
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.analyze_chain import analyze_pair
from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.test_analyze_chain import fixture


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


def shifted(value, offset):
    if isinstance(value, list):
        return [shifted(item, offset) for item in value]
    if isinstance(value, dict):
        return {key: (item+offset if key in ('started_at', 'completed_at', 'time_seconds',
            'clock_seconds', 'clock', 'observed_at') and isinstance(item, (int, float))
            else shifted(item, offset)) for key, item in value.items()}
    return value


def populated_case(root, scenario):
    root = root/scenario
    proof = dict(all_pass=True, source_initializer_updates=50000, source_feedback_slots=285,
        source_updates=456, new_updates=0, new_feedback=0, old_twelve_state_truth_read=False)
    budget = dict(new_native_prefixes=1, prefix_native_seconds=24000., pairs=6, suffixes=12,
        actual_suffix_native_seconds=3612., overrun_native_seconds=12.,
        new_training_labels=0, new_optimizer_updates=0, frozen_training_slots=285,
        frozen_updates_per_model=456)
    save(root/'model_gate.json', proof)
    save(root/'scored'/'summary.json', dict(status='complete', integrity=proof))
    save(root/'summary.json', dict(status='complete', integrity=dict(all_pass=True), budget=budget))
    native_checks = dict(full_v37_prefix_exact=True, full24000_parent=True,
        all_three_old_early_exact=True, collector_frozen=True)
    selection = {}
    for line in batch.LINES:
        for window in batch.WINDOWS:
            slot = line+'/'+window
            suffix = slot.replace('/', '_')
            low, lt, selected, lexact, lend = fixture()
            high, ht, _, hexact, hend = fixture(high=True)
            if window == 'late':
                low, high, lt, ht, selected, lexact, hexact, lend, hend = [shifted(value, 18250.)
                    for value in (low, high, lt, ht, selected, lexact, hexact, lend, hend)]
            for rows in (low, high):
                for row in rows:
                    row['transition']['line_id'] = line
            record = dict(slot=slot, selected=selected, pair_receipt=dict(all_pass=True, overlap_seconds=.9),
                native_feasibility=dict(feasible=True), old_early_replay=dict(all_pass=True) if window == 'early' else None)
            save(root/'native'/'states'/suffix/'state_summary.json', record)
            selection[slot] = dict(status='selected', reasons={}, selected=dict(issue=selected['started_at']))
            for arm, rows, ticks, exact, endpoint, hold in (
                    ('low', low, lt, lexact, lend, 0.), ('high', high, ht, hexact, hend, 1.875)):
                summary = dict(selected=selected, external_seed=106237,
                    integrity=dict(all_pass=True, checks=dict(suffix_zero_training=True)),
                    execution=dict(native_applied_hold_seconds=hold, executed_hold_seconds=hold),
                    truth=dict(fixed300=exact), real_endpoint=endpoint,
                    chain=dict(zero_duration_count=1), budget=dict(new_training_labels=0,
                        new_optimizer_updates=0, overrun_native_seconds=1.))
                save(root/'native'/'states'/suffix/arm/'summary.json', summary)
                lines(root/'native'/'states'/suffix/arm/'ticks.jsonl', ticks)
                lines(root/'scored'/'states'/suffix/arm/'events.jsonl', rows)
    save(root/'native'/'summary.json', dict(status='complete', integrity=dict(all_pass=True,
        checks=native_checks), budget=budget, selection=selection))


class BatchTests(unittest.TestCase):
    def test_generator_groups_include_every_model(self):
        low, lt, selected, lexact, lend = fixture()
        high, ht, _, hexact, hend = fixture(high=True)
        result = analyze_pair(low, high, lt, ht, selected=selected,
            exact300_by_arm=dict(low=lexact, high=hexact), endpoint_by_arm=dict(low=lend, high=hend))
        stats = batch._stats(value for value in (result, result))
        for model in batch.MODELS:
            self.assertEqual(stats[model][batch.METRIC_COLUMNS.index('states')], 2)
            self.assertEqual(stats[model][batch.METRIC_COLUMNS.index('q_regret')], 1180.)

    def test_full_batch_retains_twelve_states_and_small_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in batch.SCENARIOS:
                populated_case(root/'runs', scenario)
            result = batch.analyze(root/'runs', root/'analysis')
            self.assertEqual(result['status'], 'complete', result['integrity'])
            self.assertTrue(result['integrity']['all_pass'])
            self.assertEqual(result['observations']['available_states'], 12)
            self.assertLessEqual((root/'analysis'/'summary.json').stat().st_size, batch.MAX_BYTES)
            self.assertEqual(len(list((root/'analysis').glob('*/*.json'))), 12)
            for scenario in batch.SCENARIOS:
                for group in result['cases'][scenario]['metrics'].values():
                    self.assertEqual(group['zero_memory'][0], group['per_head_level'][0])
                    self.assertEqual(group['zero_memory'][0], group['shared_level'][0])
            self.assertFalse(result['decision']['new_training'])

    def test_missing_state_and_failed_prefix_budget_are_never_zero_imputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populated_case(root/'runs', 'stable')
            path = root/'runs'/'stable'/'native'/'states'/'102X_late'/'state_summary.json'
            path.unlink()
            failed = root/'runs'/'local_speed_drop'/'native'
            save(failed/'failure.json', dict(message='native failed after real progress', native_time=1234.,
                budget=dict(prefix_native_seconds=1234., actual_suffix_native_seconds=400.,
                    unknown_or_lower_bound=True, new_training_labels=0, new_optimizer_updates=0)))
            result = batch.analyze(root/'runs', root/'analysis')
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['observations']['available_states'], 5)
            self.assertEqual(result['cases']['local_speed_drop']['budget']['actual_suffix_native_seconds'], 400.)
            self.assertTrue(result['cases']['local_speed_drop']['budget']['unknown_or_lower_bound'])
            missing = json.loads((root/'analysis'/'stable'/'102X_late.json').read_text())
            self.assertEqual(missing['status'], 'missing')
            self.assertNotIn('analysis', missing)

    def test_cli_exception_writes_coordinator_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'analysis'
            argv = ['analyze.py', '--runs', directory, '--out', str(out)]
            with patch('sys.argv', argv), patch.object(batch, 'analyze', side_effect=ValueError('invalid real chain')):
                with self.assertRaisesRegex(ValueError, 'invalid real chain'):
                    batch.main()
            self.assertEqual(json.loads((out/'failure.json').read_text())['type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
