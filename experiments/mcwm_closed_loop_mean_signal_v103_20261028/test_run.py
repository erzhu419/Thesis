"""Synthetic collector contract checks; no SUMO, training or new labels."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mean_run',HERE/'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class FakeTrace:
    def __init__(self, path, observed, matched, remaining='', exact=True):
        self.path,self.observed_issues,self.matched_issues = path,observed,matched
        self.comparison_exact = exact
        self.reference = io.StringIO(remaining)
        self.writer = path.open('w')
        for index in range(observed):
            self.writer.write(json.dumps({'sequence':index})+'\n')

    def close(self):
        self.writer.close()
        self.reference.close()


def pair_fixture():
    checks = {key:True for key in ('same_complete_root','same_fixed_reference','same_future',
        'original_actor_root_exact','predeclared_candidate_exact','actual_departures_observed',
        'physical_departure_separation','actual_next_arrivals','both_real_terminal','native_fork',
        'actual_parallel','parent_actor_frozen','parent_at_fixed_root','trace_persisted','rng_only_change',
        'rng_used','reference_before_equals_parent','candidate_before_equals_parent',
        'paired_after_physical','paired_after_lane_rng','paired_after_other_rng')}
    rng_checks = {key:True for key in ('native_physical_serialization_unchanged','other_native_rng_unchanged',
        'python_numpy_torch_rng_unchanged','native_clock_unchanged','lane_pools_changed_as_declared')}
    trace_checks = {key:True for key in ('expected_rows','continuous_sequences','first_sequence','last_sequence')}
    pair = dict(future_id='R1',future_seed=1012471808,all_pass=True,checks=checks,
        physical_noop=False,overlap_seconds=350.1234567890123,
        delta={'wait':1234567.,'ride':-1234568.,'cost':1234566.})
    for name in ('action','remaining','physical','rng','rng_usage','trace_audit'):
        pair[name] = {}
    for role in ('reference','candidate'):
        pair['action'][role] = dict(executed_action=[-.8812345678901234],hold_seconds=3.577204942703247)
        pair['remaining'][role] = dict(dt=11997.,wait=123456789.,ride=234567891.,cost=481481469.,observed_at=24000,available_at=24030)
        pair['physical'][role] = dict(departure_clock=12007.,next_arrival_clock=12085.)
        pair['rng'][role] = dict(mode='lane_rng_replacement',seed=1012471808,lane_count=64,
            burn_per_lane=1000000,other_native_streams='unchanged',loader_symbol='resident_symbol',
            all_pass=True,checks=copy.deepcopy(rng_checks),clock=12003.,changed_lane_pools=list(range(64)),
            before_lane_counts=[123456789012]*64,after_lane_counts=[1000000]*64)
        pair['rng_usage'][role] = dict(all_pass=True,pools_advanced=64,total_draws=123456789,
            native_clock=24000.,draws_per_pool=[123456789]*64)
        pair['trace_audit'][role] = dict(all_pass=True,checks=copy.deepcopy(trace_checks),rows=5555,
            expected_rows=5555,first_sequence=8128,last_sequence=13682,first_mismatch=None)
    return pair


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE/'protocol.json').read_text())

    def test_native_root_order_is_not_protocol_line_order(self):
        rows = run.declared_roots(self.protocol,309177151,'stable')
        self.assertEqual([row['line_id'] for row in rows],['102X','7X','705X'])
        self.assertEqual([row['root']['sequence'] for row in rows],[8111,8116,8124])
        self.assertEqual(rows[0]['futures'],self.protocol['roots'][1]['futures'])

    def test_full_panel_has_declared_native_cost_and_no_F0(self):
        cost = 0
        roots = []
        for seed in self.protocol['external_seeds']:
            for scene in self.protocol['scenarios']:
                rows = run.declared_roots(self.protocol,seed,scene)
                roots.extend(rows)
                cost += 12600+sum(2*len(row['futures'])*(24000-row['root']['started_at']) for row in rows)
        self.assertEqual(cost,1777752)
        self.assertEqual(len(roots),12)
        altered = copy.deepcopy(self.protocol)
        altered['roots'][0]['futures'][0]['seed'] = None
        with self.assertRaises(ValueError):
            run.declared_roots(altered,309177151,'stable')

    def test_compaction_keeps_measured_noop_and_truth_without_mutating_resident(self):
        pair = pair_fixture()
        pair['checks']['physical_departure_separation'] = False
        pair['physical_noop'] = True
        before = copy.deepcopy(pair)
        compact = run.compact_pair(pair)
        self.assertEqual(pair,before)
        for key in ('remaining','delta','action','checks','physical','physical_noop','trace_audit'):
            self.assertEqual(compact[key],before[key])
        self.assertTrue(run.pair_integrity(compact['checks']))
        self.assertNotEqual(compact['delta']['cost'],0)
        self.assertEqual(compact['rng']['reference']['changed_lane_pool_count'],64)
        self.assertTrue(compact['rng']['reference']['all_declared_lane_pools_changed'])
        self.assertNotIn('draws_per_pool',compact['rng_usage']['reference'])
        self.assertNotIn('before_lane_counts',compact['rng']['reference'])

    def test_eighteen_compact_pairs_fit_bound_with_collector_metadata(self):
        pair = run.compact_pair(pair_fixture())
        # 8 KB exceeds the normal three-root metadata plus 36 budget-detail rows.
        value = dict(roots=[dict(pairs=[copy.deepcopy(pair) for _ in range(6)]) for _ in range(3)],
            metadata='x'*8000)
        size = len(json.dumps(value,separators=(',',':')).encode())+1
        self.assertLessEqual(size,65536)

    def test_finalize_reads_reference_before_closing_and_flushes_last_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = FakeTrace(Path(directory)/'prefix.jsonl',3,3)
            prefix,audit = run.finalize_prefix(trace,'original.jsonl')
        self.assertTrue(prefix['all_pass'])
        self.assertEqual(prefix['source_rows'],3)
        self.assertTrue(audit['all_pass'])
        self.assertEqual(audit['rows'],3)
        self.assertTrue(trace.reference.closed)
        self.assertTrue(trace.writer.closed)

    def test_partial_or_mismatched_prefix_stays_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = FakeTrace(Path(directory)/'partial.jsonl',2,2,'{"sequence":2}\n')
            prefix,audit = run.finalize_prefix(trace,'original.jsonl')
            self.assertFalse(prefix['all_pass'])
            self.assertFalse(prefix['reference_exhausted'])
            self.assertEqual(prefix['source_rows'],3)
            self.assertTrue(audit['all_pass'])
            trace = FakeTrace(Path(directory)/'mismatch.jsonl',2,1,exact=False)
            prefix,audit = run.finalize_prefix(trace,'original.jsonl')
            self.assertFalse(prefix['all_pass'])
            self.assertIsNone(prefix['source_rows'])
            self.assertTrue(audit['all_pass'])


if __name__ == '__main__':
    unittest.main()
