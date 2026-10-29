"""Pure synthetic collector checks; does not import native runtime or run SUMO."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('future_run',Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class CollectorTests(unittest.TestCase):
    def trace(self, directory, sequences):
        path = Path(directory)/'trace.jsonl'
        path.write_text(''.join(json.dumps({'sequence':sequence})+'\n' for sequence in sequences))
        return path

    def test_suffix_file_count_excludes_inherited_prefix_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            value = run.audit_trace(self.trace(directory,[8128,8129,8130]),8128,8131)
        self.assertTrue(value['all_pass'])
        self.assertEqual(value['rows'],3)
        self.assertEqual(value['expected_rows'],3)

    def test_unflushed_tail_and_duplicate_sequence_fail_durability(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = run.audit_trace(self.trace(directory,[8128,8129]),8128,8131)
            duplicate = run.audit_trace(self.trace(directory,[8128,8129,8129]),8128,8131)
        self.assertFalse(missing['all_pass'])
        self.assertFalse(missing['checks']['expected_rows'])
        self.assertFalse(duplicate['all_pass'])
        self.assertEqual(duplicate['first_mismatch'],{'row':2,'expected':8130,'observed':8129})

    def test_prefix_audit_excludes_half_issued_root(self):
        with tempfile.TemporaryDirectory() as directory:
            value = run.audit_trace(self.trace(directory,[0,1,2]),0,3)
        self.assertTrue(value['all_pass'])
        self.assertEqual(value['last_sequence'],2)

    def test_F0_requires_individual_W_R_not_merely_equal_D(self):
        original = {'reference':{'wait':100,'ride':200},'candidate':{'wait':110,'ride':220}}
        arms = {role:{'remaining':copy.deepcopy(value)} for role,value in original.items()}
        self.assertTrue(run.original_reproduction(arms,original)['all_pass'])
        for value in arms.values():
            value['remaining']['wait'] += 5
        self.assertFalse(run.original_reproduction(arms,original)['all_pass'])

    def test_complete_noop_retained_but_technical_error_stops(self):
        self.assertTrue(run.pair_integrity({'physical_departure_separation':False,'same_root':True}))
        self.assertFalse(run.pair_integrity({'physical_departure_separation':False,'same_root':False}))

    def test_paired_native_state_checks_all_three_parts_and_parent(self):
        parent = {'physical':('bus',10),'lane':{0:'old'},'other_rng':['unchanged']}
        before = {role:copy.deepcopy(parent) for role in ('reference','candidate')}
        after = copy.deepcopy(before)
        for value in after.values():
            value['lane'][0] = 'new'
        self.assertTrue(all(run.paired_native_checks(parent,before,after).values()))
        after['candidate']['lane'][0] = 'wrong'
        self.assertFalse(run.paired_native_checks(parent,before,after)['paired_after_lane_rng'])
        before['reference']['physical'] = ('bus',11)
        self.assertFalse(run.paired_native_checks(parent,before,after)['reference_before_equals_parent'])

    def test_declared_futures_roots_and_native_budget(self):
        self.assertEqual(run.FUTURES,(('F0',None),('F1',102001),('F2',202001)))
        computed = sum(value['time']+6*(24000-value['time']) for value in run.CASES.values())
        self.assertEqual(computed,167955.)
        self.assertEqual(computed,sum(value['cap'] for value in run.CASES.values()))


if __name__ == '__main__':
    unittest.main()
