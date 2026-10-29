"""Each check detects a budget, endpoint or snapshot change that blocks collection."""
import unittest

from experiments.mcwm_closed_loop_joint_support_v52_20261025.plan import build_plan, precheck_issues
from experiments.mcwm_closed_loop_physical_feedback_v49_20261025.test_plan import records


def extended(step=2.):
    issues, contexts = records(step=step)
    last = dict(issues[-1], sequence=271, started_at=9000+271*step, line_id='102X')
    issues.append(last)
    previous = contexts[-1]
    contexts.append(dict(previous, sequence=270, started_at=issues[270]['started_at'],
        line_id=issues[270]['line_id'], behavior_s=dict(previous['behavior_s'],
        action_rng=dict(stream_seed=49, draw_ordinal=270))))
    return issues, contexts


class PlanTests(unittest.TestCase):
    def test_exact_replacement_and_updates(self):
        plan = build_plan()
        b = plan['budget']
        self.assertEqual(sum(b['warmup_container_sizes']), 14)
        self.assertEqual(sum(b['response_admission_container_sizes']), 271)
        self.assertEqual(14+271, b['total_slots'])
        self.assertEqual(3*8+b['response_update_packets']*8, b['total_updates'])
        self.assertEqual(plan['replacement']['cancelled'], 'ordinary:14')
        self.assertEqual(plan['replacement']['replacement'], 'continuous:270')
        self.assertEqual(plan['target_design']['eta'], 9/270)

    def test_271st_real_closure_reaches_missing_300_endpoint(self):
        issues, contexts = extended(step=1.)
        # Keep fixed roots; only the extra genuine endpoint reaches the last root's +300.
        issues[-2]['started_at'] = 9301.
        contexts[-1]['started_at'] = 9301.
        issues[-1]['started_at'] = 9302.
        result = precheck_issues(issues, contexts)
        self.assertTrue(result['all_pass'], result['failure_reasons'])
        self.assertEqual(result['root_coverage']['705X']['first_real_300_endpoint_index'], 271)
        short = precheck_issues(issues[:-1], contexts)
        self.assertFalse(short['checks']['exact271_native_td'])
        self.assertFalse(short['checks']['all_roots_reach300'])

    def test_zero_duration_is_charged_not_skipped(self):
        issues, contexts = extended()
        issues[1]['started_at'] = issues[0]['started_at']
        contexts[1]['started_at'] = issues[1]['started_at']
        result = precheck_issues(issues, contexts)
        self.assertTrue(result['all_pass'], result['failure_reasons'])
        self.assertEqual(result['budget']['unique_native_td_slots'], 271)
        self.assertEqual(result['budget']['zero_duration_slots'], 1)
        self.assertEqual(result['budget']['prospective_total_slots'], 285)

    def test_missing_behavior_or_native_endpoint_is_preserved(self):
        issues, contexts = extended()
        missing = precheck_issues(issues, contexts[:-1])
        self.assertFalse(missing['all_pass'])
        self.assertEqual(missing['missing']['behavior_rows'], 1)
        short = precheck_issues(issues[:-1], contexts)
        self.assertEqual(short['missing']['native_closures'], 1)
        self.assertIsNone(short['selection']['endpoint_clock'])

    def test_fixed_start_and_fixed_cutoff_never_reselected(self):
        issues, contexts = extended(step=1.)
        result = precheck_issues(issues, contexts)
        self.assertFalse(result['checks']['all_roots_reach300'])
        self.assertEqual(result['selection']['start_sequence'], 0)
        self.assertEqual(result['selection']['reselection_attempts'], 0)
        late = precheck_issues(*extended(step=2.3))
        self.assertFalse(late['checks']['released_by9630'])
        issues, contexts = extended()
        issues[20]['sequence'] += 1
        self.assertFalse(precheck_issues(issues, contexts)['checks']['adjacent_native_sequences'])

    def test_old_geometry_does_not_certify_new_snapshot(self):
        result = precheck_issues(*extended())
        self.assertTrue(result['all_pass'], result['failure_reasons'])
        self.assertFalse(result['future_snapshot_available'])
        self.assertFalse(result['future_joint_support_pass'])
        plan = build_plan()
        self.assertIn('not original current-actor Y', plan['synchronization']['lambda_zero'])
        self.assertFalse(plan['synchronization']['current_actor_RN_is_one'])


if __name__ == '__main__':
    unittest.main()
