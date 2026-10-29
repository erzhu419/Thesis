"""Failures here change whether the fixed physical-coverage design can proceed."""
from copy import deepcopy
import unittest

from experiments.mcwm_closed_loop_physical_feedback_v49_20261025.plan import build_plan, precheck_issues


def records(step=2., start=9000.):
    issues = [dict(sequence=i, started_at=start+step*i,
                   line_id=('7X', '102X', '705X')[i % 3], episode_id=0) for i in range(271)]
    contexts = []
    for row in issues[:270]:
        action = -1. if row['sequence'] % 2 == 0 else -.5
        contexts.append(dict(sequence=row['sequence'], started_at=row['started_at'], line_id=row['line_id'],
            behavior_s=dict(actor_snapshot='actual_post24', intervention=False,
                policy_input=[0.]*229, execution_input=[0.]*229,
                parameters=dict(positive_probability=.6, zero_probability=.4, mean=0., std=1., gate=.4, log_std=0.),
                proposal=[action], nominal_executed_action=[action], executed_action=[action],
                action_rng=dict(stream_seed=49, draw_ordinal=row['sequence']))))
    return issues, contexts


class PlanTests(unittest.TestCase):
    def test_budget_shared_suffixes_and_ideal_bound(self):
        plan = build_plan()
        self.assertEqual(plan['budget']['total_slots'], 285)
        self.assertEqual(plan['budget']['total_updates'], 456)
        result = precheck_issues(*records())
        self.assertTrue(result['all_pass'], result['failure_reasons'])
        self.assertEqual(result['budget']['unique_native_td_slots'], 270)
        self.assertEqual(result['budget']['root_suffix_references'], 807)
        root = result['root_coverage']['7X']
        self.assertEqual(root['first_real_300_endpoint_clock'], 9300.)
        self.assertEqual(root['trace_links_to_300'], 149)
        self.assertEqual(root['ideal_c1_mixed_weight_to_300'], .9**149)

    def test_zero_duration_keeps_slot_and_strict_execution_zero(self):
        issues, behavior = records()
        issues[1]['started_at'] = issues[0]['started_at']
        behavior[1]['started_at'] = issues[1]['started_at']
        behavior[1]['behavior_s']['nominal_executed_action'] = [-1.+1e-12]
        behavior[1]['behavior_s']['executed_action'] = [-1.+1e-12]
        result = precheck_issues(issues, behavior)
        self.assertTrue(result['all_pass'])
        self.assertEqual(result['budget']['zero_duration_slots'], 1)
        self.assertEqual(result['native_action_coverage']['all']['executed_zero_strict'], 135)

    def test_short_physical_span_fails_without_moving_start(self):
        result = precheck_issues(*records(step=1.))
        self.assertFalse(result['all_pass'])
        self.assertFalse(result['checks']['all_roots_reach300'])
        self.assertEqual(result['selection']['start_clock'], 9000.)
        self.assertEqual(result['selection']['reselection_attempts'], 0)

    def test_true_endpoint_overshoot_is_retained_and_late_end_fails(self):
        result = precheck_issues(*records(step=2.1))
        self.assertTrue(result['all_pass'])
        self.assertGreater(result['root_coverage']['7X']['first_real_300_endpoint_clock'], 9300.)
        late = precheck_issues(*records(step=2.3))
        self.assertFalse(late['checks']['released_by9630'])

    def test_no_hindsight_start_shift_or_skipped_event(self):
        issues, behavior = records(step=1.)
        issues[0]['line_id'] = 'other'
        behavior[0]['line_id'] = 'other'
        issues[270]['started_at'] = 9302.
        result = precheck_issues(issues, behavior)
        self.assertEqual(result['selection']['start_sequence'], 0)
        self.assertFalse(result['checks']['all_roots_reach300'])
        issues, behavior = records()
        issues[20]['sequence'] += 1
        self.assertFalse(precheck_issues(issues, behavior)['checks']['adjacent_native_sequences'])

    def test_forced_action_or_wrong_actual_input_fails(self):
        for field, value in [('intervention', True), ('actor_snapshot', 'initial50000'),
                             ('executed_action', [.5]), ('execution_input', [1.]*229)]:
            issues, behavior = records()
            behavior[0]['behavior_s'][field] = deepcopy(value)
            self.assertFalse(precheck_issues(issues, behavior)['all_pass'])


if __name__ == '__main__':
    unittest.main()
