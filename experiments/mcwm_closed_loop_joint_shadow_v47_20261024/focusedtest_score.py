"""Check fixed F aliases, RNG isolation, signed contrasts and bounded exports."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import (
    FIXED_HOLDS_SECONDS, score_bank,
)
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.summary import (
    compact_worker, signed_stats, summarize_scores,
)
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.analyze import aggregate, compact_final
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


HERE = Path(__file__).resolve().parent


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(-.75))

    def forward(self, states, deterministic=False):
        torch.rand(1)  # Deliberately exercise probe RNG restoration.
        return self.p.expand(len(states), 1), None


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(1.))
        self.scored_sizes = []

    def forward(self, states, actions):
        self.scored_sizes.append(len(actions))
        q = self.p*states[:, :1]+.2*actions
        return torch.stack((q, q+.01))


class Execution:
    max_hold = 15.

    def __call__(self, states, actions):
        return actions.clamp(min=-.75)

    def hold_seconds(self, actions):
        return (actions+1.)*7.5


def learner_and_bank():
    learner = SimpleNamespace(actor=Actor(), critic=Critic(), target_critic=Critic(),
        value=torch.nn.Linear(1, 1), action_transform=Execution(), device='cpu',
        rng=np.random.default_rng(1), diagnostic_rng=np.random.default_rng(2),
        config=SimpleNamespace(online_backup='h2oplus', offline_objective='iql'))
    bank = []
    for index, line in enumerate(('7X', '102X', '705X')):
        for arm in ('low', 'high'):
            row = SimpleNamespace(s=np.array([9000./24000., float(index)], np.float32),
                memory_s=dict(observed_at=8999., vector=[0.]*84),
                sequence=index, started_at=9000., a=[-1. if arm == 'low' else -.75])
            bank.append(dict(bank_id=f'{line}/{arm}', line_id=line, arm=arm, row=row))
    return learner, bank


def snapshots(protocol, arm):
    result = []
    for packet, updates in enumerate(protocol['score_after_updates']):
        shift = updates/456. if arm == 'M8' else 0.
        rows = []
        for line_index, line in enumerate(('7X', '102X', '705X')):
            for root_arm in ('low', 'high'):
                q = [-25.-line_index+shift*.01+hold*(.001+shift*.0001)
                     for hold in FIXED_HOLDS_SECONDS]
                rows.append(dict(bank_id=f'{line}/{root_arm}', line_id=line,
                    root_arm=root_arm, state_alias_group=line_index, sequence=line_index,
                    issue_clock=9000., canonical_clock=9000., memory_observed_at=8999.,
                    candidate_proposals_seconds=list(FIXED_HOLDS_SECONDS),
                    candidate_executed_seconds=list(FIXED_HOLDS_SECONDS),
                    candidate_executed_actions=[2.*hold/15.-1. for hold in FIXED_HOLDS_SECONDS],
                    candidate_alias_groups=list(range(6)), q_heads=[q, [v+.01 for v in q]],
                    q_min=q, advantage_min=[value-q[0] for value in q], best_fixed_candidate=5,
                    actor_proposal_seconds=1.+shift*.1,
                    actor_executed_seconds=2.+(shift*.1 if updates >= 32 else 0.),
                    actor_proposal_action=-.8+shift*.01,
                    actor_executed_action=-.7+(shift*.01 if updates >= 32 else 0.),
                    online_backup='h2oplus', offline_objective='iql'))
        result.append(dict(packet=packet, cumulative_updates=updates,
            processed_at=0. if not packet else 900.+300.*packet if packet <= 3 else 9630., rows=rows))
    return result


class ScoreTests(unittest.TestCase):
    def test_unique_physical_actions_score_once_without_rng_or_gradient_mutation(self):
        learner, bank = learner_and_bank()
        for parameter in learner.critic.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = _rng()
        private = deepcopy(learner.rng.bit_generator.state)
        result = score_bank(learner, bank, packet=0, cumulative_updates=0, processed_at=0.)
        self.assertTrue(_equal(before, _rng()))
        self.assertEqual(private, learner.rng.bit_generator.state)
        self.assertEqual(learner.critic.scored_sizes, [4]*6)
        self.assertTrue(all(parameter.requires_grad and torch.equal(parameter.grad, torch.ones_like(parameter))
                            for parameter in learner.critic.parameters()))
        self.assertEqual([row['state_alias_group'] for row in result], [0, 0, 1, 1, 2, 2])
        self.assertEqual(result[0]['candidate_alias_groups'], [0, 0, 0, 1, 2, 3])
        self.assertEqual(result[0]['advantage_min'][:3], [0., 0., 0.])

    def test_cancellation_aliases_and_level_are_distinct(self):
        stats = signed_stats([1., -1., 0.])
        self.assertEqual(stats[:3], [3, 0., 2./3.])
        protocol = json.loads((HERE/'protocol.json').read_text())
        scores = summarize_scores(snapshots(protocol, 'M1'), snapshots(protocol, 'M8'), protocol)
        self.assertTrue(scores['checks']['initial_scores_exact'])
        self.assertEqual(len(scores['points']), 3*len(protocol['export_after_updates']))
        self.assertTrue(all(row[3] == 3 for row in scores['statistics'] if row[1] == 1))
        self.assertTrue(all(row[3] == 54 for row in scores['statistics'] if row[1] == 2))
        self.assertEqual([row[3] for row in scores['persistence']], [3, 3, 3])
        self.assertEqual(scores['configured_learning_modes'], dict(online_backup='h2oplus', offline_objective='iql'))

    def test_full_four_worker_export_stays_bounded_and_checks_actual_training_overlap(self):
        protocol = json.loads((HERE/'protocol.json').read_text())
        workers, records = {}, {}
        for scenario in protocol['scenarios']:
            workers[scenario], records[scenario] = {}, {}
            for arm in protocol['arms']:
                receipt = dict(status='complete', scenario=scenario, arm=arm, phase='formal',
                    priority='high', proof=dict(all_pass=True, checks=dict(source_exact=True)),
                    budget=dict(slots=285, updates=456), extra_compute=dict(extra_draws=3192),
                    actual_target_path=dict(offline_objective='iql', online_backup='h2oplus',
                        target_dispatch='one_online_backup_call_on_concatenated256'),
                    wall=dict(started=1.123456789123, finished=20.123456789123, seconds=19.,
                        training_started=3.123456789123, training_finished=18.123456789123),
                    paths=dict(scores='scores.jsonl', packets='packets.jsonl', final_model='final_model.pt'))
                workers[scenario][arm] = compact_worker(receipt)
                self.assertEqual(workers[scenario][arm]['wall'], receipt['wall'])
                records[scenario][arm] = snapshots(protocol, arm)
        combined = compact_final(aggregate(workers, records, protocol), protocol)
        encoded = json.dumps(combined, separators=(',', ':'), allow_nan=False)
        self.assertLess(len(encoded.encode()), 48000)
        self.assertTrue(combined['integrity']['all_pass'])
        workers['stable']['M8']['wall'].update(training_started=19., training_finished=20.)
        self.assertFalse(aggregate(workers, records, protocol)['integrity']['all_pass'])
        workers['stable']['M8'] = dict(status='failed', message='known worker failure')
        failed = compact_final(aggregate(workers, {}, protocol), protocol)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['integrity']['failed_workers'], ['stable/M8'])
        self.assertEqual(failed['compute_accounting']['failed_worker_spend'],
                         'not_inferred_from_missing_summary')


if __name__ == '__main__':
    unittest.main()
