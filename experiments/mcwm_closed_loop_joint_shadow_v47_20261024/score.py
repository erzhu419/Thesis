"""Common fixed actions on the six saved v43 root records; no target draws."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _restore, _rng


FIXED_HOLDS_SECONDS = (0., .9375, 1.875, 3.75, 7.5, 15.)
LINES = ('7X', '102X', '705X')
ROOT_ARMS = ('low', 'high')


def load_bank(receipt_path):
    """Read the first existing event of each already declared response arm."""
    from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import row_from_record
    receipt = json.loads(Path(receipt_path).read_text())
    indexed = {pair['line_id']: pair for pair in receipt['pairs']}
    bank = []
    for line in LINES:
        for arm in ROOT_ARMS:
            path = Path(indexed[line]['branches'][arm]['events_path'])
            with path.open() as handle:
                record = next(json.loads(text) for text in handle if text.strip())
            row = row_from_record(record['transition'])
            bank.append(dict(bank_id=f'{line}/{arm}', line_id=line, arm=arm, row=row))
    return bank


def _aliases(values):
    representatives, groups = [], []
    for value in values:
        if value not in representatives:
            representatives.append(value)
        groups.append(representatives.index(value))
    return groups


def _state_aliases(bank):
    representatives, groups = [], []
    for item in bank:
        row = item['row']
        match = next((index for index, previous in enumerate(representatives)
            if np.array_equal(row.s, previous.s)
            and row.memory_s['observed_at'] == previous.memory_s['observed_at']
            and np.array_equal(row.memory_s['vector'], previous.memory_s['vector'])), None)
        if match is None:
            match = len(representatives)
            representatives.append(row)
        groups.append(match)
    return groups


def _versions(learner):
    return [(id(parameter), parameter._version) for name in
        ('actor', 'critic', 'target_critic', 'value')
        for parameter in getattr(learner, name).parameters()]


@torch.no_grad()
def score_bank(learner, bank, *, packet, cumulative_updates, processed_at):
    """Score the same grid through F once, keeping the issue clock and memory.

    No call to learner._target is made. In particular this does not consume
    the M8 training estimator's isolated random stream.
    """
    if len(bank) != 6 or {item['bank_id'] for item in bank} != {
            f'{line}/{arm}' for line in LINES for arm in ROOT_ARMS}:
        raise ValueError('The same six v43 root records are required')
    saved, versions = _rng(), _versions(learner)
    private = {name: deepcopy(getattr(learner, name).bit_generator.state)
               for name in ('rng', 'diagnostic_rng')}
    state_groups = _state_aliases(bank)
    result = []
    try:
        for item, state_group in zip(bank, state_groups):
            row = item['row']
            states = torch.as_tensor(np.asarray(row.s), dtype=torch.float32,
                device=learner.device).reshape(1, -1)
            memory = torch.as_tensor(row.memory_s['vector'], dtype=states.dtype,
                device=learner.device).reshape(1, -1)
            explicit = critic_states(learner, states, memory)
            seconds = torch.as_tensor(FIXED_HOLDS_SECONDS, dtype=states.dtype,
                device=learner.device).reshape(-1, 1)
            proposals = 2.*seconds/learner.action_transform.max_hold-1.
            repeated = states.repeat(len(FIXED_HOLDS_SECONDS), 1)
            executed = learner.action_transform(repeated, proposals)
            actions = executed.reshape(-1).tolist()
            groups = _aliases(actions)
            # A physical action is evaluated once. Repeated float32 action
            # slots can differ by one ULP even with identical inputs.
            representatives = [groups.index(index) for index in range(max(groups)+1)]
            unique_actions = executed[representatives]
            unique_q = learner.critic(explicit.repeat(len(representatives), 1), unique_actions)
            unique_q = unique_q.reshape(-1, len(representatives))
            q = unique_q[:, groups]
            q_min, q_mean = q.min(0).values, q.mean(0)
            heads = q.tolist()
            minimum, mean = q_min.tolist(), q_mean.tolist()
            proposal = learner.actor(states, deterministic=True)[0]
            actor_executed = learner.action_transform(states, proposal)
            result.append(dict(bank_id=item['bank_id'], line_id=item['line_id'],
                root_arm=item['arm'], state_alias_group=state_group,
                sequence=row.sequence, issue_clock=float(row.started_at),
                canonical_clock=float(states[0, 0])*24000.,
                memory_observed_at=row.memory_s['observed_at'],
                packet=int(packet), cumulative_updates=int(cumulative_updates),
                snapshot_clock=float(processed_at),
                candidate_proposals_seconds=list(FIXED_HOLDS_SECONDS),
                candidate_executed_actions=actions,
                candidate_executed_seconds=learner.action_transform.hold_seconds(executed).reshape(-1).tolist(),
                candidate_alias_groups=groups, unique_executed_candidates=len(set(groups)),
                q_heads=heads, q_min=minimum, q_mean=mean,
                advantage_min=[value-minimum[0] for value in minimum],
                advantage_mean=[value-mean[0] for value in mean],
                best_fixed_candidate=int(q_min.argmax()),
                actor_proposal_seconds=float(learner.action_transform.hold_seconds(proposal).reshape(-1)[0]),
                actor_executed_seconds=float(learner.action_transform.hold_seconds(actor_executed).reshape(-1)[0]),
                actor_proposal_action=float(proposal.reshape(-1)[0]),
                actor_executed_action=float(actor_executed.reshape(-1)[0]),
                root_factual_action=float(row.a[0]), scoring_uses_target_estimator=False,
                candidate_scoring='unique_executed_actions_scored_once_then_aliases_expanded',
                online_backup=str(learner.config.online_backup),
                offline_objective=str(learner.config.offline_objective)))
    finally:
        for name, state in private.items():
            getattr(learner, name).bit_generator.state = state
        _restore(saved)
    if versions != _versions(learner) or not _equal(saved, _rng()):
        raise RuntimeError('Fixed-bank scoring changed model parameters or training RNG')
    return result
