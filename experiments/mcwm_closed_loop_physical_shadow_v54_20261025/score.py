"""Read-only common-action scores on the newly collected v53 factual issues."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.policy_summary import actor_scores
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.replay import decode
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import _aliases, _versions
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _restore, _rng


LINES = ('7X', '102X', '705X')
FIXED_HOLDS_SECONDS = (0., 1.875, 3.75, 7.5)
# Fixed descriptive CRNs, separate from every training/target random stream.
ACTOR_BLOCKS = tuple(dict(block_id=i,
    draw_seeds=[546202+8*i+j for j in range(8)]) for i in range(16))


def load_bank(source_root, protocol=None):
    """Keep the first actual issue per line; never read a counterfactual label."""
    window = (protocol or {}).get('issue_window', (9000., 9300.))
    found = {}
    with (Path(source_root)/'continuous_feedback.jsonl').open() as handle:
        for text in handle:
            if not text.strip():
                continue
            row = decode(json.loads(text)['transition'])
            if row.line_id in LINES and row.line_id not in found and window[0] <= row.started_at < window[1]:
                found[row.line_id] = dict(bank_id=f'{row.line_id}/root', line_id=row.line_id, row=row)
    if set(found) != set(LINES):
        raise ValueError('The three predeclared v53 line roots are required')
    return [found[line] for line in LINES]


def _snapshot(learner):
    return dict(global_rng=_rng(), versions=_versions(learner),
        private={name: deepcopy(getattr(learner, name).bit_generator.state)
            for name in ('rng', 'diagnostic_rng')}, updates=learner.total_updates)


def _finish(learner, before):
    for name, state in before['private'].items():
        getattr(learner, name).bit_generator.state = state
    _restore(before['global_rng'])
    if (before['versions'] != _versions(learner) or learner.total_updates != before['updates']
            or not _equal(before['global_rng'], _rng())):
        raise RuntimeError('Read-only action scoring changed model parameters or training RNG')


@torch.no_grad()
def fixed_executed_candidates(transform, states):
    """Invert the existing reserve/native branches without bypassing F.

    These four binary-exact durations need only native proposal h or8h.
    A completion floor can make a requested duration unreachable; retain its
    actual transformed action and mark that duration unsupported.
    """
    requested = states.new_tensor(FIXED_HOLDS_SECONDS).reshape(-1, 1)
    repeated = states.repeat(len(FIXED_HOLDS_SECONDS), 1)
    direct = 2.*requested/transform.max_hold-1.
    first = transform(repeated, direct)
    supported_direct = transform.hold_seconds(first) == requested
    native = 2.*(8.*requested)/transform.max_hold-1.
    proposals = torch.where(supported_direct, direct, native)
    executed = transform(repeated, proposals)
    supported = transform.hold_seconds(executed) == requested
    return proposals, executed, supported.reshape(-1).tolist()


@torch.no_grad()
def actor_candidates(learner, bank):
    """Small per-packet manifest for exchanging the two live actor candidates."""
    before, result = _snapshot(learner), {}
    try:
        for item in bank:
            row = item['row']
            state = torch.as_tensor(np.asarray(row.s), dtype=torch.float32,
                device=learner.device).reshape(1, -1)
            proposal = learner.actor(state, deterministic=True)[0]
            executed = learner.action_transform(state, proposal)
            distribution = actor_scores(learner, row, ACTOR_BLOCKS)
            result[item['bank_id']] = dict(proposal_action=float(proposal.reshape(-1)[0]),
                proposal_seconds=float(learner.action_transform.hold_seconds(proposal).reshape(-1)[0]),
                executed_action=float(executed.reshape(-1)[0]),
                executed_seconds=float(learner.action_transform.hold_seconds(executed).reshape(-1)[0]),
                expected_holding_seconds=distribution['expected_holding'],
                expected_holding_block_means=distribution['block_means'],
                p_hold=distribution['p_hold'], latent_mean=distribution['latent_mean'],
                latent_log_std=distribution['latent_log_std'])
    finally:
        _finish(learner, before)
    return result


@torch.no_grad()
def score_bank(learner, bank, *, packet, cumulative_updates, processed_at=9630.,
               shared_actor_proposals=None, own_candidates=None, arm=None):
    """Score F aliases once, then expand the same physical grid for comparison.

    The runner exchanges only actor manifests between concurrent workers. If
    those manifests are omitted the common fixed grid remains valid, while
    actor-candidate comparison is explicitly marked incomplete.
    """
    before, result = _snapshot(learner), []
    try:
        own = own_candidates if own_candidates is not None else actor_candidates(learner, bank)
        shared = shared_actor_proposals or {}
        for item in bank:
            row, bank_id = item['row'], item['bank_id']
            states = torch.as_tensor(np.asarray(row.s), dtype=torch.float32,
                device=learner.device).reshape(1, -1)
            memory = torch.as_tensor(row.memory_s['vector'], dtype=states.dtype,
                device=learner.device).reshape(1, -1)
            explicit = critic_states(learner, states, memory)
            fixed, _, supported = fixed_executed_candidates(learner.action_transform, states)
            proposals = fixed.reshape(-1).tolist()
            labels = [f'requested_executed:{hold:g}' for hold in FIXED_HOLDS_SECONDS]
            for name, manifest in sorted(shared.items()):
                candidate = manifest[bank_id]
                proposals.append(float(candidate['proposal_action'] if isinstance(candidate, dict) else candidate))
                labels.append(f'actor:{name}')
            proposed = states.new_tensor(proposals).reshape(-1, 1)
            executed = learner.action_transform(states.repeat(len(proposals), 1), proposed)
            actions = executed.reshape(-1).tolist()
            groups = _aliases(actions)
            representatives = [groups.index(index) for index in range(max(groups)+1)]
            unique_q = learner.critic(explicit.repeat(len(representatives), 1), executed[representatives])
            q = unique_q.reshape(-1, len(representatives))[:, groups]
            q_min, q_mean = q.min(0).values, q.mean(0)
            own_actor = own[bank_id]
            result.append(dict(bank_id=bank_id, line_id=item['line_id'], sequence=row.sequence,
                issue_clock=float(row.started_at), memory_observed_at=float(row.memory_s['observed_at']),
                packet=int(packet), cumulative_updates=int(cumulative_updates),
                snapshot_clock=float(processed_at), arm=arm,
                candidate_labels=labels, candidate_proposal_actions=proposals,
                fixed_requested_executed_seconds=list(FIXED_HOLDS_SECONDS),
                fixed_executed_supported=supported,
                candidate_executed_actions=actions,
                candidate_executed_seconds=learner.action_transform.hold_seconds(executed).reshape(-1).tolist(),
                candidate_alias_groups=groups, unique_executed_candidates=len(representatives),
                shared_actor_candidates_complete=len(shared) == 2,
                q_heads=q.tolist(), q_min=q_min.tolist(), q_mean=q_mean.tolist(),
                advantage_min=(q_min-q_min[0]).tolist(),
                advantage_mean=(q_mean-q_mean[0]).tolist(),
                best_candidate=int(q_min.argmax()),
                best_fixed_candidate=int(q_min[:len(FIXED_HOLDS_SECONDS)].argmax()),
                min_head=q.argmin(0).tolist(), actor=own_actor,
                root_factual_action=float(row.a[0]), scoring_uses_target_estimator=False))
    finally:
        _finish(learner, before)
    return result
