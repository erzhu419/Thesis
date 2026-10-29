"""Read-only v54 final models and issue-visible common physical actions.

The critics retain their long-horizon mixed-backup meaning. Comparing their
action ranking with a 300-second native cost difference is a finite-horizon
diagnostic, not a calibration of absolute Q against that cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import critic_states
from experiments.mcwm_closed_loop_joint_shadow_v47_20261024.score import _aliases
from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.score import (
    _snapshot, _finish, actor_candidates, load_bank, score_bank)
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import (
    freeze as freeze_modules, optimizer_steps, plain)
from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.learning import objects
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    _equal, _rng, _restore)


V54 = Path(__file__).resolve().parents[1]/'mcwm_closed_loop_physical_shadow_v54_20261025'
ARMS = ('one_step', 'trace')


def _read(path):
    return json.loads(Path(path).read_text())


def freeze(learner):
    freeze_modules(learner)
    # This actor supplied the training evaluator; it is never a new target here.
    if getattr(learner, 'sync_actor', None) is not None:
        learner.sync_actor.eval()
        for parameter in learner.sync_actor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    return learner


def _last_score(path):
    with Path(path).open() as handle:
        last = None
        for text in handle:
            if text.strip():
                last = json.loads(text)
    if last is None or last['packet'] != 54 or last['cumulative_updates'] != 456:
        raise RuntimeError('The v54 final packet54 score is missing')
    return last


def _final_prediction_gate(learners, scenario, root, protocol):
    """Re-score the saved factual bank, never read native counterfactual truth."""
    source = root.parent/protocol['source_version']/'runs'/scenario
    bank = load_bank(source, protocol)
    manifests = {arm: _read(root/'runs'/scenario/arm/'candidates_54.json')
                 for arm in ARMS}
    shared = {arm: manifests[arm]['candidates'] for arm in ARMS}
    result = {}
    for arm, learner in learners.items():
        if manifests[arm]['packet'] != 54:
            raise RuntimeError('The common final actor manifest differs')
        before, identities = optimizer_steps(learner), objects(learner)
        own = actor_candidates(learner, bank)
        if plain(own) != shared[arm]:
            raise RuntimeError('Frozen final actor/F manifest differs: '+arm)
        actual = score_bank(learner, bank, packet=54, cumulative_updates=456,
            processed_at=9630., shared_actor_proposals=shared,
            own_candidates=own, arm=arm)
        expected = _last_score(root/'runs'/scenario/arm/'scores.jsonl')
        if plain(actual) != expected['rows'] or len(actual) != 3:
            raise RuntimeError('Frozen final three-line full prediction differs: '+arm)
        if optimizer_steps(learner) != before or objects(learner) != identities:
            raise RuntimeError('Read-only old prediction scoring changed Adam')
        result[arm] = dict(all_pass=True, final_packet=54,
            final_three_line_prediction_rows=3, actor_and_F_exact=True,
            complete_common_candidate_heads_minQ_advantage_exact=True,
            optimizer_steps_and_objects_unchanged=True)
    return result


def build_frozen(scenario, *, source_root=None):
    """Load existing server-resident final checkpoints, with no optimizer work.

    Returns ``learners, contract, stations, proof``. All global RNG streams are
    restored, including the streams consumed while unpickling saved learners.
    """
    root, saved = Path(source_root or V54), _rng()
    try:
        protocol = _read(root/'protocol.json')
        if scenario not in protocol['scenarios'] or protocol['training_seed'] != 6202:
            raise ValueError('Frozen v54 scenario or initializer differs')
        learners, contracts, stations, arms = {}, {}, {}, {}
        for arm in ARMS:
            run = root/'runs'/scenario/arm
            summary = _read(run/'summary.json')
            if not (summary['status'] == 'complete' and summary['integrity']['all_pass']
                    and summary['scenario'] == scenario and summary['arm'] == arm
                    and summary['budget']['initializer_updates'] == 50000
                    and summary['budget']['feedback_slots'] == 285
                    and summary['budget']['optimizer_updates'] == 456
                    and summary['budget']['source_horizon_seconds'] == 24000):
                raise RuntimeError('The completed v54 budget/scenario gate differs: '+arm)
            payload = torch.load(run/'post456_model.pt', map_location='cpu', weights_only=False)
            learner = payload['learner']
            if not (payload['updates'] == 456 and payload['feedback_slots'] == 285
                    and learner.total_updates == 50456
                    and learner.replay_buffer.target_received == 285
                    and len(learner.replay_buffer.recent) == 285
                    and learner.world is None
                    and Path(payload['source_ready']).parent.name == scenario
                    and payload['protocol'] == protocol):
                raise RuntimeError('The frozen checkpoint is not v54 post456/285: '+arm)
            learners[arm] = freeze(learner)
            contracts[arm], stations[arm] = payload['contract'], payload['stations']
            arms[arm] = dict(original_feedback_slots=285, original_updates=456,
                new_optimizer_updates=0, optimizer_steps=optimizer_steps(learner),
                resident_checkpoint=str(run/'post456_model.pt'))
        if not (_equal(contracts['one_step'], contracts['trace'])
                and _equal(stations['one_step'], stations['trace'])):
            raise RuntimeError('Frozen arms use different physical contracts or stations')
        prediction = _final_prediction_gate(learners, scenario, root, protocol)
        proof = dict(all_pass=True, scenario=scenario, seed=6202, arms=arms,
            old_final_predictions=prediction, feedback_slots_per_arm=285,
            optimizer_updates_per_arm=456, new_optimizer_updates=0,
            new_feedback_slots=0, truth_used_for_training=False,
            target_installed=False, source_read_only=True)
        return learners, contracts['one_step'], stations['one_step'], proof
    finally:
        _restore(saved)


@torch.no_grad()
def executable_candidates(transform, state, holds=(0., 7.5)):
    """Invert h/8h through the original F; preserve unsupported/alias cases."""
    states = torch.as_tensor(np.asarray(state), dtype=torch.float32).reshape(1, 229)
    seconds = states.new_tensor(holds).reshape(-1, 1)
    repeated = states.repeat(len(holds), 1)
    direct = 2.*seconds/transform.max_hold-1.
    direct_reaches = transform.hold_seconds(transform(repeated, direct)) == seconds
    inverse = 2.*(8.*seconds)/transform.max_hold-1.
    proposals = torch.where(direct_reaches, direct, inverse)
    executed = transform(repeated, proposals)
    actual = transform.hold_seconds(executed)
    supported = (actual == seconds).reshape(-1).tolist()
    actions = executed.reshape(-1).tolist()
    groups = _aliases(actions)
    return dict(requested_holding_seconds=list(holds),
        proposal_actions=proposals.reshape(-1).tolist(), executed_actions=actions,
        holding_seconds=actual.reshape(-1).tolist(), supported=supported,
        alias_groups=groups, physically_distinct=len(set(actions)) == len(holds)
            and all(supported))


@torch.no_grad()
def score_frozen(learners, state, memory, executed_actions):
    """Score already executed actions once, with the actual issue memory.

    No target, replay admission or optimizer method is called. A fixed action
    is sent to Q directly; F is used only for each actor's own proposal.
    """
    result = {}
    for arm, learner in learners.items():
        before, adam, identities = _snapshot(learner), optimizer_steps(learner), objects(learner)
        try:
            states = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                device=learner.device).reshape(1, 229)
            vector = memory['vector'] if isinstance(memory, dict) else memory
            context = torch.as_tensor(np.asarray(vector), dtype=states.dtype,
                device=learner.device).reshape(1, 84)
            actions = states.new_tensor(executed_actions).reshape(-1, 1)
            aliases = _aliases(actions.reshape(-1).tolist())
            representatives = [aliases.index(i) for i in range(max(aliases)+1)]
            explicit = critic_states(learner, states, context)
            unique = learner.critic(explicit.repeat(len(representatives), 1),
                actions[representatives]).reshape(-1, len(representatives))
            q = unique[:, aliases]
            minimum, mean = q.min(0).values, q.mean(0)
            proposal = learner.actor(states, deterministic=True)[0]
            own_action = learner.action_transform(states, proposal)
            result[arm] = dict(q_heads=q.tolist(), q_min=minimum.tolist(), q_mean=mean.tolist(),
                advantage_min=(minimum-minimum[0]).tolist(),
                advantage_mean=(mean-mean[0]).tolist(), min_head=q.argmin(0).tolist(),
                predicted_choice=int(minimum.argmax()), executed_actions=actions[:, 0].tolist(),
                holding_seconds=learner.action_transform.hold_seconds(actions[:, 0]).tolist(),
                alias_groups=aliases, unique_executed_candidates=len(representatives),
                actor_proposal_action=float(proposal[0, 0]),
                actor_proposal_holding_seconds=float(learner.action_transform.hold_seconds(proposal[0, 0])),
                actor_executed_action=float(own_action[0, 0]),
                actor_holding_seconds=float(learner.action_transform.hold_seconds(own_action[0, 0])),
                scoring_uses_target_estimator=False)
            if optimizer_steps(learner) != adam or objects(learner) != identities:
                raise RuntimeError('Frozen issue scoring changed Adam')
        finally:
            _finish(learner, before)
    if len({tuple(item['holding_seconds']) for item in result.values()}) != 1:
        raise RuntimeError('The frozen critics use different physical action units')
    return result
