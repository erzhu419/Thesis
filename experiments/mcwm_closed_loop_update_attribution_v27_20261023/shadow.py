"""Paired eight-step optimizer interventions from one fresh-origin packet.

Only the critic's final quarter-batch changes. The two clones inherit the same
parameters, Adam moments, replay covariates and sampler/global RNG state. Probes
never enter replay, and every diagnostic RNG draw is restored before training.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch

from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import FIELDS


MODULES = ("actor", "critic", "target_critic", "value", "ratio")
OPTIMIZERS = ("actor_optimizer", "critic_optimizer", "value_optimizer", "alpha_optimizer")


def _rng():
    return dict(torch=torch.get_rng_state().clone(), numpy=deepcopy(np.random.get_state()),
                python=random.getstate())


def _restore(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return left == right


def _state(learner):
    modules = {name: getattr(learner, name).state_dict() for name in MODULES
               if getattr(learner, name, None) is not None}
    optimizers = {name: getattr(learner, name).state_dict() for name in OPTIMIZERS
                  if getattr(learner, name, None) is not None}
    return dict(modules=modules, optimizers=optimizers, log_alpha=learner.log_alpha,
                updates=learner.total_updates, online_updates=learner.online_critic_updates,
                sampler=learner.rng.bit_generator.state,
                actor_batches=learner._online_actor_state_batches,
                temperature_batches=learner._online_temperature_state_batches)


def _vector(module):
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _distribution(value):
    value = value.detach().double().reshape(-1)
    return dict(n=value.numel(), mean=float(value.mean()), min=float(value.min()),
                max=float(value.max()), std=float(value.std(unbiased=False)))


def _batch(rows, device):
    batch = {key: torch.as_tensor(np.asarray([getattr(row, field) for row in rows]),
                                 dtype=torch.float32, device=device)
             for key, field in FIELDS.items()}
    batch["actions"] = batch["actions"].reshape(-1, 1)
    for key in ("rewards", "delta_t_seconds", "terminals"):
        batch[key] = batch[key].reshape(-1)
    batch["discounts"] = torch.ones(len(rows), device=device)
    return batch


@torch.no_grad()
def _probe(learner, states, batch, before_proposal, diagnostic_rng):
    """Probe the same states/actions; never consume training randomness."""
    live_rng = _rng()
    try:
        _restore(diagnostic_rng)
        proposal = learner.actor(states, deterministic=True)[0]
        executed = learner.action_transform(states, proposal)
        old_executed = learner.action_transform(states, before_proposal)
        old_q = learner.critic(states, old_executed)
        new_q = learner.critic(states, executed)
        target = learner._target(batch, backup=learner.config.online_backup)
        reward = batch["rewards"] * learner.config.reward_scale
        bootstrap = target - reward
        observed_q = learner.critic(batch["observations"], batch["actions"])
        rows = []
        for index in range(len(states)):
            has_transition = index < len(target)
            row = dict(reward_scaled=float(reward[index]) if has_transition else None,
                       bootstrap_contribution=float(bootstrap[index]) if has_transition else None,
                       bellman_target=float(target[index]) if has_transition else None,
                       q_observed_mean=float(observed_q[:, index].mean()) if has_transition else None,
                       q_observed_min=float(observed_q[:, index].min()) if has_transition else None,
                       proposal_action=float(proposal[index, 0]),
                       proposal_seconds=float(learner.action_transform.hold_seconds(proposal[index, 0])),
                       executed_action=float(executed[index, 0]),
                       holding_seconds=float(learner.action_transform.hold_seconds(executed[index, 0])),
                       q_on_before_actor_mean=float(old_q[:, index].mean()),
                       q_on_before_actor_min=float(old_q[:, index].min()),
                       q_on_post_actor_mean=float(new_q[:, index].mean()),
                       q_on_post_actor_min=float(new_q[:, index].min()))
            rows.append(row)
        return rows
    finally:
        _restore(live_rng)


@dataclass
class FreshReference:
    learner: object
    global_rng: dict


def compare_after_main(origin, reference):
    """Call immediately after the real packet's eight updates, before action draws."""
    actual, expected = _state(origin), _state(reference.learner)
    checks = {name: _equal(actual[name], expected[name]) for name in actual}
    checks["global_rng"] = _equal(_rng(), reference.global_rng)
    return dict(exact=all(checks.values()), checks=checks,
                update_after=origin.total_updates)


def run_shadow_packet(origin, packet, metadata=None, *, probe_states=None, probe_metadata=None):
    """Run two copies for eight updates and return JSON record plus fresh reference.

    ``packet`` has already entered the main replay. Extra real-state probes are
    evaluation only, have no transition labels, and retain their physical clock.
    Both counterfactual treatments start from the fresh-origin optimizer state;
    no-fresh is a local intervention, not the old independently evolved arm.
    """
    if origin.replay_buffer.arm != "fresh" or origin.world is not None:
        raise ValueError("v27 starts from a fresh-origin factual learner without world updates")
    if not packet:
        raise ValueError("a packet of factual probes is required")
    start_rng = _rng()
    origin_state = deepcopy(_state(origin))
    origin_replay = deepcopy(list(origin.replay_buffer.recent))
    states = np.asarray([row.s for row in packet], dtype=np.float32)
    extras = [] if probe_states is None else np.asarray(probe_states, dtype=np.float32)
    if len(extras):
        states = np.concatenate((states, extras.reshape(-1, states.shape[1])))
    extra_metadata = [{} for _ in extras] if probe_metadata is None else probe_metadata
    if len(extra_metadata) != len(extras):
        raise ValueError("extra probe metadata must identify each extra state")
    states = torch.as_tensor(states, device=origin.device)
    batch = _batch(packet, origin.device)
    before_parameters = {name: _vector(getattr(origin, name)).clone()
                         for name in ("actor", "critic", "target_critic", "value")}
    with torch.no_grad():
        before_proposal = origin.actor(states, deterministic=True)[0]
    before = _probe(origin, states, batch, before_proposal, start_rng)
    record = dict(schema=1, metadata={} if metadata is None else dict(metadata),
                  origin="fresh", update_before=origin.total_updates,
                  probe_rows=[], arms={}, integrity={},
                  diagnostic_budget=dict(shadow_packets=1, shadow_optimizer_updates=16,
                                         counterfactual_environment_queries=0, training_labels=0))
    for index, row in enumerate(packet):
        record["probe_rows"].append(dict(index=index, role="packet_feedback", has_transition=True,
            line_id=getattr(row, "line_id", None), station_id=getattr(row, "station_id", None),
            sequence=getattr(row, "sequence", None),
            issue_time=float(getattr(row, "started_at", float(states[index, 0])*24000.)),
            canonical_clock=float(states[index, 0])*24000.,
            dt_seconds=float(row.dt), terminal=bool(row.terminal), before=before[index]))
    for index, values in enumerate(extra_metadata, len(packet)):
        extra_row = dict(index=index, role="heldout_state", has_transition=False,
            line_id=None, station_id=None, sequence=None,
            issue_time=float(states[index, 0])*24000., dt_seconds=None, terminal=None,
            before=before[index])
        extra_row.update({key: value for key, value in values.items() if key not in (
            "index", "has_transition", "dt_seconds", "terminal", "issue_time", "before")})
        record["probe_rows"].append(extra_row)
    clones, sampling, covariates = {}, {}, {}
    try:
        for arm in ("fresh", "no_fresh"):
            _restore(start_rng)
            clone = deepcopy(origin)
            clone.outcome_arm = clone.replay_buffer.arm = arm
            if arm == "no_fresh":
                clone.replay_buffer.recent = deque(
                    ({key: row[key] for key in ("s", "a")} for row in clone.replay_buffer.recent),
                    maxlen=clone.replay_buffer.recent.maxlen)
            steps, arm_sampling, arm_covariates = [], [], []
            target_method = clone._target
            def target(batch, *, backup):
                result = target_method(batch, backup=backup)
                reward = batch["rewards"] * clone.config.reward_scale
                quarter = clone.config.batch_size // 4
                partitions = (("historical_base", 0, quarter),
                    ("fresh_outcome" if arm == "fresh" else "historical_replacement", quarter, 2*quarter),
                    ("source", 2*quarter, 4*quarter))
                steps[-1]["batches"] = [dict(partition=name, n=end-start,
                    reward_scaled=_distribution(reward[start:end]),
                    bootstrap_contribution=_distribution((result-reward)[start:end]),
                    bellman_target=_distribution(result[start:end])) for name, start, end in partitions]
                return result
            clone._target = target
            try:
                for step in range(8):
                    steps.append(dict(step=step+1))
                    clone.continue_step()
                    arm_sampling.append(deepcopy(clone.replay_buffer.last_sampling))
                    arm_covariates.append({key: value.clone() for key, value in
                                           clone.replay_buffer.last_covariates.items()})
            finally:
                del clone._target
            post_rng = _rng()
            after = _probe(clone, states, batch, before_proposal, start_rng)
            for row, values in zip(record["probe_rows"], after):
                row[arm] = values
            changes = {}
            for name, initial in before_parameters.items():
                change = _vector(getattr(clone, name))-initial
                changes[name] = dict(l2=float(change.norm()), max_absolute=float(change.abs().max()))
            record["arms"][arm] = dict(updates=8, update_after=clone.total_updates,
                training_steps=steps, parameter_change=changes, replay=clone.replay_buffer.summary())
            clones[arm] = FreshReference(clone, post_rng)
            sampling[arm], covariates[arm] = arm_sampling, arm_covariates
        record["integrity"] = dict(matched_sampling_draws=_equal(sampling["fresh"], sampling["no_fresh"]),
            matched_actor_value_covariates=_equal(covariates["fresh"], covariates["no_fresh"]),
            no_fresh_outcomes_stored=all(set(row) == {"s", "a"} for row in
                                        clones["no_fresh"].learner.replay_buffer.recent),
            main_update_unchanged=origin.total_updates == record["update_before"],
            origin_learning_state_unchanged=_equal(origin_state, _state(origin)),
            origin_replay_unchanged=_equal(origin_replay, list(origin.replay_buffer.recent)))
        _restore(start_rng)
        record["integrity"]["global_rng_restored"] = _equal(start_rng, _rng())
        return record, clones["fresh"]
    finally:
        _restore(start_rng)
