"""Cumulative optimizer counterfactuals on one unchanged factual feedback stream.

Fresh and historical-replacement learners retain their parameters, Adam moments,
sampler and private global RNG between packets. Their actions never reach the
environment. Every critic scores the exact same five executed candidates.
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import (
    FreshReference, _batch, _distribution, _equal, _probe, _restore, _rng, _state,
    _vector, compare_after_main)


CANDIDATES = ["initial", "fresh", "no_fresh", "zero", "7.5s"]
ARMS = ("fresh", "no_fresh")
PARAMETER_MODULES = ("actor", "critic", "target_critic", "value")


class CumulativeShadow:
    """Create after the first packet enters main replay, before its eight steps."""

    def __init__(self, origin, fixed_states, fixed_metadata):
        if origin.replay_buffer.arm != "fresh" or origin.world is not None:
            raise ValueError("v28 requires a fresh-origin learner without world updates")
        self.states = torch.as_tensor(np.asarray(fixed_states, dtype=np.float32),
                                      device=origin.device).clone()
        if len(self.states) != len(fixed_metadata) or not len(self.states):
            raise ValueError("Each real fixed state requires its own metadata")
        self.fixed_metadata = deepcopy(fixed_metadata)
        self.learners = {arm: deepcopy(origin) for arm in ARMS}
        self.start_update = origin.total_updates
        self.packets = 0
        self.private_rng = {arm: _rng() for arm in ARMS}
        self.initial_parameters = {name: _vector(getattr(origin, name)).clone()
                                   for name in PARAMETER_MODULES}
        for arm, learner in self.learners.items():
            learner.outcome_arm = learner.replay_buffer.arm = arm
            if arm == "no_fresh":
                replay = learner.replay_buffer
                replay.recent = deque(({key: row[key] for key in ("s", "a")}
                                       for row in replay.recent), maxlen=replay.recent.maxlen)
        with torch.no_grad():
            self.initial_proposal = origin.actor(self.states, deterministic=True)[0].clone()
        self.initial_probe_rows = self._scores()

    @torch.no_grad()
    def _scores(self):
        """Score a shared candidate grid without changing any training RNG."""
        proposals = [self.initial_proposal]
        proposals.extend(self.learners[arm].actor(self.states, deterministic=True)[0]
                         for arm in ARMS)
        proposals.extend((torch.full_like(proposals[0], -1.),
                          torch.full_like(proposals[0],
                              2.*7.5/self.learners["fresh"].action_transform.max_hold-1.)))
        proposals = torch.stack(proposals, dim=1)
        states = self.states[:, None, :].expand(-1, len(CANDIDATES), -1).reshape(-1, self.states.shape[1])
        actions = proposals.reshape(-1, 1)
        transforms = {arm: learner.action_transform(states, actions)
                      for arm, learner in self.learners.items()}
        if not _equal(transforms["fresh"], transforms["no_fresh"]):
            raise AssertionError("The shared candidates must pass the same F")
        transform = self.learners["fresh"].action_transform
        executed = transforms["fresh"].reshape(len(self.states), len(CANDIDATES))
        proposal_seconds = transform.hold_seconds(proposals).reshape_as(executed)
        holding = transform.hold_seconds(executed)
        q = {arm: learner.critic(states, transforms[arm]).reshape(-1, len(self.states), len(CANDIDATES))
             for arm, learner in self.learners.items()}
        rows = []
        for index, metadata in enumerate(self.fixed_metadata):
            row = dict(metadata)
            row.update(index=index, canonical_clock=float(self.states[index, 0])*24000.,
                       proposal_action=proposals[index, :, 0].tolist(),
                       proposal_seconds=proposal_seconds[index].tolist(),
                       executed_action=executed[index].tolist(),
                       holding_seconds=holding[index].tolist(),
                       q_mean={arm: values[:, index].mean(dim=0).tolist() for arm, values in q.items()},
                       q_min={arm: values[:, index].min(dim=0).values.tolist() for arm, values in q.items()})
            rows.append(row)
        return rows

    def advance(self, origin, packet, metadata=None):
        """Train the next two eight-step packets, restoring main state and RNG."""
        if not packet:
            raise ValueError("A nonempty admitted factual packet is required")
        live_rng, main_state = _rng(), deepcopy(_state(origin))
        main_replay = deepcopy(list(origin.replay_buffer.recent))
        packet_states = torch.as_tensor(np.asarray([row.s for row in packet]), device=origin.device)
        batch = _batch(packet, origin.device)
        before, after, sampling, covariates = {}, {}, {}, {}
        record = dict(schema=1, metadata={} if metadata is None else dict(metadata),
                      packet_index=self.packets+1, update_before=origin.total_updates,
                      candidates=CANDIDATES, arms={}, integrity={},
                      diagnostic_budget=dict(shadow_packets=1, shadow_optimizer_updates=16,
                                             training_labels=0, counterfactual_environment_queries=0))
        start_exact = _equal(_state(origin), _state(self.learners["fresh"]))
        start_rng_exact = _equal(live_rng, self.private_rng["fresh"])
        try:
            for arm, learner in self.learners.items():
                if self.packets:
                    learner.replay_buffer.add_target(packet)
                _restore(self.private_rng[arm])
                with torch.no_grad():
                    before_proposal = learner.actor(packet_states, deterministic=True)[0]
                before[arm] = _probe(learner, packet_states, batch, before_proposal, self.private_rng[arm])
                steps, arm_sampling, arm_covariates = [], [], []
                target_method = learner._target

                def target(batch, *, backup):
                    value = target_method(batch, backup=backup)
                    reward = batch["rewards"] * learner.config.reward_scale
                    quarter = learner.config.batch_size // 4
                    partitions = (("historical_base", 0, quarter),
                        ("fresh_outcome" if arm == "fresh" else "historical_replacement", quarter, 2*quarter),
                        ("source", 2*quarter, 4*quarter))
                    steps[-1]["batches"] = [dict(partition=name, n=end-start,
                        reward_scaled=_distribution(reward[start:end]),
                        bootstrap_contribution=_distribution((value-reward)[start:end]),
                        bellman_target=_distribution(value[start:end])) for name, start, end in partitions]
                    return value

                learner._target = target
                try:
                    for step in range(8):
                        steps.append(dict(step=step+1))
                        learner.continue_step()
                        arm_sampling.append(deepcopy(learner.replay_buffer.last_sampling))
                        arm_covariates.append({key: value.clone() for key, value in
                                               learner.replay_buffer.last_covariates.items()})
                finally:
                    del learner._target
                self.private_rng[arm] = _rng()
                after[arm] = _probe(learner, packet_states, batch, before_proposal, live_rng)
                changes = {}
                for name, initial in self.initial_parameters.items():
                    delta = _vector(getattr(learner, name))-initial
                    changes[name] = dict(l2=float(delta.norm()), max_absolute=float(delta.abs().max()))
                record["arms"][arm] = dict(updates=8, update_after=learner.total_updates,
                    training_steps=steps, parameter_change_from_initial=changes,
                    replay=learner.replay_buffer.summary())
                sampling[arm], covariates[arm] = arm_sampling, arm_covariates
            record["probe_rows"] = self._scores()
            if not self.packets:
                record["initial_probe_rows"] = self.initial_probe_rows
            record["packet_feedback"] = [dict(index=index, role="packet_feedback",
                line_id=getattr(row, "line_id", None), sequence=getattr(row, "sequence", None),
                canonical_clock=float(packet_states[index, 0])*24000.,
                dt_seconds=float(row.dt), terminal=bool(row.terminal),
                before={arm: before[arm][index] for arm in ARMS},
                after={arm: after[arm][index] for arm in ARMS}) for index, row in enumerate(packet)]
            self.packets += 1
            record.update(update_after=self.learners["fresh"].total_updates,
                          cumulative_updates=self.packets*8)
            _restore(live_rng)
            record["integrity"] = dict(main_start_learning_state_exact=start_exact,
                main_start_rng_exact=start_rng_exact,
                matched_sampling_draws=_equal(sampling["fresh"], sampling["no_fresh"]),
                matched_actor_value_covariates=_equal(covariates["fresh"], covariates["no_fresh"]),
                matched_private_rng=_equal(self.private_rng["fresh"], self.private_rng["no_fresh"]),
                no_fresh_outcomes_stored=all(set(row) == {"s", "a"}
                                           for row in self.learners["no_fresh"].replay_buffer.recent),
                origin_learning_state_unchanged=_equal(main_state, _state(origin)),
                origin_replay_unchanged=_equal(main_replay, list(origin.replay_buffer.recent)),
                global_rng_restored=_equal(live_rng, _rng()), common_candidate_execution=True)
            return record
        finally:
            _restore(live_rng)

    def compare_after_main(self, origin):
        """Verify the retained fresh shadow equals the actual main next eight steps."""
        reference = FreshReference(self.learners["fresh"], self.private_rng["fresh"])
        result = compare_after_main(origin, reference)
        result["checks"]["replay"] = _equal(list(origin.replay_buffer.recent),
                                           list(self.learners["fresh"].replay_buffer.recent))
        result["checks"]["feedback_count"] = (origin.replay_buffer.target_received ==
            self.learners["fresh"].replay_buffer.target_received)
        result["exact"] = all(result["checks"].values())
        return result

    def update_counts(self):
        return {arm: learner.total_updates-self.start_update
                for arm, learner in self.learners.items()}
